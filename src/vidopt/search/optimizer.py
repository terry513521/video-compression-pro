"""The parameter search.

For one segment and one VMAF target, find the encoder parameters that minimise output
size subject to ``VMAF >= target`` — the constrained form of the objective, justified in
DESIGN.md section 2.

Two stages:

**A. Exploration.** ``n_explore`` Sobol points over ``(crf, aq_mode, aq_strength)``.
This is what learns which AQ setting suits this content — the reference never varied
``aq-mode`` or ``aq-strength`` at all, despite naming them as the parameters of interest.

**B. Refinement.** VMAF is monotonically non-increasing in CRF at fixed AQ, so for the
best AQ settings found in stage A the highest CRF meeting the target is located by
bisection, accelerated with a secant step: the two bracketing measurements give a local
linear model of ``VMAF(crf)`` whose root is a much better probe than the midpoint. The
secant guess is clamped into the bracket's interior, so the method keeps bisection's
guaranteed convergence while usually beating its rate.

A linear sweep of 8 CRF values (the reference's approach) costs 8 encodes and resolves
CRF only to the step size. This reaches +/-0.5 CRF in about 5.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config
from ..encoding.encoders import (
    Encoder,
    EncodeRequest,
    container_for,
    encode,
    resolve_pix_fmt,
)
from ..encoding.params import EncodeParams
from ..errors import EncodeError, SearchError, VmafError
from ..ffmpeg.probe import MediaInfo
from ..ffmpeg.toolchain import Capabilities
from ..log import get_logger
from ..quality import vmaf as vmaf_mod
from ..scoring import ScoreBreakdown, compression_score, is_feasible
from .cache import NullCache, TrialCache, TrialRecord
from .samplers import get_sampler

log = get_logger(__name__)


@dataclass
class SearchResult:
    """Outcome of searching one (segment, target) pair."""

    segment_path: str
    segment_hash: str
    target: float
    best: TrialRecord | None
    feasible: bool
    n_trials: int
    n_encodes: int
    trials: list[TrialRecord] = field(default_factory=list)

    @property
    def score(self) -> float:
        if self.best is None:
            return 0.0
        return compression_score(self.best.vmaf, self.best.rate, self.target).score

    def breakdown(self) -> ScoreBreakdown | None:
        if self.best is None:
            return None
        return compression_score(self.best.vmaf, self.best.rate, self.target)


class Evaluator:
    """Encode + measure one parameter point, with caching.

    Trials are cached independently of the VMAF target, so all targets for a segment
    share the same measurements.
    """

    def __init__(
        self,
        *,
        caps: Capabilities,
        config: Config,
        encoder: Encoder,
        cache: TrialCache | NullCache,
        work_dir: Path,
    ) -> None:
        self.caps = caps
        self.config = config
        self.encoder = encoder
        self.cache = cache
        self.work_dir = work_dir
        self.n_encodes = 0

    def evaluate(
        self, segment_path: Path, segment_hash: str, info: MediaInfo, params: EncodeParams
    ) -> TrialRecord:
        params = self.encoder.space.clamp(params)
        subsample = self.config.vmaf.n_subsample_search

        cached = self.cache.get(
            segment_hash, self.encoder.name, params, self.config.vmaf.model, subsample
        )
        if cached is not None:
            log.debug("cache hit %s %s", Path(segment_path).name, params)
            return cached

        ref_bytes = Path(segment_path).stat().st_size

        # Resolve once per segment. Dev and production must agree on this, or the
        # parameters learned here describe a different encode than the one deployed.
        pix_fmt = resolve_pix_fmt(
            self.caps.ffmpeg, self.encoder, self.config.encoder.pix_fmt,
            info.pix_fmt, context=Path(segment_path).name,
        )

        # ignore_cleanup_errors: on Windows a virus scanner or a lingering handle can keep
        # a just-written file locked for a moment. A trial directory that fails to
        # delete is litter, not a reason to abort a multi-hour search.
        with tempfile.TemporaryDirectory(
            dir=self.work_dir, prefix="trial_", ignore_cleanup_errors=True
        ) as tmp:
            out_path = Path(tmp) / f"trial.{container_for(self.encoder.name)}"
            request = EncodeRequest(
                input_path=str(segment_path),
                output_path=str(out_path),
                params=params,
                preset=self.config.encoder.preset,
                keyint_seconds=self.config.encoder.keyint_seconds,
                pix_fmt=pix_fmt,
                fps=info.fps,
                threads=self.config.ffmpeg.threads,
                loglevel=self.config.ffmpeg.loglevel,
                extra_args=tuple(self.config.encoder.extra_args),
            )

            try:
                encoded = encode(self.caps.ffmpeg, self.encoder, request)
            except Exception as exc:
                raise EncodeError(
                    f"trial encode failed for {Path(segment_path).name} "
                    f"at {params}: {exc}"
                ) from exc
            self.n_encodes += 1

            try:
                quality = vmaf_mod.measure(
                    out_path, segment_path, self.caps, self.config.vmaf,
                    n_subsample=subsample,
                    expected_frames=info.n_frames,
                )
            except VmafError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise VmafError(
                    f"VMAF failed for {Path(segment_path).name} at {params}: {exc}"
                ) from exc

            record = TrialRecord(
                segment_hash=segment_hash,
                encoder=self.encoder.name,
                params=params,
                vmaf_model=self.config.vmaf.model,
                n_subsample=subsample,
                ref_bytes=ref_bytes,
                out_bytes=encoded.size_bytes,
                vmaf=quality.score,
                encode_seconds=encoded.seconds,
                vmaf_seconds=quality.seconds,
            )

        self.cache.put(record, extra={"segment": str(segment_path)})
        log.debug(
            "trial %s %s -> vmaf %.2f rate %.4f",
            Path(segment_path).name, params, record.vmaf, record.rate,
        )
        return record


def _aq_key(params: EncodeParams) -> tuple[int, float]:
    p = params.rounded()
    return (p.aq_mode, p.aq_strength)


def _bisect_crf(
    evaluator: Evaluator,
    segment_path: Path,
    segment_hash: str,
    info: MediaInfo,
    aq_mode: int,
    aq_strength: float,
    target: float,
    known: list[TrialRecord],
    config: Config,
) -> list[TrialRecord]:
    """Find the highest CRF meeting ``target`` at fixed AQ. Returns the new trials."""
    space = evaluator.encoder.space
    tolerance = config.search.crf_tolerance
    new: list[TrialRecord] = []

    def probe(crf: float) -> TrialRecord:
        record = evaluator.evaluate(
            segment_path, segment_hash, info,
            EncodeParams(crf=crf, aq_mode=aq_mode, aq_strength=aq_strength),
        )
        new.append(record)
        return record

    # Seed the bracket from measurements we already have at this AQ setting.
    samples: dict[float, float] = {t.params.crf: t.vmaf for t in known}

    def bracket() -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        """Highest feasible (crf, vmaf), and lowest infeasible (crf, vmaf) above it."""
        feasible = [(c, v) for c, v in samples.items() if v >= target]
        low = max(feasible, key=lambda cv: cv[0]) if feasible else None
        above = [
            (c, v)
            for c, v in samples.items()
            if v < target and (low is None or c > low[0])
        ]
        high = min(above, key=lambda cv: cv[0]) if above else None
        return low, high

    low, high = bracket()

    # No feasible point known: try the most conservative CRF the encoder allows.
    if low is None:
        record = probe(space.crf_min)
        samples[record.params.crf] = record.vmaf
        if record.vmaf < target:
            # Even maximum quality misses the target. Nothing to bisect.
            return new
        low, high = bracket()

    # No infeasible point known: try the most aggressive CRF.
    if high is None:
        record = probe(space.crf_max)
        samples[record.params.crf] = record.vmaf
        if record.vmaf >= target:
            # The whole range is feasible; crf_max is optimal.
            return new
        low, high = bracket()

    if low is None or high is None:  # pragma: no cover - bracket() guarantees both here
        return new

    for _ in range(config.search.max_bisect_iters):
        (crf_lo, vmaf_lo), (crf_hi, vmaf_hi) = low, high
        if crf_hi - crf_lo <= tolerance:
            break

        midpoint = 0.5 * (crf_lo + crf_hi)
        if vmaf_lo > vmaf_hi:
            # Secant step: root of the line through the two bracketing measurements.
            guess = crf_lo + (vmaf_lo - target) * (crf_hi - crf_lo) / (vmaf_lo - vmaf_hi)
        else:
            guess = midpoint

        # Keep the probe inside the middle 60% of the bracket. This preserves
        # bisection's convergence guarantee when the secant model is poor.
        margin = 0.2 * (crf_hi - crf_lo)
        guess = min(max(guess, crf_lo + margin), crf_hi - margin)

        record = probe(guess)
        crf, achieved = record.params.crf, record.vmaf
        samples[crf] = achieved

        if crf <= crf_lo or crf >= crf_hi:
            # Quantisation collapsed the probe onto a bracket edge: no progress possible.
            break
        if achieved >= target:
            low = (crf, achieved)
        else:
            high = (crf, achieved)

    return new


def search_segment(
    segment_path: str | Path,
    segment_hash: str,
    info: MediaInfo,
    targets: list[float],
    *,
    caps: Capabilities,
    config: Config,
    encoder: Encoder,
    cache: TrialCache | NullCache,
    work_dir: str | Path,
) -> dict[float, SearchResult]:
    """Search one segment for every VMAF target.

    All targets share the exploration stage and the trial cache, so the marginal cost of
    an extra target is only its own refinement trials.

    Returns:
        Mapping of target -> :class:`SearchResult`.
    """
    segment_path = Path(segment_path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    evaluator = Evaluator(
        caps=caps, config=config, encoder=encoder, cache=cache, work_dir=work_dir
    )
    sampler = get_sampler(config.search.sampler)

    # ---- Stage A: exploration (shared by every target) ----
    explore_points = sampler(
        encoder.space, config.search.n_explore, config.search.seed
    )
    all_trials: list[TrialRecord] = []
    for params in explore_points:
        try:
            all_trials.append(
                evaluator.evaluate(segment_path, segment_hash, info, params)
            )
        except (EncodeError, VmafError) as exc:
            # One bad point should not abort a segment; a total failure is caught below.
            log.warning("trial %s failed on %s: %s", params, segment_path.name, exc)

    if not all_trials:
        raise SearchError(
            f"every exploration trial failed for {segment_path.name}; "
            "check the encoder and VMAF toolchain"
        )

    log.info(
        "%s: explored %d point(s), vmaf %.1f-%.1f",
        segment_path.name,
        len(all_trials),
        min(t.vmaf for t in all_trials),
        max(t.vmaf for t in all_trials),
    )

    # ---- Stage B: refinement, per target ----
    results: dict[float, SearchResult] = {}
    for target in targets:
        # Rank AQ settings by the best score any of their trials achieved, so refinement
        # is spent on the settings that look most promising for *this* target.
        by_aq: dict[tuple[int, float], float] = {}
        for trial in all_trials:
            key = _aq_key(trial.params)
            score = compression_score(trial.vmaf, trial.rate, target).score
            # A config whose every trial scores zero is still worth ranking, by how close
            # it got to the target -- otherwise hard segments have nothing to refine.
            proxy = score if score > 0 else 1e-6 * min(trial.vmaf, target)
            by_aq[key] = max(by_aq.get(key, 0.0), proxy)

        ranked = sorted(by_aq, key=lambda k: by_aq[k], reverse=True)
        for aq_mode, aq_strength in ranked[: config.search.n_refine_configs]:
            same_aq = [
                t for t in all_trials if _aq_key(t.params) == (aq_mode, aq_strength)
            ]
            try:
                refined = _bisect_crf(
                    evaluator, segment_path, segment_hash, info,
                    aq_mode, aq_strength, target, same_aq, config,
                )
            except (EncodeError, VmafError) as exc:
                log.warning(
                    "refinement failed for %s aq=(%d,%g): %s",
                    segment_path.name, aq_mode, aq_strength, exc,
                )
                continue
            all_trials.extend(refined)

        feasible_trials = [
            t for t in all_trials if is_feasible(t.vmaf, t.rate, target)
        ]
        if feasible_trials:
            best = max(
                feasible_trials,
                key=lambda t: compression_score(t.vmaf, t.rate, target).score,
            )
            feasible = True
        else:
            # Nothing met the target. Keep the closest attempt for the report, but mark
            # the sample infeasible so it is excluded from training rather than teaching
            # the model to aim at something unreachable.
            best = max(all_trials, key=lambda t: t.vmaf)
            feasible = False
            log.warning(
                "%s: no parameters reached VMAF %.0f (best %.2f)",
                segment_path.name, target, best.vmaf,
            )

        results[target] = SearchResult(
            segment_path=str(segment_path),
            segment_hash=segment_hash,
            target=target,
            best=best,
            feasible=feasible,
            n_trials=len(all_trials),
            n_encodes=evaluator.n_encodes,
            trials=list(all_trials),
        )
        log.info(
            "%s target %.0f: %s vmaf=%.2f ratio=%.2fx score=%.4f%s",
            segment_path.name, target, best.params, best.vmaf, best.ratio,
            results[target].score, "" if feasible else "  [INFEASIBLE]",
        )

    return results
