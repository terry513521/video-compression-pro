"""The parameter search.

For one segment and one VMAF target, find the encoder parameters that minimise output
size subject to ``VMAF >= target`` — the constrained form of the objective, justified in
DESIGN.md section 2.

Strategies (``search.strategy``):

**aq_then_crf (default).** Enumerate AQ, screen at a few CRFs, then 1-D CRF solve.
**coordinate.** Screen AQ, then walk 4-neighbours while re-solving CRF.
**sample.** 3-D space-filling design (``search.sampler``), then CRF solve on the best AQ.
**bayes.** Sobol/LHS init, then Gaussian-process proposals (P(VMAF≥target)×CRF).
**tpe.** Parzen estimator: sample where good trials were denser than bad ones.
**cmaes.** Diagonal CMA-style evolution strategy on the unit cube.

CRF at fixed AQ (``search.crf_solver``): ``bisect`` (secant+bisection), ``brent``
(inverse-quadratic interpolation), or ``golden`` (golden-section split).
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
from .adaptive import explore_adaptive
from .cache import NullCache, TrialCache, TrialRecord
from .samplers import SAMPLERS, get_sampler

log = get_logger(__name__)

STRATEGIES = ("aq_then_crf", "coordinate", "sample", "bayes", "tpe", "cmaes")
CRF_SOLVERS = ("bisect", "brent", "golden")
ADAPTIVE = {"bayes", "tpe", "cmaes"}
_PHI = (5.0 ** 0.5 - 1.0) / 2.0  # 1/φ ≈ 0.618


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
    top_candidates: list[TrialRecord] = field(default_factory=list)

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


def _trial_proxy(trial: TrialRecord, target: float) -> float:
    """Rank key: real objective if feasible-ish, else closeness of VMAF to the target."""
    score = compression_score(trial.vmaf, trial.rate, target).score
    if score > 0:
        return score
    return 1e-6 * min(trial.vmaf, target)


def _interior_crfs(space, n: int) -> list[float]:
    """CRF probes strictly inside [min, max] so bisection still owns the endpoints."""
    n = max(1, int(n))
    lo, hi = space.crf_min, space.crf_max
    if n == 1 or hi <= lo:
        return [0.5 * (lo + hi)]
    return [lo + (hi - lo) * (i + 1) / (n + 1) for i in range(n)]


def _subsample_even(items: list, keep: int) -> list:
    if keep >= len(items) or keep <= 0:
        return list(items)
    if keep == 1:
        return [items[len(items) // 2]]
    return [
        items[round(i * (len(items) - 1) / (keep - 1))]
        for i in range(keep)
    ]


def _screen_plan(space, config: Config) -> tuple[list[tuple[int, float]], list[float]]:
    """AQ settings × screen CRFs, clipped to about ``2 * n_explore`` trials."""
    aq = space.aq_grid(config.search.n_strength_steps)
    n_crf = max(1, min(int(config.search.n_screen_crfs), 4))
    budget = max(8, int(config.search.n_explore)) * 2
    while aq and len(aq) * n_crf > budget:
        if n_crf > 1:
            n_crf -= 1
            continue
        aq = _subsample_even(aq, max(4, budget))
        break
    return aq, _interior_crfs(space, n_crf)


def _try_evaluate(
    evaluator: Evaluator,
    segment_path: Path,
    segment_hash: str,
    info: MediaInfo,
    params: EncodeParams,
) -> TrialRecord | None:
    try:
        return evaluator.evaluate(segment_path, segment_hash, info, params)
    except (EncodeError, VmafError) as exc:
        log.warning("trial %s failed on %s: %s", params, segment_path.name, exc)
        return None


def _explore_sample(
    evaluator: Evaluator,
    segment_path: Path,
    segment_hash: str,
    info: MediaInfo,
    encoder: Encoder,
    config: Config,
) -> list[TrialRecord]:
    sampler = get_sampler(config.search.sampler)
    points = sampler(encoder.space, config.search.n_explore, config.search.seed)
    trials: list[TrialRecord] = []
    for params in points:
        record = _try_evaluate(evaluator, segment_path, segment_hash, info, params)
        if record is not None:
            trials.append(record)
    return trials


def _explore_aq_screen(
    evaluator: Evaluator,
    segment_path: Path,
    segment_hash: str,
    info: MediaInfo,
    encoder: Encoder,
    config: Config,
) -> list[TrialRecord]:
    aq_settings, crfs = _screen_plan(encoder.space, config)
    log.info(
        "%s: screening %d AQ setting(s) x %d CRF(s)",
        segment_path.name, len(aq_settings), len(crfs),
    )
    trials: list[TrialRecord] = []
    for aq_mode, aq_strength in aq_settings:
        for crf in crfs:
            record = _try_evaluate(
                evaluator, segment_path, segment_hash, info,
                EncodeParams(crf=crf, aq_mode=aq_mode, aq_strength=aq_strength),
            )
            if record is not None:
                trials.append(record)
    return trials


def _rank_aq(trials: list[TrialRecord], target: float) -> list[tuple[int, float]]:
    by_aq: dict[tuple[int, float], float] = {}
    for trial in trials:
        key = _aq_key(trial.params)
        by_aq[key] = max(by_aq.get(key, 0.0), _trial_proxy(trial, target))
    return sorted(by_aq, key=lambda k: by_aq[k], reverse=True)


def _refine_aq(
    evaluator: Evaluator,
    segment_path: Path,
    segment_hash: str,
    info: MediaInfo,
    aq_mode: int,
    aq_strength: float,
    target: float,
    all_trials: list[TrialRecord],
    config: Config,
) -> None:
    same_aq = [t for t in all_trials if _aq_key(t.params) == (aq_mode, aq_strength)]
    try:
        refined = _solve_crf(
            evaluator, segment_path, segment_hash, info,
            aq_mode, aq_strength, target, same_aq, config,
        )
    except (EncodeError, VmafError) as exc:
        log.warning(
            "refinement failed for %s aq=(%d,%g): %s",
            segment_path.name, aq_mode, aq_strength, exc,
        )
        return
    all_trials.extend(refined)


def _walk_aq(
    evaluator: Evaluator,
    segment_path: Path,
    segment_hash: str,
    info: MediaInfo,
    encoder: Encoder,
    target: float,
    all_trials: list[TrialRecord],
    config: Config,
) -> None:
    """Coordinate descent on the AQ grid at the current best CRF."""
    space = encoder.space
    ranked = _rank_aq(all_trials, target)
    if not ranked:
        mid = space.midpoint()
        current = (mid.aq_mode, mid.aq_strength)
    else:
        current = ranked[0]
    visited: set[tuple[int, float]] = set()
    steps = max(1, int(config.search.n_strength_steps))

    for _ in range(max(1, int(config.search.max_coordinate_rounds))):
        if current in visited:
            break
        visited.add(current)
        _refine_aq(
            evaluator, segment_path, segment_hash, info,
            current[0], current[1], target, all_trials, config,
        )
        local = [t for t in all_trials if _aq_key(t.params) == current]
        if not local:
            break
        best_local = max(local, key=lambda t: _trial_proxy(t, target))
        current_score = _trial_proxy(best_local, target)

        best_nb: tuple[int, float] | None = None
        best_nb_score = current_score
        for nb in space.aq_neighbors(current[0], current[1], steps):
            record = _try_evaluate(
                evaluator, segment_path, segment_hash, info,
                EncodeParams(
                    crf=best_local.params.crf,
                    aq_mode=nb[0],
                    aq_strength=nb[1],
                ),
            )
            if record is None:
                continue
            all_trials.append(record)
            score = _trial_proxy(record, target)
            if score > best_nb_score:
                best_nb = nb
                best_nb_score = score
        if best_nb is None:
            break
        current = best_nb


def _iq_interpolate(points: list[tuple[float, float]], target: float) -> float | None:
    """Inverse quadratic: treat CRF as a function of VMAF, evaluate at ``target``."""
    if len(points) < 3:
        return None
    (x0, y0), (x1, y1), (x2, y2) = points[-3:]
    # x = CRF, y = VMAF. Interpolate x(y) at y=target.
    if len({y0, y1, y2}) < 3:
        return None
    t = target
    try:
        a = (t - y1) * (t - y2) / ((y0 - y1) * (y0 - y2))
        b = (t - y0) * (t - y2) / ((y1 - y0) * (y1 - y2))
        c = (t - y0) * (t - y1) / ((y2 - y0) * (y2 - y1))
        return a * x0 + b * x1 + c * x2
    except ZeroDivisionError:
        return None


def _crf_guess(
    method: str,
    crf_lo: float,
    vmaf_lo: float,
    crf_hi: float,
    vmaf_hi: float,
    history: list[tuple[float, float]],
    target: float,
) -> float:
    midpoint = 0.5 * (crf_lo + crf_hi)
    if method == "golden":
        return crf_lo + (1.0 - _PHI) * (crf_hi - crf_lo)
    if method == "brent":
        iq = _iq_interpolate(history, target)
        if iq is not None:
            return iq
    if vmaf_lo > vmaf_hi:
        return crf_lo + (vmaf_lo - target) * (crf_hi - crf_lo) / (vmaf_lo - vmaf_hi)
    return midpoint


def _solve_crf(
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
    method = str(config.search.crf_solver).strip().lower()
    if method not in CRF_SOLVERS:
        raise SearchError(
            f"unknown search.crf_solver {config.search.crf_solver!r}; "
            f"available: {list(CRF_SOLVERS)}"
        )
    new: list[TrialRecord] = []

    def probe(crf: float) -> TrialRecord:
        record = evaluator.evaluate(
            segment_path, segment_hash, info,
            EncodeParams(crf=crf, aq_mode=aq_mode, aq_strength=aq_strength),
        )
        new.append(record)
        return record

    samples: dict[float, float] = {t.params.crf: t.vmaf for t in known}

    def bracket() -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
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

    if low is None:
        record = probe(space.crf_min)
        samples[record.params.crf] = record.vmaf
        if record.vmaf < target:
            return new
        low, high = bracket()

    if high is None:
        record = probe(space.crf_max)
        samples[record.params.crf] = record.vmaf
        if record.vmaf >= target:
            return new
        low, high = bracket()

    if low is None or high is None:  # pragma: no cover
        return new

    history: list[tuple[float, float]] = [low, high]
    for _ in range(config.search.max_bisect_iters):
        (crf_lo, vmaf_lo), (crf_hi, vmaf_hi) = low, high
        if crf_hi - crf_lo <= tolerance:
            break

        guess = _crf_guess(method, crf_lo, vmaf_lo, crf_hi, vmaf_hi, history, target)
        margin = 0.2 * (crf_hi - crf_lo)
        guess = min(max(guess, crf_lo + margin), crf_hi - margin)

        record = probe(guess)
        crf, achieved = record.params.crf, record.vmaf
        samples[crf] = achieved
        history.append((crf, achieved))

        if crf <= crf_lo or crf >= crf_hi:
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
    strategy = str(config.search.strategy).strip().lower()
    if strategy not in STRATEGIES:
        raise SearchError(
            f"unknown search.strategy {config.search.strategy!r}; "
            f"available: {list(STRATEGIES)}"
        )
    solver = str(config.search.crf_solver).strip().lower()
    if solver not in CRF_SOLVERS:
        raise SearchError(
            f"unknown search.crf_solver {config.search.crf_solver!r}; "
            f"available: {list(CRF_SOLVERS)}"
        )
    sampler_name = str(config.search.sampler).strip().lower()
    if strategy in ADAPTIVE or strategy == "sample":
        if sampler_name not in SAMPLERS:
            raise SearchError(
                f"unknown search.sampler {config.search.sampler!r}; "
                f"available: {sorted(SAMPLERS)}"
            )

    # ---- Stage A: exploration (shared by every target) ----
    primary_target = float(targets[0]) if targets else 85.0
    if strategy == "sample":
        all_trials = _explore_sample(
            evaluator, segment_path, segment_hash, info, encoder, config
        )
    elif strategy in ADAPTIVE:
        def _eval(params: EncodeParams) -> TrialRecord | None:
            return _try_evaluate(
                evaluator, segment_path, segment_hash, info, params
            )
        all_trials = explore_adaptive(
            _eval, encoder, config, strategy, primary_target
        )
        log.info("%s: adaptive %s used %d trial(s)", segment_path.name, strategy, len(all_trials))
    else:
        all_trials = _explore_aq_screen(
            evaluator, segment_path, segment_hash, info, encoder, config
        )

    if not all_trials:
        raise SearchError(
            f"every exploration trial failed for {segment_path.name}; "
            "check the encoder and VMAF toolchain"
        )

    log.info(
        "%s: strategy=%s explored %d point(s), vmaf %.1f-%.1f",
        segment_path.name,
        strategy,
        len(all_trials),
        min(t.vmaf for t in all_trials),
        max(t.vmaf for t in all_trials),
    )

    # ---- Stage B: refinement, per target ----
    results: dict[float, SearchResult] = {}
    for target in targets:
        if strategy == "coordinate":
            _walk_aq(
                evaluator, segment_path, segment_hash, info, encoder,
                target, all_trials, config,
            )
        else:
            ranked = _rank_aq(all_trials, target)
            for aq_mode, aq_strength in ranked[: config.search.n_refine_configs]:
                _refine_aq(
                    evaluator, segment_path, segment_hash, info,
                    aq_mode, aq_strength, target, all_trials, config,
                )

        feasible_trials = [t for t in all_trials if is_feasible(t.vmaf, t.rate, target)]
        ranked_feasible = sorted(
            feasible_trials,
            key=lambda t: compression_score(t.vmaf, t.rate, target).score,
            reverse=True,
        )
        top_k = max(1, int(config.search.top_k_per_segment))
        top_candidates = ranked_feasible[:top_k]
        if feasible_trials:
            best = top_candidates[0]
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
            top_candidates=top_candidates,
        )
        log.info(
            "%s target %.0f: %s vmaf=%.2f ratio=%.2fx score=%.4f%s",
            segment_path.name, target, best.params, best.vmaf, best.ratio,
            results[target].score, "" if feasible else "  [INFEASIBLE]",
        )

    return results
