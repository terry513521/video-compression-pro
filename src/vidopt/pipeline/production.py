"""Production mode: compress one video with per-segment predicted parameters.

    probe -> segment -> features -> predict -> encode (parallel) -> concat -> verify

No search and no VMAF measurement in the hot path — that is the entire point of dev
mode. Optional verification re-measures the final result and reports the same score dev
mode optimised, which closes the loop.
"""

from __future__ import annotations

import dataclasses
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ..config import Config
from ..encoding.encoders import (
    EncodeRequest,
    container_for,
    encode,
    get_encoder,
    resolve_pix_fmt,
)
from ..encoding.params import EncodeParams
from ..errors import EncodeError, FeatureExtractionError, VidoptError
from ..features.extract import FEATURE_NAMES, extract
from ..ffmpeg import toolchain
from ..ffmpeg.probe import MediaInfo, probe
from ..log import get_logger
from ..media.concat import concat
from ..media.segment import Segment, segment_video
from ..modeling.bundle import find_bundle
from ..quality import vmaf as vmaf_mod
from ..scoring import compression_score

log = get_logger(__name__)

# Worker processes are started with "spawn" explicitly rather than the platform default.
# Three reasons, in order of importance:
#
#   1. Windows only has spawn. Forcing it everywhere means the code path exercised in
#      development on Linux is the same one that runs in production on Windows, instead
#      of fork here and spawn there.
#   2. fork() in a process that has already initialised CUDA is unsafe — the child
#      inherits a broken context. The GPU configuration would hit that.
#   3. fork() copies whatever the parent happened to have loaded; spawn starts clean, so
#      a worker's state depends only on the payload it is given.
#
# The cost is that payloads must be picklable and workers pay ~0.5 s of start-up. Both
# are already true here: every worker takes a plain dict and rebuilds its own config.
_MP_CONTEXT = multiprocessing.get_context("spawn")


def _segments_manifest_path(work_dir: Path) -> Path:
    return work_dir / "segments.json"


def _load_segments_manifest(manifest: Path, source: Path) -> list[Segment]:
    """Best-effort resume for already cut segments."""
    if not manifest.is_file():
        return []
    try:
        raw = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    out: list[Segment] = []
    want = source.resolve()
    for item in raw:
        try:
            seg = Segment.from_dict(item)
        except (KeyError, TypeError, ValueError):
            continue
        if Path(seg.source).resolve() != want:
            continue
        if not Path(seg.path).is_file():
            return []
        out.append(seg)
    return out



@dataclass
class SegmentPlan:
    """A segment plus the parameters chosen for it."""

    segment: Segment
    params: EncodeParams
    info: MediaInfo


@dataclass
class ProductionResult:
    """Outcome of compressing one video."""

    input_path: str
    output_path: str
    target: float
    encoder: str
    n_segments: int
    input_bytes: int
    output_bytes: int
    seconds: float
    vmaf: float | None = None
    score: float | None = None
    reason: str = ""

    @property
    def rate(self) -> float:
        return (self.output_bytes / self.input_bytes) if self.input_bytes else 1.0

    @property
    def ratio(self) -> float:
        rate = self.rate
        return (1.0 / rate) if rate > 0 else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "input": self.input_path,
            "output": self.output_path,
            "target": self.target,
            "encoder": self.encoder,
            "segments": self.n_segments,
            "input_bytes": self.input_bytes,
            "output_bytes": self.output_bytes,
            "compression_rate": round(self.rate, 6),
            "compression_ratio": round(self.ratio, 3),
            "vmaf": self.vmaf,
            "score": self.score,
            "reason": self.reason,
            "seconds": round(self.seconds, 1),
        }

    def summary(self) -> str:
        lines = [
            f"input      {self.input_path}",
            f"output     {self.output_path}",
            f"encoder    {self.encoder}  target VMAF {self.target:g}",
            f"segments   {self.n_segments}",
            f"size       {self.input_bytes / 1e6:.1f} MB -> "
            f"{self.output_bytes / 1e6:.1f} MB  ({self.ratio:.2f}x)",
        ]
        if self.vmaf is not None:
            met = "MET" if self.vmaf >= self.target else "MISSED"
            lines.append(f"vmaf       {self.vmaf:.2f}  [{met}]")
        if self.score is not None:
            lines.append(f"score      {self.score:.4f}  ({self.reason})")
        lines.append(f"elapsed    {self.seconds:.1f}s")
        return "\n".join(lines)


def _encode_one(payload: dict) -> dict:
    """Worker entry point: encode one planned segment.

    Retries once at a safer CRF before giving up. A single segment failing an encode —
    a transient GPU session limit, an awkward frame count, a full disk that briefly
    recovered — should not throw away every other segment's work on a long video.
    """
    from ..config import load_config

    config = load_config(payload["config_paths"], payload["config_overrides"])
    caps = toolchain.detect(config.ffmpeg.bin_dir)
    encoder = get_encoder(config.encoder.name)
    params = EncodeParams(**payload["params"])

    def attempt(attempt_params: EncodeParams) -> object:
        request = EncodeRequest(
            input_path=payload["input_path"],
            output_path=payload["output_path"],
            params=attempt_params,
            preset=config.encoder.preset,
            keyint_seconds=config.encoder.keyint_seconds,
            pix_fmt=payload["pix_fmt"],
            fps=payload["fps"],
            threads=config.ffmpeg.threads,
            loglevel=config.ffmpeg.loglevel,
            extra_args=tuple(config.encoder.extra_args),
        )
        return encode(caps.ffmpeg, encoder, request)

    try:
        result = attempt(params)
        retried = False
    except Exception as first_error:  # noqa: BLE001
        # Back off to a conservative, widely-safe point in the space. Losing some
        # compression on one segment beats losing the whole job.
        safe = encoder.space.clamp(
            EncodeParams(
                crf=max(encoder.space.crf_min, params.crf - 4.0),
                aq_mode=encoder.space.aq_modes[0],
                aq_strength=encoder.space.aq_strength_min,
            )
        )
        log.warning(
            "segment %d failed at %s (%s); retrying at %s",
            payload["index"], params, first_error, safe,
        )
        result = attempt(safe)
        params = safe
        retried = True

    if payload.get("delete_input_after"):
        # Bound peak disk on long videos: the cut segment is dead once encoded, and
        # keeping every one alive means holding roughly 2x the input on disk.
        try:
            Path(payload["input_path"]).unlink(missing_ok=True)
        except OSError as exc:
            # Windows refuses to unlink a file another process still has open. Leaving
            # it behind costs disk; failing here would cost the whole job.
            log.debug("could not remove %s: %s", payload["input_path"], exc)

    return {
        "index": payload["index"],
        "output_path": result.output_path,
        "size_bytes": result.size_bytes,
        "seconds": result.seconds,
        "params": params.to_dict(),
        "retried": retried,
    }


def _analyse_one(payload: dict) -> dict:
    """Worker entry point: probe and analyse one segment.

    Returns plain data because it crosses a process boundary.
    """
    from ..config import load_config

    config = load_config(payload["config_paths"], payload["config_overrides"])
    caps = toolchain.detect(config.ffmpeg.bin_dir)

    info = probe(payload["segment_path"], caps.ffprobe)
    features = extract(payload["segment_path"], info, config.features)
    return {
        "index": payload["index"],
        "info": dataclasses.asdict(info),
        "vector": features.vector().tolist(),
    }


def plan_segments(
    segments: list[Segment],
    config: Config,
    target: float,
    config_paths: list[str],
    config_overrides: list[str],
) -> list[SegmentPlan]:
    """Extract features and predict parameters for every segment.

    Analysis is CPU-bound OpenCV work over independent segments, so it runs across the
    worker pool. On a long video this is otherwise the second-largest cost after encoding.
    """
    encoder = get_encoder(config.encoder.name)
    bundle = find_bundle(config.paths.models_dir, config.encoder.name, target)

    payloads = [
        {
            "index": index,
            "segment_path": segment.path,
            "config_paths": config_paths,
            "config_overrides": config_overrides,
        }
        for index, segment in enumerate(segments)
    ]
    workers = max(1, min(config.resolved_cpu_workers(), len(payloads)))

    analysed: dict[int, dict] = {}
    if workers == 1:
        for payload in payloads:
            result = _analyse_one(payload)
            analysed[result["index"]] = result
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=_MP_CONTEXT) as pool:
            futures = {pool.submit(_analyse_one, p): p for p in payloads}
            for done in as_completed(futures):
                payload = futures[done]
                try:
                    result = done.result()
                except Exception as exc:  # noqa: BLE001
                    raise FeatureExtractionError(
                        f"could not analyse {Path(payload['segment_path']).name}: {exc}"
                    ) from exc
                analysed[result["index"]] = result

    infos = [MediaInfo(**analysed[i]["info"]) for i in range(len(payloads))]
    vectors = [
        np.asarray(analysed[i]["vector"], dtype=np.float64)
        for i in range(len(payloads))
    ]

    matrix = np.vstack(vectors) if vectors else np.empty((0, len(FEATURE_NAMES)))
    predictions = bundle.predict(matrix, vmaf_target=target)

    plans: list[SegmentPlan] = []
    off_domain: dict[str, tuple[float, float]] = {}
    for segment, info, vector, params in zip(
        segments, infos, vectors, predictions, strict=True
    ):
        for name, _value, low, high in bundle.out_of_domain(vector, vmaf_target=target):
            off_domain[name] = (low, high)
        clamped = encoder.space.clamp(params)
        log.info(
            "%s (%.1fs): %s", Path(segment.path).name, segment.duration, clamped
        )
        plans.append(SegmentPlan(segment=segment, params=clamped, info=info))

    if off_domain:
        # Not fatal — the prediction may still be usable — but the user must know the
        # model is answering a question it was never asked during training.
        detail = ", ".join(
            f"{name} (trained {low:.4g}..{high:.4g})"
            for name, (low, high) in sorted(off_domain.items())
        )
        log.warning(
            "this input is outside the model's training domain: %s. "
            "Predictions are extrapolations and may miss the VMAF target — verify with "
            "--verify, and re-run `vidopt train` on a corpus that includes content like "
            "this.",
            detail,
        )
    return plans


def compress(
    input_path: str | Path,
    output_path: str | Path,
    target: float,
    config: Config,
    config_paths: list[str],
    config_overrides: list[str],
    *,
    verify: bool = False,
    keep_work: bool = False,
    resume: bool = False,
    progress: Callable[..., None] | None = None,
) -> ProductionResult:
    """Compress one video end to end."""
    started = time.monotonic()

    caps = toolchain.detect(config.ffmpeg.bin_dir)
    encoder = get_encoder(config.encoder.name)
    toolchain.require(caps, encoder=encoder.ffmpeg_encoder, vmaf=verify)

    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)

    info = probe(source, caps.ffprobe)
    log.info(
        "input: %s  %dx%d @ %.2f fps  %.1fs  %.1f MB",
        source.name, info.width, info.height, info.fps, info.duration,
        info.size_bytes / 1e6,
    )
    if progress:
        progress(
            "compress.start",
            input=str(source),
            output=str(destination),
            encoder=config.encoder.name,
            target=target,
            resume=resume,
        )

    work_dir = Path(config.paths.work_dir) / f"prod_{source.stem}"
    segments_dir = work_dir / "segments"
    encoded_dir = work_dir / "encoded"
    encoded_dir.mkdir(parents=True, exist_ok=True)
    segments_manifest = _segments_manifest_path(work_dir)
    completed = False

    try:
        segments: list[Segment] = []
        if resume:
            segments = _load_segments_manifest(segments_manifest, source)
            if segments:
                log.info("stage 1/4: resume reusing %d segment(s)", len(segments))
                if progress:
                    progress("compress.resume.segments", reused=len(segments))
        if not segments:
            log.info("stage 1/4: segmenting")
            segments = segment_video(source, segments_dir, caps, config, info=info)
        if progress:
            progress("compress.stage.segment.done", n_segments=len(segments))
        segments_manifest.write_text(
            json.dumps([s.to_dict() for s in segments], indent=2), encoding="utf-8"
        )

        log.info("stage 2/4: extracting features and predicting parameters")
        plans = plan_segments(
            segments, config, target, config_paths, config_overrides
        )
        if progress:
            progress("compress.stage.plan.done", n_segments=len(plans))

        log.info("stage 3/4: encoding %d segment(s)", len(plans))
        extension = container_for(config.encoder.name)

        # Resolve the pixel format once, from the SOURCE, and use it for every segment.
        # It has to be identical across segments or the concat demuxer refuses to join
        # them; taking it from the source (not from a segment) also means a segment that
        # happens to decode as a different format cannot skew the choice.
        target_pix_fmt = resolve_pix_fmt(
            caps.ffmpeg, encoder, config.encoder.pix_fmt, info.pix_fmt,
            context=source.name,
        )
        if target_pix_fmt != info.pix_fmt:
            log.info("pixel format: %s -> %s", info.pix_fmt, target_pix_fmt)

        payloads = [
            {
                "index": index,
                "input_path": plan.segment.path,
                "output_path": str(encoded_dir / f"seg_{index:05d}.{extension}"),
                "params": plan.params.to_dict(),
                "fps": plan.info.fps,
                "pix_fmt": target_pix_fmt,
                "config_paths": config_paths,
                "config_overrides": config_overrides,
                # Only safe to reclaim the cut segment when the caller does not want the
                # intermediates kept for inspection.
                "delete_input_after": not keep_work,
            }
            for index, plan in enumerate(plans)
        ]

        workers = config.resolved_cpu_workers()
        if encoder.is_gpu:
            workers = min(workers, max(1, config.jobs.gpu_workers))
        workers = max(1, min(workers, len(payloads)))

        encoded: dict[int, str] = {}
        retried_segments: list[int] = []
        if resume:
            for payload in payloads:
                existing = Path(payload["output_path"])
                if existing.is_file() and existing.stat().st_size > 0:
                    encoded[payload["index"]] = str(existing)
            if encoded:
                log.info(
                    "stage 3/4: resume reusing %d pre-encoded segment(s)",
                    len(encoded),
                )
                if progress:
                    progress("compress.resume.encoded", reused=len(encoded), total=len(plans))
            payloads = [p for p in payloads if p["index"] not in encoded]
        if workers == 1:
            for payload in payloads:
                outcome = _encode_one(payload)
                encoded[outcome["index"]] = outcome["output_path"]
                if outcome.get("retried"):
                    retried_segments.append(outcome["index"])
                if progress:
                    progress(
                        "compress.segment.encoded",
                        done=len(encoded),
                        total=len(plans),
                        index=outcome["index"],
                        retried=bool(outcome.get("retried")),
                    )
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=_MP_CONTEXT) as pool:
                futures = {pool.submit(_encode_one, p): p for p in payloads}
                for done in as_completed(futures):
                    payload = futures[done]
                    try:
                        outcome = done.result()
                    except Exception as exc:  # noqa: BLE001
                        raise EncodeError(
                            f"segment {payload['index']} failed to encode: {exc}"
                        ) from exc
                    encoded[outcome["index"]] = outcome["output_path"]
                    if outcome.get("retried"):
                        retried_segments.append(outcome["index"])
                    if progress:
                        progress(
                            "compress.segment.encoded",
                            done=len(encoded),
                            total=len(plans),
                            index=outcome["index"],
                            retried=bool(outcome.get("retried")),
                        )

        ordered = [encoded[i] for i in range(len(plans))]
        if retried_segments:
            log.warning(
                "%d segment(s) needed a conservative retry: %s",
                len(retried_segments), retried_segments,
            )

        log.info("stage 4/4: concatenating")
        concat(ordered, destination, caps, audio_from=source if info.has_audio else None)
        if progress:
            progress("compress.stage.concat.done", total_segments=len(plans))

        output_bytes = destination.stat().st_size
        result = ProductionResult(
            input_path=str(source),
            output_path=str(destination.resolve()),
            target=target,
            encoder=config.encoder.name,
            n_segments=len(plans),
            input_bytes=info.size_bytes,
            output_bytes=output_bytes,
            seconds=time.monotonic() - started,
        )

        if verify:
            log.info("verifying: measuring VMAF of the final output")
            measured = vmaf_mod.measure(
                destination, source, caps, config.vmaf,
                n_subsample=config.vmaf.n_subsample_verify,
                expected_frames=info.n_frames,
            )
            breakdown = compression_score(measured.score, result.rate, target)
            result.vmaf = measured.score
            result.score = breakdown.score
            result.reason = breakdown.reason
            if progress:
                progress(
                    "compress.verify.done",
                    vmaf=result.vmaf,
                    score=result.score,
                    met=bool(result.vmaf >= target),
                )

        (work_dir / "result.json").write_text(
            json.dumps(result.to_dict(), indent=2), encoding="utf-8"
        )
        if progress:
            progress(
                "compress.done",
                output=str(destination.resolve()),
                ratio=round(result.ratio, 4),
                seconds=round(result.seconds, 2),
                n_segments=result.n_segments,
            )
        completed = True
        return result

    finally:
        if not keep_work:
            if resume and not completed:
                log.info(
                    "resume mode: keeping intermediates in %s after failure",
                    work_dir,
                )
            else:
                from ..media.segment import clear_dir

                clear_dir(segments_dir)
                clear_dir(encoded_dir)
