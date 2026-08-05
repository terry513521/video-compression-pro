"""VMAF measurement.

One implementation, replacing the reference's three (see REFERENCE_ANALYSIS.md §1.7).

The essential correctness property: **the inputs are never re-encoded, trimmed or
rescaled before comparison.** The reference's primary path extracted sample clips by
re-encoding the reference at ``-crf 10`` and the distorted file at ``-crf 15``, then
compared those — damaging both sides, asymmetrically, before measuring. Whatever that
produces, it is not VMAF.

Here the two files go straight into ffmpeg's ``libvmaf`` filter.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..config import VmafConfig
from ..errors import VmafError
from ..ffmpeg.run import run
from ..ffmpeg.toolchain import Capabilities
from ..log import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class VmafResult:
    """A VMAF measurement and the context needed to interpret it."""

    score: float
    """Pooled score, using the configured pooling method."""

    mean: float
    harmonic_mean: float
    minimum: float
    n_frames: int
    model: str
    n_subsample: int
    seconds: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "vmaf": self.score,
            "vmaf_mean": self.mean,
            "vmaf_harmonic_mean": self.harmonic_mean,
            "vmaf_min": self.minimum,
            "vmaf_frames": self.n_frames,
            "vmaf_model": self.model,
            "vmaf_n_subsample": self.n_subsample,
        }


# Frame-index timestamps on a shared timebase. This is the part that makes the
# measurement trustworthy, and it is easy to get wrong.
#
# The obvious `setpts=PTS-STARTPTS` only aligns the *first* frame. When the two inputs
# carry different container timebases -- which they always do here, because segments are
# Matroska (1/1000) and encodes are MP4 (1/15360) -- later frames land on slightly
# different timestamps, and libvmaf's frame synchroniser then pads or duplicates to fill
# the gaps. On a real 66-frame segment that produced 67 scored frames and dragged the
# harmonic mean from 94.4 down to 86.2, with individual "frames" scoring as low as 43:
# entirely an artefact of comparing frame N against frame N+1.
#
# `settb=AVTB,setpts=N-STARTPTS` rewrites both streams to a common timebase with the
# frame *index* as the timestamp, so frame N is compared against frame N and nothing
# else.
_ALIGN = "settb=AVTB,setpts=N-STARTPTS"


def escape_filter_path(path: str | os.PathLike[str]) -> str:
    """Escape a filesystem path for an ffmpeg filter option value.

    Filter options are colon-separated, so an unescaped Windows drive letter
    (``C:``) is parsed as the end of the option and breaks ``log_path``. Prefer
    relative paths (see ``measure``) when you can — they need no escaping.
    Absolute paths are normalised to forward slashes and have ``:`` escaped.
    """
    p = Path(path)
    # Keep relative paths relative: resolving would reintroduce a drive letter.
    text = p.resolve().as_posix() if p.is_absolute() else p.as_posix()
    # Two backslashes: one for the filter-option lexer, one consumed by the
    # filtergraph parser — a single ``\:`` is not enough when ``-lavfi`` is used.
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\\\:")
        .replace("'", "\\'")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def _build_filter(
    *, model: str, n_subsample: int, n_threads: int, log_path: str, cuda: bool
) -> str:
    """Filter graph: force frame-exact alignment on both inputs, then libvmaf.

    Input 0 is the distorted stream and input 1 the reference — ffmpeg's convention,
    and the order libvmaf expects.
    """
    options = [
        f"model=version={model}",
        f"n_subsample={max(1, n_subsample)}",
        "log_fmt=json",
        f"log_path={escape_filter_path(log_path)}",
    ]
    if cuda:
        # libvmaf_cuda runs on hardware frames, so both inputs must be uploaded.
        chain = (
            f"[0:v]{_ALIGN},format=yuv420p,hwupload_cuda[dis];"
            f"[1:v]{_ALIGN},format=yuv420p,hwupload_cuda[ref];"
            "[dis][ref]libvmaf_cuda=" + ":".join(options)
        )
    else:
        options.append(f"n_threads={max(1, n_threads)}")
        chain = (
            f"[0:v]{_ALIGN}[dis];"
            f"[1:v]{_ALIGN}[ref];"
            "[dis][ref]libvmaf=" + ":".join(options)
        )
    return chain


def expected_scored_frames(n_frames: int, n_subsample: int) -> int:
    """How many frames libvmaf should score for an ``n_frames`` input."""
    if n_frames <= 0:
        return 0
    step = max(1, n_subsample)
    return (n_frames - 1) // step + 1


def measure(
    distorted: str | os.PathLike[str],
    reference: str | os.PathLike[str],
    caps: Capabilities,
    config: VmafConfig,
    *,
    n_subsample: int | None = None,
    expected_frames: int | None = None,
    timeout: float | None = 3600.0,
) -> VmafResult:
    """Measure VMAF of ``distorted`` against ``reference``.

    Args:
        distorted: The encoded file under test.
        reference: The original file.
        caps: Probed toolchain capabilities.
        config: VMAF settings.
        n_subsample: Override the configured subsampling (search vs. verify).
        expected_frames: Reference frame count, if the caller already knows it. When
            given, the number of scored frames is checked against it — a mismatch means
            the comparison was not frame-for-frame and the score is meaningless.

    Raises:
        VmafError: libvmaf is unavailable, the run failed, or the output is unusable.
    """
    dist_path = Path(distorted)
    ref_path = Path(reference)
    for path, label in ((dist_path, "distorted"), (ref_path, "reference")):
        if not path.is_file():
            raise VmafError(f"{label} file not found: {path}")

    if not caps.has_libvmaf:
        raise VmafError(
            "this ffmpeg build has no libvmaf filter, so VMAF cannot be measured. "
            "Install the vendored build (`python scripts/setup.py`)."
        )

    subsample = config.n_subsample_search if n_subsample is None else n_subsample
    use_cuda = bool(config.use_cuda and caps.has_libvmaf_cuda)

    with tempfile.TemporaryDirectory(prefix="vidopt_vmaf_") as tmp:
        # Relative log_path avoids Windows drive-letter colons, which ffmpeg would
        # otherwise treat as filter-option separators (``C:`` → option ends at ``C``).
        log_name = "vmaf.json"
        log_path = os.path.join(tmp, log_name)
        graph = _build_filter(
            model=config.model,
            n_subsample=subsample,
            n_threads=config.n_threads,
            log_path=log_name,
            cuda=use_cuda,
        )
        argv = [
            caps.ffmpeg, "-hide_banner", "-nostdin",
            "-loglevel", "error",
            # Absolute inputs: we set cwd to ``tmp`` so ``log_path`` can be a
            # relative name (Windows drive-letter colons break filter options).
            "-i", str(dist_path.resolve()),
            "-i", str(ref_path.resolve()),
            "-lavfi", graph,
            "-f", "null", "-",
        ]

        try:
            result = run(argv, timeout=timeout, cwd=tmp)
        except Exception as exc:
            if use_cuda:
                # A CUDA-specific failure is worth retrying on CPU: the numbers are the
                # same, only slower. Any other failure is a real error.
                log.warning("libvmaf_cuda failed (%s); retrying on CPU", exc)
                return measure(
                    distorted, reference, caps,
                    _without_cuda(config), n_subsample=subsample,
                    expected_frames=expected_frames, timeout=timeout,
                )
            raise VmafError(f"VMAF measurement failed: {exc}") from exc

        try:
            payload = json.loads(Path(log_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise VmafError(f"could not read libvmaf output: {exc}") from exc

    frames = payload.get("frames") or []
    pooled = (payload.get("pooled_metrics") or {}).get("vmaf") or {}
    if not frames or not pooled:
        raise VmafError(
            f"libvmaf produced no usable scores for {dist_path.name}. "
            "This usually means the two inputs share no comparable frames."
        )

    if expected_frames is not None and expected_frames > 0:
        wanted = expected_scored_frames(expected_frames, subsample)
        # One frame of slack absorbs off-by-one rounding in libvmaf's subsampling.
        if abs(len(frames) - wanted) > 1:
            raise VmafError(
                f"frame-count mismatch measuring {dist_path.name}: libvmaf scored "
                f"{len(frames)} frame(s) but the {expected_frames}-frame reference "
                f"at n_subsample={subsample} should give {wanted}. The two streams "
                "were not compared frame-for-frame, so the score is meaningless."
            )

    mean = float(pooled.get("mean", 0.0))
    harmonic = float(pooled.get("harmonic_mean", mean))
    minimum = float(pooled.get("min", mean))
    score = harmonic if config.pool == "harmonic_mean" else mean

    if not 0.0 <= score <= 100.0:
        raise VmafError(f"implausible VMAF score {score} for {dist_path.name}")

    log.debug(
        "vmaf %s vs %s: %.3f (%s, %d frames, subsample=%d, cuda=%s) in %.1fs",
        dist_path.name, ref_path.name, score, config.model,
        len(frames), subsample, use_cuda, result.seconds,
    )
    return VmafResult(
        score=score,
        mean=mean,
        harmonic_mean=harmonic,
        minimum=minimum,
        n_frames=len(frames),
        model=config.model,
        n_subsample=subsample,
        seconds=result.seconds,
    )


def _without_cuda(config: VmafConfig) -> VmafConfig:
    import dataclasses

    return dataclasses.replace(config, use_cuda=False)
