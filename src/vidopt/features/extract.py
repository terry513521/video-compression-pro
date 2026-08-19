"""Per-segment feature extraction.

The feature set is a compact 8-D vector aligned with the ref SI/TI/luma labels:
spatial information, temporal information, peak motion, duration, luma mean/std,
colour complexity, and a log-pixel resolution prior. Each maps onto encoder bitrate
demand (high SI/TI → lower CRF; colour/luma shape AQ behaviour).

What changed from the earlier 18-feature set:

* Dropped redundant container fields (width/height/fps/bitrate) in favour of
  ``log_pixels`` + ``duration``.
* Dropped overlapping spatial metrics (edge density, texture entropy, grain) —
  SI already captures spatial complexity.
* Kept ``motion_p95`` (peak) over mean/variance; TI covers average temporal energy.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from ..config import FeatureConfig
from ..errors import FeatureExtractionError
from ..ffmpeg.probe import MediaInfo
from ..log import get_logger

log = get_logger(__name__)

# OpenCV's FFmpeg backend prints H.264 DPB/MMCO warnings at ERROR when seeking into
# stream-copied cuts (``mmco: unref short failure``). Harmless; they drown the job log.
_AV_LOG_FATAL = "8"
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", _AV_LOG_FATAL)


@contextmanager
def _suppress_native_stderr() -> Iterator[None]:
    """Mute libc-level stderr for the duration of a decode.

    libavcodec is often linked into the OpenCV wheel, so Python logging and
    ``OPENCV_FFMPEG_LOGLEVEL`` (read only at first capture init) cannot always
    reach it — especially after a forked worker inherits an already-initialised
    ffmpeg. Dup2 on fd 2 always works. Set ``VIDOPT_SHOW_DECODER_LOGS=1`` to keep
    the chatter.
    """
    if os.environ.get("VIDOPT_SHOW_DECODER_LOGS"):
        yield
        return
    try:
        saved = os.dup(2)
    except OSError:
        yield
        return
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, 2)
        finally:
            os.close(devnull)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)

# Compact 8-feature schema (aligned with ref SI/TI/luma labels).
# Order matters: it is the column order of the model's design matrix and is stored in
# every model bundle so a mismatch is caught at load time rather than silently
# misinterpreted at predict time.
FEATURE_NAMES: tuple[str, ...] = (
    "spatial_information",  # SI — Sobel magnitude σ
    "temporal_information",  # TI — frame-diff σ
    "motion_p95",  # peak motion (ref ti_p95 analogue)
    "duration",
    "luma_mean",
    "luma_std",  # contrast / luma range proxy
    "color_complexity",  # mean per-channel entropy
    "log_pixels",  # resolution prior (log10 width*height)
)

# Unified-model schema: scene features + requested VMAF target at inference.
VMAF_TARGET_FEATURE: str = "vmaf_target"
MODEL_FEATURE_NAMES: tuple[str, ...] = FEATURE_NAMES + (VMAF_TARGET_FEATURE,)


def with_vmaf_target(scene: np.ndarray, vmaf_target: float) -> np.ndarray:
    """Append ``vmaf_target`` to an 8-D scene vector."""
    row = np.atleast_1d(np.asarray(scene, dtype=np.float64)).reshape(-1)
    if row.size != len(FEATURE_NAMES):
        raise ValueError(
            f"expected {len(FEATURE_NAMES)} scene features, got {row.size}"
        )
    return np.append(row, float(vmaf_target))


@dataclass(frozen=True)
class Features:
    """A feature vector plus its provenance."""

    values: dict[str, float]
    n_frames_analysed: int

    def vector(self) -> np.ndarray:
        """Dense array in ``FEATURE_NAMES`` order."""
        return np.array([self.values[name] for name in FEATURE_NAMES], dtype=np.float64)

    def to_dict(self) -> dict[str, float]:
        return dict(self.values)


def _entropy(channel: np.ndarray) -> float:
    """Shannon entropy of an 8-bit channel's histogram, in bits.

    Vectorised. The reference used a Python list comprehension over all 256 bins per
    channel per frame, which dominated its analysis time.
    """
    histogram = np.bincount(channel.ravel(), minlength=256).astype(np.float64)
    total = histogram.sum()
    if total <= 0:
        return 0.0
    probabilities = histogram[histogram > 0] / total
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _sample_indices(total: int, wanted: int) -> list[int]:
    """Uniformly spaced frame indices across the middle of the clip.

    The first and last few frames are skipped: segment boundaries often carry
    encoder ramp-up artefacts that are not representative of the content.
    """
    if total <= 0:
        return []
    if total <= wanted:
        return list(range(total))
    lo = int(total * 0.05)
    hi = max(lo + 1, int(total * 0.95))
    return [int(round(x)) for x in np.linspace(lo, hi - 1, num=wanted)]


def extract(
    path: str | Path, info: MediaInfo, config: FeatureConfig
) -> Features:
    """Analyse one segment.

    Raises:
        FeatureExtractionError: The file cannot be decoded or yields no frames.
    """
    path = Path(path)
    try:
        cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
    except AttributeError:
        pass

    with _suppress_native_stderr():
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise FeatureExtractionError(f"OpenCV cannot open {path}")

        try:
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                total = info.n_frames
            wanted = _sample_indices(total, config.max_frames)
            if not wanted:
                raise FeatureExtractionError(f"{path.name} reports no frames")

            color: list[float] = []
            spatial: list[float] = []
            luma_mean: list[float] = []
            luma_std: list[float] = []
            motion: list[float] = []
            temporal: list[float] = []

            previous: np.ndarray | None = None
            analysed = 0

            for index in wanted:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue

                height, width = frame.shape[:2]
                if width > config.analysis_width:
                    scale = config.analysis_width / width
                    frame = cv2.resize(
                        frame,
                        (config.analysis_width, max(2, int(round(height * scale)))),
                        interpolation=cv2.INTER_AREA,
                    )

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                color.append(float(np.mean([_entropy(c) for c in cv2.split(frame)])))

                sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                spatial.append(float(np.std(cv2.magnitude(sobel_x, sobel_y))))

                luma_mean.append(float(np.mean(gray)))
                luma_std.append(float(np.std(gray)))

                if previous is not None and previous.shape == gray.shape:
                    difference = cv2.absdiff(previous, gray)
                    motion.append(float(np.mean(difference)) / 255.0)
                    temporal.append(float(np.std(difference)))

                previous = gray
                analysed += 1
        finally:
            capture.release()

    if analysed < 2:
        raise FeatureExtractionError(
            f"decoded only {analysed} frame(s) from {path.name}; cannot characterise it"
        )
    if not motion:
        # One frame decoded per sample point but never two consecutive: treat as static.
        motion = [0.0]
        temporal = [0.0]

    motion_array = np.asarray(motion, dtype=np.float64)
    pixels = max(1, info.pixels_per_frame)

    values = {
        "spatial_information": float(np.mean(spatial)),
        "temporal_information": float(np.mean(temporal)),
        "motion_p95": float(np.percentile(motion_array, 95)),
        "duration": float(info.duration),
        "luma_mean": float(np.mean(luma_mean)),
        "luma_std": float(np.mean(luma_std)),
        "color_complexity": float(np.mean(color)),
        "log_pixels": float(np.log10(pixels)),
    }

    missing = set(FEATURE_NAMES) - set(values)
    if missing:  # pragma: no cover - guards against edits that desync the schema
        raise FeatureExtractionError(
            f"feature schema mismatch, missing: {sorted(missing)}"
        )

    log.debug("features for %s from %d frame(s)", path.name, analysed)
    return Features(values=values, n_frames_analysed=analysed)
