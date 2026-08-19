"""Scene segmentation.

Detection runs on a downscaled proxy (kept from the reference — detection does not need
4K, and building the proxy is far cheaper than detecting on the original), then the
source is cut with ffmpeg's segment muxer in stream-copy mode.

Two things the reference did not do:

* **Duration guards.** Cut points closer together than ``min_segment_seconds`` are
  merged, and runs longer than ``max_segment_seconds`` are split. Sub-second segments
  encode inefficiently and add label noise; unbounded segments defeat parallelism.
* **Boundaries are probed back, not assumed.** Stream-copy cutting snaps to keyframes,
  so the realised boundaries differ from the requested ones. The reference printed a
  warning about the count mismatch and carried on with the requested values; here the
  produced files are measured, and their real durations are what downstream sees.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..errors import SegmentationError
from ..ffmpeg.probe import MediaInfo, probe
from ..ffmpeg.run import run
from ..ffmpeg.toolchain import Capabilities
from ..log import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Segment:
    """One scene-aligned piece of a source video."""

    index: int
    path: str
    source: str
    start: float
    duration: float

    @property
    def name(self) -> str:
        return Path(self.path).name

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "path": self.path,
            "source": self.source,
            "start": round(self.start, 3),
            "duration": round(self.duration, 3),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Segment:
        return cls(
            index=int(data["index"]),
            path=str(data["path"]),
            source=str(data["source"]),
            start=float(data["start"]),
            duration=float(data["duration"]),
        )


# --------------------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------------------


def _build_proxy(source: Path, out_path: Path, caps: Capabilities, height: int) -> Path:
    """Downscaled, fast-to-decode copy used only for cut detection."""
    argv = [
        caps.ffmpeg, "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
        "-i", str(source),
        "-an", "-sn", "-dn",
        "-vf", f"scale=-2:{height}",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    run(argv, timeout=None)
    return out_path


def detect_cut_times(
    source: Path, caps: Capabilities, config: Config, work_dir: Path
) -> list[float]:
    """Return scene-cut timestamps in seconds (excluding 0.0)."""
    from scenedetect import AdaptiveDetector, ContentDetector, SceneManager, open_video

    target = source
    proxy: Path | None = None
    if config.segment.proxy_height > 0:
        proxy = work_dir / f"{source.stem}__proxy.mp4"
        try:
            target = _build_proxy(source, proxy, caps, config.segment.proxy_height)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "proxy build failed for %s (%s); detecting at full res",
                source.name, exc,
            )
            target = source
            proxy = None

    video = None
    try:
        video = open_video(str(target))
        manager = SceneManager()
        if config.segment.detector == "adaptive":
            manager.add_detector(
                AdaptiveDetector(adaptive_threshold=config.segment.adaptive_threshold)
            )
        else:
            manager.add_detector(
                ContentDetector(threshold=config.segment.content_threshold)
            )
        manager.detect_scenes(video=video, show_progress=False)
        scenes = manager.get_scene_list()
    except Exception as exc:  # noqa: BLE001
        raise SegmentationError(
            f"scene detection failed for {source.name}: {exc}"
        ) from exc
    finally:
        # Windows refuses to delete a file that still has an open handle. PySceneDetect
        # keeps the OpenCV capture alive until we release it explicitly.
        if video is not None:
            capture = getattr(video, "capture", None)
            if capture is not None:
                try:
                    capture.release()
                except Exception:  # noqa: BLE001
                    pass
            del video
        if proxy is not None:
            _unlink_with_retry(proxy)

    return [float(start.get_seconds()) for start, _ in scenes[1:]]


def _unlink_with_retry(path: Path, attempts: int = 8) -> None:
    """Delete a file, retrying briefly on Windows sharing violations."""
    import time

    for attempt in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt + 1 >= attempts:
                log.warning("could not delete temporary file %s (still locked)", path)
                return
            time.sleep(0.05 * (attempt + 1))


def plan_boundaries(
    cut_times: list[float], duration: float, config: Config
) -> list[float]:
    """Turn raw cut times into a clean, guard-respecting boundary list.

    Returns interior boundaries only (0.0 and ``duration`` are implied).
    """
    seg = config.segment
    min_len, max_len = seg.min_segment_seconds, seg.max_segment_seconds

    # 1. Drop cuts that would create a segment shorter than the minimum.
    kept: list[float] = []
    previous = 0.0
    for time in sorted(t for t in cut_times if 0.0 < t < duration):
        if time - previous >= min_len and duration - time >= min_len:
            kept.append(time)
            previous = time

    # 2. Split any run longer than the maximum into equal pieces.
    boundaries: list[float] = []
    edges = [0.0, *kept, duration]
    for start, end in zip(edges[:-1], edges[1:], strict=True):
        span = end - start
        if span > max_len:
            pieces = int(span // max_len) + 1
            step = span / pieces
            boundaries.extend(start + step * i for i in range(1, pieces))
        if end < duration:
            boundaries.append(end)

    return sorted(set(round(b, 3) for b in boundaries))


# --------------------------------------------------------------------------------------
# Cutting
# --------------------------------------------------------------------------------------


def _cut(
    source: Path,
    boundaries: list[float],
    out_dir: Path,
    caps: Capabilities,
    config: Config,
) -> list[Path]:
    """Cut with the segment muxer in stream-copy mode (no re-encode)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / f"{source.stem}__seg_%04d.{config.segment.container}")

    argv = [
        caps.ffmpeg, "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
        "-i", str(source),
        "-map", "0:v:0", "-c", "copy",
        "-an", "-sn", "-dn",
        "-f", "segment",
        "-reset_timestamps", "1",
    ]
    if boundaries:
        argv += ["-segment_times", ",".join(f"{b:.3f}" for b in boundaries)]
    argv += [pattern]

    run(argv, timeout=None)

    produced = sorted(out_dir.glob(f"{source.stem}__seg_*.{config.segment.container}"))
    if not produced:
        raise SegmentationError(f"segmentation produced no files for {source.name}")
    return produced


def segment_video(
    source: str | Path,
    out_dir: str | Path,
    caps: Capabilities,
    config: Config,
    *,
    info: MediaInfo | None = None,
) -> list[Segment]:
    """Split ``source`` into scene-aligned segments under ``out_dir``.

    Returns segments with **measured** start/duration, not the requested ones.
    """
    source = Path(source).resolve()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    media = info or probe(source, caps.ffprobe)
    if media.duration <= 0:
        raise SegmentationError(f"{source.name} reports zero duration")

    try:
        cuts = detect_cut_times(source, caps, config, out_dir)
    except SegmentationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise SegmentationError(
            f"scene detection failed for {source.name}: {exc}"
        ) from exc

    if cuts:
        log.info("%s: detected %d scene cut(s)", source.name, len(cuts))
    else:
        # No cuts is legitimate (a single continuous shot). Fall back to fixed-duration
        # splitting so long single-shot videos still parallelise -- and say so.
        log.info(
            "%s: no scene cuts detected; falling back to %.1fs fixed segments",
            source.name, config.segment.fallback_segment_seconds,
        )
        step = config.segment.fallback_segment_seconds
        cuts = [step * i for i in range(1, int(media.duration // step) + 1)]

    boundaries = plan_boundaries(cuts, media.duration, config)
    log.info(
        "%s: %.1fs -> %d segment(s)", source.name, media.duration, len(boundaries) + 1
    )

    files = _cut(source, boundaries, out_dir, caps, config)

    segments: list[Segment] = []
    elapsed = 0.0
    for path in files:
        seg_info = probe(path, caps.ffprobe)
        if seg_info.duration <= 0.05:
            log.warning(
                "dropping degenerate segment %s (%.3fs)", path.name, seg_info.duration
            )
            path.unlink(missing_ok=True)
            continue
        segments.append(
            Segment(
                index=len(segments),
                path=str(path.resolve()),
                source=str(source),
                start=elapsed,
                duration=seg_info.duration,
            )
        )
        elapsed += seg_info.duration

    if not segments:
        raise SegmentationError(f"no usable segments produced for {source.name}")
    return segments


def clear_dir(path: str | Path) -> None:
    """Remove a directory tree if it exists. Used for scratch space between runs."""
    p = Path(path)
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
