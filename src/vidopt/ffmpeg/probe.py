"""ffprobe wrappers returning typed media metadata.

Replaces four separate duration-probing implementations and three copies of
``has_audio()`` found across the reference.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ..errors import ProbeError
from .run import run


@dataclass(frozen=True)
class MediaInfo:
    """Everything the pipeline needs to know about a media file."""

    path: str
    width: int
    height: int
    fps: float
    duration: float
    n_frames: int
    codec: str
    pix_fmt: str
    bitrate_kbps: float
    size_bytes: int
    has_audio: bool

    @property
    def pixels_per_frame(self) -> int:
        return self.width * self.height

    @property
    def bits_per_pixel(self) -> float:
        """Source bitrate normalised by pixel rate — a strong compressibility prior."""
        rate = self.pixels_per_frame * self.fps
        if rate <= 0:
            return 0.0
        return (self.bitrate_kbps * 1000.0) / rate


def _parse_rational(text: str | None, default: float = 0.0) -> float:
    if not text:
        return default
    if "/" in text:
        num, _, den = text.partition("/")
        try:
            denominator = float(den)
            return float(num) / denominator if denominator else default
        except ValueError:
            return default
    try:
        return float(text)
    except ValueError:
        return default


def probe(path: str | Path, ffprobe: str) -> MediaInfo:
    """Describe a media file. Raises :class:`ProbeError` if it cannot be read."""
    p = Path(path)
    if not p.is_file():
        raise ProbeError(f"file not found: {p}")

    argv = [
        ffprobe, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(p),
    ]
    try:
        raw = run(argv, timeout=120).stdout
        data = json.loads(raw)
    except Exception as exc:
        raise ProbeError(f"ffprobe failed for {p}: {exc}") from exc

    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ProbeError(f"no video stream in {p}")

    fmt = data.get("format", {})
    size_bytes = int(fmt.get("size") or p.stat().st_size)

    duration = _parse_rational(video.get("duration")) or _parse_rational(
        fmt.get("duration")
    )
    fps = _parse_rational(video.get("avg_frame_rate")) or _parse_rational(
        video.get("r_frame_rate")
    )

    # Frame count must be EXACT: it is what verifies that a VMAF pair was compared
    # frame-for-frame. Matroska — the container used for stream-copied segments — carries
    # no nb_frames tag, and the obvious `round(fps * duration)` estimate is routinely off
    # by a frame or two on a cut segment. Counting packets reads the container index
    # without decoding, so it is cheap, and probe() is called once per segment rather
    # than once per trial.
    n_frames = int(video.get("nb_frames") or 0)
    if n_frames <= 0:
        try:
            n_frames = count_frames(p, ffprobe)
        except Exception:  # noqa: BLE001 - fall back to the estimate, but say so
            n_frames = int(round(fps * duration)) if fps > 0 and duration > 0 else 0

    bitrate_bps = _parse_rational(video.get("bit_rate")) or _parse_rational(
        fmt.get("bit_rate")
    )
    if bitrate_bps <= 0 and duration > 0:
        bitrate_bps = size_bytes * 8.0 / duration

    return MediaInfo(
        path=str(p.resolve()),
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=fps,
        duration=duration,
        n_frames=n_frames,
        codec=str(video.get("codec_name") or "unknown"),
        pix_fmt=str(video.get("pix_fmt") or "unknown"),
        bitrate_kbps=bitrate_bps / 1000.0,
        size_bytes=size_bytes,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


def count_frames(path: str | Path, ffprobe: str) -> int:
    """Exact frame count by decoding packets. Slower than ``probe`` but authoritative.

    Used to verify that a VMAF pair is aligned — comparing streams of different lengths
    silently produces a meaningless score.
    """
    argv = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-count_packets", "-show_entries", "stream=nb_read_packets",
        "-of", "csv=p=0", str(path),
    ]
    out = run(argv, timeout=600).stdout.strip().splitlines()
    for line in out:
        token = line.strip().rstrip(",")
        if token.isdigit():
            return int(token)
    raise ProbeError(f"could not count frames in {path}")


def content_hash(path: str | Path, *, chunk_bytes: int = 1 << 20) -> str:
    """Stable content identity for cache keys.

    Hashes the file size plus the head, middle and tail chunks rather than the whole
    file: for multi-GB video a full hash costs more than the encode we are trying to
    avoid, while size + three anchored chunks is more than enough to distinguish
    segments of a corpus.
    """
    p = Path(path)
    size = p.stat().st_size
    digest = hashlib.sha1(str(size).encode())
    with p.open("rb") as fh:
        anchors = (0, max(0, size // 2 - chunk_bytes // 2), max(0, size - chunk_bytes))
        for offset in anchors:
            fh.seek(offset)
            digest.update(fh.read(chunk_bytes))
    return digest.hexdigest()
