"""Lossless concatenation of encoded segments.

Uses ffmpeg's concat demuxer with ``-c copy``: no re-encode, so the bits chosen by the
per-segment parameter search survive into the final file. This requires every segment to
share codec, resolution, pixel format and timebase — which the encoder layer guarantees
by writing closed GOPs with identical settings.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ..errors import EncodeError
from ..ffmpeg.run import run
from ..ffmpeg.toolchain import Capabilities
from ..log import get_logger

log = get_logger(__name__)


def _concat_line(path: Path) -> str:
    """A concat-demuxer line with the quoting rules the format actually uses.

    Forward slashes even on Windows: the concat demuxer treats a backslash as an escape
    character, so ``C:\\work\\seg.mp4`` is read as ``C:worksseg.mp4``. ffmpeg accepts
    forward slashes on every platform, which sidesteps the whole problem.
    """
    escaped = path.resolve().as_posix().replace("'", "'\\''")
    return f"file '{escaped}'\n"


def concat(
    segments: list[str | Path],
    output: str | Path,
    caps: Capabilities,
    *,
    faststart: bool = True,
    audio_from: str | Path | None = None,
) -> Path:
    """Concatenate encoded segments into ``output`` without re-encoding.

    Args:
        segments: Encoded video segments, in order.
        output: Destination file.
        audio_from: If given, every non-video stream of this file (audio, subtitles,
            chapters) is muxed into the result with ``-c copy``.

    Why audio is taken from the *original* rather than carried through the segments:
    splitting audio at video scene cuts and re-joining it is a reliable source of drift,
    because audio frame boundaries do not line up with video frame boundaries. The video
    timeline is preserved exactly by the segment-and-concat process, so grafting the
    untouched original audio back on at the end keeps A/V sync exact and costs nothing —
    the audio is never decoded.
    """
    if not segments:
        raise EncodeError("nothing to concatenate")

    paths = [Path(s) for s in segments]
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise EncodeError(f"segment missing or empty: {path}")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    list_path: Path | None = None
    try:
        argv = [caps.ffmpeg, "-hide_banner", "-nostdin", "-y", "-loglevel", "error"]

        if len(paths) == 1:
            # Still remux rather than copying the file, so the container matches what a
            # multi-segment run produces.
            argv += ["-i", str(paths[0])]
        else:
            with tempfile.NamedTemporaryFile(
                "w", suffix=".txt", delete=False, encoding="utf-8"
            ) as handle:
                handle.writelines(_concat_line(p) for p in paths)
                list_path = Path(handle.name)
            argv += ["-f", "concat", "-safe", "0", "-i", str(list_path)]

        if audio_from is not None:
            argv += ["-i", str(Path(audio_from))]
            # Video from the concatenated stream, everything else from the original.
            # The '?' suffixes make each mapping optional, so a silent video still works.
            argv += [
                "-map", "0:v:0",
                "-map", "1:a?",
                "-map", "1:s?",
                "-map_chapters", "1",
            ]
        else:
            argv += ["-map", "0:v:0"]

        argv += ["-c", "copy"]
        if faststart:
            argv += ["-movflags", "+faststart"]
        argv += [str(out_path)]

        run(argv, timeout=None)
    finally:
        if list_path is not None:
            list_path.unlink(missing_ok=True)

    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise EncodeError(f"concat produced no output: {out_path}")

    log.info(
        "concatenated %d segment(s) -> %s (%.2f MB)%s",
        len(paths), out_path.name, out_path.stat().st_size / 1e6,
        " with original audio" if audio_from is not None else "",
    )
    return out_path
