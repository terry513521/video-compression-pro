"""Encoder registry: logical parameters -> ffmpeg arguments.

Each encoder owns its own parameter space and its own translation. Adding an encoder
means adding one subclass here; nothing else in the pipeline changes.

Two invariants every encoder must maintain, because segments are encoded independently
and then concatenated with the stream-copy concat demuxer:

1. A keyframe at frame 0 and a fixed GOP (``-g``), with scene-cut insertion disabled.
2. Constant, explicitly-set pixel format and no adaptive resolution changes.
"""

from __future__ import annotations

import shlex
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from ..errors import EncodeError
from ..log import get_logger
from . import pixfmt
from .params import EncodeParams, ParamSpace

log = get_logger(__name__)


@dataclass(frozen=True)
class EncodeRequest:
    """Everything needed to run one encode.

    ``pix_fmt`` is the ALREADY-RESOLVED format (see :func:`resolve_pix_fmt`), not the raw
    config value: resolution needs the source format and the encoder's capabilities, and
    doing it once per segment keeps it out of the per-encode hot path.
    """

    input_path: str
    output_path: str
    params: EncodeParams
    preset: str
    keyint_seconds: float
    pix_fmt: str
    fps: float
    threads: int = 0
    loglevel: str = "error"
    extra_args: tuple[str, ...] = ()


class Encoder(ABC):
    """Base class for an encoder backend."""

    name: str
    ffmpeg_encoder: str
    is_gpu: bool = False
    space: ParamSpace

    @abstractmethod
    def quality_args(self, params: EncodeParams) -> list[str]:
        """Encoder-specific flags for (crf, aq_mode, aq_strength)."""

    def gop_args(self, request: EncodeRequest) -> list[str]:
        """Closed-GOP settings so independently encoded segments concatenate cleanly."""
        fps = request.fps if request.fps > 0 else 30.0
        keyint = max(1, int(round(fps * request.keyint_seconds)))
        return ["-g", str(keyint), "-keyint_min", str(keyint), "-sc_threshold", "0"]

    def build_argv(self, ffmpeg: str, request: EncodeRequest) -> list[str]:
        """Full argument vector for the encode."""
        argv = [
            ffmpeg, "-hide_banner", "-nostdin", "-y",
            "-loglevel", request.loglevel,
            "-i", request.input_path,
            "-map", "0:v:0",
            "-an", "-sn", "-dn",  # video only: audio is copied at concat time
            "-c:v", self.ffmpeg_encoder,
            "-pix_fmt", request.pix_fmt,
        ]
        if request.preset:
            argv += ["-preset", request.preset]
        argv += self.quality_args(request.params)
        argv += self.gop_args(request)
        if request.threads > 0:
            argv += ["-threads", str(request.threads)]
        argv += list(request.extra_args)
        argv += [request.output_path]
        return argv

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Encoder {self.name} gpu={self.is_gpu}>"


class _X26xEncoder(Encoder):
    """Shared logic for libx264/libx265.

    Both take a single ``-<codec>-params`` key=value list, and both honour only the
    *last* one on the command line — so AQ and GOP settings must be merged into one
    string rather than passed as two flags.
    """

    params_flag: str
    extra_params: str = ""

    def _codec_params(self, request: EncodeRequest) -> str:
        p = self.space.clamp(request.params)
        fps = request.fps if request.fps > 0 else 30.0
        keyint = max(1, int(round(fps * request.keyint_seconds)))
        parts = [
            f"aq-mode={p.aq_mode}",
            f"aq-strength={p.aq_strength:g}",
            f"keyint={keyint}",
            f"min-keyint={keyint}",
            "scenecut=0",
        ]
        if self.extra_params:
            parts.append(self.extra_params)
        return ":".join(parts)

    def quality_args(self, params: EncodeParams) -> list[str]:
        p = self.space.clamp(params)
        return ["-crf", f"{p.crf:g}"]

    def gop_args(self, request: EncodeRequest) -> list[str]:
        # Folded into -<codec>-params instead; see _codec_params.
        return []

    def build_argv(self, ffmpeg: str, request: EncodeRequest) -> list[str]:
        p = self.space.clamp(request.params)
        argv = [
            ffmpeg, "-hide_banner", "-nostdin", "-y",
            "-loglevel", request.loglevel,
            "-i", request.input_path,
            "-map", "0:v:0", "-an", "-sn", "-dn",
            "-c:v", self.ffmpeg_encoder,
            "-pix_fmt", request.pix_fmt,
            "-preset", request.preset,
            "-crf", f"{p.crf:g}",
            self.params_flag, self._codec_params(request),
        ]
        if request.threads > 0:
            argv += ["-threads", str(request.threads)]
        argv += list(request.extra_args)
        argv += [request.output_path]
        return argv


class LibX265(_X26xEncoder):
    name = "libx265"
    ffmpeg_encoder = "libx265"
    params_flag = "-x265-params"
    extra_params = "log-level=error"
    # x265 aq-mode: 0 off, 1 variance, 2 auto-variance, 3 auto-variance + bias to dark,
    # 4 auto-variance + edge information.
    space = ParamSpace(
        crf_min=18.0, crf_max=40.0,
        aq_modes=(0, 1, 2, 3, 4),
        aq_strength_min=0.0, aq_strength_max=2.0,
    )


class LibX264(_X26xEncoder):
    name = "libx264"
    ffmpeg_encoder = "libx264"
    params_flag = "-x264-params"
    # x264 aq-mode: 0 off, 1 variance, 2 auto-variance, 3 auto-variance + bias to dark.
    space = ParamSpace(
        crf_min=18.0, crf_max=40.0,
        aq_modes=(0, 1, 2, 3),
        aq_strength_min=0.0, aq_strength_max=2.0,
    )


class LibSvtAv1(Encoder):
    """SVT-AV1.

    Modern SVT-AV1 dropped the old ``aq-mode`` knob in favour of variance boost, which
    is the same idea (spend more bits on low-variance regions) with a different name.
    The logical parameters map as:

        aq_mode     -> enable-variance-boost (0/1)
        aq_strength -> variance-boost-strength (1-4)
    """

    name = "libsvtav1"
    ffmpeg_encoder = "libsvtav1"
    space = ParamSpace(
        crf_min=20.0, crf_max=55.0,
        aq_modes=(0, 1),
        aq_strength_min=1.0, aq_strength_max=4.0,
        aq_strength_is_integer=True,
    )

    def quality_args(self, params: EncodeParams) -> list[str]:
        p = self.space.clamp(params)
        return ["-crf", f"{p.crf:g}"]

    def build_argv(self, ffmpeg: str, request: EncodeRequest) -> list[str]:
        p = self.space.clamp(request.params)
        fps = request.fps if request.fps > 0 else 30.0
        keyint = max(1, int(round(fps * request.keyint_seconds)))
        svt = (
            f"enable-variance-boost={p.aq_mode}:"
            f"variance-boost-strength={int(p.aq_strength)}:"
            f"keyint={keyint}:scd=0"
        )
        argv = [
            ffmpeg, "-hide_banner", "-nostdin", "-y",
            "-loglevel", request.loglevel,
            "-i", request.input_path,
            "-map", "0:v:0", "-an", "-sn", "-dn",
            "-c:v", self.ffmpeg_encoder,
            "-pix_fmt", request.pix_fmt,
            "-preset", request.preset,
            "-crf", f"{p.crf:g}",
            "-svtav1-params", svt,
        ]
        if request.threads > 0:
            argv += ["-threads", str(request.threads)]
        argv += list(request.extra_args)
        argv += [request.output_path]
        return argv


class _NvencEncoder(Encoder):
    """Shared logic for NVENC encoders.

    NVENC's quality knob is ``-cq`` (used with ``-rc vbr``), its AQ toggle is
    ``-spatial-aq`` and its strength is an integer 1-15 — a different scale from x264/5,
    which is precisely why the search is run per encoder rather than mapped across them.
    """

    is_gpu = True

    def quality_args(self, params: EncodeParams) -> list[str]:
        p = self.space.clamp(params)
        args = [
            "-rc", "vbr",
            "-cq", f"{p.crf:g}",
            "-b:v", "0",
            "-spatial-aq", str(1 if p.aq_mode > 0 else 0),
        ]
        if p.aq_mode > 0:
            args += ["-aq-strength", str(int(p.aq_strength))]
        return args


class HevcNvenc(_NvencEncoder):
    name = "hevc_nvenc"
    ffmpeg_encoder = "hevc_nvenc"
    space = ParamSpace(
        crf_min=18.0, crf_max=40.0,
        aq_modes=(0, 1),
        aq_strength_min=1.0, aq_strength_max=15.0,
        aq_strength_is_integer=True,
    )


class Av1Nvenc(_NvencEncoder):
    name = "av1_nvenc"
    ffmpeg_encoder = "av1_nvenc"
    space = ParamSpace(
        crf_min=18.0, crf_max=45.0,
        aq_modes=(0, 1),
        aq_strength_min=1.0, aq_strength_max=15.0,
        aq_strength_is_integer=True,
    )


class H264Nvenc(_NvencEncoder):
    name = "h264_nvenc"
    ffmpeg_encoder = "h264_nvenc"
    space = ParamSpace(
        crf_min=18.0, crf_max=38.0,
        aq_modes=(0, 1),
        aq_strength_min=1.0, aq_strength_max=15.0,
        aq_strength_is_integer=True,
    )


ENCODERS: dict[str, Encoder] = {
    encoder.name: encoder
    for encoder in (
        LibX265(), LibX264(), LibSvtAv1(),
        HevcNvenc(), Av1Nvenc(), H264Nvenc(),
    )
}


def resolve_pix_fmt(
    ffmpeg: str,
    encoder: Encoder,
    requested: str,
    source_pix_fmt: str,
    *,
    context: str = "",
) -> str:
    """Pick the pixel format for encoding this source with this encoder."""
    from ..ffmpeg.toolchain import encoder_pix_fmts

    return pixfmt.resolve(
        requested,
        source_pix_fmt,
        encoder_pix_fmts(ffmpeg, encoder.ffmpeg_encoder),
        context=context,
    )


def container_for(encoder_name: str) -> str:
    """Container extension that can carry this codec's elementary stream.

    MP4 support for AV1 is patchy across muxer versions, so AV1 output goes to Matroska.
    """
    return "mkv" if encoder_name in {"libsvtav1", "av1_nvenc"} else "mp4"


def get_encoder(name: str) -> Encoder:
    try:
        return ENCODERS[name]
    except KeyError:
        raise EncodeError(
            f"unknown encoder {name!r}. Available: {sorted(ENCODERS)}"
        ) from None


@dataclass(frozen=True)
class EncodeResult:
    output_path: str
    size_bytes: int
    seconds: float
    argv: list[str]

    @property
    def command(self) -> str:
        return shlex.join(self.argv)


def encode(ffmpeg: str, encoder: Encoder, request: EncodeRequest) -> EncodeResult:
    """Run one encode.

    There is deliberately no fallback encoder. The reference retried with ``libsvtav1``
    while keeping a CQ calibrated for a different codec and reported success — producing
    an output in an unrequested codec at an unintended quality. Here a failure is an
    error the caller must handle.
    """
    from ..ffmpeg.run import run  # local import keeps encoding/ free of a hard cycle

    argv = encoder.build_argv(ffmpeg, request)
    output = Path(request.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    result = run(argv, timeout=None)

    if not output.is_file() or output.stat().st_size == 0:
        raise EncodeError(
            f"{encoder.name} exited 0 but produced no output: {output}\n"
            f"command: {shlex.join(argv)}"
        )

    size = output.stat().st_size
    log.debug(
        "encoded %s [%s] -> %.2f MB in %.1fs",
        output.name, request.params, size / 1e6, result.seconds,
    )
    return EncodeResult(
        output_path=str(output), size_bytes=size, seconds=result.seconds, argv=argv
    )
