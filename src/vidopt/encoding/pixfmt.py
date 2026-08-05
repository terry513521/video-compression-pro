"""Pixel-format resolution.

Production input is arbitrary: 8-bit 4:2:0 from a phone, 10-bit 4:2:2 from a camera,
4:4:4 from a screen capture. Forcing everything to ``yuv420p`` — which the first version
of this project did — silently throws away bit depth and chroma resolution, and the loss
never shows up in the VMAF score because VMAF compares the *downconverted* result against
a reference it also downconverts.

So the default is ``auto``: keep the source format when the encoder supports it, and when
it does not, step down along an explicit preference order and **say so in the log**.
"""

from __future__ import annotations

from ..log import get_logger

log = get_logger(__name__)

# Fallback order within a chroma family, most preferred first. The aim is to lose as
# little as possible: prefer keeping bit depth over keeping chroma resolution, because
# banding from an 8-bit downconvert is more visible than 4:2:0 chroma on real content.
_FALLBACKS: dict[str, tuple[str, ...]] = {
    "yuv444p12le": ("yuv444p10le", "yuv422p10le", "yuv420p10le", "yuv444p", "yuv420p"),
    "yuv422p12le": ("yuv422p10le", "yuv420p10le", "yuv422p", "yuv420p"),
    "yuv420p12le": ("yuv420p10le", "yuv420p"),
    "yuv444p10le": ("yuv422p10le", "yuv420p10le", "yuv444p", "yuv420p"),
    "yuv422p10le": ("yuv420p10le", "yuv422p", "yuv420p"),
    "yuv420p10le": ("yuv420p",),
    "p010le": ("yuv420p10le", "yuv420p"),
    "yuv444p": ("yuv422p", "yuv420p"),
    "yuv422p": ("yuv420p",),
    "yuvj420p": ("yuv420p",),
    "yuvj422p": ("yuv422p", "yuv420p"),
    "yuvj444p": ("yuv444p", "yuv420p"),
    "gbrp": ("yuv444p", "yuv420p"),
    "nv12": ("yuv420p",),
    "yuva420p": ("yuv420p",),
}

DEFAULT = "yuv420p"


def is_high_bit_depth(pix_fmt: str) -> bool:
    return any(marker in pix_fmt for marker in ("10le", "12le", "10be", "12be", "p010"))


def resolve(
    requested: str,
    source_pix_fmt: str,
    supported: frozenset[str],
    *,
    context: str = "",
) -> str:
    """Choose the pixel format to encode with.

    Args:
        requested: ``encoder.pix_fmt`` from the config. ``"auto"`` means follow the
            source; anything else is an explicit override and is honoured as given.
        source_pix_fmt: What the input actually is.
        supported: What this encoder build accepts. Empty means the probe failed, in
            which case the source format is passed through and ffmpeg decides.
        context: File name or similar, for log messages.

    Returns:
        A pixel-format name to hand to ``-pix_fmt``.
    """
    label = f" for {context}" if context else ""

    if requested and requested != "auto":
        # An explicit setting is a deliberate choice; do not second-guess it, but do warn
        # when it silently discards bit depth.
        if is_high_bit_depth(source_pix_fmt) and not is_high_bit_depth(requested):
            log.warning(
                "encoder.pix_fmt=%s downconverts a %s source%s to 8-bit. "
                "Set encoder.pix_fmt=auto to preserve it.",
                requested, source_pix_fmt, label,
            )
        return requested

    if not source_pix_fmt or source_pix_fmt == "unknown":
        return DEFAULT

    if not supported:
        # Probe failed. Passing the source format through is the least surprising thing;
        # ffmpeg will reject it loudly if the encoder really cannot take it.
        log.debug(
            "no pixel-format list for this encoder; passing through %s", source_pix_fmt
        )
        return source_pix_fmt

    if source_pix_fmt in supported:
        return source_pix_fmt

    for candidate in _FALLBACKS.get(source_pix_fmt, ()):
        if candidate in supported:
            level = log.warning if is_high_bit_depth(source_pix_fmt) else log.info
            level(
                "encoder cannot take %s%s; using %s instead",
                source_pix_fmt, label, candidate,
            )
            return candidate

    if DEFAULT in supported:
        log.warning(
            "encoder cannot take %s%s and no close match is available; "
            "falling back to %s",
            source_pix_fmt, label, DEFAULT,
        )
        return DEFAULT

    chosen = sorted(supported)[0]
    log.warning("unusual encoder pixel-format set%s; using %s", label, chosen)
    return chosen
