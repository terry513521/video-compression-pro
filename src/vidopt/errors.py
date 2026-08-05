"""Typed exception hierarchy.

Every failure mode in the pipeline maps to one of these. Nothing in this package
returns a sentinel or a fabricated value on failure — the reference implementation
did (see REFERENCE_ANALYSIS.md), and a plausible-looking wrong answer is far more
expensive than a crash.
"""

from __future__ import annotations


class VidoptError(Exception):
    """Base class for every error raised by this package."""


class ConfigError(VidoptError):
    """Configuration is missing, malformed, or internally inconsistent."""


class ToolchainError(VidoptError):
    """ffmpeg/ffprobe is missing or lacks a required capability."""


class CommandError(VidoptError):
    """An external command exited non-zero."""

    def __init__(self, argv: list[str], returncode: int, stderr: str) -> None:
        self.argv = argv
        self.returncode = returncode
        self.stderr = stderr
        # Tails are what matter: ffmpeg prints the real diagnosis last.
        tail = stderr.strip().splitlines()[-12:]
        super().__init__(
            f"command failed (exit {returncode}): {' '.join(argv[:6])} ...\n"
            + "\n".join(tail)
        )


class ProbeError(VidoptError):
    """ffprobe could not describe a media file."""


class SegmentationError(VidoptError):
    """Scene segmentation failed or produced no usable segments."""


class FeatureExtractionError(VidoptError):
    """A segment could not be decoded or analysed."""


class EncodeError(VidoptError):
    """An encode failed. Never recovered by substituting a different encoder."""


class VmafError(VidoptError):
    """A VMAF measurement failed or is not trustworthy."""


class SearchError(VidoptError):
    """The parameter search could not produce a usable result."""


class ModelError(VidoptError):
    """Training failed, or a model bundle is missing/incompatible."""
