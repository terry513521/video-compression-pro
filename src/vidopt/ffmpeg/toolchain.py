"""ffmpeg/ffprobe discovery and capability probing.

The reference scattered bare ``subprocess.run(["ffmpeg", ...])`` calls across ~15 files
and inferred GPU support by grepping ``ffmpeg -encoders``. That listing reports
``av1_nvenc`` on machines with no NVIDIA device at all, which is how a "GPU pipeline"
ends up silently swapping to a CPU codec mid-run.

Here discovery happens once, in one place, and NVENC is probed by *actually encoding a
frame*.
"""

from __future__ import annotations

import functools
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import ToolchainError
from ..log import get_logger
from .run import run

log = get_logger(__name__)

_VENDOR_SUBPATH = Path("vendor") / "ffmpeg" / "bin"

# Windows names executables with a suffix and puts a venv's scripts in Scripts\ rather
# than bin/. Both differences matter only here, in discovery.
EXE_SUFFIX = ".exe" if os.name == "nt" else ""
_VENV_BIN = "Scripts" if os.name == "nt" else "bin"


def _binary_names() -> tuple[str, str]:
    return f"ffmpeg{EXE_SUFFIX}", f"ffprobe{EXE_SUFFIX}"


@dataclass(frozen=True)
class Capabilities:
    """What this ffmpeg build can actually do, on this machine."""

    ffmpeg: str
    ffprobe: str
    version: str
    encoders: frozenset[str] = field(default_factory=frozenset)
    filters: frozenset[str] = field(default_factory=frozenset)

    @property
    def has_libvmaf(self) -> bool:
        return "libvmaf" in self.filters

    @property
    def has_libvmaf_cuda(self) -> bool:
        return "libvmaf_cuda" in self.filters

    def has_encoder(self, name: str) -> bool:
        return name in self.encoders

    def describe(self) -> str:
        interesting = sorted(
            e
            for e in self.encoders
            if e.startswith(("libx26", "libsvtav1", "libvpx", "libaom"))
            or e.endswith(("_nvenc", "_qsv", "_vaapi"))
        )
        return (
            f"ffmpeg={self.ffmpeg}\n"
            f"version={self.version}\n"
            f"libvmaf={self.has_libvmaf} libvmaf_cuda={self.has_libvmaf_cuda}\n"
            f"encoders={', '.join(interesting)}"
        )


def _search_roots() -> list[Path]:
    """Repo root candidates that might contain vendor/ffmpeg/bin."""
    here = Path(__file__).resolve()
    return [p for p in here.parents]


def _find_binaries(bin_dir: str | None) -> tuple[str, str]:
    """Resolve (ffmpeg, ffprobe).

    Precedence: config -> ``VIDOPT_FFMPEG_DIR`` -> the active environment's script
    directory -> ``vendor/ffmpeg/bin`` relative to the source tree -> ``PATH``.

    The environment entry matters more than it looks. Walking up from ``__file__`` only
    reaches ``vendor/`` for an editable install; after a real wheel install the package
    lives in ``site-packages`` and that search finds nothing, so resolution used to fall
    through to ``PATH`` and pick up the *system* ffmpeg — which on most Linux
    distributions is built without libvmaf. The result was a working-looking install
    that could not measure quality.

    Windows differs in two ways, both handled here: executables carry ``.exe``, and a
    virtualenv keeps them in ``Scripts\\`` rather than ``bin/``.
    """
    ffmpeg_name, ffprobe_name = _binary_names()
    candidates: list[Path] = []

    if bin_dir:
        candidates.append(Path(bin_dir).expanduser())

    env_dir = os.environ.get("VIDOPT_FFMPEG_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())

    prefix = Path(sys.prefix)
    candidates.append(prefix / _VENV_BIN)
    if os.name == "nt":
        # Some Windows layouts put tools directly in the prefix.
        candidates.append(prefix)

    for root in _search_roots():
        candidates.append(root / _VENDOR_SUBPATH)

    for directory in candidates:
        ffmpeg = directory / ffmpeg_name
        ffprobe = directory / ffprobe_name
        if ffmpeg.is_file() and ffprobe.is_file():
            # os.access(X_OK) is meaningful on POSIX and near-meaningless on Windows,
            # so only enforce it where it says something.
            if os.name == "nt" or os.access(ffmpeg, os.X_OK):
                return str(ffmpeg), str(ffprobe)

    # shutil.which applies PATHEXT on Windows, so the bare names are correct here.
    path_ffmpeg = shutil.which("ffmpeg")
    path_ffprobe = shutil.which("ffprobe")
    if path_ffmpeg and path_ffprobe:
        return path_ffmpeg, path_ffprobe

    tried = ", ".join(str(c) for c in candidates) or "(none)"
    raise ToolchainError(
        "could not find ffmpeg and ffprobe.\n"
        f"Searched: {tried}, then PATH.\n"
        "Run `python scripts/setup.py` to fetch a suitable ffmpeg into vendor/ffmpeg/, "
        "or set ffmpeg.bin_dir in the config / VIDOPT_FFMPEG_DIR in the environment."
    )


def _parse_listing(text: str, name_column: int) -> frozenset[str]:
    """Parse `ffmpeg -encoders` / `-filters` table output into a set of names."""
    names: set[str] = set()
    body = False
    for line in text.splitlines():
        if not body:
            if line.strip().startswith("---") or line.strip().startswith("==="):
                body = True
            continue
        parts = line.split()
        if len(parts) > name_column:
            names.add(parts[name_column])
    return frozenset(names)


@functools.lru_cache(maxsize=8)
def detect(bin_dir: str | None = None) -> Capabilities:
    """Probe the toolchain. Cached per ``bin_dir`` for the process lifetime."""
    ffmpeg, ffprobe = _find_binaries(bin_dir)

    version_out = run([ffmpeg, "-hide_banner", "-version"]).stdout
    version = version_out.splitlines()[0] if version_out else "unknown"

    encoders = _parse_listing(
        run([ffmpeg, "-hide_banner", "-encoders"], check=False).stdout, name_column=1
    )
    filters = _parse_listing(
        run([ffmpeg, "-hide_banner", "-filters"], check=False).stdout, name_column=1
    )

    caps = Capabilities(
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        version=version,
        encoders=encoders,
        filters=filters,
    )
    log.info("toolchain: %s", version)
    log.info(
        "capabilities: libvmaf=%s libvmaf_cuda=%s encoders=%d",
        caps.has_libvmaf,
        caps.has_libvmaf_cuda,
        len(caps.encoders),
    )
    return caps


@functools.lru_cache(maxsize=32)
def encoder_pix_fmts(ffmpeg: str, encoder: str) -> frozenset[str]:
    """Pixel formats this build's ``encoder`` accepts.

    Parsed from ``ffmpeg -h encoder=NAME`` rather than hard-coded, because the answer
    depends on how the binary was compiled: an x265 built without high-bit-depth support
    lists no 10-bit formats, and NVENC's list varies by SDK version. Hard-coding it would
    mean silently downconverting 10-bit sources on builds that could have handled them.
    """
    try:
        out = run([ffmpeg, "-hide_banner", "-h", f"encoder={encoder}"], check=False)
    except Exception as exc:  # noqa: BLE001
        log.debug("could not probe pixel formats for %s: %s", encoder, exc)
        return frozenset()

    for line in (out.stdout + out.stderr).splitlines():
        if "Supported pixel formats:" in line:
            _, _, listed = line.partition("Supported pixel formats:")
            return frozenset(listed.split())
    return frozenset()


@functools.lru_cache(maxsize=16)
def encoder_works(ffmpeg: str, encoder: str) -> bool:
    """Test an encoder by encoding one real frame with it.

    This is the check that distinguishes 'the build has NVENC compiled in' from 'this
    machine has a working NVIDIA device'. Cached, because it costs ~200 ms.
    """
    argv = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1:r=10",
        "-frames:v", "1", "-c:v", encoder, "-f", "null", "-",
    ]
    try:
        run(argv, timeout=60)
    except Exception as exc:  # noqa: BLE001 - any failure means "unusable here"
        log.debug("encoder %s unusable: %s", encoder, exc)
        return False
    return True


def require(
    caps: Capabilities, *, encoder: str | None = None, vmaf: bool = False
) -> None:
    """Assert the capabilities a run needs, with an actionable message if absent."""
    if vmaf and not caps.has_libvmaf:
        raise ToolchainError(
            f"this ffmpeg has no libvmaf filter, so VMAF cannot be measured.\n"
            f"  binary: {caps.ffmpeg}\n"
            f"  {caps.version}\n"
            "Most prebuilt ffmpeg packages omit libvmaf. Fetch a suitable build with "
            "`python scripts/setup.py`."
        )

    if encoder is not None:
        if not caps.has_encoder(encoder):
            raise ToolchainError(
                f"encoder {encoder!r} is not present in this ffmpeg build.\n"
                f"Available: {', '.join(sorted(caps.encoders))[:400]}"
            )
        if not encoder_works(caps.ffmpeg, encoder):
            hint = (
                " NVENC encoders are listed by every ffmpeg build with NVIDIA support "
                "compiled in, but need an actual NVIDIA device plus a matching driver. "
                "Check `nvidia-smi`, or switch to a CPU encoder "
                "(--set encoder.name=libx265)."
                if encoder.endswith("_nvenc")
                else ""
            )
            raise ToolchainError(
                f"encoder {encoder!r} is present but failed a one-frame test encode on "
                f"this machine.{hint}"
            )
