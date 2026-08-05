#!/usr/bin/env python3
"""Cross-platform environment setup for vidopt.

Works on Windows, Linux and macOS with nothing but a Python 3.10+ interpreter — no
bash, no make, no Docker. Written against the standard library only, so it runs before
any dependency is installed.

    python scripts/setup.py                 # fetch ffmpeg + install vidopt
    python scripts/setup.py --ffmpeg-only   # just the ffmpeg toolchain
    python scripts/setup.py --check         # verify an existing setup

What it does:

1. Downloads a prebuilt ffmpeg for this platform into ``vendor/ffmpeg`` and verifies it
   has ``libvmaf`` plus the encoders the pipeline needs. This matters: most ffmpeg
   packages — including the ones on Windows package managers and in Linux distributions
   — are built WITHOUT libvmaf, and then quality cannot be measured at all.
2. Installs vidopt and its dependencies into the current interpreter (or a virtualenv
   you have already activated).

Rerunning is safe; existing pieces are reused.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import NoReturn

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor"
FFMPEG_DIR = VENDOR / "ffmpeg"

EXE = ".exe" if os.name == "nt" else ""

# Builds that include libvmaf. BtbN publishes GPL builds for Windows and Linux with
# libvmaf, libx264, libx265, SVT-AV1 and NVENC all compiled in.
FFMPEG_URLS = {
    ("Windows", "AMD64"): (
        "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
        "ffmpeg-master-latest-win64-gpl.zip"
    ),
    ("Linux", "x86_64"): (
        "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
        "ffmpeg-master-latest-linux64-gpl.tar.xz"
    ),
    ("Linux", "aarch64"): (
        "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
        "ffmpeg-master-latest-linuxarm64-gpl.tar.xz"
    ),
}

REQUIRED_FILTERS = ("libvmaf",)
REQUIRED_ENCODERS = ("libx264", "libx265")


def info(message: str) -> None:
    print(f"==> {message}", flush=True)


def detail(message: str) -> None:
    print(f"    {message}", flush=True)


def fail(message: str) -> NoReturn:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def platform_key() -> tuple[str, str]:
    return platform.system(), platform.machine()


def ffmpeg_binary() -> Path:
    """Prefer the vendored build; fall back to VIDOPT_FFMPEG_DIR / PATH."""
    vendored = FFMPEG_DIR / "bin" / f"ffmpeg{EXE}"
    if vendored.is_file():
        return vendored

    env_dir = os.environ.get("VIDOPT_FFMPEG_DIR")
    if env_dir:
        candidate = Path(env_dir).expanduser() / f"ffmpeg{EXE}"
        if candidate.is_file():
            return candidate

    found = shutil.which(f"ffmpeg{EXE}")
    if found:
        return Path(found)
    return vendored


def ffprobe_binary() -> Path:
    ffmpeg = ffmpeg_binary()
    sibling = ffmpeg.parent / f"ffprobe{EXE}"
    if sibling.is_file():
        return sibling
    found = shutil.which(f"ffprobe{EXE}")
    return Path(found) if found else sibling


def vendor_from_system() -> bool:
    """Copy a working system ffmpeg (with libvmaf) into vendor/ for offline use."""
    src = ffmpeg_binary()
    if not src.is_file():
        return False
    # Already vendored.
    vendored_ffmpeg = FFMPEG_DIR / "bin" / f"ffmpeg{EXE}"
    if vendored_ffmpeg.is_file():
        return True
    try:
        if FFMPEG_DIR.exists() and src.resolve().is_relative_to(FFMPEG_DIR.resolve()):
            return True
    except (ValueError, OSError):
        pass

    info(f"vendoring system ffmpeg from {src.parent}")
    dest_bin = FFMPEG_DIR / "bin"
    dest_bin.mkdir(parents=True, exist_ok=True)
    for name in (f"ffmpeg{EXE}", f"ffprobe{EXE}"):
        source = src.parent / name
        if not source.is_file():
            fail(f"expected {source} next to ffmpeg")
        shutil.copy2(source, dest_bin / name)
    detail(f"copied into {dest_bin}")
    return True


# --------------------------------------------------------------------------------------
# ffmpeg
# --------------------------------------------------------------------------------------


def download(url: str, dest: Path) -> None:
    detail(f"downloading {url}")

    def report(block: int, block_size: int, total: int) -> None:
        if total > 0:
            done = min(100, block * block_size * 100 // total)
            print(f"\r    {done:3d}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=report)  # noqa: S310
    print(flush=True)


def extract(archive: Path, into: Path) -> Path:
    """Unpack and return the single top-level directory the archive contains."""
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(into)
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(into)  # noqa: S202 - trusted release artifact

    entries = [p for p in into.iterdir() if p.is_dir()]
    if len(entries) != 1:
        fail(f"unexpected archive layout: {[p.name for p in entries]}")
    return entries[0]


def install_ffmpeg(force: bool = False) -> None:
    if ffmpeg_binary().is_file() and not force:
        info("ffmpeg already present, skipping download")
        return

    key = platform_key()
    url = FFMPEG_URLS.get(key)
    if url is None:
        fail(
            f"no prebuilt ffmpeg configured for {key[0]}/{key[1]}.\n"
            "       Install an ffmpeg built with --enable-libvmaf yourself, then point\n"
            "       vidopt at it:  set VIDOPT_FFMPEG_DIR=<dir containing ffmpeg>"
        )

    info(f"fetching ffmpeg for {key[0]}/{key[1]}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / Path(url).name
        download(url, archive)

        staging = tmp_path / "unpacked"
        staging.mkdir()
        extracted = extract(archive, staging)

        if FFMPEG_DIR.exists():
            shutil.rmtree(FFMPEG_DIR)
        FFMPEG_DIR.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extracted), str(FFMPEG_DIR))

    # ffplay drags in GUI libraries this project never uses and is ~140 MB.
    for junk in FFMPEG_DIR.glob(f"bin/ffplay{EXE}"):
        junk.unlink(missing_ok=True)

    if os.name != "nt":
        for tool in ("ffmpeg", "ffprobe"):
            binary = FFMPEG_DIR / "bin" / tool
            if binary.is_file():
                binary.chmod(0o755)


def check_ffmpeg() -> bool:
    binary = ffmpeg_binary()
    if not binary.is_file():
        detail(f"missing: {binary}")
        return False

    def listing(flag: str) -> str:
        # Read fully rather than piping into a search: on POSIX a short-circuiting
        # reader makes ffmpeg die of SIGPIPE, which looks like a missing feature.
        result = subprocess.run(
            [str(binary), "-hide_banner", flag],
            capture_output=True, text=True, check=False,
        )
        return result.stdout + result.stderr

    version = subprocess.run(
        [str(binary), "-hide_banner", "-version"],
        capture_output=True, text=True, check=False,
    ).stdout.splitlines()
    detail(version[0] if version else "unknown version")

    filters = listing("-filters")
    encoders = listing("-encoders")

    ok = True
    for name in REQUIRED_FILTERS:
        if name not in filters:
            detail(f"MISSING filter: {name} — VMAF cannot be measured")
            ok = False
    for name in REQUIRED_ENCODERS:
        if name not in encoders:
            detail(f"MISSING encoder: {name}")
            ok = False

    if ok:
        extras = [n for n in ("libsvtav1", "hevc_nvenc", "av1_nvenc") if n in encoders]
        detail("libvmaf, libx264, libx265: present")
        if extras:
            detail(f"also available: {', '.join(extras)}")
    return ok


# --------------------------------------------------------------------------------------
# Python package
# --------------------------------------------------------------------------------------


def in_virtualenv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def install_package() -> None:
    """Install deps + vidopt into the current interpreter.

    Offline (wheelhouse present): install pinned wheels, then copy ``src/vidopt``
    into site-packages. Avoids editable/wheel builds that fail on Windows
    (``WinError 32`` / old setuptools ``project.license``).

    Online: editable ``pip install -e .``.
    """
    info("installing vidopt and its dependencies")
    if not in_virtualenv():
        detail("note: not in a virtualenv; installing into the current interpreter")

    wheelhouse = VENDOR / "wheelhouse"
    offline = wheelhouse.is_dir() and any(wheelhouse.iterdir())
    deps = [
        "numpy==2.1.3",
        "scipy==1.14.1",
        "scikit-learn==1.5.2",
        "joblib==1.4.2",
        "opencv-python-headless==4.10.0.84",
        "scenedetect==0.6.5",
        "PyYAML==6.0.2",
    ]

    if offline:
        detail(f"using the offline wheelhouse at {wheelhouse}")
        result = subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "--no-index", "--find-links", str(wheelhouse),
                "--no-warn-script-location",
                *deps,
            ],
            check=False,
        )
        if result.returncode != 0:
            fail("pip install of dependencies failed; see the output above")
        _install_vidopt_by_copy()
        return

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(ROOT)],
        check=False,
    )
    if result.returncode != 0:
        fail("pip install failed; see the output above")


def _install_vidopt_by_copy() -> None:
    """Copy application code into site-packages (no wheel / editable build)."""
    import sysconfig

    src = ROOT / "src" / "vidopt"
    if not (src / "__init__.py").is_file():
        fail(f"missing application source at {src}")

    dest = Path(sysconfig.get_path("purelib")) / "vidopt"
    detail(f"copying {src} -> {dest}")
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)

    scripts = Path(sysconfig.get_path("scripts"))
    scripts.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        launcher = scripts / "vidopt.cmd"
        launcher.write_text(
            f'@echo off\r\n"{sys.executable}" -m vidopt %*\r\n',
            encoding="utf-8",
        )
    else:
        launcher = scripts / "vidopt"
        launcher.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" -m vidopt "$@"\n',
            encoding="utf-8",
        )
        launcher.chmod(0o755)
    detail(f"launcher: {launcher}")


def check_package() -> bool:
    try:
        subprocess.run(
            [sys.executable, "-c", "import vidopt; print(vidopt.__version__)"],
            check=True, capture_output=True, text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        detail("vidopt is not importable from this interpreter")
        return False
    return True


def verify_toolchain() -> bool:
    """Encode a synthetic clip and measure VMAF on it, end to end.

    Stronger than --check: it proves the toolchain can do the actual work rather than
    that the right strings appear in a capability listing. Safe to use as a gate.
    """
    binary = ffmpeg_binary()
    if not binary.is_file():
        detail(f"missing: {binary}")
        return False

    info("running a real encode and VMAF measurement")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ref = tmp_path / "ref.mp4"
        dis = tmp_path / "dis.mp4"
        log_name = "vmaf.json"
        log_path = tmp_path / log_name

        def ffmpeg(*argv: str, cwd: Path | None = None) -> bool:
            result = subprocess.run(
                [str(binary), "-hide_banner", "-loglevel", "error", "-y", *argv],
                capture_output=True, text=True, check=False,
                cwd=str(cwd) if cwd else None,
            )
            if result.returncode != 0:
                detail(f"ffmpeg failed: {result.stderr.strip()[:300]}")
            return result.returncode == 0

        if not ffmpeg(
            "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=2",
            "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", str(ref),
        ):
            return False
        if not ffmpeg(
            "-i", str(ref), "-c:v", "libx265", "-crf", "38",
            "-pix_fmt", "yuv420p", str(dis),
        ):
            return False
        detail("encode: OK")

        # Relative log_path + cwd: Windows drive letters (C:) break filter options.
        graph = (
            "[0:v]settb=AVTB,setpts=N-STARTPTS[d];"
            "[1:v]settb=AVTB,setpts=N-STARTPTS[r];"
            "[d][r]libvmaf=model=version=vmaf_v0.6.1neg:log_fmt=json:"
            f"log_path={log_name}"
        )
        if not ffmpeg(
            "-i", str(dis), "-i", str(ref), "-lavfi", graph, "-f", "null", "-",
            cwd=tmp_path,
        ):
            return False

        import json

        data = json.loads(log_path.read_text())
        frames = len(data.get("frames", []))
        score = data["pooled_metrics"]["vmaf"]["harmonic_mean"]
        detail(f"VMAF: {score:.2f} over {frames} frame(s)")

        # The reference is 60 frames. A different count means the two streams were not
        # compared frame-for-frame, and any score would be meaningless.
        if frames != 60:
            detail(f"FAIL: expected 60 scored frames, got {frames} — inputs misaligned")
            return False
        if not 20.0 < score < 95.0:
            detail(f"FAIL: implausible VMAF {score} for a crf 10 vs crf 38 pair")
            return False

    detail("VMAF measurement is working and frame-aligned")
    return True


# --------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ffmpeg-only", action="store_true", help="Only fetch ffmpeg.")
    parser.add_argument(
        "--skip-ffmpeg", action="store_true", help="Only install the Python package."
    )
    parser.add_argument("--force", action="store_true", help="Re-download ffmpeg.")
    parser.add_argument(
        "--vendor-system-ffmpeg",
        action="store_true",
        help="Copy ffmpeg/ffprobe from VIDOPT_FFMPEG_DIR or PATH into vendor/ffmpeg.",
    )
    parser.add_argument(
        "--check", action="store_true", help="Verify an existing setup and exit."
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Deeper check: run a real encode and a real VMAF measurement.",
    )
    args = parser.parse_args()

    print(f"vidopt setup — {platform.system()} {platform.machine()}, "
          f"Python {sys.version.split()[0]}")
    print(f"project: {ROOT}\n")

    if args.verify:
        binary = ffmpeg_binary()
        detail(f"using {binary}")
        return 0 if verify_toolchain() else 1

    if args.check:
        info("checking ffmpeg")
        detail(f"using {ffmpeg_binary()}")
        ffmpeg_ok = check_ffmpeg()
        info("checking the Python package")
        package_ok = check_package()
        if ffmpeg_ok and package_ok:
            print("\nSetup looks good. Next: vidopt doctor")
            return 0
        print("\nSetup is incomplete. Run: python scripts/setup.py")
        return 1

    # Deliberately not guarded by the project's requires-python: this script is what
    # runs BEFORE anything is installed, so it may well be started by an older
    # interpreter and must say so clearly rather than crash on new syntax.
    if sys.version_info < (3, 10):  # noqa: UP036
        fail(f"Python 3.10+ required, this is {sys.version.split()[0]}")

    if args.vendor_system_ffmpeg:
        if not vendor_from_system():
            fail(
                "no system ffmpeg found. Set VIDOPT_FFMPEG_DIR or put ffmpeg on PATH."
            )
        info("verifying the vendored ffmpeg build")
        if not check_ffmpeg():
            fail("the system ffmpeg lacks libvmaf or a required encoder.")
        if args.ffmpeg_only:
            print()
            info("done")
            return 0

    if not args.skip_ffmpeg and not args.vendor_system_ffmpeg:
        install_ffmpeg(force=args.force)
        info("verifying the ffmpeg build")
        if not check_ffmpeg():
            fail(
                "the downloaded ffmpeg lacks something the pipeline needs.\n"
                "       Point vidopt at a build made with --enable-libvmaf via\n"
                "       VIDOPT_FFMPEG_DIR, or use --vendor-system-ffmpeg."
            )

    if not args.ffmpeg_only:
        install_package()

    print()
    info("done")
    detail("next:  vidopt doctor")
    detail("       vidopt dev <corpus-dir>")
    detail("       vidopt compress in.mp4 -o out.mp4 --target 89 --verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
