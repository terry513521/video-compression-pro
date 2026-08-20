#!/usr/bin/env python3
"""Download offline installables for a Windows x64 production package.

On a connected Windows machine, run once before ``install.bat`` and
``pack_production.bat``:

    python scripts/prepare_offline_bundle.py
    install.bat
    scripts/pack_production.bat

Downloads into ``vendor/installers/``, ``vendor/wheelhouse/``, and fetches ffmpeg
into ``vendor/ffmpeg/`` (via ``scripts/setup.py --ffmpeg-only``).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor"
INSTALLERS = VENDOR / "installers"
WHEELHOUSE = VENDOR / "wheelhouse"
FFMPEG_BIN = VENDOR / "ffmpeg" / "bin"

PYTHON_VERSION = "3.11.9"
EMBED_NAME = f"python-{PYTHON_VERSION}-embed-amd64.zip"
INSTALLER_NAME = f"python-{PYTHON_VERSION}-amd64.exe"
GET_PIP_NAME = "get-pip.py"

URLS = {
    EMBED_NAME: (
        f"https://www.python.org/ftp/python/{PYTHON_VERSION}/{EMBED_NAME}"
    ),
    INSTALLER_NAME: (
        f"https://www.python.org/ftp/python/{PYTHON_VERSION}/{INSTALLER_NAME}"
    ),
    GET_PIP_NAME: "https://bootstrap.pypa.io/get-pip.py",
}

PINNED = [
    "pip",
    "setuptools",
    "wheel",
    "numpy==2.1.3",
    "scipy==1.14.1",
    "scikit-learn==1.5.2",
    "joblib==1.4.2",
    "opencv-python-headless==4.10.0.84",
    "scenedetect==0.6.5",
    "PyYAML==6.0.2",
]


def info(msg: str) -> None:
    print(f"==> {msg}", flush=True)


def detail(msg: str) -> None:
    print(f"    {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def download(url: str, dest: Path, *, force: bool) -> None:
    if dest.is_file() and not force:
        detail(f"already have {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    detail(f"downloading {url}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310


def download_installers(*, force: bool) -> None:
    info("installers -> vendor/installers/")
    INSTALLERS.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        download(url, INSTALLERS / name, force=force)


def download_wheelhouse(*, force: bool) -> None:
    info("Python wheels -> vendor/wheelhouse/")
    WHEELHOUSE.mkdir(parents=True, exist_ok=True)
    if not force and any(WHEELHOUSE.glob("*.whl")):
        detail("wheelhouse already populated (use --force to re-download)")
        return
    if force:
        for whl in WHEELHOUSE.glob("*.whl"):
            whl.unlink()

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "-d",
        str(WHEELHOUSE),
        *PINNED,
    ]
    detail(" ".join(cmd))
    result = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if result.returncode != 0:
        fail("pip download failed; ensure pip and network access on this machine")
    count = sum(1 for _ in WHEELHOUSE.glob("*.whl"))
    detail(f"{count} wheel(s) in {WHEELHOUSE}")


def fetch_ffmpeg(*, skip: bool, vendor_system: bool) -> None:
    if skip:
        info("skipping ffmpeg download")
        return
    if FFMPEG_BIN.joinpath("ffmpeg.exe").is_file() and FFMPEG_BIN.joinpath(
        "ffprobe.exe"
    ).is_file():
        detail("vendor/ffmpeg/bin already present")
        return

    setup = ROOT / "scripts" / "setup.py"
    cmd = [sys.executable, str(setup), "--ffmpeg-only"]
    if vendor_system:
        cmd.append("--vendor-system-ffmpeg")
    info("ffmpeg -> vendor/ffmpeg/ (via scripts/setup.py --ffmpeg-only)")
    result = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if result.returncode != 0:
        fail(
            "ffmpeg fetch failed. Copy libvmaf builds manually into "
            "vendor/ffmpeg/bin/ or set VIDOPT_FFMPEG_DIR and retry with "
            "--vendor-system-ffmpeg"
        )
    if not FFMPEG_BIN.joinpath("ffmpeg.exe").is_file() and not vendor_system:
        detail("vendor/ffmpeg/bin still empty — copying from system ffmpeg")
        cmd = [sys.executable, str(setup), "--ffmpeg-only", "--vendor-system-ffmpeg"]
        result = subprocess.run(cmd, cwd=str(ROOT), check=False)
        if result.returncode != 0:
            fail("failed to vendor system ffmpeg into vendor/ffmpeg/bin/")


def run_install(*, skip: bool) -> None:
    if skip:
        info("skipping install.bat (run it manually before pack_production.bat)")
        return
    install_bat = ROOT / "install.bat"
    if not install_bat.is_file():
        fail(f"missing {install_bat}")
    info("running install.bat (offline install into vendor/python/)")
    result = subprocess.run(
        ["cmd", "/c", str(install_bat)],
        cwd=str(ROOT),
        check=False,
    )
    if result.returncode != 0:
        fail("install.bat failed; fix errors above before pack_production.bat")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download installers and wheels even if present.",
    )
    parser.add_argument(
        "--skip-ffmpeg",
        action="store_true",
        help="Do not fetch ffmpeg (copy vendor/ffmpeg/bin manually).",
    )
    parser.add_argument(
        "--vendor-system-ffmpeg",
        action="store_true",
        help="Copy ffmpeg from PATH / VIDOPT_FFMPEG_DIR instead of downloading.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Only download artifacts; do not run install.bat.",
    )
    args = parser.parse_args()

    if sys.platform != "win32":
        fail("prepare_offline_bundle is for Windows x64 only")

    info(f"vidopt offline bundle prep (root: {ROOT})")
    download_installers(force=args.force)
    download_wheelhouse(force=args.force)
    fetch_ffmpeg(skip=args.skip_ffmpeg, vendor_system=args.vendor_system_ffmpeg)
    run_install(skip=args.skip_install)

    print()
    info("offline bundle prep complete")
    detail("next:  install.bat          (if you used --skip-install)")
    detail("       scripts\\pack_production.bat")
    detail("       scripts\\pack_production.bat --with-models   (include trained models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
