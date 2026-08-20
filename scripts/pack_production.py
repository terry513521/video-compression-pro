#!/usr/bin/env python3
"""Build an offline Windows production zip (vendor nested inside).

Build flow on a connected machine::

    scripts\\prepare_offline_bundle.bat
    install.bat
    vidopt train ... --resume
    scripts\\pack_production.bat --with-models

Pack flow (two compression steps)::

    1. Compress ``vendor/`` -> ``vendor-windows-x64.zip``
    2. Compress the project tree **including** that zip (no raw ``vendor/`` folder)

Deploy flow on an offline machine::

    1. Extract ``dist/vidopt-offline-windows-x64.zip``
    2. Run ``install.bat``  (extracts vendor zip + installs vidopt into bundled Python)
    3. ``vidopt.bat train ...`` / ``vidopt.bat compress ...``
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from pack_archive import (
    VENDOR_ARCHIVE_NAME,
    ZIP_COMPRESSLEVEL,
    compress_vendor,
    fmt_size,
    require_vendor,
)

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
STAGING_NAME = "vidopt-offline-windows-x64"

ROOT_FILES = (
    "vidopt.bat",
    "install.bat",
    "activate_vidopt.bat",
    "pyproject.toml",
    "START_HERE.txt",
    "README.md",
    "USAGE.md",
    "OFFLINE_GUIDE.md",
    "COMPRESS_GUIDE.md",
    "SYSTEM_GUIDE.md",
    "REPAIR.txt",
)

SCRIPT_FILES = (
    "scripts/setup.py",
    "scripts/prepare_offline_bundle.py",
    "scripts/prepare_offline_bundle.bat",
    "scripts/pack_production.py",
    "scripts/pack_production.bat",
    "scripts/pack_compress.py",
    "scripts/pack_compress.bat",
    "scripts/pack_project.py",
    "scripts/pack_project.bat",
    "scripts/pack_archive.py",
)


def info(msg: str) -> None:
    print(f"==> {msg}", flush=True)


def detail(msg: str) -> None:
    print(f"    {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return
    for item in sorted(src.rglob("*")):
        rel = item.relative_to(src)
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def _copy_models(src: Path, dst: Path) -> list[dict]:
    bundles: list[dict] = []
    for meta in sorted(src.glob("*/target_*/metadata.json")):
        bundle = meta.parent
        rel = bundle.relative_to(src)
        _copy_tree(bundle, dst / rel)
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        bundles.append({
            "path": str(rel).replace("/", "\\"),
            "encoder": data.get("encoder", bundle.parent.name),
            "target": data.get("target"),
            "hit_rate": (data.get("metrics") or {}).get("crf_hit_rate"),
        })
    return bundles


def _write_manifest(
    staging: Path,
    *,
    with_models: bool,
    bundles: list[dict],
    vendor_zip_size: int,
) -> None:
    manifest = {
        "package": "vidopt-offline",
        "platform": "windows-x64",
        "modes": ["train", "compress"],
        "description": "Full offline production: train models and compress videos (no network after install).",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "with_models": with_models,
        "vendor_archive": VENDOR_ARCHIVE_NAME,
        "vendor_archive_bytes": vendor_zip_size,
        "models": bundles,
        "install_steps": [
            f"Extract {STAGING_NAME}.zip",
            "Run install.bat (extracts vendor archive and installs vidopt)",
            "Run vidopt.bat doctor",
            "Copy videos to video\\corpus\\ (if not bundled)",
            "Run vidopt.bat train video\\corpus ... --resume",
            "Run vidopt.bat compress in.mp4 -o out\\out.mp4 ... --verify",
        ],
    }
    (staging / "PACKAGE.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def _zip_dir(staging: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=ZIP_COMPRESSLEVEL,
    ) as zf:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                arcname = f"{STAGING_NAME}/{path.relative_to(staging).as_posix()}"
                zf.write(path, arcname)


def pack(*, with_models: bool, output: Path | None = None) -> Path:
    require_vendor(ROOT)

    archive = output or (DIST / f"{STAGING_NAME}.zip")
    models_src = ROOT / "models"
    bundles: list[dict] = []

    if with_models:
        if not any(models_src.glob("*/target_*/metadata.json")):
            fail(
                f"no trained models under {models_src}\n"
                "Finish training first, or omit --with-models."
            )

    DIST.mkdir(parents=True, exist_ok=True)
    vendor_sidecar = DIST / VENDOR_ARCHIVE_NAME

    with tempfile.TemporaryDirectory(prefix="vidopt-prod-pack-") as tmp:
        staging = Path(tmp) / STAGING_NAME
        staging.mkdir(parents=True)

        info(f"step 1/2: compress vendor/ -> {VENDOR_ARCHIVE_NAME}")
        n_vendor, raw_vendor = compress_vendor(ROOT, staging / VENDOR_ARCHIVE_NAME)
        shutil.copy2(staging / VENDOR_ARCHIVE_NAME, vendor_sidecar)
        vendor_zip_size = (staging / VENDOR_ARCHIVE_NAME).stat().st_size
        detail(
            f"{n_vendor} file(s), {fmt_size(raw_vendor)} raw -> "
            f"{fmt_size(vendor_zip_size)} zip"
        )

        info("step 2/2: compress project (vendor archive embedded, no vendor/ folder)")
        for name in ROOT_FILES:
            src = ROOT / name
            if src.is_file():
                shutil.copy2(src, staging / name)

        for rel in SCRIPT_FILES:
            src = ROOT / rel
            if src.is_file():
                dst = staging / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

        _copy_tree(ROOT / "src", staging / "src")

        corpus = staging / "video" / "corpus"
        corpus.mkdir(parents=True, exist_ok=True)
        readme = ROOT / "video" / "corpus" / "README.txt"
        if readme.is_file():
            shutil.copy2(readme, corpus / "README.txt")

        out_dir = staging / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / ".gitkeep").write_text("", encoding="utf-8")

        models_dst = staging / "models"
        if with_models:
            info("including trained models")
            bundles = _copy_models(models_src, models_dst)
        else:
            models_dst.mkdir(parents=True, exist_ok=True)
            (models_dst / ".gitkeep").write_text("", encoding="utf-8")

        _write_manifest(
            staging,
            with_models=with_models,
            bundles=bundles,
            vendor_zip_size=vendor_zip_size,
        )
        info(f"creating {archive}")
        _zip_dir(staging, archive)

    size_mb = archive.stat().st_size / (1024 * 1024)
    print()
    info(f"created {archive} ({size_mb:.1f} MB)")
    detail(f"also wrote {vendor_sidecar}")
    if with_models:
        detail("models included:")
        for b in bundles:
            hr = b.get("hit_rate")
            hr_s = f" hit-rate={hr:.0%}" if isinstance(hr, (int, float)) else ""
            t = b.get("target")
            t_s = f" target={t}" if t is not None else ""
            detail(f"  {b['path']}  encoder={b['encoder']}{t_s}{hr_s}")
    else:
        detail("no models (train on offline PC or re-pack with --with-models)")
    detail("offline PC: extract zip -> install.bat -> vidopt.bat doctor -> train / compress")
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-models",
        action="store_true",
        help="Include trained models/ in the zip.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output zip path (default: dist/vidopt-offline-windows-x64.zip)",
    )
    args = parser.parse_args()

    if sys.platform != "win32":
        fail("pack_production is for Windows x64 only")

    pack(with_models=args.with_models, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
