#!/usr/bin/env python3
"""Compress vendor first, then the whole project with vendor embedded.

Creates under ``dist/``:

  vendor-windows-x64.zip              step 1 — runtime + repair kit only
  vidopt-project-windows-x64.zip      step 2 — full project + embedded vendor zip

Offline PC after extracting the **project** zip::

    install.bat
    vidopt.bat doctor
    vidopt.bat train ...
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
STAGING_NAME = "vidopt-project-windows-x64"

PROJECT_TOP = (
    "src",
    "models",
    "scripts",
    "video",
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

ARCHIVE_VENDOR = DIST / VENDOR_ARCHIVE_NAME
ARCHIVE_PROJECT = DIST / f"{STAGING_NAME}.zip"


def info(msg: str) -> None:
    print(f"==> {msg}", flush=True)


def detail(msg: str) -> None:
    print(f"    {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _list_models() -> list[dict]:
    models = ROOT / "models"
    out: list[dict] = []
    if not models.is_dir():
        return out
    for meta in sorted(models.glob("*/target_*/metadata.json")):
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        rel = meta.parent.relative_to(models)
        out.append({
            "path": str(rel).replace("/", "\\"),
            "encoder": data.get("encoder"),
            "target": data.get("target"),
        })
    return out


def pack_vendor(output: Path) -> Path:
    require_vendor(ROOT)
    info(f"step 1: compress vendor/ -> {output.name}")
    n, raw = compress_vendor(ROOT, output)
    size = output.stat().st_size
    detail(f"{n} file(s), {fmt_size(raw)} raw -> {fmt_size(size)} zip")
    return output


def _zip_staging(staging: Path, archive: Path) -> None:
    if archive.is_file():
        archive.unlink()
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


def pack_project(
    output: Path,
    *,
    vendor_zip: Path,
    with_models: bool,
    with_corpus: bool,
    with_runs: bool,
) -> Path:
    extra_skip: set[str] = {"vendor", "dist", "runs"} if not with_runs else {"vendor", "dist"}
    if not with_models:
        extra_skip.add("models")
    if not with_corpus:
        extra_skip.add("video")

    include = [name for name in PROJECT_TOP if (ROOT / name).exists()]
    if not with_models and "models" in include:
        include.remove("models")
    if not with_corpus and "video" in include:
        include.remove("video")

    info(f"step 2: compress project -> {output.name}")
    detail(f"including: {VENDOR_ARCHIVE_NAME}, {', '.join(include)}")
    if with_runs:
        include.append("runs")

    with tempfile.TemporaryDirectory(prefix="vidopt-proj-pack-") as tmp:
        staging = Path(tmp) / STAGING_NAME
        staging.mkdir(parents=True)

        shutil.copy2(vendor_zip, staging / VENDOR_ARCHIVE_NAME)

        for name in include:
            src = ROOT / name
            dst = staging / name
            if src.is_file():
                shutil.copy2(src, dst)
            elif src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)

        manifest = {
            "package": "vidopt-project",
            "platform": "windows-x64",
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "vendor_archive": VENDOR_ARCHIVE_NAME,
            "with_models": with_models,
            "with_corpus": with_corpus,
            "with_runs": with_runs,
            "models": _list_models() if with_models else [],
            "install_steps": [
                f"Extract {STAGING_NAME}.zip",
                "Run install.bat",
                "Run vidopt.bat doctor",
            ],
        }
        (staging / "PACKAGE.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        _zip_staging(staging, output)

    size = output.stat().st_size
    detail(f"project zip size: {fmt_size(size)}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vendor-only",
        action="store_true",
        help="Only create dist/vendor-windows-x64.zip",
    )
    parser.add_argument(
        "--project-only",
        action="store_true",
        help="Only create project zip (vendor zip must already exist in dist/)",
    )
    parser.add_argument(
        "--with-models",
        action="store_true",
        default=True,
        help="Include models/ (default: on).",
    )
    parser.add_argument(
        "--no-models",
        action="store_true",
        help="Exclude models/.",
    )
    parser.add_argument(
        "--with-corpus",
        action="store_true",
        default=True,
        help="Include video/corpus (default: on).",
    )
    parser.add_argument(
        "--no-corpus",
        action="store_true",
        help="Exclude video/.",
    )
    parser.add_argument(
        "--with-runs",
        action="store_true",
        help="Include runs/ (large).",
    )
    args = parser.parse_args()

    if sys.platform != "win32":
        fail("pack_project is for Windows x64 only")

    with_models = args.with_models and not args.no_models
    with_corpus = args.with_corpus and not args.no_corpus
    do_vendor = not args.project_only
    do_project = not args.vendor_only

    DIST.mkdir(parents=True, exist_ok=True)

    vendor_zip = ARCHIVE_VENDOR
    if do_vendor:
        pack_vendor(vendor_zip)
    elif do_project and not vendor_zip.is_file():
        fail(f"missing {vendor_zip} — run without --project-only first")

    if do_project:
        pack_project(
            ARCHIVE_PROJECT,
            vendor_zip=vendor_zip,
            with_models=with_models,
            with_corpus=with_corpus,
            with_runs=args.with_runs,
        )

    print()
    info("done")
    if do_vendor or vendor_zip.is_file():
        detail(str(vendor_zip))
    if do_project:
        detail(str(ARCHIVE_PROJECT))
        detail("offline PC: extract -> install.bat -> vidopt.bat doctor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
