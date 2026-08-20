#!/usr/bin/env python3
"""Build a single offline **compress-only** archive (runtime + trained models).

Excludes training corpus, dev runs, and training-only scripts. Produces:

  dist\\vidopt-compress-windows-x64.zip

Usage:

    python scripts\\pack_compress.py
    scripts\\pack_compress.bat
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
    require_vendor,
)

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"

# Paths never shipped in a compress-only package.
EXCLUDE_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "corpus",
    "_sources",
    "test",
    "tests",
    "runs",
    "node_modules",
}
EXCLUDE_FILE_NAMES = {
    "download_corpus.py",
    "train_matrix.py",
    "pack_compress.py",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".mkv", ".mp4", ".mov", ".webm", ".avi"}

DOC_FILES = (
    "COMPRESS_GUIDE.md",
    "REPAIR.txt",
    "README.md",
)

START_HERE_COMPRESS = """\
vidopt — compress-only package (offline production, Windows)
============================================================

No training corpus included. Pre-trained models are in models\\.

  Extract vidopt-compress-windows-x64.zip
  install.bat
  vidopt.bat doctor
  vidopt.bat inspect
  vidopt.bat compress in.mp4 -o out\\output.mp4 --encoder ENCODER --level 2 --verify

Replace ENCODER with a name from `vidopt inspect` (must match training).
See PACKAGE.json for bundled models. Full guide: COMPRESS_GUIDE.md
"""


def _has_models(models_dir: Path) -> bool:
    return any(models_dir.glob("*/target_*/metadata.json"))


def _list_bundles(models_dir: Path) -> list[dict]:
    out: list[dict] = []
    for meta in sorted(models_dir.glob("*/target_*/metadata.json")):
        rel = str(meta.parent.relative_to(models_dir))
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        out.append({
            "path": rel,
            "encoder": data.get("encoder", meta.parent.parent.name),
            "target": data.get("target"),
            "hit_rate": (data.get("metrics") or {}).get("crf_hit_rate"),
        })
    return out


def _should_skip(rel: Path, *, is_dir: bool) -> bool:
  parts = rel.parts
  if not parts:
      return False
  if parts[0] == "video":
      return True
  if "corpus" in parts or "_sources" in parts:
      return True
  if any(p in EXCLUDE_DIR_NAMES for p in parts):
      return True
  if not is_dir:
      if rel.name in EXCLUDE_FILE_NAMES:
          return True
      if rel.suffix.lower() in EXCLUDE_SUFFIXES and rel.parts[0] not in ("vendor", "models"):
          return True
      if rel.name.startswith("output.") and rel.parent == Path("."):
          return True
  return False


def _copy_tree(src: Path, dst: Path, *, rel_base: Path | None = None) -> None:
    rel_base = rel_base or src
    if not src.exists():
        return
    if src.is_file():
        rel = src.relative_to(rel_base)
        if _should_skip(rel, is_dir=False):
            return
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        return
    for item in sorted(src.rglob("*")):
        rel = item.relative_to(rel_base)
        if item.is_dir():
            if _should_skip(rel, is_dir=True):
                continue
        else:
            if _should_skip(rel, is_dir=False):
                continue
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def _write_manifest(staging: Path, bundles: list[dict], plat: str) -> None:
    manifest = {
        "package": "vidopt-compress",
        "platform": plat,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "models": bundles,
        "excludes": [
            "video/corpus",
            "video/test",
            "runs/",
            "training scripts",
        ],
    }
    (staging / "PACKAGE.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def _copy_models(src_root: Path, dst_root: Path) -> None:
    """Copy only complete bundles (directories with metadata.json)."""
    found = False
    for meta in sorted(src_root.glob("*/target_*/metadata.json")):
        bundle = meta.parent
        rel = bundle.relative_to(src_root)
        _copy_tree(bundle, dst_root / rel)
        found = True
    if not found:
        raise SystemExit(f"no model bundles under {src_root}")


def _stage_windows(staging: Path, models_dir: Path) -> None:
    require_vendor(ROOT)

    _copy_tree(ROOT / "src", staging / "src")

    vendor_zip = staging / VENDOR_ARCHIVE_NAME
    compress_vendor(ROOT, vendor_zip)

    for name in ("vidopt.bat", "activate_vidopt.bat", "install.bat", "pyproject.toml"):
        if (ROOT / name).is_file():
            shutil.copy2(ROOT / name, staging / name)
    (staging / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "scripts" / "setup.py", staging / "scripts" / "setup.py")

    for doc in DOC_FILES:
        if (ROOT / doc).is_file():
            shutil.copy2(ROOT / doc, staging / doc)
    (staging / "START_HERE.txt").write_text(START_HERE_COMPRESS, encoding="utf-8")

    (staging / "out").mkdir(parents=True, exist_ok=True)
    (staging / "out" / ".gitkeep").write_text("", encoding="utf-8")
    _copy_models(models_dir, staging / "models")


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
                zf.write(path, path.relative_to(staging.parent).as_posix())


def pack(*, models_dir: Path, output: Path | None = None) -> Path:
    plat = "windows"
    if not _has_models(models_dir):
        raise SystemExit(
            f"no trained models under {models_dir} "
            "(expected models/<encoder>/metadata.json or legacy target_<T>/ layout)"
        )

    bundles = _list_bundles(models_dir)
    DIST.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="vidopt-pack-") as tmp:
        staging = Path(tmp) / f"vidopt-compress-{plat}-x64"
        staging.mkdir(parents=True)

        _stage_windows(staging, models_dir)
        archive = output or (DIST / "vidopt-compress-windows-x64.zip")
        _write_manifest(staging, bundles, plat)
        _zip_dir(staging, archive)

    size_mb = archive.stat().st_size / (1024 * 1024)
    print(f"created {archive} ({size_mb:.1f} MB)")
    print("models included:")
    for b in bundles:
        hr = b.get("hit_rate")
        hr_s = f" hit-rate={hr:.0%}" if isinstance(hr, (int, float)) else ""
        t = b.get("target")
        t_s = f" target={t}" if t is not None else ""
        print(f"  {b['path']}  encoder={b['encoder']}{t_s}{hr_s}")
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=ROOT / "models",
        help="models root (default: models/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output zip path (default: dist/vidopt-compress-windows-x64.zip)",
    )
    args = parser.parse_args()

    models_dir = args.models_dir.resolve()
    pack(models_dir=models_dir, output=args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
