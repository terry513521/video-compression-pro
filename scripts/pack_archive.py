"""Shared zip helpers for offline packaging scripts."""

from __future__ import annotations

import zipfile
from pathlib import Path

# Strong deflate; binaries compress modestly but text/wheels benefit.
ZIP_COMPRESSLEVEL = 9

# Embedded in production / project zips; extracted by install.bat on first run.
VENDOR_ARCHIVE_NAME = "vendor-windows-x64.zip"

SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
}

SKIP_FILE_NAMES = {
    ".DS_Store",
    "Thumbs.db",
}


def should_skip(rel: Path, *, is_dir: bool) -> bool:
    parts = rel.parts
    if not parts:
        return False
    if parts[0] == "dist":
        return True
    if any(p in SKIP_DIR_NAMES for p in parts):
        return True
    if not is_dir and rel.name in SKIP_FILE_NAMES:
        return True
    if not is_dir and rel.suffix.lower() in {".pyc", ".pyo"}:
        return True
    return False


def zip_path(
    src: Path,
    archive: Path,
    *,
    arc_prefix: str = "",
    root: Path | None = None,
    extra_skip: set[str] | None = None,
) -> tuple[int, int]:
    """Add ``src`` (file or directory tree) to ``archive``. Returns (files, bytes)."""
    root = root or src.parent
    extra_skip = extra_skip or set()
    files = 0
    bytes_total = 0

    archive.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if archive.is_file() else "w"
    with zipfile.ZipFile(
        archive,
        mode,
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=ZIP_COMPRESSLEVEL,
    ) as zf:
        if src.is_file():
            rel = src.relative_to(root)
            if should_skip(rel, is_dir=False) or rel.parts[0] in extra_skip:
                return 0, 0
            arc = "/".join(p for p in (arc_prefix, *rel.parts) if p)
            zf.write(src, arc)
            return 1, src.stat().st_size

        for path in sorted(src.rglob("*")):
            rel = path.relative_to(root)
            if path.is_dir():
                if should_skip(rel, is_dir=True) or rel.parts[0] in extra_skip:
                    continue
                continue
            if should_skip(rel, is_dir=False) or rel.parts[0] in extra_skip:
                continue
            if any(part in extra_skip for part in rel.parts):
                continue
            arc = "/".join(p for p in (arc_prefix, *rel.parts) if p)
            zf.write(path, arc)
            files += 1
            bytes_total += path.stat().st_size
    return files, bytes_total


def write_zip(
    root: Path,
    archive: Path,
    *,
    include_top_level: list[str],
    arc_folder: str,
    extra_skip: set[str] | None = None,
) -> tuple[int, int]:
    """Create ``archive`` from selected top-level paths under ``root``."""
    if archive.is_file():
        archive.unlink()
    total_files = 0
    total_bytes = 0
    for name in include_top_level:
        src = root / name
        if not src.exists():
            continue
        n, b = zip_path(
            src,
            archive,
            arc_prefix=arc_folder,
            root=root,
            extra_skip=extra_skip,
        )
        total_files += n
        total_bytes += b
    return total_files, total_bytes


def fmt_size(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MB"
    return f"{n / 1024:.0f} KB"


def require_vendor(root: Path) -> None:
    """Ensure ``vendor/`` is ready on the build machine."""
    vendor = root / "vendor"
    py = vendor / "python" / "python.exe"
    ff = vendor / "ffmpeg" / "bin" / "ffmpeg.exe"
    wh = vendor / "wheelhouse"
    if not py.is_file() or not ff.is_file():
        raise SystemExit(
            "vendor/ is incomplete (need vendor\\python\\python.exe and "
            "vendor\\ffmpeg\\bin\\ffmpeg.exe).\n"
            "Run scripts\\prepare_offline_bundle.bat and install.bat first."
        )
    if not wh.is_dir() or not any(wh.glob("*.whl")):
        raise SystemExit(
            "vendor/wheelhouse is empty — run scripts\\prepare_offline_bundle.bat first."
        )


def compress_vendor(root: Path, archive: Path) -> tuple[int, int]:
    """Zip ``vendor/`` so archive entries are ``vendor/...`` (for install.bat extract)."""
    if archive.is_file():
        archive.unlink()
    return write_zip(
        root,
        archive,
        include_top_level=["vendor"],
        arc_folder="",
        extra_skip=set(),
    )
