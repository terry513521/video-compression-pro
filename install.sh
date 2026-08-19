#!/usr/bin/env bash
# vidopt — Linux CPU install / repair
#
# Online path (this machine): create .venv, fetch a libvmaf ffmpeg into vendor/,
# install Python deps. No NVIDIA GPU is used.
#
# Offline path: if vendor/wheelhouse/*.whl and vendor/ffmpeg/bin/ffmpeg already
# exist, setup.py stays off the network.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo
echo "============================================================"
echo " vidopt Linux CPU install / repair"
echo " root: $ROOT"
echo "============================================================"
echo

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Install Python 3.10+:" >&2
  echo "  sudo apt install python3 python3-venv python3-pip" >&2
  exit 1
fi

PY_VER="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' \
  || { echo "ERROR: Python 3.10+ required, this is $PY_VER" >&2; exit 1; }

echo "Python: $(python3 -c 'import sys; print(sys.version.split()[0])') ($(command -v python3))"

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "NVIDIA GPU detected — this install still defaults to CPU (libx265)."
  echo "Use --encoder hevc_nvenc --gpu-workers 1 only after ./vidopt.sh doctor shows NVENC OK."
else
  echo "No NVIDIA GPU — CPU-only (libx265). Overlay: --config cpu"
fi

# ---- venv ------------------------------------------------------------------
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo
  echo "[1/3] Creating virtualenv at .venv ..."
  python3 -m venv "$ROOT/.venv"
else
  echo
  echo "[1/3] Reusing virtualenv at .venv"
fi

# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
python -m pip install --upgrade pip

# ---- ffmpeg + package ------------------------------------------------------
echo
echo "[2/3] Fetching libvmaf ffmpeg (if needed) and installing vidopt ..."
export VIDOPT_FFMPEG_DIR="${VIDOPT_FFMPEG_DIR:-$ROOT/vendor/ffmpeg/bin}"
python "$ROOT/scripts/setup.py"

# ---- verify ----------------------------------------------------------------
echo
echo "[3/3] Verifying (CPU encode + VMAF) ..."
export PATH="$VIDOPT_FFMPEG_DIR:$PATH"
python "$ROOT/scripts/setup.py" --verify
python -m vidopt doctor --config cpu

chmod +x "$ROOT/vidopt.sh" "$ROOT/activate_vidopt.sh" 2>/dev/null || true

echo
echo "============================================================"
echo " Linux CPU install complete."
echo "============================================================"
echo
echo "  ./vidopt.sh doctor --config cpu"
echo "  copy training videos into video/corpus/"
echo "  ./vidopt.sh dev video/corpus --config cpu --encoder libx265 --cpu-workers 0"
echo "  ./vidopt.sh compress in.mp4 -o out/out.mp4 --target 89 --encoder libx265 --verify"
echo
echo "Activate this shell:  source ./activate_vidopt.sh"
echo "Guide: LINUX.md"
echo
