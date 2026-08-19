#!/usr/bin/env bash
# Relocatable Linux launcher — works after moving this folder anywhere.
# Uses the project venv + vendored ffmpeg (libvmaf). No system ffmpeg required.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VIDOPT_ROOT="$ROOT"
export VIDOPT_FFMPEG_DIR="${VIDOPT_FFMPEG_DIR:-$ROOT/vendor/ffmpeg/bin}"

PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "ERROR: project venv missing: $PY" >&2
  echo "Run:  ./install.sh" >&2
  exit 1
fi

if [[ ! -x "$VIDOPT_FFMPEG_DIR/ffmpeg" ]]; then
  echo "ERROR: vendored ffmpeg missing: $VIDOPT_FFMPEG_DIR/ffmpeg" >&2
  echo "Run:  ./install.sh" >&2
  exit 1
fi

export PATH="$VIDOPT_FFMPEG_DIR:$ROOT/.venv/bin:$PATH"
exec "$PY" -m vidopt "$@"
