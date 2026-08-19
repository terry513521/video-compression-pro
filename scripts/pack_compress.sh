#!/usr/bin/env bash
# Build dist/vidopt-compress-linux-x64.tar.gz (runtime + models, no corpus).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/pack_compress.py" --platform linux "$@"
