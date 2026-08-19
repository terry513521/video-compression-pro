# Source this file to put vidopt + vendored ffmpeg on PATH for the current shell.
#   source ./activate_vidopt.sh
#
# Do not execute it:  ./activate_vidopt.sh   (that would not change your shell)

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "source this file instead:" >&2
  echo "  source $(cd "$(dirname "$0")" && pwd)/activate_vidopt.sh" >&2
  exit 1
fi

_VIDOPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VIDOPT_ROOT="$_VIDOPT_ROOT"
export VIDOPT_FFMPEG_DIR="${VIDOPT_FFMPEG_DIR:-$_VIDOPT_ROOT/vendor/ffmpeg/bin}"
export PATH="$VIDOPT_FFMPEG_DIR:$_VIDOPT_ROOT/.venv/bin:$PATH"

if [[ -f "$_VIDOPT_ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$_VIDOPT_ROOT/.venv/bin/activate"
fi

unset _VIDOPT_ROOT

echo "vidopt ready (venv Python + vendored ffmpeg, CPU). Examples:"
echo "  ./vidopt.sh doctor --config cpu"
echo "  ./vidopt.sh dev video/corpus --config cpu --encoder libx265 --cpu-workers 0"
echo "  ./vidopt.sh compress in.mp4 -o out/out.mp4 --target 89 --encoder libx265 --verify"
echo
