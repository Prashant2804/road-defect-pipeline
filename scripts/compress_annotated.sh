#!/usr/bin/env bash
# Compress a huge OpenCV mp4v annotated.mp4 to H.264 for Drive preview.
#
#   ./scripts/compress_annotated.sh 'runs/rfdetr_infer/ROAD-1 Gopro/annotated.mp4'
#
# Writes annotated_h264.mp4 next to it (much smaller). Then upload that file,
# or use --replace to overwrite annotated.mp4 before re-uploading.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
  PY="$ROOT/.venv/bin/python"
else
  PY="${PYTHON:-python3}"
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 path/to/annotated.mp4 [--replace] [--crf 23]" >&2
  exit 1
fi

INPUT="$1"
shift

exec "$PY" -m tools.rfdetr_infer.compress_video --input "$INPUT" "$@"
