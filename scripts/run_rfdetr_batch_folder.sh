#!/usr/bin/env bash
# Batch: all MP4s in a Drive folder → custom Stage-2 Medium @ conf 0.30 → optional upload.
#
#   tmux new -s rfdetr_batch_1iAd2Nie
#   ./scripts/run_rfdetr_batch_folder.sh --upload
#
# Default source folder is baked into the Python module; override with --folder.
# Share the source Drive folder as Anyone with the link.
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

export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "==> Using: $PY ($("$PY" --version 2>&1))"
echo "==> Batch Stage-2 Medium (conf 0.30, extended ROI)"

exec "$PY" -m tools.rfdetr_infer.run_batch_folder "$@"
