#!/usr/bin/env bash
# Inspect Stage-1 training outputs (checkpoints + logs + metric hints).
#
# Run on the VM after training finishes, then paste the full stdout into chat:
#
#   ./scripts/check_stage1_run.sh
#   ./scripts/check_stage1_run.sh --run-dir runs/rfdetr_stage1
#
# Optional: also dump JSON
#   ./scripts/check_stage1_run.sh --json-out runs/rfdetr_stage1/check_report.json
#
# If text logs are empty but you trained in tmux:
#   tmux capture-pane -t rfdetr -p -S -3000 > /tmp/rfdetr_train_pane.txt
#   then attach that file or paste it too.
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

echo "==> nvidia-smi (context)"
nvidia-smi || true
echo ""

exec "$PY" -m tools.rfdetr_train.check_run "$@"
