#!/usr/bin/env bash
# Print RT-DETR Large training metrics (precision, recall, mAP, losses).
#
#   ./scripts/check_rtdetr_run.sh
#   ./scripts/check_rtdetr_run.sh --run-dir runs/rtdetr_stage2
#   ./scripts/check_rtdetr_run.sh --val
#   ./scripts/check_rtdetr_run.sh --json-out runs/rtdetr_stage2/metrics_report.json
#
# Paste the full stdout into chat if you want help interpreting the numbers.
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

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "==> nvidia-smi (context)"
nvidia-smi || true
echo ""

exec "$PY" -m tools.rfdetr_train.report_rtdetr "$@"
