#!/usr/bin/env bash
# Inspect RF-DETR Medium 50-epoch 6-class training (checkpoints + mAP/AP hints).
#
#   ./scripts/check_rfdetr_medium_50ep.sh
#   ./scripts/check_rfdetr_medium_50ep.sh --json-out runs/rfdetr_medium_6class_50ep/check_report.json
#
# If text logs are thin and you trained in tmux:
#   tmux capture-pane -t rfdetr_medium_50 -p -S -5000 > /tmp/rfdetr_medium_50_pane.txt
#   then paste that file / stdout into chat too.
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

RUN_DIR="${RUN_DIR:-$ROOT/runs/rfdetr_medium_6class_50ep}"
DATASET_DIR="${DATASET_DIR:-}"
if [[ -z "$DATASET_DIR" ]]; then
  if [[ -f "$ROOT/data/rfdetr/stage2/train/_annotations.coco.json" ]]; then
    DATASET_DIR="$ROOT/data/rfdetr/stage2"
  else
    DATASET_DIR="$ROOT/data/rfdetr/stage1"
  fi
fi

echo "==> nvidia-smi (context)"
nvidia-smi || true
echo ""
echo "==> run_dir=$RUN_DIR"
echo "==> dataset_dir=$DATASET_DIR"
echo ""

exec "$PY" -m tools.rfdetr_train.check_run \
  --run-dir "$RUN_DIR" \
  --dataset-dir "$DATASET_DIR" \
  "$@"
