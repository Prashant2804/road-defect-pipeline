#!/usr/bin/env bash
# Parallel companion: Ultralytics RT-DETR-l on Stage-2 data (same GPU as RF-DETR Large).
#
# Leave the existing rfdetr_stage2 tmux alone. In a 2nd SSH session:
#   cd ~/road-defect-pipeline && git pull
#   tmux new -s rtdetr_stage2
#   ./scripts/run_rtdetr_parallel.sh
#   # Ctrl-b d
#
# Defaults: rtdetr-l.pt, batch=8, workers=4, epochs=100, memory_fraction=0.45
# If RF-DETR OOMs:  EXTRA_TRAIN_ARGS="--batch 4 --memory-fraction 0.35" ./scripts/run_rtdetr_parallel.sh
# Skip export:      SKIP_EXPORT=1 ./scripts/run_rtdetr_parallel.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
  PY="$ROOT/.venv/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PY="${VIRTUAL_ENV}/bin/python"
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "==> Using: $PY ($("$PY" --version 2>&1))"
echo "==> Ensuring ultralytics is installed"
"$PY" -m pip install -q "ultralytics>=8.3.0"

echo "==> nvidia-smi (expect RF-DETR Large already using ~7GB)"
nvidia-smi || true

COCO_DIR="${COCO_DIR:-$ROOT/data/rfdetr/stage2}"
YOLO_DIR="${YOLO_DIR:-$ROOT/data/rfdetr/stage2_yolo}"

if [[ ! -f "$COCO_DIR/train/_annotations.coco.json" ]]; then
  echo "ERROR: Stage-2 COCO missing at $COCO_DIR" >&2
  echo "Finish / wait for Stage-2 download, or point COCO_DIR=..." >&2
  exit 1
fi

if [[ "${SKIP_EXPORT:-0}" != "1" ]]; then
  echo ""
  echo "==> Export COCO → YOLO ($COCO_DIR → $YOLO_DIR)"
  "$PY" -m tools.rfdetr_train.export_yolo --coco-dir "$COCO_DIR" --out-dir "$YOLO_DIR"
else
  echo "==> SKIP_EXPORT=1 — using existing $YOLO_DIR"
fi

if [[ ! -f "$YOLO_DIR/data.yaml" ]]; then
  echo "ERROR: missing $YOLO_DIR/data.yaml" >&2
  exit 1
fi

echo ""
echo "==> Train Ultralytics RT-DETR-l (parallel-safe VRAM)"
# shellcheck disable=SC2086
"$PY" -m tools.rfdetr_train.train_rtdetr \
  --data "$YOLO_DIR/data.yaml" \
  --output-dir "$ROOT/runs/rtdetr_stage2" \
  ${EXTRA_TRAIN_ARGS:-}

echo ""
echo "Done. Weights under: $ROOT/runs/rtdetr_stage2/weights/"
ls -lah "$ROOT/runs/rtdetr_stage2/weights/"*.pt 2>/dev/null || true
