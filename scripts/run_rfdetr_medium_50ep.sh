#!/usr/bin/env bash
# Train RFDETRMedium for 50 epochs on the prepared 6-class COCO dataset,
# with a large batch sized to use >20 GiB GPU VRAM (RTX 5090 32GB class).
#
# Default dataset: data/rfdetr/stage2 (merged 6-class). Falls back to stage1.
# Does NOT re-download data. Does NOT overwrite runs/rfdetr_stage1.
#
#   tmux new -s rfdetr_medium_50
#   ./scripts/run_rfdetr_medium_50ep.sh
#   # detach: Ctrl-b d    reattach: tmux attach -t rfdetr_medium_50
#
# Overrides:
#   DATASET_DIR=data/rfdetr/stage2 ./scripts/run_rfdetr_medium_50ep.sh
#   BATCH=32 ./scripts/run_rfdetr_medium_50ep.sh          # push VRAM harder
#   BATCH=24 ./scripts/run_rfdetr_medium_50ep.sh          # if OOM at 28
#   OUTPUT_DIR=runs/rfdetr_medium_6class_50ep ./scripts/run_rfdetr_medium_50ep.sh
#   RESUME=runs/rfdetr_medium_6class_50ep/checkpoint.pth ./scripts/run_rfdetr_medium_50ep.sh
#   EXTRA_TRAIN_ARGS="--lr 5e-5" ./scripts/run_rfdetr_medium_50ep.sh
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

EPOCHS="${EPOCHS:-50}"
# Medium @ 576: batch 28 typically sits well above 20 GiB on a 32GB card.
# Drop to 24 on OOM; raise to 32 if nvidia-smi still shows headroom.
BATCH="${BATCH:-28}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
WORKERS="${WORKERS:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/rfdetr_medium_6class_50ep}"

pick_dataset() {
  if [[ -n "${DATASET_DIR:-}" ]]; then
    echo "$DATASET_DIR"
    return
  fi
  local stage2="$ROOT/data/rfdetr/stage2"
  local stage1="$ROOT/data/rfdetr/stage1"
  if [[ -f "$stage2/train/_annotations.coco.json" ]]; then
    echo "$stage2"
  elif [[ -f "$stage1/train/_annotations.coco.json" ]]; then
    echo "$stage1"
  else
    echo ""
  fi
}

DATASET_DIR="$(pick_dataset)"
if [[ -z "$DATASET_DIR" ]]; then
  echo "ERROR: No prepared 6-class COCO found." >&2
  echo "Expected: data/rfdetr/stage2/train/_annotations.coco.json" >&2
  echo "       or data/rfdetr/stage1/train/_annotations.coco.json" >&2
  echo "Prepare first: SKIP_DOWNLOAD=0 ./scripts/run_stage2.sh  (or ./scripts/run_stage1.sh)" >&2
  echo "Or pass: DATASET_DIR=/path/to/coco ./scripts/run_rfdetr_medium_50ep.sh" >&2
  exit 1
fi
DATASET_DIR="$(cd "$DATASET_DIR" && pwd)"

if [[ ! -f "$DATASET_DIR/train/_annotations.coco.json" ]]; then
  echo "ERROR: Missing $DATASET_DIR/train/_annotations.coco.json" >&2
  exit 1
fi

echo "==> Using: $PY ($("$PY" --version 2>&1))"
echo "==> dataset: $DATASET_DIR"
echo "==> output:  $OUTPUT_DIR"
echo "==> epochs=$EPOCHS batch=$BATCH grad_accum=$GRAD_ACCUM workers=$WORKERS"
echo "==> target: use >20 GiB GPU VRAM (watch nvidia-smi during train)"
echo "==> nvidia-smi"
nvidia-smi || true

ARGS=(
  --dataset-dir "$DATASET_DIR"
  --output-dir "$OUTPUT_DIR"
  --epochs "$EPOCHS"
  --batch "$BATCH"
  --grad-accum "$GRAD_ACCUM"
  --workers "$WORKERS"
  --no-early-stop
)
if [[ -n "${RESUME:-}" ]]; then
  ARGS+=(--resume "$RESUME")
fi

echo ""
echo "==> Train RFDETRMedium (50 epochs, high-VRAM batch)"
# shellcheck disable=SC2086
$PY -m tools.rfdetr_train.train "${ARGS[@]}" ${EXTRA_TRAIN_ARGS:-}

echo ""
echo "Done. Checkpoints under: $OUTPUT_DIR"
ls -lah "$OUTPUT_DIR"/*.pth 2>/dev/null || true
echo ""
echo "Infer with:"
echo "  ./scripts/run_rfdetr_infer.sh \\"
echo "    --weights $OUTPUT_DIR/checkpoint_best_total.pth \\"
echo "    --video ... --srt ... --z-far 5 \\"
echo "    --out-dir runs/rfdetr_infer/ROAD-1-Gopro-medium-50ep"
