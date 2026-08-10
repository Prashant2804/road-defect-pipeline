#!/usr/bin/env bash
# Train RFDETRMedium for 50 epochs on the prepared 6-class COCO dataset,
# with a batch sized to use substantial GPU VRAM on a 32GB card.
#
# Default dataset: data/rfdetr/stage2 (merged 6-class). Falls back to stage1.
# Does NOT re-download data. Does NOT overwrite runs/rfdetr_stage1.
#
#   tmux new -s rfdetr_medium_50
#   ./scripts/run_rfdetr_medium_50ep.sh
#   # detach: Ctrl-b d    reattach: tmux attach -t rfdetr_medium_50
#
# If the process dies with "Killed" (no Python traceback), that is usually the
# Linux OOM killer (CPU RAM) from too many DataLoader workers — not CUDA OOM.
# Retry with fewer workers first, then lower batch:
#   WORKERS=2 BATCH=20 ./scripts/run_rfdetr_medium_50ep.sh
#   WORKERS=2 BATCH=16 ./scripts/run_rfdetr_medium_50ep.sh
#
# Overrides:
#   DATASET_DIR=data/rfdetr/stage2 ./scripts/run_rfdetr_medium_50ep.sh
#   BATCH=24 WORKERS=2 ./scripts/run_rfdetr_medium_50ep.sh   # push GPU VRAM
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
# Cap host-side thread fan-out (helps avoid RAM spikes with many workers).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

EPOCHS="${EPOCHS:-50}"
# Medium + multi-scale (~736): batch 20 + few workers is the stable >~15–22 GiB
# GPU path. Batch 28 + workers=10 was OOM-killed on host RAM during dataloader init.
BATCH="${BATCH:-20}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
WORKERS="${WORKERS:-4}"
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
echo "==> host RAM (free -h):"
free -h || true
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
echo "==> Train RFDETRMedium (50 epochs)"
set +e
# shellcheck disable=SC2086
$PY -m tools.rfdetr_train.train "${ARGS[@]}" ${EXTRA_TRAIN_ARGS:-}
rc=$?
set -e

if [[ $rc -ne 0 ]]; then
  echo "" >&2
  echo "ERROR: training exited with code $rc." >&2
  if [[ $rc -eq 137 ]] || [[ $rc -eq 9 ]]; then
    echo "This looks like Linux OOM-kill (host RAM), not a Python CUDA error." >&2
    echo "Retry with fewer workers / smaller batch, e.g.:" >&2
    echo "  WORKERS=2 BATCH=16 ./scripts/run_rfdetr_medium_50ep.sh" >&2
  fi
  dmesg -T 2>/dev/null | tail -n 20 | grep -i -E 'oom|killed process' >&2 || true
  exit "$rc"
fi

echo ""
echo "Done. Checkpoints under: $OUTPUT_DIR"
ls -lah "$OUTPUT_DIR"/*.pth 2>/dev/null || true
echo ""
echo "Infer with (this run may only have checkpoint_best_ema.pth):"
echo "  WEIGHTS=$OUTPUT_DIR/checkpoint_best_total.pth"
echo "  [[ -f \"\$WEIGHTS\" ]] || WEIGHTS=$OUTPUT_DIR/checkpoint_best_ema.pth"
echo "  ./scripts/run_rfdetr_infer.sh \\"
echo "    --weights \"\$WEIGHTS\" --conf 0.20 \\"
echo "    --video ... --srt ... --z-far 5 \\"
echo "    --out-dir runs/rfdetr_infer/ROAD-1-Gopro-medium-50ep"
