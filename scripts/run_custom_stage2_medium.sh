#!/usr/bin/env bash
# Custom Stage-2: analyze/merge Drive zips + fine-tune RFDETRMedium from Stage-1.
#
# IMMUTABLE: never writes under runs/rfdetr_stage1, data/rfdetr/stage1|stage2*,
# or existing ROAD-1 infer folders. New paths only:
#   data/rfdetr/custom_raw|custom_parts|custom_stage2|custom_stage2_aug|custom_stage2_stress
#   runs/rfdetr_medium_custom_stage2
#
#   tmux new -s rfdetr_custom_s2
#   ./scripts/run_custom_stage2_medium.sh
#
# Drive zips must be shared as "Anyone with the link".
# Defaults push a 32GB card: batch=28 workers=8.
# GPU OOM:   BATCH=24 WORKERS=8 ./scripts/run_custom_stage2_medium.sh
# Host OOM:  BATCH=20 WORKERS=4 ./scripts/run_custom_stage2_medium.sh
# Reuse downloaded zips: SKIP_DOWNLOAD=1 ./scripts/run_custom_stage2_medium.sh
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

INIT_WEIGHTS="${INIT_WEIGHTS:-$ROOT/runs/rfdetr_stage1/checkpoint_best_total.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/rfdetr_medium_custom_stage2}"
DATASET_DIR="${DATASET_DIR:-$ROOT/data/rfdetr/custom_stage2_aug}"
EPOCHS="${EPOCHS:-50}"
# Medium @ ~576 on 32GB: batch 28 fills most of the card; workers 8 keeps GPU fed
# without the host-RAM kill seen at workers≈10.
BATCH="${BATCH:-28}"
WORKERS="${WORKERS:-8}"
LR="${LR:-1e-5}"
AUG_PRESET="${AUG_PRESET:-custom_road}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

assert_safe_output() {
  local out="$1"
  local resolved
  resolved="$(cd "$(dirname "$out")" 2>/dev/null && pwd)/$(basename "$out")"
  case "$resolved" in
    */runs/rfdetr_stage1|*/runs/rfdetr_stage1/*|\
    */runs/rfdetr_stage2|*/runs/rfdetr_stage2/*|\
    */runs/rtdetr_stage2|*/runs/rtdetr_stage2/*|\
    */runs/rfdetr_medium_6class_50ep|*/runs/rfdetr_medium_6class_50ep/*|\
    */runs/rfdetr_infer/ROAD-1*|*/data/rfdetr/stage1|*/data/rfdetr/stage1/*|\
    */data/rfdetr/stage2|*/data/rfdetr/stage2/*|*/data/rfdetr/stage2_yolo|*/data/rfdetr/stage2_yolo/*)
      echo "ERROR: refusing to write into protected path: $resolved" >&2
      echo "Use OUTPUT_DIR=runs/rfdetr_medium_custom_stage2 (default)." >&2
      exit 1
      ;;
  esac
}

assert_safe_output "$OUTPUT_DIR"

if [[ ! -f "$INIT_WEIGHTS" ]]; then
  alt="$ROOT/runs/rfdetr_stage1/checkpoint_best_ema.pth"
  if [[ -f "$alt" ]]; then
    INIT_WEIGHTS="$alt"
  else
    echo "ERROR: Stage-1 weights not found: $INIT_WEIGHTS" >&2
    exit 1
  fi
fi

echo "==> Using: $PY ($("$PY" --version 2>&1))"
echo "==> init (read-only): $INIT_WEIGHTS"
echo "==> output (new only): $OUTPUT_DIR"
echo "==> dataset: $DATASET_DIR"
echo "==> epochs=$EPOCHS batch=$BATCH workers=$WORKERS lr=$LR aug=$AUG_PRESET (early-stop ON)"
echo "==> nvidia-smi"
nvidia-smi || true

echo ""
echo "==> Ensure albumentations (rfdetr[augment])"
"$PY" -m pip install -q 'rfdetr[augment]' albumentations opencv-python-headless

echo ""
echo "==> Prepare custom Stage-2 (download/analyze/merge/offline train augs)"
PREP_ARGS=()
if [[ "${SKIP_DOWNLOAD:-0}" == "1" ]]; then
  PREP_ARGS+=(--skip-download)
fi
"$PY" -m tools.rfdetr_train.prepare_custom_stage2 "${PREP_ARGS[@]}"

if [[ ! -f "$DATASET_DIR/train/_annotations.coco.json" ]]; then
  echo "ERROR: missing $DATASET_DIR/train/_annotations.coco.json after prepare" >&2
  exit 1
fi

echo ""
echo "==> Fine-tune RFDETRMedium (anti-overfit: low LR + early stop + augs)"
echo "    warm-start via --pretrain-weights (NOT --resume; resume restores epoch and can exit at once)"
# Drop instant-fail artifacts from a prior --resume warm-start mistake
if [[ "${CLEAN_OUTPUT:-1}" == "1" ]]; then
  if [[ -d "$OUTPUT_DIR" ]]; then
    echo "    clearing stale *.pth under $OUTPUT_DIR (CLEAN_OUTPUT=1)"
    rm -f "$OUTPUT_DIR"/*.pth
  fi
fi
"$PY" -m tools.rfdetr_train.train \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --pretrain-weights "$INIT_WEIGHTS" \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --workers "$WORKERS" \
  --lr "$LR" \
  --aug-preset "$AUG_PRESET" \
  --early-stopping-patience 10

echo ""
echo "Done. New checkpoints under: $OUTPUT_DIR"
ls -lah "$OUTPUT_DIR"/*.pth 2>/dev/null || true
echo "Stage-1 source left untouched: $INIT_WEIGHTS"
