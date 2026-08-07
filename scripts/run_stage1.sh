#!/usr/bin/env bash
# One-shot Stage-1: download CRRI → remap → train RFDETRMedium.
#
# Run inside tmux/screen on the VM so SSH drops do not kill the job:
#   tmux new -s rfdetr
#   ./scripts/run_stage1.sh
#   # detach: Ctrl-b d    reattach: tmux attach -t rfdetr
#
# Defaults (RTX 5090 32GB): batch=16, grad_accum=1, epochs=50, workers=8.
# Bump batch if VRAM allows:  EXTRA_TRAIN_ARGS="--batch 24" ./scripts/run_stage1.sh
# Resume:                     EXTRA_TRAIN_ARGS="--resume runs/rfdetr_stage1/checkpoint.pth" ./scripts/run_stage1.sh
# Extra data:                 EXTRA_DOWNLOAD_ARGS="--bharatpothole --road-crack" ./scripts/run_stage1.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

if [[ -z "${ROBOFLOW_API_KEY:-}" ]]; then
  echo "ERROR: ROBOFLOW_API_KEY not set. Copy .env.example → .env and paste your key." >&2
  exit 1
fi

PY="${PYTHON:-python}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "==> nvidia-smi"
nvidia-smi || true

echo ""
echo "==> Download + prepare Stage 1 (CRRI)"
$PY -m tools.rfdetr_train.download ${EXTRA_DOWNLOAD_ARGS:-}

echo ""
echo "==> Train RFDETRMedium Stage 1"
$PY -m tools.rfdetr_train.train ${EXTRA_TRAIN_ARGS:-}

echo ""
echo "Done. Checkpoints under: $ROOT/runs/rfdetr_stage1"
ls -lah "$ROOT/runs/rfdetr_stage1"/*.pth 2>/dev/null || true
