#!/usr/bin/env bash
# Setup a headless venv for RF-DETR Stage-1 training on an RTX 5090 VM.
#
# Usage (from repo root, over SSH):
#   cp .env.example .env    # paste ROBOFLOW_API_KEY
#   ./scripts/setup_rfdetr_vm.sh
#   tmux new -s rfdetr
#   ./scripts/run_stage1.sh
#
# Long runs should live in tmux/screen so SSH disconnects do not kill training.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-$ROOT/.venv}"

echo "==> Repo: $ROOT"
echo "==> Python: $($PYTHON --version 2>&1)"
echo "==> Venv: $VENV"

if [[ ! -d "$VENV" ]]; then
  "$PYTHON" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

python -m pip install --upgrade pip wheel setuptools

# Core train stack. Torch comes via rfdetr[train] / pip resolution.
# On bleeding-edge GPUs (5090), install a CUDA wheel that matches your driver if needed.
python -m pip install \
  "rfdetr[train]" \
  "roboflow>=1.1.0" \
  "kaggle>=1.6.0" \
  "opencv-python-headless>=4.8" \
  "Pillow>=9.0" \
  "PyYAML>=6.0" \
  "python-dotenv>=1.0" \
  "tqdm"

echo ""
echo "==> Verifying imports"
python - <<'PY'
from rfdetr import RFDETRMedium
import torch
print("RFDETRMedium OK")
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

if [[ ! -f "$ROOT/.env" && -f "$ROOT/.env.example" ]]; then
  echo ""
  echo "NOTE: copy secrets next:"
  echo "  cp .env.example .env   # then edit ROBOFLOW_API_KEY"
fi

echo ""
echo "Setup done. Activate with:  source $VENV/bin/activate"
echo "Then run:                   ./scripts/run_stage1.sh"
