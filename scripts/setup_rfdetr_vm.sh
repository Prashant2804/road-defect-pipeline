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
#
# Ubuntu prerequisite (once):  sudo apt install -y python3.12-venv
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-$ROOT/.venv}"

echo "==> Repo: $ROOT"
echo "==> Python: $($PYTHON --version 2>&1)"
echo "==> Venv: $VENV"

if ! "$PYTHON" -c "import ensurepip" 2>/dev/null; then
  echo "ERROR: ensurepip missing. On Ubuntu run:" >&2
  echo "  sudo apt install -y python3.12-venv" >&2
  exit 1
fi

# Recreate if missing or left broken (failed ensurepip can leave a half-dir
# with a python symlink but no activate script).
if [[ ! -f "$VENV/bin/activate" || ! -x "$VENV/bin/python" ]]; then
  echo "==> Recreating venv (missing or incomplete)"
  rm -rf "$VENV"
  "$PYTHON" -m venv "$VENV"
fi

# Prefer venv python explicitly (Ubuntu often has no bare `python`)
PY="$VENV/bin/python"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

"$PY" -m pip install --upgrade pip wheel setuptools

# Core train stack. Torch comes via rfdetr[train] / pip resolution.
# On bleeding-edge GPUs (5090), install a CUDA wheel that matches your driver if needed.
"$PY" -m pip install \
  "rfdetr[train]" \
  "roboflow>=1.1.0" \
  "kaggle>=1.6.0" \
  "opencv-python-headless>=4.8" \
  "Pillow>=9.0" \
  "PyYAML>=6.0" \
  "python-dotenv>=1.0" \
  "tqdm" \
  "gdown>=5.0" \
  "ultralytics>=8.3.0" \
  "google-api-python-client>=2.0" \
  "google-auth>=2.0" \
  "google-auth-httplib2>=0.2" \
  "google-auth-oauthlib>=1.0"

echo ""
echo "==> Verifying imports"
"$PY" - <<'PY'
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
