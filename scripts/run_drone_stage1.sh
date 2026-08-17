#!/usr/bin/env bash
# One-shot drone Stage-1: download UAV-PDD2023 + UAPD + HighRPD + Roboflow
# pothole-drone → remap to the 6-class taxonomy → train RFDETRMedium.
#
# Run inside tmux/screen on the VM so SSH drops do not kill the job:
#   tmux new -s rfdetr-drone
#   ./scripts/run_drone_stage1.sh
#   # detach: Ctrl-b d    reattach: tmux attach -t rfdetr-drone
#
# Defaults (RTX 5090 32GB): batch=16, grad_accum=1, epochs=50, workers=8 —
# same budget as the dashcam Stage-1 run.
#   Bump batch:   EXTRA_TRAIN_ARGS="--batch 24" ./scripts/run_drone_stage1.sh
#   Resume:       EXTRA_TRAIN_ARGS="--resume runs/rfdetr_drone_stage1/checkpoint.pth" ./scripts/run_drone_stage1.sh
#   Skip a source: EXTRA_DOWNLOAD_ARGS="--no-uapd" ./scripts/run_drone_stage1.sh
#
# drainage_issue and edge_damage have no public drone source — see
# docs/DRONE_DATASETS.md before relying on those two classes from this run.
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

if [[ -z "${ROBOFLOW_API_KEY:-}" ]]; then
  echo "ERROR: ROBOFLOW_API_KEY not set. Copy .env.example → .env and paste your key." >&2
  exit 1
fi

if ! command -v "$PY" >/dev/null 2>&1 && [[ ! -x "$PY" ]]; then
  echo "ERROR: Python not found ($PY). Run ./scripts/setup_rfdetr_vm.sh first." >&2
  exit 1
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "==> Using: $PY ($("$PY" --version 2>&1))"
echo "==> nvidia-smi"
nvidia-smi || true

echo ""
echo "==> Download + prepare drone Stage 1 (UAV-PDD2023 + UAPD + HighRPD + Roboflow pothole-drone)"
# shellcheck disable=SC2086
$PY -m tools.rfdetr_train.download_drone ${EXTRA_DOWNLOAD_ARGS:-}

echo ""
echo "==> Train RFDETRMedium on drone Stage 1"
# shellcheck disable=SC2086
$PY -m tools.rfdetr_train.train_drone ${EXTRA_TRAIN_ARGS:-}

echo ""
echo "Done. Checkpoints under: $ROOT/runs/rfdetr_drone_stage1"
ls -lah "$ROOT/runs/rfdetr_drone_stage1"/*.pth 2>/dev/null || true
