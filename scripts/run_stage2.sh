#!/usr/bin/env bash
# Stage-2: multi-source merge → RFDETRLarge @ 100 epochs (overnight).
#
#   tmux new -s rfdetr_stage2
#   ./scripts/run_stage2.sh
#   # detach: Ctrl-b d    reattach: tmux attach -t rfdetr_stage2
#
# Defaults (RTX 5090): batch=4, grad_accum=4 (eff. 16), epochs=100, Large.
# OOM:   EXTRA_TRAIN_ARGS="--batch 2 --grad-accum 8" ./scripts/run_stage2.sh
# Skip download if data ready:
#   SKIP_DOWNLOAD=1 EXTRA_TRAIN_ARGS="--epochs 100" ./scripts/run_stage2.sh
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

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "==> Using: $PY ($("$PY" --version 2>&1))"
echo "==> nvidia-smi"
nvidia-smi || true

if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
  echo ""
  echo "==> Download + prepare Stage 2 (multi-source merge + pothole cap)"
  # shellcheck disable=SC2086
  $PY -m tools.rfdetr_train.download_stage2 ${EXTRA_DOWNLOAD_ARGS:-}
else
  echo "==> SKIP_DOWNLOAD=1 — using existing data/rfdetr/stage2"
fi

echo ""
echo "==> Train RFDETRLarge Stage 2 (100 epochs default)"
# shellcheck disable=SC2086
$PY -m tools.rfdetr_train.train_stage2 ${EXTRA_TRAIN_ARGS:-}

echo ""
echo "Done. Checkpoints under: $ROOT/runs/rfdetr_stage2"
ls -lah "$ROOT/runs/rfdetr_stage2"/*.pth 2>/dev/null || true
