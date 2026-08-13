#!/usr/bin/env bash
# Extended near-field ROI infer (taller trapezoid). Does NOT change default
# run_rfdetr_infer.sh / InferConfig defaults used by prior POC runs.
#
# Stage-1 RF-DETR Medium @ conf 0.30, road_top_y=0.32, z_far=8.
# New out-dir only: runs/rfdetr_infer/ROAD-1-Gopro-medium-stage1-c030-extroi
#
#   tmux new -s rfdetr_infer_extroi
#   ./scripts/run_rfdetr_infer_extended_roi.sh
#
# Optional overrides are forwarded to the Python module, e.g.:
#   ./scripts/run_rfdetr_infer_extended_roi.sh --max-frames 300 --stride 5
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

export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "==> Using: $PY ($("$PY" --version 2>&1))"
echo "==> Extended ROI infer (road_top_y=0.32 z_far=8 conf=0.30)"
if command -v gcloud >/dev/null 2>&1; then
  echo "==> gcloud: $(gcloud --version 2>/dev/null | head -1 || true)"
elif command -v gsutil >/dev/null 2>&1; then
  echo "==> gsutil available"
else
  echo "==> NOTE: no gcloud/gsutil — gs:// downloads will fail (https:// still OK)"
fi

exec "$PY" -m tools.rfdetr_infer.run_extended_nearfield "$@"
