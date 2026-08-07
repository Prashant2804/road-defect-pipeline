#!/usr/bin/env bash
# RF-DETR near-field dashcam inference (Phase 1).
#
# Prerequisites: Stage-1 weights + video (+ optional .srt sidecar).
# Long videos: run inside tmux so SSH disconnects do not kill the job.
#
#   tmux new -s rfdetr_infer
#   ./scripts/run_rfdetr_infer.sh \
#     --video /path/to/dashcam.mp4 \
#     --weights runs/rfdetr_stage1/checkpoint_best_total.pth \
#     --z-far 5
#
# Outputs under runs/rfdetr_infer/<video_stem>/ :
#   annotated.mp4  defects.csv  defects.json  map_trail.html  summary.json
#
# Phase 2 (later): wire inference.backend=rfdetr into the full rdd pipeline
# and optional 3-panel dashboard (sidebar + overlay + map).
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

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 --video PATH --weights PATH [--srt PATH] [--z-far 5] ..." >&2
  echo "Example:" >&2
  echo "  $0 --video clip.mp4 --weights runs/rfdetr_stage1/checkpoint_best_total.pth" >&2
  exit 1
fi

echo "==> Using: $PY ($("$PY" --version 2>&1))"
exec "$PY" -m tools.rfdetr_infer.run "$@"
