#!/usr/bin/env bash
# RF-DETR near-field dashcam inference (Phase 1).
#
# Prerequisites: Stage-1 weights + video (+ optional .srt).
# --video / --srt accept local paths, gs:// URIs, or https:// URLs.
# Long videos: run inside tmux so SSH disconnects do not kill the job.
#
#   tmux new -s rfdetr_infer
#   ./scripts/run_rfdetr_infer.sh \
#     --video 'https://drive.google.com/drive/folders/FOLDER_ID' \
#     --srt   'https://drive.google.com/drive/folders/FOLDER_ID' \
#     --weights runs/rfdetr_stage1/checkpoint_best_total.pth \
#     --z-far 5
#
# Google Drive folders/files need sharing = "Anyone with the link".
# Same folder URL for --video and --srt downloads once and picks .mp4 + .srt.
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
  echo "Usage: $0 --video PATH_OR_URL --weights PATH [--srt PATH_OR_URL] [--z-far 5] ..." >&2
  echo "Example (GCS):" >&2
  echo "  $0 --video gs://bucket/clip.mp4 --srt gs://bucket/clip.srt \\" >&2
  echo "     --weights runs/rfdetr_stage1/checkpoint_best_total.pth" >&2
  exit 1
fi

echo "==> Using: $PY ($("$PY" --version 2>&1))"
if command -v gcloud >/dev/null 2>&1; then
  echo "==> gcloud: $(gcloud --version 2>/dev/null | head -1 || true)"
elif command -v gsutil >/dev/null 2>&1; then
  echo "==> gsutil available"
else
  echo "==> NOTE: no gcloud/gsutil — gs:// downloads will fail (https:// still OK)"
fi

exec "$PY" -m tools.rfdetr_infer.run "$@"
