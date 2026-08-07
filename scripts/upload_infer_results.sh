#!/usr/bin/env bash
# Upload RF-DETR inference outputs to a Google Drive folder.
#
# One-time auth on the VM (browser / device login):
#   gcloud auth application-default login \
#     --scopes=https://www.googleapis.com/auth/drive.file,https://www.googleapis.com/auth/cloud-platform
#
# Then share the destination Drive folder with that same Google account (Editor).
#
# Usage:
#   ./scripts/upload_infer_results.sh \
#     --run-dir 'runs/rfdetr_infer/ROAD-1 Gopro' \
#     --folder  'https://drive.google.com/drive/folders/YOUR_FOLDER_ID'
#
# Uploads: annotated.mp4, defects.csv, defects.json, map_trail.html, summary.json
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

"$PY" -m pip install -q google-api-python-client google-auth google-auth-httplib2

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 --run-dir PATH --folder DRIVE_FOLDER_URL" >&2
  exit 1
fi

exec "$PY" -m tools.rfdetr_infer.upload_drive "$@"
