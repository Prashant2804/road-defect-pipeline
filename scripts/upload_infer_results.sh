#!/usr/bin/env bash
# Upload RF-DETR inference outputs.
#
# Browser OAuth to Drive is blocked by Google ("This app is blocked") for the
# default Cloud SDK client + full Drive scope. Prefer GCS, or a service account.
#
# --- A) Google Cloud Storage (recommended) ---
#   ./scripts/upload_infer_to_gcs.sh \
#     --run-dir 'runs/rfdetr_infer/ROAD-1 Gopro' \
#     --gcs gs://YOUR_BUCKET/rfdetr_infer/ROAD-1-Gopro
#
# --- B) Google Drive via service account (no browser) ---
#   1. Create SA key JSON in GCP Console
#   2. Share the Drive folder with the SA email as Editor
#   3. ./scripts/upload_infer_results.sh \
#        --run-dir 'runs/rfdetr_infer/ROAD-1 Gopro' \
#        --folder  'https://drive.google.com/drive/folders/FOLDER_ID' \
#        --service-account /path/to/sa.json
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
  echo "Usage (Drive + SA): $0 --run-dir PATH --folder URL --service-account sa.json" >&2
  echo "Or use GCS:         ./scripts/upload_infer_to_gcs.sh --run-dir PATH --gcs gs://..." >&2
  exit 1
fi

exec "$PY" -m tools.rfdetr_infer.upload_drive "$@"
