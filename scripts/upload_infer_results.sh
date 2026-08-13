#!/usr/bin/env bash
# Upload RF-DETR inference outputs to Google Drive (compliant Desktop OAuth).
#
# Do NOT use gcloud application-default login for Drive — Google blocks that
# client for full Drive scope ("This app is blocked").
#
# === One-time Console setup ===
# 1. GCP project → enable Google Drive API
# 2. OAuth consent screen → External → Testing → add YOUR Gmail as Test user
# 3. Credentials → Create OAuth client ID → Desktop app → Download JSON
# 4. Save on VM:  mkdir -p ~/secrets && mv ~/Downloads/client_secret_*.json ~/secrets/drive_oauth_client.json
# 5. Share destination Drive folder with that Gmail (Editor)
#
# === Upload ROAD-1 results ===
#   ./scripts/upload_infer_results.sh \
#     --run-dir 'runs/rfdetr_infer/ROAD-1 Gopro' \
#     --folder  'https://drive.google.com/drive/folders/1gFw80e4fMdL3ztDlUxVdQinNQlskpoz-' \
#     --client-secret ~/secrets/drive_oauth_client.json
#
# === Upload dashboard pack to a NEW Drive subfolder (POC untouched) ===
#   ./scripts/upload_infer_results.sh \
#     --run-dir 'runs/rfdetr_infer/ROAD-1-Gopro-v3_dashboard' \
#     --folder  'https://drive.google.com/drive/folders/1gFw80e4fMdL3ztDlUxVdQinNQlskpoz-' \
#     --subfolder 'ROAD-1-Gopro-v3-dashboard' \
#     --dashboard \
#     --client-secret ~/secrets/drive_oauth_client.json
#
# First-time auth from a laptop (SSH tunnel so localhost redirect works):
#   ssh -L 8090:localhost:8090 ubuntu@YOUR_VM_IP
#   # in that session:
#   ./scripts/upload_infer_results.sh ... --client-secret ~/secrets/drive_oauth_client.json
# Open the printed URL in your laptop browser, approve, return to SSH.
#
# Uploads: annotated.mp4, defects.csv, defects.json, map_trail.html, summary.json
# Dashboard mode: index.html, annotated.mp4, defects.json, route.json, ...
#
# Fallback (no API): scp the run folder to your laptop, drag into drive.google.com
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

"$PY" -m pip install -q \
  google-api-python-client \
  google-auth \
  google-auth-httplib2 \
  google-auth-oauthlib

if [[ $# -lt 1 ]]; then
  echo "Usage:" >&2
  echo "  $0 --run-dir 'runs/rfdetr_infer/ROAD-1 Gopro' \\" >&2
  echo "     --folder 'https://drive.google.com/drive/folders/FOLDER_ID' \\" >&2
  echo "     --client-secret ~/secrets/drive_oauth_client.json" >&2
  exit 1
fi

exec "$PY" -m tools.rfdetr_infer.upload_drive "$@"
