#!/usr/bin/env bash
# Upload inference results WITHOUT the blocked Drive OAuth browser flow.
#
# Recommended (works with your existing gcloud login):
#   ./scripts/upload_infer_to_gcs.sh \
#     --run-dir 'runs/rfdetr_infer/ROAD-1 Gopro' \
#     --gcs gs://YOUR_BUCKET/rfdetr_infer/ROAD-1-Gopro
#
# Then open the GCS console / copy files into Drive if needed.
#
# Optional Drive path (no browser OAuth): use a service account JSON —
# see ./scripts/upload_infer_results.sh --service-account ...
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_DIR=""
GCS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) RUN_DIR="${2:-}"; shift 2 ;;
    --gcs) GCS="${2:-}"; shift 2 ;;
    -h|--help)
      sed -n '2,14p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$RUN_DIR" || -z "$GCS" ]]; then
  echo "Usage: $0 --run-dir PATH --gcs gs://BUCKET/PREFIX" >&2
  exit 1
fi

if [[ ! -d "$RUN_DIR" ]]; then
  echo "ERROR: run dir not found: $RUN_DIR" >&2
  exit 1
fi

if [[ "$GCS" != gs://* ]]; then
  echo "ERROR: --gcs must start with gs://" >&2
  exit 1
fi

# Normalize trailing slash
GCS="${GCS%/}"

FILES=(annotated.mp4 defects.csv defects.json map_trail.html summary.json)
echo "==> Uploading from: $RUN_DIR"
echo "==> Destination:    $GCS/"

UPLOADED=0
for f in "${FILES[@]}"; do
  src="$RUN_DIR/$f"
  if [[ ! -f "$src" ]]; then
    echo "  skip missing: $f"
    continue
  fi
  echo "  -> $f"
  gcloud storage cp "$src" "$GCS/$f"
  UPLOADED=$((UPLOADED + 1))
done

if [[ "$UPLOADED" -eq 0 ]]; then
  echo "ERROR: nothing uploaded" >&2
  exit 1
fi

echo ""
echo "Done ($UPLOADED files)."
echo "List:  gcloud storage ls $GCS/"
echo "Open:  https://console.cloud.google.com/storage/browser/${GCS#gs://}"
