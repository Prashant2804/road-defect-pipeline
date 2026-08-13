#!/usr/bin/env bash
# Ultralytics RT-DETR near-field dashcam inference (Stage-2 weights).
#
#   tmux new -s rtdetr_infer
#   ./scripts/run_rtdetr_infer.sh \
#     --video 'https://drive.google.com/drive/folders/FOLDER_ID' \
#     --srt   'https://drive.google.com/drive/folders/FOLDER_ID' \
#     --weights runs/rtdetr_stage2/weights/best.pt \
#     --z-far 5 \
#     --out-dir 'runs/rfdetr_infer/ROAD-1-Gopro-rtdetr-v2'
#
# Never overwrite prior ROAD-1-Gopro / v2 / v3 / rtdetr POC folders — use a new --out-dir.
#
# Injects strict gates unless already passed: conf=0.5, min-overlap=0.50, nms-iou=0.5.
# Optional: set GOOGLE_MAPS_API_KEY in .env for dashboard Google Maps tiles.
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

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$PY" -m pip install -q "ultralytics>=8.3.0"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 --video PATH_OR_URL --weights PATH [--srt PATH_OR_URL] [--out-dir DIR] ..." >&2
  echo "Example:" >&2
  echo "  $0 --video gs://bucket/clip.mp4 --srt gs://bucket/clip.srt \\" >&2
  echo "     --weights runs/rtdetr_stage2/weights/best.pt \\" >&2
  echo "     --out-dir runs/rfdetr_infer/ROAD-1-Gopro-rtdetr-v2" >&2
  exit 1
fi

# Inject strict RT-DETR defaults only when the user did not already pass them.
USER_ARGS=("$@")
_has() {
  local flag="$1"
  local a
  for a in "${USER_ARGS[@]}"; do
    [[ "$a" == "$flag" ]] && return 0
  done
  return 1
}
EXTRA=()
_has --conf || EXTRA+=(--conf 0.5)
_has --min-overlap || EXTRA+=(--min-overlap 0.50)
_has --nms-iou || EXTRA+=(--nms-iou 0.5)

echo "==> Using: $PY ($("$PY" --version 2>&1))"
echo "==> backend=rtdetr"
if ((${#EXTRA[@]})); then
  echo "==> strict gates: ${EXTRA[*]}"
else
  echo "==> strict gates: (user-supplied)"
fi
if [[ -n "${GOOGLE_MAPS_API_KEY:-}" ]]; then
  echo "==> GOOGLE_MAPS_API_KEY is set (dashboard will use Google Maps)"
else
  echo "==> GOOGLE_MAPS_API_KEY unset — dashboard falls back to Leaflet"
fi

exec "$PY" -m tools.rfdetr_infer.run --backend rtdetr "${EXTRA[@]}" "$@"
