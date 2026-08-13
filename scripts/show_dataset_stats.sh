#!/usr/bin/env bash
# Class-wise instance/image tables for Stage-1 and Stage-2 training datasets.
#
#   ./scripts/show_dataset_stats.sh
#   ./scripts/show_dataset_stats.sh --only stage1,custom_stage2_aug
#   ./scripts/show_dataset_stats.sh --json-out /tmp/dataset_stats.json
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

exec "$PY" -m tools.rfdetr_train.dataset_stats "$@"
