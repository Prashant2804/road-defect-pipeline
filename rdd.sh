#!/usr/bin/env bash
# Road Defect Detection Pipeline - simple task runner (Linux / macOS).
#
# Ultralytics/Roboflow-style command line:
#     ./rdd.sh <task> key=value key=value ...
#
# Runs inside .venv automatically - no activation needed.
# Run  ./rdd.sh help  for the full list.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Prefer .venv, but accept any interpreter that already has the dependencies —
# plenty of people manage their own conda or system environment and should not be
# forced into a second one. RDD_PYTHON overrides both.
VENV_PY=".venv/bin/python"
[ -x "$VENV_PY" ] || VENV_PY=".venv/Scripts/python.exe"     # Git Bash on Windows
if [ ! -x "$VENV_PY" ]; then
  VENV_PY=""
  if [ -n "${RDD_PYTHON:-}" ]; then
    VENV_PY="$RDD_PYTHON"
  else
    for c in python3 python; do
      command -v "$c" >/dev/null 2>&1 || continue
      if "$c" -c 'import ultralytics, cv2, torch' >/dev/null 2>&1; then
        VENV_PY="$(command -v "$c")"; break
      fi
    done
  fi
  if [ -n "$VENV_PY" ]; then
    printf '  \033[33m(no .venv - using %s, which already has the dependencies)\033[0m\n' "$VENV_PY"
  else
    printf '\n  \033[31mNo usable Python environment - run setup first:\033[0m\n      ./setup.sh\n\n'
    exit 1
  fi
fi

TASK="${1:-help}"; shift || true

declare -A OPT
OVERRIDES=()
for a in "$@"; do
  [ -z "$a" ] && continue
  case "$a" in
    *=*) ;;
    *) printf '  \033[31mBad argument "%s" - expected key=value\033[0m\n' "$a"; exit 1 ;;
  esac
  k="${a%%=*}"; v="${a#*=}"
  # Dotted keys (roadseg.backend=sam) pass straight through as config overrides, so
  # every knob in config.yaml is reachable without editing files.
  if [[ "$k" == *.* ]]; then OVERRIDES+=(--set "$k=$v"); else OPT["${k,,}"]="$v"; fi
done

opt() { local k="$1" d="${2-}"; printf '%s' "${OPT[$k]:-$d}"; }

# Sets the global NEED and exits the SCRIPT when the key is absent.
# Deliberately not `v=$(need ...)`: `exit` inside a command substitution only ends
# the subshell, so the script would carry on and use the error message itself as the
# value — which showed up as "Input video not found: Missing required source=".
need() {
  local k="$1" hint="$2"
  NEED="${OPT[$k]:-}"
  if [ -z "$NEED" ]; then
    printf '  \033[31mMissing required "%s="   e.g. %s\033[0m\n' "$k" "$hint"
    exit 1
  fi
}

run() {
  local all=("$@" ${OVERRIDES[@]+"${OVERRIDES[@]}"})
  printf '  \033[90m> python run.py %s\033[0m\n' "${all[*]}"
  "$VENV_PY" run.py "${all[@]}"
  exit $?
}

common_args() {
  local a=()
  [ -n "$(opt view)"    ] && a+=(--view "$(opt view)")
  [ -n "$(opt device)"  ] && a+=(--device "$(opt device)")
  [ -n "$(opt project)" ] && a+=(--output "$(opt project)")
  [ -n "$(opt name)"    ] && a+=(--set "run.name=$(opt name)")
  [ -n "$(opt conf)"    ] && a+=(--set "inference.conf=$(opt conf)")
  [ -n "$(opt imgsz)"   ] && a+=(--set "inference.imgsz=$(opt imgsz)")
  printf '%s\n' "${a[@]+"${a[@]}"}"
}
# shellcheck disable=SC2207
read_common() { COMMON=(); while IFS= read -r l; do [ -n "$l" ] && COMMON+=("$l"); done < <(common_args); }

case "${TASK,,}" in

  detect|predict|run)
    need source 'rdd.sh detect source=road.mp4'; SRC="$NEED"
    read_common
    A=(--input "$SRC" "${COMMON[@]+"${COMMON[@]}"}")
    [ -n "$(opt weights)" ] && A+=(--set "inference.weights=$(opt weights)")
    run "${A[@]}"
    ;;

  # Look at the road mask before trusting anything downstream.
  check|roadseg|preview)
    need source 'rdd.sh check source=road.mp4'; SRC="$NEED"
    read_common
    run roadseg --input "$SRC" --n "$(opt n 8)" "${COMMON[@]+"${COMMON[@]}"}"
    ;;

  validity|coverage)
    need source 'rdd.sh validity source=road.mp4'; SRC="$NEED"
    read_common
    A=(validity --input "$SRC" --every "$(opt every 10)" --stride "$(opt stride 3)"
       "${COMMON[@]+"${COMMON[@]}"}")
    [ "$(opt traffic false)" = "false" ] && A+=(--no-traffic)
    [ -n "$(opt json)" ] && A+=(--json "$(opt json)")
    run "${A[@]}"
    ;;

  quality)
    need source 'rdd.sh quality source=road.mp4'; SRC="$NEED"
    read_common
    A=(quality --input "$SRC" "${COMMON[@]+"${COMMON[@]}"}")
    [ -n "$(opt csv)" ] && A+=(--csv "$(opt csv)")
    run "${A[@]}"
    ;;

  preprocess|prep)
    need source 'rdd.sh preprocess source=road.mp4'; SRC="$NEED"
    read_common
    run preprocess --input "$SRC" "${COMMON[@]+"${COMMON[@]}"}"
    ;;

  label|annotate)
    run annotate --frames "$(opt frames data/rectified)"
    ;;

  train)
    A=(train --labels "$(opt data data/labels)")
    [ -n "$(opt device)"   ] && A+=(--device "$(opt device)")
    [ -n "$(opt epochs)"   ] && A+=(--set "model.train.epochs=$(opt epochs)")
    [ -n "$(opt batch)"    ] && A+=(--set "model.train.batch=$(opt batch)")
    [ -n "$(opt imgsz)"    ] && A+=(--set "model.train.imgsz=$(opt imgsz)")
    [ -n "$(opt patience)" ] && A+=(--set "model.train.patience=$(opt patience)")
    run "${A[@]}"
    ;;

  # Measures precision per unique defect and writes calibration.yaml.
  val|validate|evaluate|eval)
    DEFECTS="$(opt defects)"
    [ -z "$DEFECTS" ] && DEFECTS="$(opt project out)/$(opt name default)/defects.csv"
    need truth 'truth=ground_truth.csv  (columns: class,first_frame,last_frame)'; TRUTH="$NEED"
    A=(evaluate --defects "$DEFECTS" --truth "$TRUTH")
    SUM="$(opt summary)"
    [ -z "$SUM" ] && [ -f "$(dirname "$DEFECTS")/summary.json" ] && SUM="$(dirname "$DEFECTS")/summary.json"
    [ -n "$SUM" ] && A+=(--summary "$SUM")
    [ -n "$(opt out)" ]    && A+=(--out "$(opt out)")
    [ -n "$(opt target)" ] && A+=(--set "eval.target_precision=$(opt target)")
    run "${A[@]}"
    ;;

  # End-to-end on generated footage, so the install can be exercised with no
  # footage of your own.
  demo)
    printf '\n  \033[36mGenerating synthetic road footage with known ground truth...\033[0m\n'
    "$VENV_PY" tools/make_synthetic_road.py --out data/raw --frames 90 --only car || exit $?
    printf '\n  \033[36mRoad + surface masks (check these look right):\033[0m\n'
    "$VENV_PY" run.py roadseg --input data/raw/synthetic_road_car.mp4 --view car_flat --n 4
    printf '\n  \033[36mFull pipeline...\033[0m\n'
    "$VENV_PY" run.py --input data/raw/synthetic_road_car.mp4 --view car_flat \
        --set run.name=demo --set validity.traffic.enabled=false
    rc=$?
    printf '\n  \033[32mArtifacts in out/demo/  - open out/demo/report.html\033[0m\n'
    exit $rc
    ;;

  doctor|env)
    "$VENV_PY" tools/doctor.py
    exit $?
    ;;

  test|tests)
    "$VENV_PY" -m pytest tests/ -q
    exit $?
    ;;

  synthetic|gen)
    A=(tools/make_synthetic_road.py --out "$(opt out data/raw)" --frames "$(opt frames 120)")
    [ "$(opt scenarios true)" != "false" ] && A+=(--scenarios)
    [ "$(opt degraded true)"  != "false" ] && A+=(--degraded)
    "$VENV_PY" "${A[@]}"
    exit $?
    ;;

  help|-h|--help)
    ;;
  *)
    printf '  \033[31mUnknown task "%s"\033[0m\n' "$TASK"
    ;;
esac

cat <<'HELP'

  Road Defect Detection Pipeline

  USAGE
      ./rdd.sh <task> key=value ...

  TASKS
      demo                     end-to-end on generated footage (start here)
      doctor                   check the environment is sane
      check    source=VIDEO    road + surface mask previews  <- DO THIS FIRST
      validity source=VIDEO    which frames are assessable, and why not
      quality  source=VIDEO    sharpness / exposure / noise report
      detect   source=VIDEO    full pipeline -> annotated video + report
      preprocess source=VIDEO  reproject + sample frames for labelling
      label    frames=DIR      pick the most useful frames to label first
      train    data=DIR        fine-tune on your labels
      val      truth=GT.csv    measure precision, calibrate thresholds
      synthetic                generate test footage
      test                     run the test suite

  COMMON KEYS
      source=PATH              input video
      view=car_flat            car_flat | car_360 | drone_nadir
      weights=best.pt          trained model
      conf=0.35                confidence threshold
      imgsz=1280               inference size
      device=cpu               cpu | cuda | cuda:0
      name=myrun               output goes to out/<name>/
      project=out              output root

  ANY config.yaml key works as an override - dotted keys pass straight through:
      ./rdd.sh detect source=road.mp4 geometry.camera.height_m=1.5
      ./rdd.sh detect source=road.mp4 roadseg.backend=sam surface.mud.min_warmer_z=3.0

  TYPICAL FIRST RUN
      ./rdd.sh doctor
      ./rdd.sh demo
      ./rdd.sh check  source=/videos/road.mp4 view=car_flat
      ./rdd.sh detect source=/videos/road.mp4 view=car_flat

  MEASURE THE ROAD IN METRES (needed for m2 areas and IRC severity)
      ./rdd.sh detect source=road.mp4 geometry.camera.height_m=1.35 geometry.camera.h_fov_deg=95

HELP
exit 0
