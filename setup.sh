#!/usr/bin/env bash
# One-time setup for the Road Defect Detection Pipeline on Linux / macOS.
#
# Installs everything the pipeline needs and verifies it actually works:
#   1. Python 3.10-3.13
#   2. FFmpeg with the v360 filter  - REQUIRED, not optional
#   3. A virtual environment in .venv
#   4. PyTorch - CUDA build if an NVIDIA GPU is present, CPU build otherwise
#   5. Everything in requirements.txt
#   6. A smoke test, so a green finish means green for real
#
# Safe to re-run: every step is skipped if already satisfied.
#
# Usage:
#   ./setup.sh              full setup, auto-detects GPU
#   ./setup.sh --cpu        force the CPU build of PyTorch
#   ./setup.sh --skip-system    don't touch system packages
#   ./setup.sh --dry-run    show what would happen

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

FORCE_CPU=0
SKIP_SYSTEM=0
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --cpu)          FORCE_CPU=1 ;;
    --skip-system)  SKIP_SYSTEM=1 ;;
    --dry-run)      DRY_RUN=1 ;;
    -h|--help)      sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg (try --help)"; exit 1 ;;
  esac
done

C_RESET=$'\033[0m'; C_CYAN=$'\033[36m'; C_GREEN=$'\033[32m'
C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'; C_BOLD=$'\033[1m'
say()  { printf '%s%s%s\n' "$C_CYAN" "$1" "$C_RESET"; }
ok()   { printf '  %sOK%s    %s\n' "$C_GREEN" "$C_RESET" "$1"; }
warn() { printf '  %sWARN%s  %s\n' "$C_YELLOW" "$C_RESET" "$1"; }
die()  { printf '  %sFAIL%s  %s\n' "$C_RED" "$C_RESET" "$1"; exit 1; }
step() { printf '\n%s[%s/6] %s%s\n' "$C_BOLD" "$1" "$2" "$C_RESET"; }
have() { command -v "$1" >/dev/null 2>&1; }

# sudo only when we are not already root and it exists.
SUDO=""
if [ "$(id -u)" -ne 0 ] && have sudo; then SUDO="sudo"; fi

pkg_install() {
  # $@ = package names
  if [ "$DRY_RUN" = 1 ]; then warn "DRY RUN - would install: $*"; return 0; fi
  if   have apt-get; then $SUDO apt-get update -qq && $SUDO apt-get install -y "$@"
  elif have dnf;     then $SUDO dnf install -y "$@"
  elif have yum;     then $SUDO yum install -y "$@"
  elif have pacman;  then $SUDO pacman -Sy --noconfirm "$@"
  elif have zypper;  then $SUDO zypper install -y "$@"
  elif have apk;     then $SUDO apk add "$@"
  elif have brew;    then brew install "$@"
  else return 1; fi
}

find_python() {
  for cand in python3.12 python3.11 python3.13 python3.10 python3 python; do
    have "$cand" || continue
    if "$cand" -c 'import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] < (3,14) else 1)' 2>/dev/null; then
      command -v "$cand"; return 0
    fi
  done
  return 1
}

printf '\n%s  Road Defect Detection Pipeline - Linux/macOS setup%s\n' "$C_BOLD" "$C_RESET"
printf '  %s\n' "$PWD"
[ "$DRY_RUN" = 1 ] && warn "DRY RUN - nothing will be installed or modified"

# ------------------------------------------------------------- 1. Python ----
step 1 "Python 3.10 - 3.13"
if PY="$(find_python)"; then
  ok "Python $("$PY" -c 'import sys;print("%d.%d.%d"%sys.version_info[:3])') at $PY"
elif [ "$SKIP_SYSTEM" = 1 ]; then
  die "No supported Python and --skip-system was given. Install python3.12."
else
  warn "No supported Python found - installing"
  pkg_install python3 python3-venv python3-pip || die "Could not install Python automatically. Install python3.12 and re-run."
  PY="$(find_python)" || die "Python installed but still not usable"
  ok "Python installed at $PY"
fi

# `python3-venv` is a separate package on Debian/Ubuntu and its absence is a
# confusing failure later, so check for it now rather than at venv creation.
if ! "$PY" -c 'import venv, ensurepip' 2>/dev/null; then
  warn "The venv module is incomplete (common on Debian/Ubuntu)"
  pkg_install python3-venv || die "Install python3-venv (or your distro's equivalent) and re-run"
fi

# ------------------------------------------------------------- 2. FFmpeg ----
step 2 "FFmpeg (with the v360 filter)"
# Genuinely required: v360 does the 360-to-flat reprojection, and FFmpeg is also
# the H.264 encoder for the annotated output video.
need_ffmpeg=1
if have ffmpeg; then
  if ffmpeg -hide_banner -filters 2>/dev/null | grep -q v360; then
    ok "FFmpeg present with v360"; need_ffmpeg=0
  else
    warn "FFmpeg is installed but has no v360 filter - a fuller build is needed"
  fi
fi
if [ "$need_ffmpeg" = 1 ]; then
  if [ "$SKIP_SYSTEM" = 1 ]; then
    warn "--skip-system given; install ffmpeg yourself"
  else
    pkg_install ffmpeg || warn "Could not install ffmpeg automatically - install it manually"
    if have ffmpeg; then ok "FFmpeg installed"; fi
  fi
fi

# ------------------------------------------------------------ 3. Virtualenv -
step 3 "Virtual environment (.venv)"
VENV_PY=".venv/bin/python"
if [ -x "$VENV_PY" ]; then
  ok ".venv already exists"
elif [ "$DRY_RUN" = 1 ]; then
  warn "DRY RUN - would create .venv"; VENV_PY="$PY"
else
  "$PY" -m venv .venv || die "Could not create .venv"
  ok ".venv created"
fi

if [ "$DRY_RUN" != 1 ]; then
  "$VENV_PY" -m pip install --upgrade pip setuptools wheel --quiet
  ok "pip upgraded"
fi

# --------------------------------------------------------------- 4. Torch ---
step 4 "PyTorch"
HAS_GPU=0
if [ "$FORCE_CPU" = 0 ] && have nvidia-smi && nvidia-smi >/dev/null 2>&1; then HAS_GPU=1; fi

if [ "$DRY_RUN" = 1 ]; then
  warn "DRY RUN - would install $([ "$HAS_GPU" = 1 ] && echo CUDA || echo CPU) PyTorch"
elif TORCH_VER="$("$VENV_PY" -c 'import torch;print(torch.__version__)' 2>/dev/null)"; then
  ok "PyTorch $TORCH_VER already installed"
  if [ "$HAS_GPU" = 1 ] && "$VENV_PY" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    :
  elif [ "$HAS_GPU" = 1 ]; then
    warn "An NVIDIA GPU is present but this PyTorch is CPU-only."
    warn "To switch: .venv/bin/python -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124"
  fi
elif [ "$HAS_GPU" = 1 ]; then
  say "  NVIDIA GPU detected - installing the CUDA build (large download)"
  "$VENV_PY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 \
    || { warn "CUDA wheel failed - falling back to CPU"; "$VENV_PY" -m pip install torch torchvision; }
  ok "PyTorch installed"
else
  say "  No NVIDIA GPU detected - installing the CPU build"
  "$VENV_PY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    || "$VENV_PY" -m pip install torch torchvision
  ok "PyTorch installed (CPU)"
fi

# -------------------------------------------------------- 5. Requirements ---
step 5 "Project requirements"
if [ "$DRY_RUN" = 1 ]; then
  warn "DRY RUN - would: pip install -r requirements.txt"
else
  # OpenCV needs libGL, which headless server images routinely lack. Failing here
  # with "ImportError: libGL.so.1" much later is a bad experience, so pre-empt it.
  if ! "$VENV_PY" -c 'import ctypes.util,sys; sys.exit(0 if ctypes.util.find_library("GL") else 1)' 2>/dev/null; then
    if [ "$SKIP_SYSTEM" != 1 ] && have apt-get; then
      warn "libGL missing (headless image?) - installing OpenCV's runtime deps"
      pkg_install libgl1 libglib2.0-0 || warn "Could not install libGL; if OpenCV fails to import, install libgl1"
    fi
  fi
  "$VENV_PY" -m pip install -r requirements.txt || die "requirements.txt install failed"
  ok "requirements.txt installed"
fi

# --------------------------------------------------------------- 6. Verify --
step 6 "Verifying the install"
if [ "$DRY_RUN" = 1 ]; then
  warn "DRY RUN - would import every stage and run the test suite"
else
  "$VENV_PY" - <<'PYEOF' || die "Import check failed - the install is incomplete"
import sys, importlib
sys.path.insert(0, 'src')
for m in ['rdd.config','rdd.geometry','rdd.validity','rdd.detect','rdd.eval',
          'rdd.roadseg','rdd.surface','rdd.quality','rdd.pipeline']:
    importlib.import_module(m)
import torch, ultralytics, cv2
print(f'  python      {sys.version.split()[0]}')
print(f'  torch       {torch.__version__}  cuda={torch.cuda.is_available()}')
print(f'  ultralytics {ultralytics.__version__}')
print(f'  opencv      {cv2.__version__}')
PYEOF
  ok "All pipeline stages import"

  if "$VENV_PY" -m pytest tests/ -q --no-header -x 2>&1 | tail -3; then
    ok "Test suite passed"
  else
    warn "Some tests failed - the pipeline may still run, but check the output above"
  fi
fi

printf '\n  %sSetup complete.%s\n\n' "$C_GREEN" "$C_RESET"
printf '  %sTry it without any footage of your own:%s\n' "$C_BOLD" "$C_RESET"
printf '      ./rdd.sh demo\n\n'
printf '  %sThen on your own video:%s\n' "$C_BOLD" "$C_RESET"
printf '      ./rdd.sh check   source=/path/to/road.mp4    # look at the road mask FIRST\n'
printf '      ./rdd.sh detect  source=/path/to/road.mp4\n\n'
printf '  All commands:  ./rdd.sh help\n\n'
have ffmpeg || warn "FFmpeg is not on PATH - install it before running the pipeline."
