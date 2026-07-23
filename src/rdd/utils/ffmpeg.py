"""Thin FFmpeg/ffprobe wrapper with clear errors when the binary is missing.

The whole preprocess/ stage depends on ffmpeg's `v360` filter. We detect its
absence up front and raise an actionable message instead of a cryptic
FileNotFoundError deep in a subprocess call.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .logging import get_logger

log = get_logger("rdd.ffmpeg")

_INSTALL_HINT = (
    "FFmpeg not found on PATH. Install it and ensure the `v360` filter is built in:\n"
    "  Windows : winget install Gyan.FFmpeg   (or download from https://www.gyan.dev/ffmpeg/builds/)\n"
    "  macOS   : brew install ffmpeg\n"
    "  Linux   : sudo apt install ffmpeg\n"
    "Verify with: ffmpeg -hide_banner -filters | findstr v360   (Windows)  /  grep v360  (unix)"
)


class FFmpegNotFound(RuntimeError):
    pass


def ensure_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise FFmpegNotFound(_INSTALL_HINT)
    return exe


def has_v360() -> bool:
    if not shutil.which("ffmpeg"):
        return False
    try:
        res = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            capture_output=True, text=True, timeout=15,
        )
        return "v360" in res.stdout
    except Exception:
        return False


def run(args: list[str], desc: str = "ffmpeg") -> None:
    """Run an ffmpeg command (args already exclude the leading 'ffmpeg')."""
    ensure_ffmpeg()
    cmd = ["ffmpeg", "-hide_banner", "-y", *args]
    log.info("%s: %s", desc, " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        log.error("ffmpeg failed (%d):\n%s", res.returncode, res.stderr[-2000:])
        raise RuntimeError(f"{desc} failed; see log above")


def probe(path: str | Path) -> dict:
    """ffprobe -> parsed json (streams + format). Empty dict if ffprobe missing."""
    if not shutil.which("ffprobe"):
        log.warning("ffprobe not found; skipping metadata probe of %s", path)
        return {}
    res = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        return {}
    try:
        return json.loads(res.stdout)
    except json.JSONDecodeError:
        return {}
