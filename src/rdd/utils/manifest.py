"""Run manifest + reproducibility helpers.

Writes a manifest.json capturing: config used, package/tool versions, resolved
device, git commit, and per-stage outputs. Also seeds RNGs.
"""
from __future__ import annotations

import json
import os
import platform
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .logging import get_logger

log = get_logger("rdd.manifest")

_TRACKED_PKGS = ["ultralytics", "supervision", "opencv-python", "numpy", "pandas", "torch"]
_TRACKED_BINS = ["ffmpeg", "ffprobe", "exiftool"]


def set_seeds(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _pkg_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in _TRACKED_PKGS:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def _bin_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in _TRACKED_BINS:
        path = shutil.which(name)
        if not path:
            out[name] = None
            continue
        try:
            res = subprocess.run(
                [name, "-version"], capture_output=True, text=True, timeout=10
            )
            out[name] = (res.stdout or res.stderr).splitlines()[0].strip()
        except Exception:
            out[name] = "present (version unknown)"
    return out


def _git_commit() -> str | None:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return res.stdout.strip() or None
    except Exception:
        return None


class Manifest:
    def __init__(self, run_dir: Path, config: dict[str, Any]):
        self.run_dir = Path(run_dir)
        self.data: dict[str, Any] = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "git_commit": _git_commit(),
            "packages": _pkg_versions(),
            "binaries": _bin_versions(),
            "config": config,
            "stages": {},
        }

    def record(self, stage: str, **info: Any) -> None:
        self.data["stages"][stage] = info

    def save(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / "manifest.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, default=str)
        log.info("Manifest written: %s", path)
        return path
