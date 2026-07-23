"""Config loading, validation, and dotted access.

Loads config.yaml into a lightweight object that supports both attribute and
dict access, so `cfg.preprocess.reproject.pitch_deg` and
`cfg["preprocess"]["reproject"]` both work. Validation is intentionally light:
we check the handful of invariants that would otherwise fail deep in a stage.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Cfg(dict):
    """dict with attribute access and recursive wrapping."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return Cfg(val) if isinstance(val, dict) else val

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def load_config(path: str | Path) -> Cfg:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = Cfg(raw)
    _validate(cfg)
    return cfg


def _validate(cfg: Cfg) -> None:
    """Fail fast on the invariants that matter most."""
    errors: list[str] = []

    classes = cfg.get_path("model.classes")
    if not classes or not isinstance(classes, list):
        errors.append("model.classes must be a non-empty list")

    split_mode = cfg.get_path("model.train.split.mode")
    if split_mode == "random":
        errors.append(
            "model.train.split.mode == 'random' is forbidden: adjacent video "
            "frames leak between train/val/test. Use 'segment' or 'time'."
        )

    samp_mode = cfg.get_path("preprocess.sampling.mode")
    if samp_mode not in (None, "distance", "time", "every_n"):
        errors.append(f"preprocess.sampling.mode invalid: {samp_mode!r}")

    fov = cfg.get_path("preprocess.reproject.h_fov_deg")
    if fov is not None and not (0 < fov < 180):
        errors.append(f"preprocess.reproject.h_fov_deg must be in (0,180): {fov}")

    tracker = cfg.get_path("inference.tracker")
    if tracker not in (None, "botsort", "bytetrack"):
        errors.append(f"inference.tracker must be botsort|bytetrack: {tracker!r}")

    if errors:
        raise ValueError("Invalid config:\n  - " + "\n  - ".join(errors))
