"""Config loading, validation, and dotted access.

Loads config.yaml into a lightweight object that supports both attribute and
dict access, so `cfg.preprocess.reproject.pitch_deg` and
`cfg["preprocess"]["reproject"]` both work.

Nested dicts are converted to `Cfg` **once, at load time** (not lazily on each
attribute read). That matters: a lazy wrapper returns a fresh throwaway object
per access, so `cfg.run.name = "x"` would write to a temporary and silently
vanish. Converting in place makes nested writes stick.

Validation is intentionally light: we check the invariants that would otherwise
fail deep inside a stage, plus the geometry ones that fail *silently* by
producing subtly wrong images.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

VIEWPOINTS = ("car_360", "car_flat", "drone_nadir")
ROADSEG_BACKENDS = ("geometric", "classical", "sam", "none")
OCCLUSION_POLICIES = ("abstain", "flag", "exclude")


class Cfg(dict):
    """dict with attribute access. Nested dicts are Cfg instances already."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = _wrap(value)

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        """Set a nested key, creating intermediate dicts as needed."""
        parts = dotted.split(".")
        node: Cfg = self
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = Cfg()
                node[part] = nxt
            node = nxt  # type: ignore[assignment]
        node[parts[-1]] = _wrap(value)


def _wrap(value: Any) -> Any:
    """Recursively convert plain dicts/lists into Cfg-backed structures."""
    if isinstance(value, Cfg):
        return value
    if isinstance(value, dict):
        return Cfg({k: _wrap(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def load_config(path: str | Path) -> Cfg:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = _wrap(raw)
    if not isinstance(cfg, Cfg):
        raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}")
    _validate(cfg)
    return cfg


def _validate(cfg: Cfg) -> None:
    """Fail fast on the invariants that matter most."""
    errors: list[str] = []
    warnings: list[str] = []

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

    view = cfg.get_path("view.profile")
    if view is not None and view not in VIEWPOINTS:
        errors.append(f"view.profile must be one of {VIEWPOINTS}: {view!r}")

    backend = cfg.get_path("roadseg.backend")
    if backend is not None and backend not in ROADSEG_BACKENDS:
        errors.append(f"roadseg.backend must be one of {ROADSEG_BACKENDS}: {backend!r}")

    policy = cfg.get_path("surface.occlusion_policy")
    if policy is not None and policy not in OCCLUSION_POLICIES:
        errors.append(
            f"surface.occlusion_policy must be one of {OCCLUSION_POLICIES}: {policy!r}"
        )

    gate = cfg.get_path("roadseg.gating.mode")
    if gate not in (None, "gate", "mask", "off"):
        errors.append(f"roadseg.gating.mode must be gate|mask|off: {gate!r}")

    tracker = cfg.get_path("inference.tracker")
    if tracker not in (None, "botsort", "bytetrack"):
        errors.append(f"inference.tracker must be botsort|bytetrack: {tracker!r}")

    errors.extend(_validate_reproject(cfg, warnings))

    occluders = cfg.get_path("surface.occluder_classes") or []
    if classes and isinstance(classes, list):
        unknown = [c for c in occluders if c not in classes]
        if unknown:
            errors.append(
                f"surface.occluder_classes not present in model.classes: {unknown}"
            )

    if errors:
        raise ValueError("Invalid config:\n  - " + "\n  - ".join(errors))

    if warnings:
        from .utils.logging import get_logger

        log = get_logger("rdd.config")
        for w in warnings:
            log.warning("config: %s", w)


def _validate_reproject(cfg: Cfg, warnings: list[str]) -> list[str]:
    """Validate 360->flat geometry. Bad FOV/size pairs distort silently."""
    errors: list[str] = []
    rc = cfg.get_path("preprocess.reproject", {}) or {}

    h_fov = rc.get("h_fov_deg")
    v_fov = rc.get("v_fov_deg")
    for name, fov in (("h_fov_deg", h_fov), ("v_fov_deg", v_fov)):
        if fov is not None and not (0 < fov < 180):
            errors.append(f"preprocess.reproject.{name} must be in (0,180): {fov}")

    for name in ("out_width", "out_height"):
        val = rc.get(name)
        if val in (None, "auto"):
            continue
        if not isinstance(val, int) or val <= 0:
            errors.append(
                f"preprocess.reproject.{name} must be a positive int or 'auto': {val!r}"
            )

    # A rectilinear (gnomonic) view has square pixels only when
    #   w/h == tan(h_fov/2) / tan(v_fov/2).
    # Any other pair stretches the image, which quietly changes defect shape.
    ow, oh = rc.get("out_width"), rc.get("out_height")
    if (
        not errors
        and isinstance(ow, int)
        and isinstance(oh, int)
        and h_fov
        and v_fov
        and not rc.get("preserve_aspect", True)
    ):
        want = math.tan(math.radians(h_fov) / 2) / math.tan(math.radians(v_fov) / 2)
        got = ow / oh
        if abs(want - got) / want > 0.02:
            warnings.append(
                f"reproject {ow}x{oh} has aspect {got:.3f} but h_fov/v_fov "
                f"({h_fov}/{v_fov}) imply {want:.3f} — the flat view will be "
                f"stretched. Set preserve_aspect: true to derive out_height."
            )
    return errors
