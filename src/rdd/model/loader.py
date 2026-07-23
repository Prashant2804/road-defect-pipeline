"""Load a YOLO seg model with config-driven arch + automatic fallback.

Resolution order for the *architecture* weights (when no trained/warm-start
checkpoint is given):
  1. cfg.model.arch  (e.g. yolo26-seg -> yolo26m-seg.pt)
  2. cfg.model.fallback_arch  (e.g. yolo11-seg -> yolo11m-seg.pt)

YOLO26 may not exist in the installed ultralytics build yet; if instantiating it
raises, we log a warning and drop to the fallback. Warm-start weights (a road-
damage checkpoint) take precedence over the bare arch when provided.
"""
from __future__ import annotations

from pathlib import Path

from ..utils.logging import get_logger

log = get_logger("rdd.model.loader")


def _arch_to_weight(arch: str, size: str) -> str:
    """'yolo26-seg' + 'm' -> 'yolo26m-seg.pt'. Passthrough if already a filename."""
    if arch.endswith(".pt"):
        return arch
    if "-seg" in arch:
        base, _, _ = arch.partition("-seg")
        return f"{base}{size}-seg.pt"
    return f"{arch}{size}.pt"


def load_model(cfg, weights: str | None = None):
    """Return an ultralytics YOLO model. `weights` overrides everything (used at
    inference to load a trained .pt)."""
    from ultralytics import YOLO

    mc = cfg.get_path("model", {}) or {}
    size = mc.get("size", "m")

    # 1. explicit weights (trained model) win.
    if weights:
        log.info("Loading model from explicit weights: %s", weights)
        return YOLO(weights)

    # 2. warm-start checkpoint (road-damage pretrain) if given.
    warm = mc.get("warm_start_weights")
    if warm:
        wp = Path(warm)
        if wp.exists() or str(warm).startswith(("http://", "https://")):
            log.info("Warm-starting from road-damage checkpoint: %s", warm)
            return YOLO(str(warm))
        log.warning("warm_start_weights %s not found — falling back to arch weights", warm)

    # 3. arch with fallback.
    primary = _arch_to_weight(mc.get("arch", "yolo11-seg"), size)
    fallback = _arch_to_weight(mc.get("fallback_arch", "yolo11-seg"), size)
    try:
        log.info("Loading architecture weights: %s", primary)
        return YOLO(primary)
    except Exception as e:
        log.warning("Primary arch %s unavailable (%s); falling back to %s",
                    primary, e, fallback)
        return YOLO(fallback)
