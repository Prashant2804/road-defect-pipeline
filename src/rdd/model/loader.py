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


def model_class_names(model) -> list[str]:
    """The model's own class names, in index order."""
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names)]
    if names:
        return [str(n) for n in names]
    return []


def check_class_alignment(model, cfg) -> bool:
    """Warn when the checkpoint's classes do not match `model.classes`.

    Worth shouting about because the failure is silent and the output *looks*
    plausible. Detections are reported by index, so an off-the-shelf COCO
    checkpoint against a four-class road config yields rows labelled "71" — or,
    worse, indices 0-3 relabelled as `pothole`/`water_logging`/... when the model
    actually found a person and a car. Counts, severities and the whole report
    would be confidently wrong with nothing in the logs to explain why.
    """
    configured = [str(c) for c in (cfg.get_path("model.classes") or [])]
    actual = model_class_names(model)
    if not actual or not configured:
        return True

    if len(actual) != len(configured):
        log.warning(
            "CLASS MISMATCH: the loaded checkpoint has %d classes %s but "
            "model.classes lists %d %s. Detections are mapped by index, so any "
            "output will be mislabelled. This is expected for a stock COCO "
            "checkpoint — fine-tune on your own labels (python run.py train) "
            "before trusting counts.",
            len(actual), actual[:6] + (["..."] if len(actual) > 6 else []),
            len(configured), configured,
        )
        return False

    if [n.lower() for n in actual] != [c.lower() for c in configured]:
        log.warning(
            "CLASS ORDER MISMATCH: checkpoint classes %s differ from "
            "model.classes %s. Same count, so nothing will error — but labels "
            "will be silently swapped. Align the order.", actual, configured,
        )
        return False
    return True


def load_model(cfg, weights: str | None = None):
    """Return an ultralytics YOLO model. `weights` overrides everything (used at
    inference to load a trained .pt)."""
    from ultralytics import YOLO

    mc = cfg.get_path("model", {}) or {}
    size = mc.get("size", "m")
    model = None

    # 1. explicit weights (trained model) win.
    if weights:
        log.info("Loading model from explicit weights: %s", weights)
        model = YOLO(weights)

    # 2. warm-start checkpoint (road-damage pretrain) if given.
    if model is None:
        warm = mc.get("warm_start_weights")
        if warm:
            wp = Path(warm)
            if wp.exists() or str(warm).startswith(("http://", "https://")):
                log.info("Warm-starting from road-damage checkpoint: %s", warm)
                model = YOLO(str(warm))
            else:
                log.warning("warm_start_weights %s not found — falling back to "
                            "arch weights", warm)

    # 3. arch with fallback.
    if model is None:
        primary = _arch_to_weight(mc.get("arch", "yolo11-seg"), size)
        fallback = _arch_to_weight(mc.get("fallback_arch", "yolo11-seg"), size)
        try:
            log.info("Loading architecture weights: %s", primary)
            model = YOLO(primary)
        except Exception as e:
            log.warning("Primary arch %s unavailable (%s); falling back to %s",
                        primary, e, fallback)
            model = YOLO(fallback)

    check_class_alignment(model, cfg)
    return model
