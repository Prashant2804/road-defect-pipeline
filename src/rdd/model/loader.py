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

    has_map = bool(cfg.get_path("model.class_map"))
    if len(actual) != len(configured):
        if has_map:
            # A count mismatch is the NORMAL case for a mapped third-party checkpoint,
            # and shouting "any output will be mislabelled" here would be wrong — the
            # map is what makes it right.
            log.info(
                "Checkpoint has %d classes %s; model.classes lists %d. Resolved by "
                "name through model.class_map.", len(actual), actual[:6], len(configured))
            return True
        log.warning(
            "CLASS MISMATCH: the loaded checkpoint has %d classes %s but "
            "model.classes lists %d %s. Detections are mapped BY INDEX, so labels "
            "will be wrong. Set model.class_map to translate by name, or fine-tune "
            "on your own labels.",
            len(actual), actual[:6] + (["..."] if len(actual) > 6 else []),
            len(configured), configured,
        )
        return False

    if [n.lower() for n in actual] != [c.lower() for c in configured]:
        if has_map:
            return True
        log.warning(
            "CLASS ORDER MISMATCH: checkpoint classes %s differ from "
            "model.classes %s. Same count, so nothing will error — but labels "
            "will be silently swapped. Align the order.", actual, configured,
        )
        return False
    return True


def build_class_resolver(model, cfg):
    """Map the checkpoint's own class ids onto the pipeline taxonomy.

    Without this, a public checkpoint is unusable. Detections carry an integer class
    id, and the pipeline was resolving it positionally against `model.classes` — so an
    RDD2022 model (D00, D10, D20, D40, Repair) against this 9-class config would label
    D00 "pothole" simply because both sit at index 0. Nothing errors; every row in the
    report is just wrong.

    With `model.class_map` set, ids are resolved through the checkpoint's *own* names
    and translated by name. Mapping a name to null drops that class — useful for
    categories a public model predicts that this pipeline does not report, such as
    RDD2022's "Repair", which is a past intervention rather than a defect.

    Returns a callable id -> class name, or None to discard the detection.
    """
    configured = [str(c) for c in (cfg.get_path("model.classes") or [])]
    raw_map = cfg.get_path("model.class_map") or {}
    class_map = {str(k): (None if v in (None, "", "null") else str(v))
                 for k, v in dict(raw_map).items()}
    model_names = model_class_names(model) if model is not None else []

    if not class_map or not model_names:
        def by_index(cid: int):
            if 0 <= cid < len(configured):
                return configured[cid]
            return f"UNMAPPED_CLASS_{cid}"
        return by_index

    def _brief(items, limit: int = 8) -> str:
        """Log lists compactly — an 80-class COCO checkpoint makes a log line unreadable."""
        items = list(items)
        head = ", ".join(str(i) for i in items[:limit])
        return head if len(items) <= limit else f"{head}, ... (+{len(items) - limit} more)"

    unknown = [n for n in model_names if n not in class_map]
    if unknown:
        log.warning(
            "model.class_map has no entry for %d checkpoint class(es): %s — those "
            "detections keep their raw name. Add them to the map, or map them to null "
            "to drop them.", len(unknown), _brief(unknown))
    dropped = sorted(k for k, v in class_map.items() if v is None)
    if dropped:
        log.info("Dropping checkpoint classes not reported here: %s", _brief(dropped))
    mapped = {n: class_map[n] for n in model_names if n in class_map and class_map[n]}
    log.info("Class map: %s", _brief(f"{k}->{v}" for k, v in mapped.items()))

    def by_name(cid: int):
        if not (0 <= cid < len(model_names)):
            return f"UNMAPPED_CLASS_{cid}"
        name = model_names[cid]
        return class_map.get(name, name)
    return by_name


def load_model(cfg, weights: str | None = None, task: str | None = None):
    """Return an ultralytics YOLO model. `weights` overrides everything (used at
    inference to load a trained .pt).

    `task` forces detect or segment regardless of what `model.arch` says. Needed
    because the architecture and the LABELS have to agree: handing box annotations to
    a `-seg` model fails, and the dataset decides which you have, not the config.
    """
    from ultralytics import YOLO

    mc = cfg.get_path("model", {}) or {}
    size = mc.get("size", "m")
    model = None

    def _for_task(arch: str) -> str:
        """Adjust an arch string to the requested task ('yolo11-seg' <-> 'yolo11')."""
        if task == "detect":
            return arch.replace("-seg", "")
        if task == "segment" and "-seg" not in arch and not arch.endswith(".pt"):
            return f"{arch}-seg"
        return arch

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
        primary = _arch_to_weight(_for_task(mc.get("arch", "yolo11-seg")), size)
        fallback = _arch_to_weight(_for_task(mc.get("fallback_arch", "yolo11-seg")), size)
        try:
            log.info("Loading architecture weights: %s", primary)
            model = YOLO(primary)
        except Exception as e:
            log.warning("Primary arch %s unavailable (%s); falling back to %s",
                        primary, e, fallback)
            model = YOLO(fallback)

    check_class_alignment(model, cfg)
    return model
