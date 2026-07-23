"""Fine-tune the seg model on your labels (warm-started from arch/road-damage)."""
from __future__ import annotations

from pathlib import Path

from ..utils.device import resolve_device
from ..utils.logging import get_logger
from .loader import load_model
from .split import build_split

log = get_logger("rdd.model.train")


def train(cfg, labels_root: str | Path | None = None, fps: float = 30.0):
    mc = cfg.get_path("model", {}) or {}
    tc = mc.get("train", {})

    labels_root = labels_root or cfg.get_path("annotate.labels_dir", "data/labels")
    data_yaml = build_split(labels_root, cfg, fps=fps)

    model = load_model(cfg)  # warm-start / arch resolved inside
    device = resolve_device(cfg.get_path("run.device", "auto"))
    seed = cfg.get_path("run.seed", 0)

    log.info("Fine-tuning on %s (device=%s)", data_yaml, device)
    results = model.train(
        data=str(data_yaml),
        epochs=int(tc.get("epochs", 100)),
        imgsz=int(tc.get("imgsz", 960)),
        batch=int(tc.get("batch", 8)),
        patience=int(tc.get("patience", 20)),
        device=device,
        seed=seed,
        project=str(Path(cfg.get_path("run.output_dir", "out")) / cfg.get_path("run.name", "default")),
        name="train",
        exist_ok=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt" if hasattr(results, "save_dir") else None
    log.info("Training done. Best weights: %s", best)
    return best
