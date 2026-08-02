"""Fine-tune the seg model on your labels (warm-started from arch/road-damage)."""
from __future__ import annotations

from pathlib import Path

from ..utils.device import resolve_device
from ..utils.logging import get_logger
from .loader import load_model
from .split import build_split

log = get_logger("rdd.model.train")


def _infer_task(data_yaml: Path) -> str:
    """Read the labels and see whether they are boxes or polygons.

    The architecture and the annotations must agree, and only the annotations know
    which they are. Trusting `model.arch` here is how you end up feeding four-value
    box rows to a segmentation head.
    """
    import yaml

    try:
        doc = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8")) or {}
    except Exception:
        return "segment"
    root = Path(doc.get("path") or Path(data_yaml).parent)
    train_rel = str(doc.get("train", "train/images"))
    lbl_dir = (root / train_rel).parent / "labels"
    if not lbl_dir.is_dir():
        return "segment"
    for f in sorted(lbl_dir.glob("*.txt"))[:200]:
        for line in f.read_text(encoding="utf-8").splitlines():
            n = len(line.split())
            if n > 5:
                return "segment"
            if n == 5:
                return "detect"
    return "segment"


def train(cfg, labels_root: str | Path | None = None, fps: float = 30.0,
          data_yaml: str | Path | None = None, task: str | None = None):
    """Fine-tune on labels.

    `data_yaml` trains on an already-split dataset (a Roboflow export, say) instead of
    building a split here. Worth preferring when the export is already split *and*
    that split has been checked for leakage, because `build_split` groups frames by
    index parsed from the filename — meaningless for exports that rename files to
    content hashes, which would silently degrade to an arbitrary split.
    """
    mc = cfg.get_path("model", {}) or {}
    tc = mc.get("train", {})

    if data_yaml:
        data_yaml = Path(data_yaml)
        log.info("Using prepared dataset %s (skipping segment split)", data_yaml)
    else:
        labels_root = labels_root or cfg.get_path("annotate.labels_dir", "data/labels")
        data_yaml = build_split(labels_root, cfg, fps=fps)

    task = task or _infer_task(data_yaml)
    log.info("Training as task=%s (from the label geometry)", task)

    model = load_model(cfg, task=task)  # warm-start / arch resolved inside
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
        # Absolute: ultralytics prepends its own runs/<task>/ to a RELATIVE project,
        # so "out/<name>" lands in runs/detect/out/<name> and the artifacts go
        # somewhere nobody is looking.
        project=str((Path(cfg.get_path("run.output_dir", "out"))
                     / cfg.get_path("run.name", "default")).resolve()),
        name="train",
        exist_ok=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt" if hasattr(results, "save_dir") else None
    log.info("Training done. Best weights: %s", best)
    return best
