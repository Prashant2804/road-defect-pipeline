"""Segment-aware train/val/test split -> Ultralytics dataset.yaml.

CRITICAL: adjacent video frames are near-identical. A random frame split leaks
almost-duplicate images across train/val/test and massively inflates metrics.
We split by *segment* (a contiguous run of frames) or by *time range*, so an
entire stretch of road lands wholly in one split.

Expects a labeled dataset laid out as:
    <labels_root>/images/*.jpg   (or .png)
    <labels_root>/labels/*.txt   (YOLO-seg polygons, same stem)

Frame index / timestamp is parsed from the filename stem `frame_0001234`.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import yaml

from ..utils.logging import get_logger

log = get_logger("rdd.model.split")

_FRAME_RE = re.compile(r"(\d+)")


def _frame_index(stem: str) -> int:
    m = _FRAME_RE.search(stem)
    return int(m.group(1)) if m else 0


def _assign_segments(indices: list[int], fps: float, gap_s: float) -> dict[int, int]:
    """Group sorted frame indices into segments; a gap > gap_s starts a new one."""
    gap_frames = max(1, int(gap_s * fps))
    seg_of: dict[int, int] = {}
    seg = 0
    prev: int | None = None
    for idx in sorted(indices):
        if prev is not None and idx - prev > gap_frames:
            seg += 1
        seg_of[idx] = seg
        prev = idx
    return seg_of


def build_split(labels_root: str | Path, cfg, fps: float = 30.0) -> Path:
    labels_root = Path(labels_root)
    img_dir = labels_root / "images"
    lbl_dir = labels_root / "labels"
    if not img_dir.exists():
        raise FileNotFoundError(f"Expected images at {img_dir}")

    split_cfg = cfg.get_path("model.train.split", {}) or {}
    mode = split_cfg.get("mode", "segment")
    if mode == "random":
        raise ValueError("split.mode 'random' is forbidden (frame leakage).")
    ratios = split_cfg.get("ratios", {"train": 0.7, "val": 0.15, "test": 0.15})

    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"}])
    if not images:
        raise FileNotFoundError(f"No images found in {img_dir}")

    idx_to_img = {_frame_index(p.stem): p for p in images}
    indices = sorted(idx_to_img)
    seg_of = _assign_segments(indices, fps, split_cfg.get("segment_gap_s", 5.0))
    segments = sorted(set(seg_of.values()))
    log.info("Found %d images across %d segments", len(images), len(segments))

    # Assign whole segments to splits in order until ratio budget is filled.
    n = len(images)
    budget = {k: int(v * n) for k, v in ratios.items()}
    counts = {k: 0 for k in ratios}
    seg_split: dict[int, str] = {}
    order = ["train", "val", "test"]
    for seg in segments:
        size = sum(1 for i in indices if seg_of[i] == seg)
        target = min(
            (k for k in order if counts[k] + size <= budget[k] or k == "test"),
            key=lambda k: counts[k] / max(budget[k], 1),
        )
        seg_split[seg] = target
        counts[target] += size

    out_root = labels_root / "_split"
    if out_root.exists():
        shutil.rmtree(out_root)
    for split in order:
        (out_root / split / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split / "labels").mkdir(parents=True, exist_ok=True)

    for idx in indices:
        img = idx_to_img[idx]
        split = seg_split[seg_of[idx]]
        shutil.copy2(img, out_root / split / "images" / img.name)
        lbl = lbl_dir / f"{img.stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, out_root / split / "labels" / lbl.name)

    classes = cfg.get_path("model.classes")
    dataset = {
        "path": str(out_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {i: c for i, c in enumerate(classes)},
    }
    data_yaml = Path(cfg.get_path("model.train.data_yaml", "data/dataset.yaml"))
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    with data_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dataset, f, sort_keys=False)
    log.info("dataset.yaml -> %s  (train=%d val=%d test=%d)",
             data_yaml, counts["train"], counts["val"], counts["test"])
    return data_yaml
