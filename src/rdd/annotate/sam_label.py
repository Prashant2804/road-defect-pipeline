"""SAM-assisted labeling: prompt on a frame -> mask -> propagate -> YOLO-seg txt.

Uses ultralytics SAM (SAM2/SAM2.1). Two entry points:
  * segment_frame : point/box prompt on a single frame -> masks (interactive use)
  * masks_to_yolo_seg : convert masks to YOLO-seg polygon label lines

Video propagation (one prompt -> mask across many frames) uses ultralytics
SAM2 video predictor when available; a per-frame fallback re-prompts each frame.
This module is a labeling *helper* — never imported at inference time.
"""
from __future__ import annotations

from pathlib import Path

from ..utils.logging import get_logger

log = get_logger("rdd.annotate.sam")


def load_sam(cfg):
    from ultralytics import SAM

    ckpt = cfg.get_path("annotate.sam_model", "sam2.1_b.pt")
    log.info("Loading SAM: %s", ckpt)
    return SAM(ckpt)


def segment_frame(sam, image_path, points=None, labels=None, boxes=None):
    """Return ultralytics Results. points: [[x,y],...]; labels: [1/0,...];
    boxes: [[x1,y1,x2,y2],...]."""
    kwargs = {}
    if points is not None:
        kwargs["points"] = points
        kwargs["labels"] = labels if labels is not None else [1] * len(points)
    if boxes is not None:
        kwargs["bboxes"] = boxes
    return sam(str(image_path), **kwargs)[0]


def masks_to_yolo_seg(result, cls_id: int) -> list[str]:
    """Convert a Results object's masks to YOLO-seg lines:
    '<cls> x1 y1 x2 y2 ...' with normalized polygon coords."""
    lines: list[str] = []
    if result.masks is None:
        return lines
    h, w = result.orig_shape
    for xy in result.masks.xy:  # list of (N,2) pixel polygons
        if len(xy) < 3:
            continue
        norm = []
        for x, y in xy:
            norm.append(f"{x / w:.6f}")
            norm.append(f"{y / h:.6f}")
        lines.append(f"{cls_id} " + " ".join(norm))
    return lines


def write_label(lines: list[str], image_path: Path, labels_dir: Path) -> Path:
    labels_dir.mkdir(parents=True, exist_ok=True)
    out = labels_dir / f"{Path(image_path).stem}.txt"
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out


def propagate_video(sam, video_path, prompt_frame: int, points=None, boxes=None):
    """Propagate a prompt across video frames using SAM2's video mode.

    ultralytics exposes video segmentation via SAM2; API surface varies across
    versions, so this is guarded. On failure the caller should fall back to
    labeling picked frames individually with segment_frame().
    """
    try:
        return sam(
            str(video_path),
            points=points,
            bboxes=boxes,
            stream=True,
        )
    except Exception as e:
        log.warning("SAM video propagation unavailable (%s); label picked frames "
                    "individually instead.", e)
        return None
