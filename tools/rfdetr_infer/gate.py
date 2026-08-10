"""Gate detections to the near-field assessable mask + cross-class NMS."""
from __future__ import annotations

import numpy as np


def box_mask_overlap(xyxy, mask: np.ndarray) -> float:
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    h, w = mask.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    patch = mask[y1:y2, x1:x2]
    return float(patch.mean()) if patch.size else 0.0


def box_bottom_center_in_mask(xyxy, mask: np.ndarray) -> bool:
    x1, y1, x2, y2 = xyxy
    cx = int(round(0.5 * (x1 + x2)))
    cy = int(round(y2))
    h, w = mask.shape
    if not (0 <= cx < w and 0 <= cy < h):
        return False
    return bool(mask[cy, cx])


def box_center_in_mask(xyxy, mask: np.ndarray) -> bool:
    x1, y1, x2, y2 = xyxy
    cx = int(round(0.5 * (x1 + x2)))
    cy = int(round(0.5 * (y1 + y2)))
    h, w = mask.shape
    if not (0 <= cx < w and 0 <= cy < h):
        return False
    return bool(mask[cy, cx])


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def nms_boxes(
    xyxy: np.ndarray,
    class_id: np.ndarray | None,
    confidence: np.ndarray | None,
    iou_thresh: float = 0.5,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Class-agnostic NMS: higher confidence wins when IoU >= threshold."""
    if xyxy is None or len(xyxy) == 0:
        empty = np.zeros((0, 4), dtype=np.float32)
        return empty, class_id, confidence
    if iou_thresh <= 0:
        return xyxy, class_id, confidence

    confs = (
        np.asarray(confidence, dtype=np.float32)
        if confidence is not None
        else np.ones(len(xyxy), dtype=np.float32)
    )
    order = np.argsort(-confs)
    keep: list[int] = []
    suppressed = np.zeros(len(xyxy), dtype=bool)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(int(i))
        for j in order:
            if suppressed[j] or j == i:
                continue
            if _iou(xyxy[i], xyxy[j]) >= iou_thresh:
                suppressed[j] = True

    keep_a = np.asarray(keep, dtype=np.int64)
    cid = class_id[keep_a] if class_id is not None else None
    conf = confs[keep_a]
    return xyxy[keep_a].astype(np.float32), cid, conf


def clip_boxes_to_mask(
    xyxy: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Clip each box to the bounding box of its intersection with the assess mask.

    Keeps drawn boxes visually inside the green near-field wash.
    """
    if xyxy is None or len(xyxy) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    h, w = mask.shape
    out = []
    for box in xyxy:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        patch = mask[y1:y2, x1:x2]
        if not patch.any():
            continue
        ys, xs = np.where(patch)
        # Local coords → image coords
        nx1 = float(x1 + int(xs.min()))
        ny1 = float(y1 + int(ys.min()))
        nx2 = float(x1 + int(xs.max()) + 1)
        ny2 = float(y1 + int(ys.max()) + 1)
        if nx2 - nx1 < 2 or ny2 - ny1 < 2:
            continue
        out.append([nx1, ny1, nx2, ny2])

    if not out:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(out, dtype=np.float32)


def gate_boxes(
    xyxy: np.ndarray,
    class_id: np.ndarray | None,
    confidence: np.ndarray | None,
    near_mask: np.ndarray,
    min_overlap: float = 0.15,
    require_bottom_center: bool = True,
    require_center: bool = False,
    clip_to_mask: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, int]:
    """Return filtered (and optionally clipped) arrays and count of dropped detections."""
    if xyxy is None or len(xyxy) == 0:
        empty = np.zeros((0, 4), dtype=np.float32)
        return empty, class_id, confidence, 0

    keep = []
    for box in xyxy:
        ok = box_mask_overlap(box, near_mask) >= min_overlap
        if ok and require_bottom_center:
            ok = box_bottom_center_in_mask(box, near_mask)
        if ok and require_center:
            ok = box_center_in_mask(box, near_mask)
        keep.append(ok)
    keep_a = np.asarray(keep, dtype=bool)
    n_drop = int((~keep_a).sum())
    if not keep_a.any():
        empty = np.zeros((0, 4), dtype=np.float32)
        return empty, None, None, n_drop

    boxed = xyxy[keep_a].astype(np.float32)
    cid = class_id[keep_a] if class_id is not None else None
    conf = confidence[keep_a] if confidence is not None else None

    if clip_to_mask:
        clipped = []
        keep_clip = []
        for i, box in enumerate(boxed):
            c = clip_boxes_to_mask(box.reshape(1, 4), near_mask)
            if len(c) == 0:
                n_drop += 1
                continue
            clipped.append(c[0])
            keep_clip.append(i)
        if not clipped:
            empty = np.zeros((0, 4), dtype=np.float32)
            return empty, None, None, n_drop
        boxed = np.asarray(clipped, dtype=np.float32)
        idx = np.asarray(keep_clip, dtype=np.int64)
        cid = cid[idx] if cid is not None else None
        conf = conf[idx] if conf is not None else None

    return boxed, cid, conf, n_drop
