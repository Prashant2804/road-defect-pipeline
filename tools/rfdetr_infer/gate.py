"""Gate RF-DETR boxes to the near-field assessable mask."""
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


def gate_boxes(
    xyxy: np.ndarray,
    class_id: np.ndarray | None,
    confidence: np.ndarray | None,
    near_mask: np.ndarray,
    min_overlap: float = 0.25,
    require_bottom_center: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, int]:
    """Return filtered arrays and count of dropped detections."""
    if xyxy is None or len(xyxy) == 0:
        empty = np.zeros((0, 4), dtype=np.float32)
        return empty, class_id, confidence, 0

    keep = []
    for i, box in enumerate(xyxy):
        ok = box_mask_overlap(box, near_mask) >= min_overlap
        if ok and require_bottom_center:
            ok = box_bottom_center_in_mask(box, near_mask)
        keep.append(ok)
    keep_a = np.asarray(keep, dtype=bool)
    n_drop = int((~keep_a).sum())
    if not keep_a.any():
        empty = np.zeros((0, 4), dtype=np.float32)
        return empty, None, None, n_drop
    cid = class_id[keep_a] if class_id is not None else None
    conf = confidence[keep_a] if confidence is not None else None
    return xyxy[keep_a], cid, conf, n_drop
