"""Frame annotation: masks + class + track ID + running unique-count HUD.

Uses supervision annotators when available; falls back to plain OpenCV drawing
so the pipeline still renders without supervision installed.
"""
from __future__ import annotations

from ..utils.logging import get_logger

log = get_logger("rdd.inference.render")

# Distinct BGR colors per class index (cycled).
_PALETTE = [
    (60, 76, 231),   # red-ish  -> pothole
    (219, 152, 52),  # blue     -> water_logging
    (39, 174, 96),   # green    -> rut_erosion
    (0, 196, 255),   # amber    -> crack
    (156, 89, 182),
    (241, 196, 15),
]


def color_for(cls_id: int):
    return _PALETTE[cls_id % len(_PALETTE)]


def draw_frame(frame, detections, class_names, cfg):
    """detections: list of dicts {track_id, cls_id, conf, mask(np bool HxW|None), bbox}."""
    import cv2
    import numpy as np

    rc = cfg.get_path("inference.render", {}) or {}
    alpha = float(rc.get("alpha", 0.45))
    draw_masks = rc.get("draw_masks", True)
    draw_boxes = rc.get("draw_boxes", False)
    draw_id = rc.get("draw_track_id", True)

    overlay = frame.copy()
    for det in detections:
        cls_id = det["cls_id"]
        col = color_for(cls_id)
        mask = det.get("mask")
        if draw_masks and mask is not None:
            overlay[mask] = col
        if draw_boxes and det.get("bbox"):
            x1, y1, x2, y2 = map(int, det["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)

    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # labels drawn on top (no blending) for legibility
    for det in detections:
        col = color_for(det["cls_id"])
        name = class_names[det["cls_id"]] if det["cls_id"] < len(class_names) else str(det["cls_id"])
        x1, y1, x2, y2 = map(int, det.get("bbox", (0, 0, 0, 0)))
        parts = [name]
        if draw_id and det.get("track_id") is not None:
            parts.append(f"#{det['track_id']}")
        parts.append(f"{det.get('conf', 0):.2f}")
        label = " ".join(parts)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), col, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_hud(frame, running_total: int, per_class: dict[str, int], frame_idx: int):
    """Top-left HUD: running unique count total + per-class breakdown."""
    import cv2

    lines = [f"UNIQUE DEFECTS: {running_total}"]
    lines += [f"  {k}: {v}" for k, v in per_class.items() if v]
    x, y = 12, 26
    pad = 6
    w = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0] for l in lines) + 2 * pad
    h = 24 * len(lines) + pad
    cv2.rectangle(frame, (x - pad, y - 20), (x - pad + w, y - 20 + h), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 2, cv2.LINE_AA)
    return frame
