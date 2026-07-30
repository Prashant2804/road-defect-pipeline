"""Frame annotation: road surface, surface condition, defect masks, track IDs, HUD.

The annotated video is the artifact a human actually uses to sanity-check a run,
so it shows the pipeline's *reasoning*, not just its conclusions: the road
boundary it settled on, the areas it considers unassessable, and which defects it
abstained from scoring. A count in a CSV cannot be audited; a video where the
road outline is visibly wrong can be, immediately.

Occluded defects are drawn hatched rather than filled, so "found something,
cannot measure it" is visually distinct from "found and measured it".
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

_ROAD_TINT = (90, 200, 90)       # soft green wash over the drivable surface
_WATER_TINT = (230, 170, 60)     # blue
_MUD_TINT = (40, 90, 130)        # brown
_ROAD_EDGE = (0, 255, 120)


def color_for(cls_id: int):
    return _PALETTE[cls_id % len(_PALETTE)]


def _hatch(shape, spacing: int = 7):
    """Diagonal stripe mask — marks regions we cannot measure."""
    import numpy as np

    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    return ((xx + yy) % spacing) < 2


def draw_road(frame, road, cfg):
    """Wash the road surface and outline it."""
    import cv2
    import numpy as np

    rc = cfg.get_path("inference.render", {}) or {}
    alpha = float(rc.get("road_alpha", 0.18))
    if road is None or alpha <= 0 or road.is_empty():
        return frame

    overlay = frame.copy()
    overlay[road.mask] = _ROAD_TINT
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    if rc.get("draw_road_outline", True):
        contours, _ = cv2.findContours(road.mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, _ROAD_EDGE, 2)
    return frame


def draw_surface(frame, surface, cfg):
    """Tint water and mud, hatched to read as 'cannot inspect underneath'."""
    import cv2

    rc = cfg.get_path("inference.render", {}) or {}
    alpha = float(rc.get("surface_alpha", 0.35))
    if surface is None or alpha <= 0:
        return frame

    hatch = None
    overlay = frame.copy()
    touched = False
    for mask, tint in ((surface.water, _WATER_TINT), (surface.mud, _MUD_TINT)):
        if mask is None or not mask.any():
            continue
        if hatch is None:
            hatch = _hatch(frame.shape)
        overlay[mask & hatch] = tint
        touched = True
    if not touched:
        return frame
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def draw_frame(frame, detections, class_names, cfg, road=None, surface=None):
    """detections: {track_id, cls_id, conf, mask, bbox, occluded, area_m2, ...}."""
    import cv2

    rc = cfg.get_path("inference.render", {}) or {}
    alpha = float(rc.get("alpha", 0.45))
    draw_masks = rc.get("draw_masks", True)
    draw_boxes = rc.get("draw_boxes", False)
    draw_id = rc.get("draw_track_id", True)

    if rc.get("draw_road", True):
        frame = draw_road(frame, road, cfg)
    if rc.get("draw_surface", True):
        frame = draw_surface(frame, surface, cfg)

    hatch = None
    overlay = frame.copy()
    for det in detections:
        col = color_for(det["cls_id"])
        mask = det.get("mask")
        if draw_masks and mask is not None:
            if det.get("occluded"):
                if hatch is None:
                    hatch = _hatch(frame.shape)
                overlay[mask & hatch] = col
            else:
                overlay[mask] = col
        if draw_boxes and det.get("bbox"):
            x1, y1, x2, y2 = map(int, det["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Labels last, unblended, so they stay legible over any overlay.
    for det in detections:
        col = color_for(det["cls_id"])
        cid = det["cls_id"]
        name = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
        x1, y1, _, _ = map(int, det.get("bbox", (0, 0, 0, 0)))
        parts = [name]
        if draw_id and det.get("track_id") is not None:
            parts.append(f"#{det['track_id']}")
        parts.append(f"{det.get('conf', 0):.2f}")
        if det.get("area_m2") is not None:
            parts.append(f"{det['area_m2']:.2f}m2")
        if det.get("occluded"):
            parts.append("OCCLUDED")
        label = " ".join(parts)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(th + 6, y1)
        cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 4, ty), col, -1)
        cv2.putText(frame, label, (x1 + 2, ty - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_hud(frame, running_total: int, per_class: dict[str, int], frame_idx: int,
             occluded_frac: float = 0.0, road_conf: float = 0.0):
    """Top-left HUD: running unique count, per-class breakdown, assessability."""
    import cv2

    lines = [f"UNIQUE DEFECTS: {running_total}"]
    lines += [f"  {k}: {v}" for k, v in per_class.items() if v]
    if occluded_frac > 0.005:
        lines.append(f"UNASSESSABLE: {occluded_frac:.0%} of road")
    lines.append(f"road mask conf: {road_conf:.2f}")

    x, y, pad, lh = 12, 26, 6, 24
    w = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
            for l in lines) + 2 * pad
    h = lh * len(lines) + pad
    cv2.rectangle(frame, (x - pad, y - 20), (x - pad + w, y - 20 + h), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        colour = (0, 255, 180)
        if line.startswith("UNASSESSABLE"):
            colour = (60, 200, 255)      # amber: a caveat, not a defect count
        elif line.startswith("road mask"):
            colour = (200, 200, 200)
        cv2.putText(frame, line, (x, y + i * lh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
    return frame


def draw_quality_banner(frame, reasons):
    """Mark a frame that was too poor to analyse.

    Kept in the output rather than dropped: the annotated video should be a
    faithful record of the survey, including where it could not see.
    """
    import cv2

    h, w = frame.shape[:2]
    text = "FRAME NOT ANALYSED: " + (", ".join(reasons) if reasons else "low quality")
    bar = max(28, h // 18)
    cv2.rectangle(frame, (0, 0), (w, bar), (0, 0, 0), -1)
    scale = max(0.45, min(0.8, w / 1600.0))
    cv2.putText(frame, text[:120], (10, int(bar * 0.7)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (80, 160, 255), 2, cv2.LINE_AA)
    return frame
