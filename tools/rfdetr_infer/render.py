"""Draw near-field overlay, far-field tint, boxes, and HUD."""
from __future__ import annotations

import cv2
import numpy as np

from .near_field import NearField

# BGR per class index (cycled)
_PALETTE = [
    (60, 76, 231),
    (219, 152, 52),
    (39, 174, 96),
    (0, 196, 255),
    (156, 89, 182),
    (241, 196, 15),
]
_FAR_TINT = (90, 200, 90)
_OUTLINE = (0, 255, 120)


def color_for(cls_id: int):
    return _PALETTE[cls_id % len(_PALETTE)]


def draw_near_field(frame: np.ndarray, nf: NearField, far_alpha: float = 0.35) -> np.ndarray:
    out = frame.copy()
    if nf.far_tint is not None and nf.far_tint.any() and far_alpha > 0:
        overlay = out.copy()
        overlay[nf.far_tint] = _FAR_TINT
        out = cv2.addWeighted(overlay, far_alpha, out, 1.0 - far_alpha, 0)
    cv2.polylines(out, [nf.outline], isClosed=True, color=_OUTLINE, thickness=2)
    x0, y0 = int(nf.outline[:, 0].min()), int(nf.outline[:, 1].min())
    cv2.putText(
        out,
        "near-field assess",
        (max(8, x0), max(24, y0 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        _OUTLINE,
        2,
        cv2.LINE_AA,
    )
    return out


def draw_boxes(
    frame: np.ndarray,
    boxes,
    class_ids,
    confs,
    class_names: list[str],
) -> np.ndarray:
    out = frame
    if boxes is None or len(boxes) == 0:
        return out
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cid = int(class_ids[i]) if class_ids is not None else 0
        conf = float(confs[i]) if confs is not None else 0.0
        name = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
        color = color_for(cid)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out,
            label,
            (x1 + 2, max(th + 2, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return out


def draw_hud(
    frame: np.ndarray,
    *,
    counts: dict[str, int],
    chainage_m: float | None,
    t_s: float,
    z_far_m: float,
    gps_ok: bool,
) -> np.ndarray:
    out = frame
    lines = [
        f"t={t_s:6.1f}s  assess<= {z_far_m:.1f}m",
        f"GPS={'yes' if gps_ok else 'no'}"
        + (f"  chainage={chainage_m:.1f}m" if chainage_m is not None else ""),
    ]
    top = ", ".join(f"{k}:{v}" for k, v in counts.items() if v) or "no defects yet"
    lines.append(top[:90])
    y = 22
    for line in lines:
        cv2.putText(
            out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
        )
        y += 22
    return out
