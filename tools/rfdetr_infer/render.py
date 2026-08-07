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
_NEAR_TINT = (60, 180, 60)
_OUTLINE = (0, 255, 120)


def color_for(cls_id: int):
    return _PALETTE[cls_id % len(_PALETTE)]


def _scale_for_frame(h: int) -> dict:
    """Readable overlays on GoPro / 1080p+ frames."""
    return {
        "box_thickness": max(3, h // 360),
        "outline_thickness": max(3, h // 400),
        "font_scale": max(0.7, h / 1080.0 * 0.9),
        "text_thickness": max(2, h // 540),
        "label_pad": max(4, h // 270),
        "hud_scale": max(0.7, h / 1080.0 * 0.85),
        "hud_thickness": max(2, h // 540),
        "hud_line": max(28, h // 40),
    }


def draw_near_field(
    frame: np.ndarray,
    nf: NearField,
    far_alpha: float = 0.35,
    near_alpha: float = 0.10,
) -> np.ndarray:
    out = frame.copy()
    h = out.shape[0]
    sc = _scale_for_frame(h)

    # Light wash on assess (near) so corridor reads continuous with far shade
    if nf.mask is not None and nf.mask.any() and near_alpha > 0:
        overlay = out.copy()
        overlay[nf.mask] = _NEAR_TINT
        out = cv2.addWeighted(overlay, near_alpha, out, 1.0 - near_alpha, 0)

    if nf.far_tint is not None and nf.far_tint.any() and far_alpha > 0:
        overlay = out.copy()
        overlay[nf.far_tint] = _FAR_TINT
        out = cv2.addWeighted(overlay, far_alpha, out, 1.0 - far_alpha, 0)

    cv2.polylines(
        out,
        [nf.outline],
        isClosed=True,
        color=_OUTLINE,
        thickness=sc["outline_thickness"],
    )
    x0, y0 = int(nf.outline[:, 0].min()), int(nf.outline[:, 1].min())
    label = "near-field assess"
    fs = sc["font_scale"] * 0.85
    th = sc["text_thickness"]
    cv2.putText(
        out,
        label,
        (max(8, x0), max(int(28 * fs), y0 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        fs,
        (0, 0, 0),
        th + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        label,
        (max(8, x0), max(int(28 * fs), y0 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        fs,
        _OUTLINE,
        th,
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
    h = out.shape[0]
    sc = _scale_for_frame(h)
    bt = sc["box_thickness"]
    fs = sc["font_scale"]
    tt = sc["text_thickness"]
    pad = sc["label_pad"]

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cid = int(class_ids[i]) if class_ids is not None else 0
        conf = float(confs[i]) if confs is not None else 0.0
        name = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
        color = color_for(cid)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, bt)
        label = f"{name} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, tt)
        top = max(0, y1 - th - pad * 2 - baseline)
        cv2.rectangle(
            out,
            (x1, top),
            (x1 + tw + pad * 2, y1),
            color,
            -1,
        )
        cv2.putText(
            out,
            label,
            (x1 + pad, y1 - pad - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            fs,
            (0, 0, 0),
            tt,
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
    h = out.shape[0]
    sc = _scale_for_frame(h)
    fs = sc["hud_scale"]
    tt = sc["hud_thickness"]
    line_h = sc["hud_line"]

    lines = [
        f"t={t_s:6.1f}s  assess<= {z_far_m:.1f}m",
        f"GPS={'yes' if gps_ok else 'no'}"
        + (f"  chainage={chainage_m:.1f}m" if chainage_m is not None else ""),
    ]
    top = ", ".join(f"{k}:{v}" for k, v in counts.items() if v) or "no defects yet"
    lines.append(top[:110])
    y = line_h
    for line in lines:
        cv2.putText(
            out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), tt + 2, cv2.LINE_AA
        )
        cv2.putText(
            out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), tt, cv2.LINE_AA
        )
        y += line_h
    return out
