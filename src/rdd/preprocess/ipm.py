"""Inverse Perspective Mapping (bird's-eye view) of the rectified road frame.

Optional. IPM makes defect scale roughly distance-invariant, which helps size
estimation. Config gives src (trapezoid on the road plane) and dst (rectangle)
as fractions of image size; we compute a homography and warp.
"""
from __future__ import annotations

from ..utils.logging import get_logger

log = get_logger("rdd.preprocess.ipm")


def build_homography(cfg, in_w: int, in_h: int):
    """Return (H, out_w, out_h) or None if IPM disabled."""
    import numpy as np
    import cv2

    ic = cfg.get_path("preprocess.ipm", {}) or {}
    if not ic.get("enabled", False):
        return None

    out_w = int(ic.get("out_width", 640))
    out_h = int(ic.get("out_height", 960))
    src = np.array(
        [[x * in_w, y * in_h] for x, y in ic["src_points"]], dtype=np.float32
    )
    dst = np.array(
        [[x * out_w, y * out_h] for x, y in ic["dst_points"]], dtype=np.float32
    )
    H = cv2.getPerspectiveTransform(src, dst)
    log.info("IPM homography built: out %dx%d", out_w, out_h)
    return H, out_w, out_h


def warp(frame, H, out_w: int, out_h: int):
    import cv2

    return cv2.warpPerspective(frame, H, (out_w, out_h))
