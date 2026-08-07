"""Near-field assessable mask: road trapezoid ∩ optional distance band."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import InferConfig


@dataclass
class NearField:
    mask: np.ndarray  # bool HxW — assessable near field on road
    prior: np.ndarray  # bool HxW — geometric near trapezoid
    outline: np.ndarray  # Nx2 int32 polygon for drawing (near)
    road: np.ndarray  # bool HxW — road after optional classical grow
    far_tint: np.ndarray  # bool HxW — pixels to wash green (out of scope)


def road_trapezoid(
    h: int,
    w: int,
    bottom_y: float,
    top_y: float,
    bottom_half_w: float,
    top_half_w: float,
    center_x: float,
) -> tuple[np.ndarray, np.ndarray]:
    cx = center_x * w
    by, ty = bottom_y * h, top_y * h
    bhw, thw = bottom_half_w * w, top_half_w * w
    # Perspective: narrower at the farther (smaller) y
    pts = np.array(
        [
            [int(cx - bhw), int(by)],
            [int(cx + bhw), int(by)],
            [int(cx + thw), int(ty)],
            [int(cx - thw), int(ty)],
        ],
        dtype=np.int32,
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool), pts


def classical_grow_road(
    frame_bgr: np.ndarray,
    prior_bool: np.ndarray,
    work_width: int = 480,
    distance_tau: float = 2.5,
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    scale = work_width / float(w)
    small = cv2.resize(
        frame_bgr, (work_width, int(round(h * scale))), interpolation=cv2.INTER_AREA
    )
    sh, sw = small.shape[:2]
    prior_s = cv2.resize(
        prior_bool.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    er = max(1, int(0.05 * sw))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (er * 2 + 1, er * 2 + 1))
    seed = cv2.erode(prior_s.astype(np.uint8), kernel, iterations=1).astype(bool)
    if not seed.any():
        seed = prior_s
    search = cv2.dilate(prior_s.astype(np.uint8), kernel, iterations=1).astype(bool)

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tex = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    tex = cv2.GaussianBlur(np.abs(tex), (7, 7), 0)
    feats = np.dstack([lab[..., 0], lab[..., 1], lab[..., 2], tex])
    med = np.median(feats[seed], axis=0)
    mad = np.median(np.abs(feats[seed] - med), axis=0) + 1e-3
    z = np.abs(feats - med) / mad
    dist = (1.0 * z[..., 0] + 1.0 * z[..., 1] + 1.0 * z[..., 2] + 1.3 * z[..., 3]) / 4.3
    cand = (dist < distance_tau) & search
    filled = cand.astype(np.uint8)
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hole = np.zeros_like(filled)
    cv2.drawContours(hole, contours, -1, 1, thickness=-1)
    road_s = hole.astype(bool)
    if road_s.mean() < 0.02 or road_s.mean() > 0.95:
        road_s = prior_s
    return cv2.resize(
        road_s.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
    ).astype(bool)


def _approx_z_map_from_trapezoid(h: int, w: int, cfg: InferConfig) -> np.ndarray:
    """Linear depth proxy: frame bottom ≈ z_near, road_top_y ≈ z_far, above → farther."""
    yy = np.arange(h, dtype=np.float32)[:, None]
    by = cfg.road_bottom_y * h
    ty = cfg.road_top_y * h
    span = max(by - ty, 1.0)
    # Allow t > 1 above the near trapezoid so far band is z > z_far
    t = (by - yy) / span
    z = cfg.z_near_m + t * (cfg.z_far_m - cfg.z_near_m)
    return np.broadcast_to(z, (h, w)).copy()


def _metric_z_map(h: int, w: int, cfg: InferConfig) -> np.ndarray | None:
    """Optional pinhole ground-plane depth if height/pitch/vfov are set."""
    if cfg.camera_height_m is None or cfg.vfov_deg is None:
        return None
    height = float(cfg.camera_height_m)
    pitch = float(cfg.camera_pitch_deg or 0.0)
    vfov = np.deg2rad(float(cfg.vfov_deg))
    pitch_r = np.deg2rad(pitch)
    ys = (np.arange(h, dtype=np.float32) + 0.5) / h
    alpha = (ys - 0.5) * vfov + pitch_r
    with np.errstate(divide="ignore", invalid="ignore"):
        z = height / np.tan(np.clip(alpha, 1e-3, np.pi / 2 - 1e-3))
    z = np.where(alpha <= 0, np.nan, z)
    return np.broadcast_to(z[:, None], (h, w)).copy()


def build_near_field(frame_bgr: np.ndarray, cfg: InferConfig) -> NearField:
    h, w = frame_bgr.shape[:2]
    prior, pts = road_trapezoid(
        h,
        w,
        cfg.road_bottom_y,
        cfg.road_top_y,
        cfg.road_bottom_half_w,
        cfg.road_top_half_w,
        cfg.road_center_x,
    )
    # Extended trapezoid toward the horizon for far-field tint (inspiration green wash).
    # Same half-widths as near so the corridor reads as one continuous road ribbon.
    far_top = min(cfg.road_top_y, max(0.05, cfg.road_top_y - 0.25))
    far_prior, _ = road_trapezoid(
        h,
        w,
        cfg.road_bottom_y,
        far_top,
        cfg.road_bottom_half_w,
        cfg.road_top_half_w,
        cfg.road_center_x,
    )

    grow_seed = prior | far_prior
    if cfg.use_classical_road:
        road = classical_grow_road(frame_bgr, grow_seed)
    else:
        road = grow_seed

    z_map = _metric_z_map(h, w, cfg)
    if z_map is None:
        z_map = _approx_z_map_from_trapezoid(h, w, cfg)

    with np.errstate(invalid="ignore"):
        in_band = (z_map >= cfg.z_near_m) & (z_map <= cfg.z_far_m)
        beyond = z_map > cfg.z_far_m
    in_band = np.nan_to_num(in_band, nan=False).astype(bool)
    beyond = np.nan_to_num(beyond, nan=False).astype(bool)

    # Assessable = geometric near trapezoid ∩ distance band.
    # Do NOT require classical "road" here — cracked/rutted asphalt often fails
    # the color/texture grow and was dropping 2nd-lane defects outside the wash.
    mask = prior & in_band
    # Optional far wash (usually disabled; green belongs in the assess polygon)
    far_tint = far_prior & beyond & ~mask
    if cfg.use_classical_road:
        far_tint = far_tint & road

    return NearField(
        mask=mask,
        prior=prior,
        outline=pts,
        road=road,
        far_tint=far_tint,
    )
