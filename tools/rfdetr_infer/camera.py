"""Dashcam geometry for defect area: CameraModel, undistort, GPS speed GSD check.

Area in m² needs a measured camera (height, pitch, HFOV). Vehicle speed does not
give area; it only checks whether the longitudinal GSD implied by GPS+flow agrees
with the camera model.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rdd.geometry.calibration import CameraModel, Extrinsics, Intrinsics

# Approximate 16:9 horizontal FOV. Marketing "170°" is usually diagonal fisheye.
# Prefer Linear / Narrow mode, then measure. These are starting points only.
GOPRO_HFOV_16_9 = {
    "linear": 86.0,
    "wide": 118.0,
    "narrow": 45.0,
}

POTHOLE_MEDIUM_M2 = 0.10
POTHOLE_HIGH_M2 = 0.50


def pothole_irc_band(area_m2: float | None) -> str | None:
    if area_m2 is None:
        return None
    if area_m2 >= POTHOLE_HIGH_M2:
        return "high"
    if area_m2 >= POTHOLE_MEDIUM_M2:
        return "medium"
    return "low"


def apply_camera_json(cfg, data: dict[str, Any]) -> None:
    """Fill InferConfig metric fields from camera.json without clobbering CLI."""
    if cfg.camera_height_m is None and data.get("height_m") is not None:
        cfg.camera_height_m = float(data["height_m"])
    if cfg.camera_pitch_deg is None and data.get("pitch_deg") is not None:
        cfg.camera_pitch_deg = float(data["pitch_deg"])
    if data.get("yaw_deg") is not None:
        cfg.camera_yaw_deg = float(data["yaw_deg"])
    if cfg.h_fov_deg is None and data.get("h_fov_deg") is not None:
        cfg.h_fov_deg = float(data["h_fov_deg"])
    if cfg.vfov_deg is None and data.get("vfov_deg") is not None:
        cfg.vfov_deg = float(data["vfov_deg"])
    if data.get("k1") is not None and abs(cfg.k1) < 1e-12:
        cfg.k1 = float(data["k1"])
    if data.get("k2") is not None and abs(cfg.k2) < 1e-12:
        cfg.k2 = float(data["k2"])


def camera_from_infer_cfg(cfg, width: int, height: int) -> CameraModel | None:
    """None when height is unknown — pixel areas only."""
    if cfg.camera_height_m is None:
        return None
    return build_camera_model(
        width,
        height,
        height_m=float(cfg.camera_height_m),
        pitch_deg=float(cfg.camera_pitch_deg if cfg.camera_pitch_deg is not None else 5.0),
        yaw_deg=float(getattr(cfg, "camera_yaw_deg", 0.0) or 0.0),
        h_fov_deg=cfg.h_fov_deg,
        vfov_deg=cfg.vfov_deg,
        fx=getattr(cfg, "camera_fx", None),
        fy=getattr(cfg, "camera_fy", None),
    )


def load_camera_json(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"camera json must be an object: {path}")
    return data


def hfov_from_vfov(vfov_deg: float, width: int, height: int) -> float:
    vf = math.radians(float(vfov_deg))
    hf = 2.0 * math.atan(math.tan(vf / 2.0) * (width / max(height, 1)))
    return math.degrees(hf)


def build_camera_model(
    width: int,
    height: int,
    *,
    height_m: float,
    pitch_deg: float = 5.0,
    yaw_deg: float = 0.0,
    h_fov_deg: float | None = None,
    vfov_deg: float | None = None,
    fx: float | None = None,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
) -> CameraModel:
    """Pinhole camera looking at the road plane Y = height_m."""
    if fx and fy:
        intr = Intrinsics(
            fx=float(fx),
            fy=float(fy),
            cx=float(cx if cx is not None else width / 2.0),
            cy=float(cy if cy is not None else height / 2.0),
            width=int(width),
            height=int(height),
        )
    else:
        hfov = h_fov_deg
        if hfov is None and vfov_deg is not None:
            hfov = hfov_from_vfov(vfov_deg, width, height)
        if hfov is None:
            hfov = GOPRO_HFOV_16_9["linear"]
        intr = Intrinsics.from_hfov(width, height, float(hfov))
    extr = Extrinsics(
        height_m=float(height_m),
        pitch_deg=float(pitch_deg),
        yaw_deg=float(yaw_deg),
    )
    return CameraModel(intr, extr)


def area_map_m2(camera: CameraModel) -> np.ndarray:
    """m² of ground covered by each source pixel (0 above the horizon)."""
    x, z, valid = camera.ground_maps()
    dx_du = np.zeros_like(x)
    dz_du = np.zeros_like(z)
    dx_dv = np.zeros_like(x)
    dz_dv = np.zeros_like(z)
    dx_du[:, 1:-1] = 0.5 * (x[:, 2:] - x[:, :-2])
    dz_du[:, 1:-1] = 0.5 * (z[:, 2:] - z[:, :-2])
    dx_dv[1:-1, :] = 0.5 * (x[2:, :] - x[:-2, :])
    dz_dv[1:-1, :] = 0.5 * (z[2:, :] - z[:-2, :])
    # One-sided edges
    dx_du[:, 0] = x[:, 1] - x[:, 0]
    dz_du[:, 0] = z[:, 1] - z[:, 0]
    dx_du[:, -1] = x[:, -1] - x[:, -2]
    dz_du[:, -1] = z[:, -1] - z[:, -2]
    dx_dv[0, :] = x[1, :] - x[0, :]
    dz_dv[0, :] = z[1, :] - z[0, :]
    dx_dv[-1, :] = x[-1, :] - x[-2, :]
    dz_dv[-1, :] = z[-1, :] - z[-2, :]
    area = np.abs(dx_du * dz_dv - dx_dv * dz_du)
    area = np.where(valid & np.isfinite(area), area, 0.0)
    return area.astype(np.float32)


def mask_area_m2(mask: np.ndarray, area_map: np.ndarray | None) -> float | None:
    if area_map is None:
        return None
    if mask.shape[:2] != area_map.shape[:2]:
        return None
    return float(np.asarray(area_map)[np.asarray(mask, dtype=bool)].sum())


def undistort_maps(
    width: int,
    height: int,
    *,
    h_fov_deg: float,
    k1: float = 0.0,
    k2: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
):
    """cv2 remap maps for a simple Brown-Conrady model. Identity when k=p=0."""
    import cv2

    fx = (width / 2.0) / math.tan(math.radians(h_fov_deg) / 2.0)
    K = np.array(
        [[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.array([k1, k2, p1, p2, 0.0], dtype=np.float64)
    return cv2.initUndistortRectifyMap(
        K, dist, None, K, (width, height), cv2.CV_32FC1
    )


def remap_frame(frame, map1, map2):
    import cv2

    return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)


def speed_mps_at(gps, t: float) -> float | None:
    """Instantaneous ground speed from neighbouring GPS fixes, m/s."""
    if gps is None or len(gps) < 2:
        return None
    try:
        from rdd.utils.geo import haversine_m
    except Exception:
        return None
    if not hasattr(gps, "index_at_time"):
        return None
    i = gps.index_at_time(t)
    if i is None:
        return None
    fixes = gps.fixes
    i0 = max(0, i - 1)
    i1 = min(len(fixes) - 1, i + 1)
    if i0 == i1:
        return None
    a, b = fixes[i0], fixes[i1]
    dt = float(b.t) - float(a.t)
    if dt < 1e-3:
        return None
    return haversine_m(a.lat, a.lon, b.lat, b.lon) / dt


@dataclass
class GsdCheck:
    z_m: float
    camera_gsd_long_m: float
    implied_gsd_long_m: float | None
    speed_mps: float | None
    flow_px: float | None
    ratio: float | None
    ok: bool
    note: str

    def as_dict(self) -> dict:
        return {
            "z_m": round(self.z_m, 2),
            "camera_gsd_long_m": round(self.camera_gsd_long_m, 5),
            "implied_gsd_long_m": (
                None if self.implied_gsd_long_m is None
                else round(self.implied_gsd_long_m, 5)
            ),
            "speed_mps": None if self.speed_mps is None else round(self.speed_mps, 2),
            "flow_px": None if self.flow_px is None else round(self.flow_px, 3),
            "ratio": None if self.ratio is None else round(self.ratio, 3),
            "ok": self.ok,
            "note": self.note,
        }


def _median_vertical_flow(prev, curr, roi_mask) -> float | None:
    import cv2

    if prev is None or curr is None:
        return None
    g0 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    vy = flow[..., 1]
    sel = roi_mask.astype(bool) & np.isfinite(vy)
    if sel.sum() < 200:
        return None
    med = float(np.median(np.abs(vy[sel])))
    return med if med > 0.05 else None


def check_gsd_with_speed(
    camera: CameraModel,
    frames: list,
    gps,
    fps: float,
    z_m: float = 3.0,
    max_ratio: float = 1.2,
) -> GsdCheck:
    """Compare camera-model longitudinal GSD to GPS speed / optical flow."""
    gsd = camera.gsd_at(z_m)
    cam_long = float(gsd.longitudinal)
    if not math.isfinite(cam_long) or cam_long <= 0:
        return GsdCheck(
            z_m, cam_long, None, None, None, None, False,
            "camera GSD unusable at this range",
        )

    u, v = camera.pixel_from_ground(0.0, z_m)
    if not (math.isfinite(u) and math.isfinite(v)):
        return GsdCheck(
            z_m, cam_long, None, None, None, None, False,
            "range does not project into the frame",
        )

    speed = None
    t_mid = 0.0
    if gps is not None and len(gps) >= 2:
        speed = speed_mps_at(gps, t_mid)
        if speed is None or speed < 0.5:
            # try a later fix
            t_mid = float(gps.fixes[min(3, len(gps.fixes) - 1)].t)
            speed = speed_mps_at(gps, t_mid)

    if speed is None:
        return GsdCheck(
            z_m, cam_long, None, None, None, None, True,
            "no GPS speed — GSD check skipped; measure camera height/pitch instead",
        )

    flow_px = None
    if len(frames) >= 2:
        h, w = frames[0].shape[:2]
        row = int(np.clip(v, 8, h - 9))
        roi = np.zeros((h, w), dtype=bool)
        roi[row - 8 : row + 9, w // 4 : 3 * w // 4] = True
        flow_px = _median_vertical_flow(frames[0], frames[1], roi)

    if flow_px is None:
        return GsdCheck(
            z_m, cam_long, None, speed, None, None, True,
            "GPS speed available but optical flow too weak to check GSD",
        )

    metres_per_frame = speed / max(fps, 1e-6)
    implied = metres_per_frame / flow_px
    ratio = implied / cam_long if cam_long > 0 else None
    ok = ratio is not None and (1.0 / max_ratio) <= ratio <= max_ratio
    note = (
        "camera GSD agrees with GPS+flow"
        if ok
        else (
            f"GPS+flow GSD differs from camera model by {ratio:.2f}x — "
            "re-measure height_m and pitch_deg before trusting m²"
        )
    )
    return GsdCheck(z_m, cam_long, implied, speed, flow_px, ratio, ok, note)
