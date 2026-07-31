"""One calibration pass over a clip, before any detection runs.

Samples frames, segments the road, estimates the vanishing point from the road
edges, and derives camera pitch/yaw from it. Everything metric downstream depends
on the result, so it is worth a short dedicated pass rather than being inferred
frame by frame while detecting.

Why a separate pass rather than per-frame refinement: a per-frame vanishing point is
noisy (roads curve, masks wobble, the vehicle changes lane) and letting the ground
scale drift frame to frame would make defect areas jitter for no physical reason. The
camera pose is fixed for a clip, so estimate it once and hold it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..utils.logging import get_logger
from .calibration import CameraModel, Intrinsics, build_camera
from .zones import ZoneSet, build_zones

log = get_logger("rdd.geometry.autocal")


@dataclass
class CalibrationResult:
    camera: CameraModel
    zones: ZoneSet
    vanishing_point: tuple[float, float] | None
    n_frames_used: int
    width: int
    height: int

    def summary(self) -> dict:
        return {
            "camera": self.camera.as_dict(),
            "vanishing_point_estimated": self.vanishing_point is not None,
            "vp_frames_used": self.n_frames_used,
            "zones": self.zones.summary(),
            "unachievable_classes": self.zones.unachievable(),
        }


def calibrate_video(video_path: str | Path, cfg, view=None, spec=None
                    ) -> CalibrationResult:
    """Estimate camera pose and assessment zones for one clip."""
    import cv2

    from ..quality.enhance import enhance_frame
    from ..roadseg.base import build_segmenter
    from .calibration import estimate_vanishing_point

    cc = cfg.get_path("geometry.camera", {}) or {}
    n_target = max(4, int(cc.get("vp_sample_frames", 40)))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for calibration: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    intr_probe = Intrinsics.from_hfov(width, height,
                                     float(cc.get("h_fov_deg", 78.0)))
    segmenter = build_segmenter(cfg, view)
    segmenter.reset()

    stride = max(1, total // n_target) if total else 1
    masks = []
    idx = -1
    while len(masks) < n_target:
        ok, raw = cap.read()
        if not ok:
            break
        idx += 1
        if idx % stride:
            continue
        frame = enhance_frame(raw, spec) if (spec and spec.enabled) else raw
        rm = segmenter.segment(frame)
        # Only fitted masks are usable: the geometric prior is a *symmetric*
        # trapezoid, so its edges intersect exactly where the prior assumed the
        # vanishing point already was. Feeding those in would "confirm" the
        # configured pitch no matter what the road actually does.
        if not rm.fell_back and rm.mask.any():
            masks.append(rm.mask)
    cap.release()

    vp = estimate_vanishing_point(masks, intr_probe) if masks else None
    if vp is None and masks:
        log.warning("Could not estimate a vanishing point from %d road masks — "
                    "falling back to configured pitch/yaw", len(masks))
    elif not masks:
        log.warning("No confidently-segmented road masks in the calibration pass — "
                    "using configured camera pose. Metric outputs are only as good "
                    "as geometry.camera.pitch_deg and height_m.")

    camera = build_camera(cfg, width, height, vp=vp)
    zones = build_zones(cfg, camera)
    return CalibrationResult(camera=camera, zones=zones, vanishing_point=vp,
                             n_frames_used=len(masks), width=width, height=height)
