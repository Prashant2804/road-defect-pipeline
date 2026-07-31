"""Camera geometry: the calibrated bridge between pixels and the road plane."""
from .calibration import (
    CameraModel,
    Extrinsics,
    GsdSample,
    Intrinsics,
    build_camera,
    estimate_vanishing_point,
    extrinsics_from_vanishing_point,
    vanishing_point_from_road_mask,
)
from .autocal import CalibrationResult, calibrate_video
from .zones import AssessmentZone, ZoneSet, build_zones

__all__ = [
    "CameraModel", "Extrinsics", "GsdSample", "Intrinsics", "build_camera",
    "estimate_vanishing_point", "extrinsics_from_vanishing_point",
    "vanishing_point_from_road_mask",
    "AssessmentZone", "ZoneSet", "build_zones",
    "CalibrationResult", "calibrate_video",
]
