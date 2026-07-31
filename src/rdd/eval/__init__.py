"""Evaluation: measure precision per unique defect and calibrate thresholds to a target."""
from .precision import (
    CertificationReport,
    ClassCalibration,
    GroundTruthDefect,
    MatchResult,
    calibrate_class,
    certify,
    load_ground_truth,
    match_tracks,
    sweep,
    wilson_interval,
)

__all__ = [
    "CertificationReport", "ClassCalibration", "GroundTruthDefect", "MatchResult",
    "calibrate_class", "certify", "load_ground_truth", "match_tracks", "sweep",
    "wilson_interval",
]
