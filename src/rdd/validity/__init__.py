"""Frame validity: refuse to assess what cannot be seen.

Blocks detection when the road is buried under water/mud, cannot be located, the
vehicle has left the carriageway, traffic fills the lane, or the image itself is
unusable — and reports route coverage so a precision figure can state its subset.
"""
from .checker import StaticStructureDetector, ValidityChecker
from .egomotion import EgoMotion, EgoMotionEstimator
from .gates import ALL_GATES, FrameContext
from .traffic import TrafficDetector, TrafficResult
from .verdict import Action, FrameVerdict, GateResult, ValidityStats

__all__ = [
    "ValidityChecker", "StaticStructureDetector",
    "EgoMotion", "EgoMotionEstimator",
    "ALL_GATES", "FrameContext",
    "TrafficDetector", "TrafficResult",
    "Action", "FrameVerdict", "GateResult", "ValidityStats",
]
