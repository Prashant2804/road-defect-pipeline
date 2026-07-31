"""Detector families.

The seven requested distress categories are four different problems, and mixing them
into one flat class list caps accuracy on all of them:

  instance objects   potholes           -> instance segmentation (inference/detect_track)
  thin linear        cracks             -> linear.py: ground-plane orientation + cells
  area / texture     ravelling, rutting -> texture.py: fixed-metre grid statistics
  boundary geometry  edge damage        -> boundary.py: road-mask edge deviation

`confusers.py` rejects the recurring dashcam look-alikes, and `tiling.py` recovers the
resolution that thin cracks need.
"""
from .boundary import BoundaryConfig, BoundaryResult, EdgeDefect, detect_edge_damage
from .confusers import ConfuserStats, Rejection, check as check_confusers
from .aggregate import ConditionAggregator
from .linear import (
    ALLIGATOR,
    CRACK_SOURCES,
    LONGITUDINAL,
    TRANSVERSE,
    CrackGeometry,
    LinearConfig,
    LinearStats,
    classify_crack,
)
from .texture import (
    RavellingResult,
    RuttingResult,
    TextureConfig,
    detect_drainage,
    detect_ravelling,
    detect_rutting_proxy,
)
from .tiling import TilingConfig, merge_detections, plan_tiles, run_tiled

CRACK_CLASSES = (LONGITUDINAL, TRANSVERSE, ALLIGATOR)

__all__ = [
    "BoundaryConfig", "BoundaryResult", "EdgeDefect", "detect_edge_damage",
    "ConditionAggregator",
    "ConfuserStats", "Rejection", "check_confusers",
    "ALLIGATOR", "LONGITUDINAL", "TRANSVERSE", "CRACK_CLASSES", "CRACK_SOURCES",
    "CrackGeometry", "LinearConfig", "LinearStats", "classify_crack",
    "RavellingResult", "RuttingResult", "TextureConfig",
    "detect_drainage", "detect_ravelling", "detect_rutting_proxy",
    "TilingConfig", "merge_detections", "plan_tiles", "run_tiled",
]
