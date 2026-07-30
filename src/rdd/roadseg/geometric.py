"""Geometric road prior — the backend that cannot fail.

Pure geometry from the viewpoint profile: a trapezoid for forward-facing car
views, a band for drone nadir. No pixels are examined, so it is immune to mud,
glare, shadow and low contrast — which is exactly why it is the fallback for
every other backend and the safety net when they degenerate.

It is a weak segmenter on its own (it will happily include a verge or exclude a
bend), so treat it as a floor on quality, not a target.
"""
from __future__ import annotations

from ..utils.logging import get_logger
from .base import RoadMask
from .ops import polygon_mask

log = get_logger("rdd.roadseg.geometric")


class GeometricSegmenter:
    def __init__(self, cfg, view, full_frame: bool = False):
        self.cfg = cfg
        self.view = view
        self.full_frame = full_frame
        self.name = "whole-frame (no road gating)" if full_frame else "geometric-prior"
        self._axis: str | None = None

    def reset(self) -> None:
        self._axis = None

    def prior_mask(self, w: int, h: int, axis: str | None = None):
        import numpy as np

        if self.full_frame:
            return np.ones((h, w), dtype=bool)
        poly = self.view.road_prior.polygon(w, h, axis=axis)
        return polygon_mask(poly, w, h)

    def segment(self, frame) -> RoadMask:
        h, w = frame.shape[:2]
        prior = self.prior_mask(w, h, axis=self._axis)
        return RoadMask(
            mask=prior.copy(), prior=prior, backend=self.name,
            # A prior is an assumption, not an observation: never claim high
            # confidence, so downstream code can tell it apart from a real fit.
            confidence=1.0 if self.full_frame else 0.35,
            fell_back=False, baseline=None, axis=self._axis,
        )
