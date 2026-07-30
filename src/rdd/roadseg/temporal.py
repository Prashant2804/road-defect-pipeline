"""Temporal smoothing of the road mask.

A road does not change shape between consecutive frames, so a mask that jumps
around is measurement noise — a passing shadow, a compression artifact, one
frame of glare. Smoothing an exponential moving average of the per-pixel road
probability suppresses that flicker, which matters because an unstable road mask
makes defect gating unstable too: a defect flickering in and out of the road
region breaks its track and inflates the unique count.

The assumption is that the road occupies roughly the same *image* region frame to
frame. That holds for a forward-facing car and for a drone following the road,
which are the viewpoints this pipeline targets. It breaks on sharp turns, where
the mask lags by roughly `1/alpha` frames; keep alpha at 0.5 or above unless the
footage is very shaky.
"""
from __future__ import annotations

from ..utils.logging import get_logger
from .base import RoadMask

log = get_logger("rdd.roadseg.temporal")


class TemporalSmoother:
    def __init__(self, inner, cfg):
        self.inner = inner
        tc = cfg.get_path("roadseg.temporal", {}) or {}
        # alpha is the weight of the *current* frame: 1.0 = no smoothing.
        self.alpha = min(1.0, max(0.01, float(tc.get("alpha", 0.5))))
        self.threshold = min(0.99, max(0.01, float(tc.get("threshold", 0.5))))
        self._prob = None
        self.name = f"{getattr(inner, 'name', 'road')}+temporal(alpha={self.alpha:g})"

    def reset(self) -> None:
        self._prob = None
        if hasattr(self.inner, "reset"):
            self.inner.reset()

    @property
    def fallback_rate(self) -> float:
        return getattr(self.inner, "fallback_rate", 0.0)

    def segment(self, frame) -> RoadMask:
        import numpy as np

        rm: RoadMask = self.inner.segment(frame)
        cur = rm.mask.astype(np.float32)

        if self._prob is None or self._prob.shape != cur.shape:
            self._prob = cur
        else:
            self._prob = self.alpha * cur + (1.0 - self.alpha) * self._prob

        rm.mask = self._prob >= self.threshold
        if not rm.mask.any():
            # Smoothing must never erase the road entirely; fall back to this
            # frame's own estimate rather than reporting no road at all.
            rm.mask = cur.astype(bool)
            self._prob = cur
        return rm
