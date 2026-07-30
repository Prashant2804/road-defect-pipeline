"""Road-mask types and the backend contract.

Segmenting the drivable surface *before* looking for defects changes what the
rest of the pipeline can do:

  * off-road false positives disappear — a roadside puddle, a dark bush or a
    reprojection artifact in the sky is no longer a candidate pothole;
  * defect area becomes meaningful as a *fraction of road surface*, which is
    what a road-condition report actually needs;
  * the road mask is the denominator for "how much of this road could we even
    assess", once mud and water are taken into account.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class RoadBaseline:
    """Robust appearance statistics of the road surface for one frame.

    Produced by the segmenter and consumed by the surface-condition stage, so
    that mud/water are judged *relative to this road* rather than against
    absolute colour thresholds that only hold for one soil type and one light.
    """

    stats: dict = field(default_factory=dict)   # channel -> (median, sigma)

    def get(self, channel: str) -> tuple[float, float]:
        return self.stats.get(channel, (0.0, 1.0))

    @property
    def is_empty(self) -> bool:
        return not self.stats


@dataclass
class RoadMask:
    """The road surface for one frame, plus how much to trust it."""

    mask: "object"                    # bool HxW — the drivable surface
    prior: "object"                   # bool HxW — geometric expectation used
    backend: str = "geometric"
    confidence: float = 0.0           # 0..1
    fell_back: bool = False           # backend degenerated; prior used instead
    baseline: RoadBaseline | None = None
    axis: str | None = None           # resolved band axis, drone views only

    @property
    def area_px(self) -> float:
        return float(self.mask.sum())

    def coverage(self) -> float:
        """Fraction of the frame classified as road."""
        import numpy as np

        total = float(np.prod(self.mask.shape[:2])) or 1.0
        return self.area_px / total

    def is_empty(self) -> bool:
        return self.area_px <= 0


@runtime_checkable
class RoadSegmenter(Protocol):
    """Anything that can turn a BGR frame into a RoadMask."""

    name: str

    def segment(self, frame) -> RoadMask:
        ...

    def reset(self) -> None:
        """Clear per-clip state (temporal history, cached axis)."""
        ...


def build_segmenter(cfg, view) -> RoadSegmenter:
    """Construct the configured segmenter, wrapped in temporal smoothing.

    Backends degrade rather than fail: an unavailable model backend logs and
    drops to `classical`, which itself falls back to the pure geometric prior on
    any frame where it produces something implausible.
    """
    from ..utils.logging import get_logger
    from .geometric import GeometricSegmenter
    from .temporal import TemporalSmoother

    log = get_logger("rdd.roadseg")
    backend = (cfg.get_path("roadseg.backend", "classical") or "classical").lower()

    if backend == "none":
        log.warning(
            "roadseg.backend: none — defects will NOT be constrained to the road "
            "surface, so off-road false positives are expected."
        )
        seg: RoadSegmenter = GeometricSegmenter(cfg, view, full_frame=True)
    elif backend == "geometric":
        seg = GeometricSegmenter(cfg, view)
    elif backend == "classical":
        from .classical import ClassicalSegmenter

        seg = ClassicalSegmenter(cfg, view)
    elif backend == "sam":
        from .sam import build_sam_segmenter

        seg = build_sam_segmenter(cfg, view)
    else:
        raise ValueError(f"Unknown roadseg.backend {backend!r}")

    smoothing = float(cfg.get_path("roadseg.temporal.alpha", 0.0) or 0.0)
    if smoothing > 0:
        seg = TemporalSmoother(seg, cfg)
    log.info("Road segmenter: %s", seg.name)
    return seg
