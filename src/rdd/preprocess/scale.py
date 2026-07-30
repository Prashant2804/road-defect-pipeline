"""Turning pixel areas into ground areas.

Severity is meaningless without scale. If defect size is only ever measured in
pixels, then "how bad is this pothole" can only be answered *relative to the
other potholes in the same clip* — so the largest defect in any video is always
the worst one, even if it is trivial, and identical roads shot at different
zoom levels get different verdicts. Neither is acceptable in a survey report.

Two viewpoints, two mechanisms, one interface:

  drone nadir  — top-down, so scale is uniform across the frame and comes
                 straight from the ground sample distance.
  car views    — strong perspective, so scale varies per pixel and comes from
                 the IPM homography's local area Jacobian.

When neither is available we say so explicitly (`NoScale`) rather than quietly
falling back to pixels and letting the report imply physical units.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..utils.logging import get_logger

log = get_logger("rdd.preprocess.scale")


@runtime_checkable
class AreaScaler(Protocol):
    kind: str
    has_scale: bool

    def area_m2(self, mask) -> float | None:
        ...

    def describe(self) -> str:
        ...


class NoScale:
    kind = "none"
    has_scale = False

    def __init__(self, why: str = "no ground scale configured"):
        self.why = why

    def area_m2(self, mask) -> float | None:
        return None

    def describe(self) -> str:
        return f"pixel areas only ({self.why})"


class UniformScaler:
    """Constant metres-per-pixel — correct for a nadir (top-down) camera."""

    kind = "uniform"
    has_scale = True

    def __init__(self, m_per_px: float, source: str = "gsd"):
        if m_per_px <= 0:
            raise ValueError("m_per_px must be positive")
        self.m_per_px = float(m_per_px)
        self.m2_per_px = self.m_per_px ** 2
        self.source = source

    def area_m2(self, mask) -> float | None:
        return float(mask.sum()) * self.m2_per_px

    def describe(self) -> str:
        return f"uniform {self.m_per_px:.5f} m/px from {self.source} " \
               f"({self.m2_per_px:.3e} m²/px)"


class PerPixelScaler:
    """Per-pixel ground area — correct for a perspective camera via IPM."""

    kind = "perspective"
    has_scale = True

    def __init__(self, area_map, source: str = "ipm"):
        self.area_map = area_map
        self.source = source

    def area_m2(self, mask) -> float | None:
        import numpy as np

        if mask.shape[:2] != self.area_map.shape[:2]:
            log.warning("Mask %s does not match area map %s — skipping ground area",
                        mask.shape[:2], self.area_map.shape[:2])
            return None
        return float(np.asarray(self.area_map)[mask].sum())

    def describe(self) -> str:
        import numpy as np

        vals = np.asarray(self.area_map)
        nz = vals[vals > 0]
        if nz.size == 0:
            return f"per-pixel ground area from {self.source} (empty)"
        return (f"per-pixel ground area from {self.source}: "
                f"{nz.min():.3e}–{nz.max():.3e} m²/px "
                f"(near-field to far-field, {nz.max() / nz.min():.0f}x range)")


def build_area_scaler(cfg, view, frame_w: int, frame_h: int) -> AreaScaler:
    """Pick the right scaler for this viewpoint and configuration."""
    if view is not None and view.scale_source == "gsd":
        if view.has_scale:
            s = UniformScaler(view.m_per_px, source="drone GSD")
            log.info("Ground scale: %s", s.describe())
            return s
        return NoScale("drone altitude/camera intrinsics not set")

    from .ipm import build_transform

    try:
        transform = build_transform(cfg, frame_w, frame_h)
    except ValueError as e:
        log.warning("IPM misconfigured (%s) — falling back to pixel areas", e)
        return NoScale(f"IPM invalid: {e}")

    if transform is None:
        return NoScale("preprocess.ipm.enabled is false")
    area_map = transform.area_map(frame_w, frame_h)
    if area_map is None:
        return NoScale("IPM has no ground_extent_m")

    s = PerPixelScaler(area_map, source="IPM homography")
    log.info("Ground scale: %s", s.describe())
    return s
