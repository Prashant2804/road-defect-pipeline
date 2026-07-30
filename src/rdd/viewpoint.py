"""Viewpoint profiles — what kind of camera shot this footage, and what that implies.

The same road looks geometrically nothing alike from a survey car and from a
drone, so almost every downstream stage needs to branch on viewpoint:

                    car_360            car_flat          drone_nadir
  input projection  equirectangular    rectilinear       rectilinear
  360->flat pass    yes (v360)         no                no
  road shape        trapezoid          trapezoid         band through frame
  perspective       strong             strong            negligible (top-down)
  ground scale      IPM homography     IPM homography    GSD from altitude
  defect scale      varies with range  varies with range roughly constant

Rather than scatter `if view == ...` through the stages, each stage asks a
`ViewProfile` for the one thing it needs. Config supplies overrides; the profile
supplies defaults that are sane for that camera.

Ground scale matters more than it looks: with metres-per-pixel known, defect
size becomes an absolute measurement (m²) and severity can use fixed physical
thresholds. Without it, severity can only ever be *relative to this clip*.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from .utils.logging import get_logger

log = get_logger("rdd.viewpoint")

ViewName = Literal["car_360", "car_flat", "drone_nadir"]


@dataclass(frozen=True)
class RoadPrior:
    """Where the road is *expected* to be, before looking at any pixels.

    kind == "trapezoid": forward-facing perspective view. Wide at the bottom of
    the frame (road right under the camera), narrowing toward the horizon.

    kind == "band": nadir view. The road is an elongated strip crossing the
    frame; `axis` is the direction it runs. "auto" infers it per clip from the
    road-candidate pixels rather than trusting the flight line.
    """

    kind: Literal["trapezoid", "band"]
    # trapezoid
    bottom_y: float = 1.0
    top_y: float = 0.55
    bottom_half_width: float = 0.48
    top_half_width: float = 0.12
    center_x: float = 0.5
    # band
    axis: Literal["vertical", "horizontal", "auto"] = "auto"
    band_center: float = 0.5
    band_half_width: float = 0.30

    def polygon(self, w: int, h: int, axis: str | None = None) -> "Any":
        """Prior as an (N,2) int array of pixel coords, ready for fillPoly."""
        import numpy as np

        if self.kind == "trapezoid":
            cx = self.center_x * w
            by, ty = self.bottom_y * h, self.top_y * h
            bhw, thw = self.bottom_half_width * w, self.top_half_width * w
            pts = [
                (cx - bhw, by), (cx + bhw, by),
                (cx + thw, ty), (cx - thw, ty),
            ]
        else:
            use = axis or self.axis
            if use == "auto":
                use = "vertical"
            lo = (self.band_center - self.band_half_width)
            hi = (self.band_center + self.band_half_width)
            if use == "vertical":      # road runs top->bottom, band spans x
                x0, x1 = lo * w, hi * w
                pts = [(x0, 0), (x1, 0), (x1, h), (x0, h)]
            else:                      # road runs left->right, band spans y
                y0, y1 = lo * h, hi * h
                pts = [(0, y0), (w, y0), (w, y1), (0, y1)]
        return np.array([[int(round(x)), int(round(y))] for x, y in pts], dtype=np.int32)


@dataclass(frozen=True)
class ViewProfile:
    name: str
    input_projection: Literal["equirect", "rectilinear"]
    needs_reprojection: bool
    road_prior: RoadPrior
    scale_source: Literal["ipm", "gsd", "none"]
    m_per_px: float | None = None      # ground metres per pixel, if resolvable
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_topdown(self) -> bool:
        return self.name == "drone_nadir"

    @property
    def has_scale(self) -> bool:
        return self.m_per_px is not None and self.m_per_px > 0

    def px_to_m2(self, area_px: float) -> float | None:
        """Convert a pixel area to m². None when scale is unknown."""
        if not self.has_scale:
            return None
        return float(area_px) * (self.m_per_px ** 2)


_DEFAULT_PRIORS: dict[str, RoadPrior] = {
    "car_360": RoadPrior(kind="trapezoid"),
    "car_flat": RoadPrior(kind="trapezoid"),
    "drone_nadir": RoadPrior(kind="band"),
}

_PROJECTION: dict[str, str] = {
    "car_360": "equirect",
    "car_flat": "rectilinear",
    "drone_nadir": "rectilinear",
}


def gsd_m_per_px(altitude_m: float, focal_mm: float,
                 sensor_width_mm: float, image_width_px: int) -> float:
    """Ground sample distance for a nadir camera, in metres per pixel.

    Similar triangles: the sensor width maps to a ground swath of
    altitude * sensor_width / focal, spread over image_width pixels.
    """
    if min(altitude_m, focal_mm, sensor_width_mm, image_width_px) <= 0:
        raise ValueError("altitude, focal, sensor width and image width must be > 0")
    return (altitude_m * sensor_width_mm) / (focal_mm * float(image_width_px))


def _resolve_drone_scale(vc: dict, image_width_px: int | None) -> tuple[float | None, tuple[str, ...]]:
    explicit = vc.get("gsd_m_per_px")
    if explicit:
        return float(explicit), (f"scale: explicit GSD {float(explicit):.5f} m/px",)

    cam = vc.get("camera", {}) or {}
    alt = vc.get("altitude_m")
    focal, sensor = cam.get("focal_mm"), cam.get("sensor_width_mm")
    if alt and focal and sensor and image_width_px:
        try:
            g = gsd_m_per_px(float(alt), float(focal), float(sensor), int(image_width_px))
        except ValueError as e:
            return None, (f"scale: unusable drone camera params ({e})",)
        return g, (
            f"scale: GSD {g:.5f} m/px from altitude {alt} m, "
            f"focal {focal} mm, sensor {sensor} mm, width {image_width_px} px",
        )
    return None, (
        "scale: unknown — set view.drone.gsd_m_per_px, or altitude_m plus "
        "view.drone.camera.{focal_mm,sensor_width_mm}. Severity will be "
        "relative to this clip instead of absolute m².",
    )


def _resolve_ipm_scale(cfg, frame_w: int | None, frame_h: int | None) -> tuple[float | None, tuple[str, ...]]:
    """Metres-per-pixel for the IPM (bird's-eye) plane.

    The IPM dst rectangle is a top-down view of the src trapezoid on the road.
    If the user tells us how big that trapezoid is on the ground
    (`ground_extent_m: [width, length]`), the warped image has a known scale.
    """
    ic = cfg.get_path("preprocess.ipm", {}) or {}
    if not ic.get("enabled", False):
        return None, ("scale: unknown — enable preprocess.ipm and set "
                      "preprocess.ipm.ground_extent_m to measure defects in m².",)
    extent = ic.get("ground_extent_m")
    if not extent or len(extent) != 2:
        return None, ("scale: IPM enabled but preprocess.ipm.ground_extent_m unset "
                      "— defect areas stay in pixels.",)
    ow = float(ic.get("out_width", 640))
    oh = float(ic.get("out_height", 960))
    mx = float(extent[0]) / ow
    my = float(extent[1]) / oh
    if mx <= 0 or my <= 0:
        return None, ("scale: ground_extent_m must be positive.",)
    if abs(mx - my) / max(mx, my) > 0.15:
        log.warning(
            "IPM ground scale is anisotropic (%.4f vs %.4f m/px). Set out_width/"
            "out_height proportional to ground_extent_m for square ground pixels.",
            mx, my,
        )
    g = math.sqrt(mx * my)   # geometric mean: correct for area conversion
    return g, (f"scale: IPM {g:.5f} m/px from ground extent {extent} m over "
               f"{int(ow)}x{int(oh)} px",)


def _prior_from_cfg(base: RoadPrior, overrides: dict | None) -> RoadPrior:
    if not overrides:
        return base
    known = {f for f in RoadPrior.__dataclass_fields__}
    kwargs = {k: v for k, v in overrides.items() if k in known}
    unknown = set(overrides) - known
    if unknown:
        log.warning("Ignoring unknown road-prior keys: %s", sorted(unknown))
    return RoadPrior(**{**{f: getattr(base, f) for f in known}, **kwargs})


def resolve_view(cfg, frame_w: int | None = None, frame_h: int | None = None) -> ViewProfile:
    """Build the ViewProfile for this run.

    frame_w/h are the dimensions of the frames the *detector* will see (i.e.
    after reprojection). They are only needed to turn drone camera intrinsics
    into a GSD, so passing None just leaves scale unresolved.
    """
    vc = cfg.get_path("view", {}) or {}
    name = vc.get("profile", "car_360")
    if name not in _PROJECTION:
        raise ValueError(f"Unknown view.profile {name!r}; expected one of {tuple(_PROJECTION)}")

    prior = _prior_from_cfg(_DEFAULT_PRIORS[name], vc.get("road_prior"))

    if name == "drone_nadir":
        m_per_px, notes = _resolve_drone_scale(vc.get("drone", {}) or {}, frame_w)
        scale_source = "gsd"
        needs_reproj = False
    else:
        m_per_px, notes = _resolve_ipm_scale(cfg, frame_w, frame_h)
        scale_source = "ipm"
        needs_reproj = (
            name == "car_360"
            and bool(cfg.get_path("preprocess.reproject.enabled", True))
        )

    profile = ViewProfile(
        name=name,
        input_projection=_PROJECTION[name],  # type: ignore[arg-type]
        needs_reprojection=needs_reproj,
        road_prior=prior,
        scale_source=scale_source,  # type: ignore[arg-type]
        m_per_px=m_per_px,
        notes=notes,
    )
    log.info("Viewpoint '%s': projection=%s reproject=%s road_prior=%s",
             profile.name, profile.input_projection, profile.needs_reprojection,
             profile.road_prior.kind)
    for n in notes:
        log.info("  %s", n)
    return profile
