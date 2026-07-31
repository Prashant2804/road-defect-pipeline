"""Camera model: pixels <-> the road plane, and what that plane costs in resolution.

Everything metric in this pipeline bottoms out here. Defect area in m², crack width
in mm, whether a crack is longitudinal or transverse, rut depth, edge loss, texture
at a fixed ground scale — none of it means anything without a calibrated mapping
between image pixels and ground coordinates. Until now that mapping was four
hand-typed trapezoid corners plus a guessed `ground_extent_m`, which silently
scales every measurement if it is wrong.

Conventions (standard computer-vision camera frame):

    X right, Y down, Z forward.  Camera at the origin.
    The road is the plane Y = height_m.
    `pitch_deg` is *downward* tilt: larger pitch looks further down at the road.

Two derived quantities do most of the work downstream:

**GSD (ground sample distance)** — how much ground one pixel covers at a given
range. It is strongly anisotropic in a forward-facing view: at 15 m a pixel might
span 12 mm across the road but 90 mm along it, because of foreshortening. Reporting
a single averaged number would hide exactly the direction that limits transverse
crack detection, so both are computed and the *worst* is used for resolution
budgets.

**Maximum useful range** — invert the GSD curve to answer "how far ahead can this
camera still resolve a 3 mm crack?". Usually much closer than people assume, and
knowing it turns a silent false negative ("no cracks at 25 m") into an honest
"not assessed beyond 9 m".
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..utils.logging import get_logger

log = get_logger("rdd.geometry")


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics in pixels."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_hfov(cls, width: int, height: int, h_fov_deg: float) -> "Intrinsics":
        """Approximate intrinsics from horizontal field of view.

        The fallback when no checkerboard calibration exists. Assumes square
        pixels and a centred principal point — good enough for ground geometry,
        and far better than treating the image as already-rectified. Real
        calibration should replace it when available.
        """
        if not (0 < h_fov_deg < 180):
            raise ValueError(f"h_fov_deg must be in (0,180): {h_fov_deg}")
        fx = (width / 2.0) / math.tan(math.radians(h_fov_deg) / 2.0)
        return cls(fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0,
                   width=int(width), height=int(height))

    @property
    def h_fov_deg(self) -> float:
        return 2.0 * math.degrees(math.atan((self.width / 2.0) / self.fx))

    @property
    def v_fov_deg(self) -> float:
        return 2.0 * math.degrees(math.atan((self.height / 2.0) / self.fy))

    def as_dict(self) -> dict:
        return {"fx": round(self.fx, 3), "fy": round(self.fy, 3),
                "cx": round(self.cx, 2), "cy": round(self.cy, 2),
                "width": self.width, "height": self.height,
                "h_fov_deg": round(self.h_fov_deg, 2)}


@dataclass(frozen=True)
class Extrinsics:
    """Camera pose relative to the road plane."""

    height_m: float = 1.3       # dashcam height above the road surface
    pitch_deg: float = 5.0      # downward tilt; > 0 looks toward the road
    yaw_deg: float = 0.0        # 0 = aligned with the direction of travel
    roll_deg: float = 0.0

    def as_dict(self) -> dict:
        return {"height_m": self.height_m, "pitch_deg": round(self.pitch_deg, 3),
                "yaw_deg": round(self.yaw_deg, 3), "roll_deg": self.roll_deg}


@dataclass(frozen=True)
class GsdSample:
    """Ground resolution at one range, in metres per pixel."""

    distance_m: float
    lateral: float          # across the road
    longitudinal: float     # along the road — the foreshortened, limiting one

    @property
    def worst(self) -> float:
        return max(self.lateral, self.longitudinal)


class CameraModel:
    """Maps pixels to the road plane and reports the resolution cost of range."""

    def __init__(self, intr: Intrinsics, extr: Extrinsics):
        self.intr = intr
        self.extr = extr
        self._th = math.radians(extr.pitch_deg)
        self._yaw = math.radians(extr.yaw_deg)

    # -- core geometry -----------------------------------------------------
    def _ray_world(self, u: float, v: float) -> tuple[float, float, float]:
        """Unit-ish ray direction in world axes for pixel (u, v)."""
        xn = (u - self.intr.cx) / self.intr.fx
        yn = (v - self.intr.cy) / self.intr.fy
        c, s = math.cos(self._th), math.sin(self._th)
        # Rotate the camera ray down by pitch: the optical axis (0,0,1) must gain
        # a positive (downward) Y component.
        dx, dy, dz = xn, c * yn + s, -s * yn + c
        if self._yaw:
            cy_, sy_ = math.cos(self._yaw), math.sin(self._yaw)
            dx, dz = cy_ * dx + sy_ * dz, -sy_ * dx + cy_ * dz
        return dx, dy, dz

    @property
    def horizon_row(self) -> float:
        """Image row of the horizon: where a ground ray becomes parallel to the plane."""
        return self.intr.cy - self.intr.fy * math.tan(self._th)

    def ground_from_pixel(self, u: float, v: float) -> tuple[float, float] | None:
        """(x_lateral_m, z_forward_m) on the road, or None above the horizon.

        Above the horizon the ray never meets the plane — returning None rather
        than a huge number keeps "sky" from becoming "road 8 km away".
        """
        dx, dy, dz = self._ray_world(u, v)
        if dy <= 1e-9:
            return None
        t = self.extr.height_m / dy
        z = t * dz
        if z <= 0:
            return None
        return t * dx, z

    def pixel_from_ground(self, x: float, z: float) -> tuple[float, float]:
        """Project a road point (x lateral, z forward) back to a pixel."""
        c, s = math.cos(self._th), math.sin(self._th)
        y = self.extr.height_m
        if self._yaw:
            cy_, sy_ = math.cos(self._yaw), math.sin(self._yaw)
            x, z = cy_ * x - sy_ * z, sy_ * x + cy_ * z
        # Inverse of the pitch rotation applied in _ray_world.
        yc = c * y - s * z
        zc = s * y + c * z
        if abs(zc) < 1e-9:
            return float("nan"), float("nan")
        return (self.intr.fx * x / zc + self.intr.cx,
                self.intr.fy * yc / zc + self.intr.cy)

    def vanishing_point(self) -> tuple[float, float]:
        """Where road-parallel lines converge — the direction of travel at infinity."""
        return (self.intr.cx + self.intr.fx * math.tan(self._yaw), self.horizon_row)

    # -- resolution --------------------------------------------------------
    def gsd_at(self, z: float) -> GsdSample:
        """Ground metres per pixel at forward distance `z`.

        Lateral is analytic. Longitudinal is a central difference on the
        pixel->ground map, which is exact enough and avoids a brittle closed form.
        """
        u0, v0 = self.pixel_from_ground(0.0, z)
        if not (math.isfinite(u0) and math.isfinite(v0)):
            return GsdSample(z, float("inf"), float("inf"))

        near = self.ground_from_pixel(u0, v0 + 0.5)
        far = self.ground_from_pixel(u0, v0 - 0.5)
        if near is None or far is None:
            longitudinal = float("inf")
        else:
            longitudinal = abs(far[1] - near[1])

        left = self.ground_from_pixel(u0 - 0.5, v0)
        right = self.ground_from_pixel(u0 + 0.5, v0)
        lateral = abs(right[0] - left[0]) if (left and right) else float("inf")
        return GsdSample(z, lateral, longitudinal)

    def max_range_for_gsd(self, target_m_per_px: float,
                          z_min: float = 1.0, z_max: float = 120.0) -> float:
        """Furthest distance at which resolution is still better than `target`.

        GSD grows monotonically with range, so this is a clean bisection. Returns
        z_min when even the nearest usable ground fails the budget — which is a
        real answer, not an error: that camera cannot resolve that feature at all.
        """
        if target_m_per_px <= 0:
            return z_min
        if self.gsd_at(z_min).worst > target_m_per_px:
            return z_min

        lo, hi = z_min, z_max
        if self.gsd_at(hi).worst <= target_m_per_px:
            return hi
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if self.gsd_at(mid).worst <= target_m_per_px:
                lo = mid
            else:
                hi = mid
        return lo

    def visible_range(self) -> tuple[float, float]:
        """(nearest, furthest) ground distance actually inside the frame.

        The near limit is the bottom image row — often several metres out on a
        dashcam, since the bonnet and the camera's own height put the ground
        under the vehicle out of view.
        """
        near = self.ground_from_pixel(self.intr.cx, self.intr.height - 0.5)
        far = self.ground_from_pixel(self.intr.cx, self.horizon_row + 2.0)
        return (near[1] if near else 0.0), (far[1] if far else float("inf"))

    # -- bird's-eye --------------------------------------------------------
    def ipm_homography(self, x_half_width_m: float, z_near_m: float, z_far_m: float,
                       out_w: int, out_h: int):
        """Homography image -> metric bird's-eye view, with exact known scale.

        This replaces hand-picked trapezoid corners: the ground rectangle is
        specified in metres, so metres-per-pixel is known by construction rather
        than guessed. Returns (H, m_per_px_x, m_per_px_z).
        """
        import cv2
        import numpy as np

        if z_far_m <= z_near_m or x_half_width_m <= 0:
            raise ValueError("need z_far > z_near > 0 and x_half_width > 0")

        # Ground rectangle corners -> their image positions.
        ground = [(-x_half_width_m, z_far_m), (x_half_width_m, z_far_m),
                  (x_half_width_m, z_near_m), (-x_half_width_m, z_near_m)]
        src = np.array([self.pixel_from_ground(x, z) for x, z in ground],
                       dtype=np.float32)
        if not np.isfinite(src).all():
            raise ValueError("ground rectangle does not project into the image")

        # Bird's-eye: x across the width, far range at the top.
        dst = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]],
                       dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        return H, (2 * x_half_width_m) / out_w, (z_far_m - z_near_m) / out_h

    def ground_maps(self, width: int | None = None, height: int | None = None):
        """Per-pixel ground coordinates: (x_m, z_m, valid).

        Vectorised because the per-pixel form is needed every frame — for
        assessment-zone masks, for converting crack width and defect area into
        ground units, and for laying a fixed-ground-resolution texture grid.
        `valid` is False above the horizon and behind the camera, where no ground
        point exists at all.
        """
        import numpy as np

        w = int(width or self.intr.width)
        h = int(height or self.intr.height)
        us = np.arange(w, dtype=np.float64)[None, :] + 0.5
        vs = np.arange(h, dtype=np.float64)[:, None] + 0.5

        xn = (us - self.intr.cx) / self.intr.fx
        yn = (vs - self.intr.cy) / self.intr.fy
        c, s = math.cos(self._th), math.sin(self._th)

        dx = np.broadcast_to(xn, (h, w)).astype(np.float64)
        dy = np.broadcast_to(c * yn + s, (h, w)).astype(np.float64)
        dz = np.broadcast_to(-s * yn + c, (h, w)).astype(np.float64)
        if self._yaw:
            cy_, sy_ = math.cos(self._yaw), math.sin(self._yaw)
            dx, dz = cy_ * dx + sy_ * dz, -sy_ * dx + cy_ * dz

        valid = dy > 1e-9
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(valid, self.extr.height_m / dy, np.nan)
            x = t * dx
            z = t * dz
        valid &= np.isfinite(z) & (z > 0)
        return (np.where(valid, x, np.nan), np.where(valid, z, np.nan), valid)

    # -- reporting ---------------------------------------------------------
    def describe(self) -> str:
        near, far = self.visible_range()
        g5 = self.gsd_at(min(max(near, 5.0), far))
        return (f"camera h={self.extr.height_m:.2f}m pitch={self.extr.pitch_deg:.2f}° "
                f"yaw={self.extr.yaw_deg:.2f}° | horizon row {self.horizon_row:.0f} | "
                f"visible {near:.1f}–{far:.0f}m | GSD@{g5.distance_m:.0f}m "
                f"{1000 * g5.lateral:.1f}mm lat / {1000 * g5.longitudinal:.1f}mm long")

    def as_dict(self) -> dict:
        near, far = self.visible_range()
        return {
            "intrinsics": self.intr.as_dict(),
            "extrinsics": self.extr.as_dict(),
            "horizon_row": round(self.horizon_row, 1),
            "vanishing_point": [round(v, 1) for v in self.vanishing_point()],
            "visible_range_m": [round(near, 2), round(far, 1)],
        }


# -- auto extrinsics from the vanishing point ---------------------------------

def extrinsics_from_vanishing_point(intr: Intrinsics, vp_u: float, vp_v: float,
                                    height_m: float) -> Extrinsics:
    """Recover pitch and yaw from an observed vanishing point.

    The VP is the image of the travel direction at infinity, so it depends only on
    orientation — invert the projection and the angles fall out:

        pitch = atan((cy - v_vp) / fy)
        yaw   = atan((u_vp - cx) / fx)

    Worth doing per clip rather than trusting config: dashcams get knocked, remounted
    and stuck to differently-raked windscreens, and a 2° pitch error is a large
    range error. Camera height cannot be recovered this way (it does not affect the
    VP) and must still be measured.
    """
    pitch = math.degrees(math.atan((intr.cy - vp_v) / intr.fy))
    yaw = math.degrees(math.atan((vp_u - intr.cx) / intr.fx))
    return Extrinsics(height_m=height_m, pitch_deg=pitch, yaw_deg=yaw)


def _edge_points(mask, n_rows: int = 24):
    """Left/right road-boundary points sampled over the lower part of the mask."""
    import numpy as np

    h, w = mask.shape[:2]
    rows = np.linspace(int(0.55 * h), h - 2, num=n_rows).astype(int)
    left, right = [], []
    for r in rows:
        cols = np.nonzero(mask[r])[0]
        if cols.size < 8:
            continue
        left.append((float(cols[0]), float(r)))
        right.append((float(cols[-1]), float(r)))
    return left, right


def _fit_line(points):
    """Fit u = a*v + b (x as a function of row) — robust for near-vertical edges."""
    import numpy as np

    if len(points) < 4:
        return None
    u = np.array([p[0] for p in points], dtype=np.float64)
    v = np.array([p[1] for p in points], dtype=np.float64)
    a, b = np.polyfit(v, u, 1)
    resid = np.abs(a * v + b - u)
    # One robust re-fit: road edges are ragged, and a pothole at the verge or a
    # patch of erosion should not tilt the whole line.
    keep = resid <= max(2.0, 2.0 * float(np.median(resid)))
    if keep.sum() >= 4:
        a, b = np.polyfit(v[keep], u[keep], 1)
    return float(a), float(b)


def vanishing_point_from_road_mask(mask) -> tuple[float, float] | None:
    """Intersect the fitted left and right road edges.

    Uses the road mask this pipeline already produces, so it needs no extra
    detector. Near-parallel edges (a straight road seen head-on with a narrow
    lens, or a badly-cropped mask) give an unstable intersection and are rejected
    rather than returning a wild estimate.
    """
    left, right = _edge_points(mask)
    fl, fr = _fit_line(left), _fit_line(right)
    if fl is None or fr is None:
        return None
    al, bl = fl
    ar, br = fr
    if abs(al - ar) < 1e-3:
        return None
    v = (br - bl) / (al - ar)
    u = al * v + bl
    return u, v


def estimate_vanishing_point(masks, intr: Intrinsics) -> tuple[float, float] | None:
    """Median VP across frames.

    Per-frame estimates are noisy — roads curve, masks wobble, the vehicle changes
    lane — so the median over a clip is the stable quantity. Estimates falling
    outside the frame are discarded before averaging.
    """
    import numpy as np

    us, vs = [], []
    for m in masks:
        vp = vanishing_point_from_road_mask(m)
        if vp is None:
            continue
        u, v = vp
        if not (0 <= u < intr.width and 0 <= v < intr.height):
            continue
        us.append(u)
        vs.append(v)

    if len(us) < 3:
        log.warning("Vanishing point: only %d usable estimates — keeping configured "
                    "pitch/yaw instead of auto-calibrating", len(us))
        return None
    u_med, v_med = float(np.median(us)), float(np.median(vs))
    spread = float(np.median(np.abs(np.array(vs) - v_med)))
    log.info("Vanishing point (%d frames): (%.1f, %.1f), row spread ±%.1f px",
             len(us), u_med, v_med, spread)
    if spread > 0.08 * intr.height:
        log.warning("Vanishing-point row is unstable (±%.0f px). Footage may be very "
                    "shaky or the road mask unreliable; auto-calibration will be "
                    "poor — verify camera pitch manually.", spread)
    return u_med, v_med


def _num(mapping: dict, key: str, default: float) -> float:
    """Read a numeric config value, treating an explicit `null` as absent.

    `dict.get(key, default)` only substitutes the default when the key is *missing* —
    a key present with value `None` returns None and then blows up in `float()`.
    Config templates list optional keys explicitly as `null` (so users can see what
    is available), which makes that the normal case here, not an edge case.
    """
    val = mapping.get(key)
    return float(default) if val is None else float(val)


def build_camera(cfg, width: int, height: int,
                 vp: tuple[float, float] | None = None) -> CameraModel:
    """Assemble the camera model from config, refined by a vanishing point if given."""
    cc = cfg.get_path("geometry.camera", {}) or {}

    fx, fy = cc.get("fx"), cc.get("fy")
    if fx and fy:
        intr = Intrinsics(fx=float(fx), fy=float(fy),
                          cx=_num(cc, "cx", width / 2.0),
                          cy=_num(cc, "cy", height / 2.0),
                          width=width, height=height)
        source = "explicit intrinsics"
    else:
        h_fov = _num(cc, "h_fov_deg", 78.0)
        intr = Intrinsics.from_hfov(width, height, h_fov)
        source = f"h_fov {h_fov:g}° (approximate)"

    height_m = _num(cc, "height_m", 1.3)
    extr = Extrinsics(height_m=height_m,
                      pitch_deg=_num(cc, "pitch_deg", 5.0),
                      yaw_deg=_num(cc, "yaw_deg", 0.0))

    if vp is not None and cc.get("auto_pitch_from_vp", True):
        auto = extrinsics_from_vanishing_point(intr, vp[0], vp[1], height_m)
        delta = abs(auto.pitch_deg - extr.pitch_deg)
        limit = _num(cc, "max_auto_pitch_correction_deg", 12.0)
        if delta <= limit:
            log.info("Auto-calibrated pitch %.2f° (config said %.2f°), yaw %.2f°",
                     auto.pitch_deg, extr.pitch_deg, auto.yaw_deg)
            extr = auto
            source += " + VP-derived pitch/yaw"
        else:
            log.warning(
                "Vanishing point implies pitch %.2f°, which is %.1f° from the "
                "configured %.2f° — beyond the %.0f° sanity limit. Keeping config; "
                "check geometry.camera.pitch_deg and the road mask.",
                auto.pitch_deg, delta, extr.pitch_deg, limit)

    cam = CameraModel(intr, extr)
    log.info("Camera model from %s: %s", source, cam.describe())
    return cam
