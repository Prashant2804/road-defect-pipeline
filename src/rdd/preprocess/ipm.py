"""Inverse Perspective Mapping — the road plane, seen from above.

Previously this module was never called: `preprocess.ipm.enabled` was a config
knob wired to nothing. It is now used for the thing it is actually good for.

The naive use of IPM is to warp every frame to bird's-eye and detect on that. We
do not, for two reasons: warping resamples (costing sharpness on the very small
defects we care about), and a bird's-eye annotated video is much harder for a
human reviewer to sanity-check than the natural camera view.

What IPM is genuinely needed for is **measurement**. In a perspective view, a
pothole 30 m down the road covers a fraction of the pixels of an identical one at
5 m, so pixel area is not a defect size — it is a defect size confounded with
range. The homography that flattens the road plane tells us exactly how much
ground each pixel covers, so we can measure in m² while still detecting and
drawing in the original view.

That conversion is done analytically rather than by warping each mask. For a
homography H, the local area scale at a point is |det H| / w³ where
w = h20·x + h21·y + h22 — the Jacobian determinant. Precomputing that as a
per-pixel map turns "ground area of this defect" into a single masked sum.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..utils.logging import get_logger

log = get_logger("rdd.preprocess.ipm")


@dataclass
class IpmTransform:
    """Homography from the camera image to a metric top-down road plane."""

    H: "object"                 # 3x3 float64
    out_w: int
    out_h: int
    m_per_px_x: float | None    # metres per pixel in the warped plane, x
    m_per_px_y: float | None
    ref_point: tuple[float, float] = (0.0, 0.0)   # a pixel known to be on the road
    _area_map: "object" = None   # lazily built, m² per source pixel

    @property
    def has_scale(self) -> bool:
        return bool(self.m_per_px_x and self.m_per_px_y)

    def warp(self, frame):
        """Bird's-eye view of a frame — useful for debugging the calibration."""
        import cv2

        return cv2.warpPerspective(frame, self.H, (self.out_w, self.out_h))

    def area_map(self, in_w: int, in_h: int):
        """m² of ground covered by each source pixel; None when scale is unknown.

        Cached, since it depends only on the homography and frame size.
        """
        import numpy as np

        if not self.has_scale:
            return None
        if self._area_map is not None and self._area_map.shape == (in_h, in_w):
            return self._area_map

        H = np.asarray(self.H, dtype=np.float64)
        det = float(np.linalg.det(H))
        if abs(det) < 1e-12:
            log.warning("IPM homography is degenerate (det=%.3g) — no area scaling", det)
            return None

        ys, xs = np.mgrid[0:in_h, 0:in_w]
        w = H[2, 0] * xs + H[2, 1] * ys + H[2, 2]

        # w == 0 is the horizon line. It splits the image into the road plane and
        # everything beyond it, and which side carries which *sign* depends on the
        # point ordering — so we cannot assume w > 0 means "road". Take the sign at
        # a pixel known to be on the road (the src trapezoid centroid) as the valid
        # side, and reject the other. Magnitude is what sets the area scale.
        rx, ry = self.ref_point
        ref_w = H[2, 0] * float(rx) + H[2, 1] * float(ry) + H[2, 2]
        sign = 1.0 if ref_w >= 0 else -1.0

        eps = 1e-6
        valid = (w * sign) > eps
        with np.errstate(divide="ignore", invalid="ignore"):
            dst_px_per_src_px = np.abs(det) / np.power(np.abs(w), 3.0)
        dst_px_per_src_px[~np.isfinite(dst_px_per_src_px)] = 0.0
        dst_px_per_src_px[~valid] = 0.0

        if not valid.any():
            log.warning("IPM: no pixels fall on the road side of the horizon — "
                        "check src_points/dst_points ordering")

        m2_per_dst_px = float(self.m_per_px_x) * float(self.m_per_px_y)
        self._area_map = (dst_px_per_src_px * m2_per_dst_px).astype(np.float32)
        return self._area_map

    def describe(self) -> str:
        if not self.has_scale:
            return f"IPM {self.out_w}x{self.out_h} (no ground scale)"
        return (f"IPM {self.out_w}x{self.out_h}, "
                f"{self.m_per_px_x:.4f}x{self.m_per_px_y:.4f} m/px on the road plane")


def _points(name: str, raw, w: float, h: float):
    import numpy as np

    if not raw or len(raw) != 4:
        raise ValueError(f"preprocess.ipm.{name} must be 4 [x,y] fractions, got {raw!r}")
    for p in raw:
        if len(p) != 2:
            raise ValueError(f"preprocess.ipm.{name} entries must be [x,y]: {p!r}")
    return np.array([[float(x) * w, float(y) * h] for x, y in raw], dtype=np.float32)


def build_transform(cfg, in_w: int, in_h: int) -> IpmTransform | None:
    """Build the IPM transform, or None when disabled/unconfigured."""
    import cv2
    import numpy as np

    ic = cfg.get_path("preprocess.ipm", {}) or {}
    if not ic.get("enabled", False):
        return None

    out_w = int(ic.get("out_width", 640))
    out_h = int(ic.get("out_height", 960))
    if out_w <= 0 or out_h <= 0:
        raise ValueError("preprocess.ipm.out_width/out_height must be positive")

    src = _points("src_points", ic.get("src_points"), in_w, in_h)
    dst = _points("dst_points", ic.get("dst_points"), out_w, out_h)
    H = cv2.getPerspectiveTransform(src, dst)
    if abs(float(np.linalg.det(H))) < 1e-12:
        raise ValueError(
            "preprocess.ipm src_points/dst_points give a degenerate homography — "
            "check that the four points are in the same order and not collinear."
        )

    mx = my = None
    extent = ic.get("ground_extent_m")
    if extent and len(extent) == 2 and float(extent[0]) > 0 and float(extent[1]) > 0:
        mx = float(extent[0]) / out_w
        my = float(extent[1]) / out_h
    else:
        log.info(
            "IPM enabled without preprocess.ipm.ground_extent_m — geometry is "
            "available but defect areas stay in pixels. Set it to the real-world "
            "[width, length] in metres of the src_points trapezoid to get m²."
        )

    centroid = (float(src[:, 0].mean()), float(src[:, 1].mean()))
    t = IpmTransform(H=H, out_w=out_w, out_h=out_h, m_per_px_x=mx, m_per_px_y=my,
                     ref_point=centroid)
    log.info("%s", t.describe())
    return t


# Backwards-compatible helpers -------------------------------------------------

def build_homography(cfg, in_w: int, in_h: int):
    """Legacy shape: (H, out_w, out_h) or None."""
    t = build_transform(cfg, in_w, in_h)
    return None if t is None else (t.H, t.out_w, t.out_h)


def warp(frame, H, out_w: int, out_h: int):
    import cv2

    return cv2.warpPerspective(frame, H, (out_w, out_h))
