"""Synthetic road scenes for testing segmentation and surface analysis.

Real footage is not available in CI, and these tests need *known ground truth* —
"is the water patch at (x,y) detected" is only answerable if we put it there. The
builders below encode the physical cues the pipeline relies on, so a test failing
here means a detector rule broke, not that a real clip happened to be awkward:

  vegetation : high texture, green, saturated
  road       : low texture, neutral grey-brown
  water      : near-zero texture (specular), brighter
  mud        : reduced texture, warm chroma shift, darker
  shadow     : darker but texture PRESERVED and chroma neutral (the decoy)
"""
from __future__ import annotations

import numpy as np

RNG = np.random.default_rng(1337)

W, H = 640, 480


def _noisy(shape, bgr, sigma, rng=None):
    rng = rng or RNG
    base = np.zeros((*shape, 3), dtype=np.float32)
    base[:] = np.asarray(bgr, dtype=np.float32)
    return base + rng.normal(0.0, sigma, size=(*shape, 3)).astype(np.float32)


def _clip(a):
    return np.clip(a, 0, 255).astype(np.uint8)


def trapezoid_pts(w=W, h=H, bottom_hw=0.46, top_hw=0.13, top_y=0.56):
    cx = w / 2
    return np.array([
        [cx - bottom_hw * w, h],
        [cx + bottom_hw * w, h],
        [cx + top_hw * w, top_y * h],
        [cx - top_hw * w, top_y * h],
    ], dtype=np.int32)


def car_scene(w=W, h=H, road_bgr=(120, 125, 130), road_sigma=6.0,
              veg_bgr=(40, 95, 45), veg_sigma=26.0):
    """Forward-facing car view: textured vegetation, smooth grey road trapezoid."""
    import cv2

    frame = _noisy((h, w), veg_bgr, veg_sigma)
    road = _noisy((h, w), road_bgr, road_sigma)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [trapezoid_pts(w, h)], 1)
    m = mask.astype(bool)
    frame[m] = road[m]
    return _clip(frame), m


def drone_scene(w=W, h=H, axis="vertical", road_bgr=(118, 124, 128),
                road_sigma=6.0, veg_bgr=(38, 92, 44), veg_sigma=26.0,
                half_width=0.18):
    """Nadir view: a straight road band crossing the frame."""
    frame = _noisy((h, w), veg_bgr, veg_sigma)
    road = _noisy((h, w), road_bgr, road_sigma)
    mask = np.zeros((h, w), dtype=bool)
    if axis == "vertical":
        x0, x1 = int((0.5 - half_width) * w), int((0.5 + half_width) * w)
        mask[:, x0:x1] = True
    else:
        y0, y1 = int((0.5 - half_width) * h), int((0.5 + half_width) * h)
        mask[y0:y1, :] = True
    frame[mask] = road[mask]
    return _clip(frame), mask


def _ellipse(shape, cx, cy, rx, ry):
    import cv2

    m = np.zeros(shape[:2], dtype=np.uint8)
    cv2.ellipse(m, (int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, 1, -1)
    return m.astype(bool)


def add_patch(frame, centre, radii, bgr, sigma):
    """Stamp an elliptical patch with its own colour and texture level."""
    frame = frame.copy()
    m = _ellipse(frame.shape, centre[0], centre[1], radii[0], radii[1])
    patch = _clip(_noisy(frame.shape[:2], bgr, sigma))
    frame[m] = patch[m]
    return frame, m


def add_water(frame, centre=(320, 420), radii=(70, 30)):
    """Specular: texture destroyed, markedly brighter (sky reflection)."""
    return add_patch(frame, centre, radii, bgr=(205, 200, 190), sigma=0.6)


def add_mud(frame, centre=(230, 430), radii=(60, 26)):
    """Wet soil: warm chroma shift, darker, somewhat smoother."""
    return add_patch(frame, centre, radii, bgr=(42, 72, 118), sigma=2.5)


def add_pothole(frame, centre=(400, 440), radii=(34, 16)):
    """A dark, still-textured depression — an appearance outlier on the road."""
    return add_patch(frame, centre, radii, bgr=(58, 60, 62), sigma=7.0)


def add_shadow(frame, centre=(320, 400), radii=(110, 45), gain=0.45):
    """The decoy: much darker, but physically a shadow rather than a wet surface.

    Modelled *multiplicatively*, which is what a shadow actually is — less light
    reaching the surface scales the reflected signal. That means absolute texture
    energy drops in proportion to brightness, so a detector keying on absolute
    smoothness will call this water. Only the brightness-relative texture (which
    a multiplicative change leaves invariant) separates the two, which is exactly
    what the surface stage keys on.
    """
    frame = frame.copy()
    m = _ellipse(frame.shape, centre[0], centre[1], radii[0], radii[1])
    darkened = _clip(frame.astype(np.float32) * float(gain))
    frame[m] = darkened[m]
    return frame, m


def blur(frame, sigma=4.0):
    import cv2

    return cv2.GaussianBlur(frame, (0, 0), sigma)


def overexpose(frame, gain=2.4):
    return _clip(frame.astype(np.float32) * gain)


def make_road_mask(mask, baseline=None, confidence=0.9, prior=None):
    """Wrap a boolean array as a RoadMask for surface-stage tests."""
    from rdd.roadseg.base import RoadMask

    return RoadMask(mask=mask, prior=prior if prior is not None else mask.copy(),
                    backend="test", confidence=confidence, fell_back=False,
                    baseline=baseline)


def surface_map(road_px: float, water_px: float, mud_px: float):
    """A SurfaceMap with only the area counters set — for aggregation tests."""
    from rdd.surface.condition import SurfaceMap

    z = np.zeros((2, 2), dtype=bool)
    return SurfaceMap(water=z, mud=z.copy(), dry=z.copy(), occlusion=z.copy(),
                      road_area_px=road_px, water_px=water_px, mud_px=mud_px)


def _filled_like(frame, mask, bgr, sigma):
    """Replace `mask` pixels with a uniform colour + texture level.

    Used to build fully-contaminated roads, where relative detection is blind and
    only absolute plausibility can help.
    """
    out = frame.copy()
    layer = _clip(_noisy(frame.shape[:2], bgr, sigma))
    out[mask] = layer[mask]
    return out
