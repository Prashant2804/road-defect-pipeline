"""Camera model: ground projection, resolution curve, and auto-calibration.

Checked against independently-derivable values, not against the implementation's own
output. Several use a 45° pitch, where the geometry collapses to values you can work
out on paper — if the sign conventions were wrong these would fail loudly rather
than being quietly self-consistent.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from rdd.geometry.calibration import (
    CameraModel,
    Extrinsics,
    Intrinsics,
    build_camera,
    estimate_vanishing_point,
    extrinsics_from_vanishing_point,
    vanishing_point_from_road_mask,
)


def _cam(pitch=5.0, height=1.3, yaw=0.0, w=1920, h=1080, hfov=78.0):
    return CameraModel(Intrinsics.from_hfov(w, h, hfov),
                       Extrinsics(height_m=height, pitch_deg=pitch, yaw_deg=yaw))


# -- intrinsics ---------------------------------------------------------------

def test_hfov_roundtrips():
    intr = Intrinsics.from_hfov(1920, 1080, 78.0)
    assert abs(intr.h_fov_deg - 78.0) < 1e-6
    assert intr.cx == 960 and intr.cy == 540
    assert intr.fx == intr.fy, "square pixels assumed"


def test_narrower_fov_means_longer_focal_length():
    assert (Intrinsics.from_hfov(1920, 1080, 40.0).fx
            > Intrinsics.from_hfov(1920, 1080, 120.0).fx)


def test_absurd_fov_rejected():
    with pytest.raises(ValueError):
        Intrinsics.from_hfov(1920, 1080, 190.0)


# -- ground projection --------------------------------------------------------

def test_projection_roundtrips():
    cam = _cam()
    for x, z in [(0.0, 8.0), (-1.5, 12.0), (2.0, 25.0), (0.5, 40.0)]:
        u, v = cam.pixel_from_ground(x, z)
        back = cam.ground_from_pixel(u, v)
        assert back is not None
        assert abs(back[0] - x) < 1e-6, f"x drifted at z={z}"
        assert abs(back[1] - z) < 1e-6, f"z drifted at z={z}"


def test_forty_five_degree_pitch_gives_the_hand_calculable_answer():
    """At 45° down, the optical axis meets the ground at exactly the camera height."""
    cam = _cam(pitch=45.0, height=2.0)
    ground = cam.ground_from_pixel(cam.intr.cx, cam.intr.cy)
    assert ground is not None
    x, z = ground
    assert abs(x) < 1e-9
    assert abs(z - 2.0) < 1e-9, f"expected z = height = 2.0 m, got {z}"


def test_above_the_horizon_has_no_ground_point():
    """Sky must be None, not a point 8 km away."""
    cam = _cam(pitch=5.0)
    assert cam.ground_from_pixel(cam.intr.cx, cam.horizon_row - 20) is None
    assert cam.ground_from_pixel(cam.intr.cx, cam.horizon_row + 20) is not None


def test_horizon_rises_in_the_image_as_pitch_increases():
    """Looking further down puts the horizon higher up the frame (smaller row)."""
    rows = [_cam(pitch=p).horizon_row for p in (0.0, 5.0, 15.0, 30.0)]
    assert rows == sorted(rows, reverse=True)
    assert abs(_cam(pitch=0.0).horizon_row - 540.0) < 1e-9, "level camera: horizon at cy"


def test_lower_rows_are_closer_ground():
    cam = _cam()
    zs = [cam.ground_from_pixel(cam.intr.cx, v)[1]
          for v in (1079, 900, 800, 700)]
    assert zs == sorted(zs), "further down the image must be nearer ground"


def test_taller_mount_sees_further_at_the_same_pixel():
    a = _cam(height=1.3).ground_from_pixel(960, 900)[1]
    b = _cam(height=2.6).ground_from_pixel(960, 900)[1]
    assert abs(b - 2 * a) < 1e-6, "ground distance scales linearly with height"


def test_yaw_shifts_the_vanishing_point_sideways():
    straight = _cam(yaw=0.0).vanishing_point()
    turned = _cam(yaw=6.0).vanishing_point()
    assert turned[0] > straight[0]
    assert abs(turned[1] - straight[1]) < 1e-9, "yaw must not move the horizon row"


# -- resolution ---------------------------------------------------------------

def test_gsd_degrades_with_distance():
    cam = _cam()
    near, far = cam.gsd_at(5.0), cam.gsd_at(25.0)
    assert far.lateral > near.lateral
    assert far.longitudinal > near.longitudinal


def test_longitudinal_gsd_is_the_limiting_direction():
    """Foreshortening makes along-road resolution far worse than across-road.

    This is why a single averaged GSD would be misleading — it hides the direction
    that actually limits transverse crack detection.
    """
    g = _cam().gsd_at(15.0)
    assert g.longitudinal > g.lateral
    assert g.worst == g.longitudinal


def test_max_range_respects_the_budget():
    cam = _cam()
    for target in (0.005, 0.01, 0.02):
        z = cam.max_range_for_gsd(target)
        assert cam.gsd_at(z).worst <= target * 1.05, "returned range breaks its budget"
        assert cam.gsd_at(z + 3.0).worst > target, "range should be maximal"


def test_tighter_budget_means_shorter_range():
    cam = _cam()
    assert cam.max_range_for_gsd(0.005) < cam.max_range_for_gsd(0.020)


def test_crack_range_is_short_at_1080p():
    """The physical fact the assessment zones exist to encode.

    A 5 mm/px budget is only met within a few metres of a 1080p dashcam, so cracks
    genuinely cannot be assessed at 25 m. This is a sensor limit, not a model limit.
    """
    z = _cam(w=1920, h=1080, hfov=78.0, height=1.3, pitch=5.0).max_range_for_gsd(0.005)
    assert 1.0 < z < 20.0, f"expected a short crack range, got {z:.1f} m"


def test_higher_resolution_extends_range():
    lo = _cam(w=1280, h=720).max_range_for_gsd(0.005)
    hi = _cam(w=3840, h=2160).max_range_for_gsd(0.005)
    assert hi > lo, "more pixels must buy more range"


def test_visible_range_is_ordered_and_finite_near():
    near, far = _cam().visible_range()
    assert 0 < near < far


# -- per-pixel maps -----------------------------------------------------------

def test_ground_maps_agree_with_the_scalar_projection():
    cam = _cam(w=320, h=180)
    x_map, z_map, valid = cam.ground_maps()
    assert x_map.shape == (180, 320)
    for v in (120, 150, 179):
        u = 160
        scalar = cam.ground_from_pixel(u + 0.5, v + 0.5)
        assert valid[v, u], f"row {v} should be below the horizon"
        assert abs(z_map[v, u] - scalar[1]) < 1e-6
        assert abs(x_map[v, u] - scalar[0]) < 1e-6


def test_ground_maps_mark_sky_invalid():
    cam = _cam(w=320, h=180)
    _, z_map, valid = cam.ground_maps()
    top = int(max(0, cam.horizon_row - 10))
    assert not valid[:top, :].any()
    assert np.isnan(z_map[:top, :]).all()


# -- auto-calibration ---------------------------------------------------------

def test_pitch_recovered_from_a_synthesised_vanishing_point():
    """Round-trip: project the VP the model implies, then recover the pose from it."""
    intr = Intrinsics.from_hfov(1920, 1080, 78.0)
    for true_pitch, true_yaw in [(3.0, 0.0), (8.0, 0.0), (5.0, 4.0), (12.0, -3.0)]:
        cam = CameraModel(intr, Extrinsics(1.3, true_pitch, true_yaw))
        vp_u, vp_v = cam.vanishing_point()
        rec = extrinsics_from_vanishing_point(intr, vp_u, vp_v, 1.3)
        assert abs(rec.pitch_deg - true_pitch) < 1e-6
        assert abs(rec.yaw_deg - true_yaw) < 1e-6


def _road_mask(h=480, w=640, vp_u=320, vp_v=200, half_at_bottom=190):
    """A trapezoid converging on a chosen vanishing point."""
    import cv2

    m = np.zeros((h, w), dtype=np.uint8)
    poly = np.array([
        [vp_u - half_at_bottom, h], [vp_u + half_at_bottom, h],
        [int(vp_u + 0.06 * half_at_bottom), int(vp_v + 0.06 * (h - vp_v))],
        [int(vp_u - 0.06 * half_at_bottom), int(vp_v + 0.06 * (h - vp_v))],
    ], dtype=np.int32)
    cv2.fillPoly(m, [poly], 1)
    return m.astype(bool)


def test_vanishing_point_found_from_road_edges():
    vp = vanishing_point_from_road_mask(_road_mask(vp_u=320, vp_v=200))
    assert vp is not None
    assert abs(vp[0] - 320) < 25, f"u off: {vp[0]}"
    assert abs(vp[1] - 200) < 40, f"v off: {vp[1]}"


def test_vanishing_point_tracks_an_offset_road():
    left = vanishing_point_from_road_mask(_road_mask(vp_u=250, vp_v=200))
    right = vanishing_point_from_road_mask(_road_mask(vp_u=400, vp_v=200))
    assert left and right and right[0] > left[0] + 50


def test_parallel_edges_are_rejected_not_extrapolated():
    """A rectangular 'road' has no intersection; guessing one would be wrong."""
    m = np.zeros((480, 640), dtype=bool)
    m[240:, 200:440] = True
    assert vanishing_point_from_road_mask(m) is None


def test_empty_mask_is_safe():
    assert vanishing_point_from_road_mask(np.zeros((480, 640), dtype=bool)) is None


def test_median_vp_ignores_outlier_frames():
    intr = Intrinsics.from_hfov(640, 480, 78.0)
    masks = [_road_mask(vp_u=320, vp_v=200) for _ in range(8)]
    masks.append(_road_mask(vp_u=120, vp_v=380))     # one bad frame
    vp = estimate_vanishing_point(masks, intr)
    assert vp is not None
    assert abs(vp[0] - 320) < 30, "median should ignore the outlier"


def test_too_few_estimates_returns_none():
    intr = Intrinsics.from_hfov(640, 480, 78.0)
    assert estimate_vanishing_point([np.zeros((480, 640), bool)] * 5, intr) is None


# -- config assembly ----------------------------------------------------------

def test_build_camera_uses_config(cfg):
    cfg.set_path("geometry.camera.height_m", 1.55)
    cfg.set_path("geometry.camera.pitch_deg", 7.0)
    cam = build_camera(cfg, 1920, 1080)
    assert cam.extr.height_m == 1.55
    assert abs(cam.extr.pitch_deg - 7.0) < 1e-9


def test_build_camera_prefers_explicit_intrinsics(cfg):
    cfg.set_path("geometry.camera.fx", 1500.0)
    cfg.set_path("geometry.camera.fy", 1500.0)
    cam = build_camera(cfg, 1920, 1080)
    assert cam.intr.fx == 1500.0


def test_vp_correction_applied_when_plausible(cfg):
    cfg.set_path("geometry.camera.pitch_deg", 5.0)
    intr = Intrinsics.from_hfov(1920, 1080, 78.0)
    target = CameraModel(intr, Extrinsics(1.3, 9.0, 0.0)).vanishing_point()
    cam = build_camera(cfg, 1920, 1080, vp=target)
    assert abs(cam.extr.pitch_deg - 9.0) < 0.1, "VP-derived pitch should be adopted"


def test_wild_vp_correction_is_refused(cfg):
    """A VP implying a 60° pitch on a dashcam means the estimate is broken."""
    cfg.set_path("geometry.camera.pitch_deg", 5.0)
    cfg.set_path("geometry.camera.max_auto_pitch_correction_deg", 10.0)
    intr = Intrinsics.from_hfov(1920, 1080, 78.0)
    absurd = CameraModel(intr, Extrinsics(1.3, 60.0, 0.0)).vanishing_point()
    cam = build_camera(cfg, 1920, 1080, vp=absurd)
    assert abs(cam.extr.pitch_deg - 5.0) < 1e-9, "should keep the configured pitch"


# -- calibrated IPM -----------------------------------------------------------

def test_ipm_scale_is_exact_by_construction():
    """The point of calibrated IPM: metres-per-pixel is known, not guessed."""
    cam = _cam()
    H, mx, mz = cam.ipm_homography(x_half_width_m=3.0, z_near_m=4.0, z_far_m=20.0,
                                   out_w=300, out_h=800)
    assert abs(mx - 6.0 / 300) < 1e-12
    assert abs(mz - 16.0 / 800) < 1e-12
    assert H.shape == (3, 3)


def test_ipm_maps_ground_corners_to_image_corners():
    import cv2

    cam = _cam()
    H, _, _ = cam.ipm_homography(3.0, 4.0, 20.0, 300, 800)
    u, v = cam.pixel_from_ground(-3.0, 20.0)
    out = cv2.perspectiveTransform(np.array([[[u, v]]], dtype=np.float32), H)[0][0]
    assert abs(out[0]) < 1.0 and abs(out[1]) < 1.0, f"far-left corner -> {out}"


def test_ipm_rejects_a_bad_range():
    with pytest.raises(ValueError):
        _cam().ipm_homography(3.0, 20.0, 4.0, 300, 800)
