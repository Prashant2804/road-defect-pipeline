"""Viewpoint scale, reprojection sizing, IPM area mapping, and GPS indexing."""
from __future__ import annotations

import math

import numpy as np
import pytest

from rdd.preprocess.ipm import build_transform
from rdd.preprocess.reproject import native_angular_width, plan_output_size
from rdd.preprocess.scale import NoScale, PerPixelScaler, UniformScaler, build_area_scaler
from rdd.utils.geo import GpsFix, GpsTrack, haversine_m
from rdd.viewpoint import gsd_m_per_px, resolve_view


# -- reprojection sizing ------------------------------------------------------

def test_native_angular_width_is_the_source_share_of_the_sphere():
    assert native_angular_width(5760, 110.0) == 1760
    assert native_angular_width(5760, 360.0) == 5760


def test_auto_width_preserves_source_angular_detail():
    """The bug this replaced: a hard-coded 1280 threw away a quarter of the detail."""
    w, h, note = plan_output_size(
        {"h_fov_deg": 110.0, "v_fov_deg": 70.0, "out_width": "auto",
         "min_width": 960, "max_width": 3840, "preserve_aspect": True},
        src_w=5760, src_h=2880,
    )
    assert w == 1760, f"expected the native 1760 px, got {w}"
    assert "auto width" in note


def test_auto_width_is_clamped():
    w, _, note = plan_output_size(
        {"h_fov_deg": 110.0, "v_fov_deg": 70.0, "out_width": "auto",
         "min_width": 960, "max_width": 1200}, src_w=5760, src_h=2880)
    assert w == 1200
    assert "clamped" in note


def test_explicit_width_below_native_is_warned_about():
    _, _, note = plan_output_size(
        {"h_fov_deg": 110.0, "v_fov_deg": 70.0, "out_width": 1280,
         "preserve_aspect": True}, src_w=5760, src_h=2880)
    assert "WARNING" in note and "discards detail" in note


def test_preserve_aspect_gives_square_pixels():
    """A gnomonic view has square pixels only at w/h == tan(hfov/2)/tan(vfov/2)."""
    h_fov, v_fov = 110.0, 70.0
    w, h, _ = plan_output_size(
        {"h_fov_deg": h_fov, "v_fov_deg": v_fov, "out_width": 1760,
         "preserve_aspect": True}, src_w=5760, src_h=2880)

    want = math.tan(math.radians(h_fov) / 2) / math.tan(math.radians(v_fov) / 2)
    assert abs((w / h) - want) / want < 0.01, f"aspect {w/h:.3f} should be {want:.3f}"


def test_the_old_default_was_geometrically_stretched():
    """Documents the defect: 1280x720 at 110/70 deg was never square-pixel."""
    want = math.tan(math.radians(110.0) / 2) / math.tan(math.radians(70.0) / 2)
    assert abs((1280 / 720) - want) / want > 0.10


def test_output_dimensions_are_even_for_h264():
    w, h, _ = plan_output_size(
        {"h_fov_deg": 97.0, "v_fov_deg": 61.0, "out_width": "auto",
         "min_width": 100, "max_width": 9999}, src_w=3333, src_h=1666)
    assert w % 2 == 0 and h % 2 == 0


def test_missing_source_size_falls_back_without_crashing():
    w, h, note = plan_output_size({"h_fov_deg": 110.0, "v_fov_deg": 70.0,
                                   "out_width": "auto"}, src_w=None, src_h=None)
    assert w == 1920 and h > 0
    assert "source width unknown" in note


# -- drone ground sample distance ---------------------------------------------

def test_gsd_matches_the_similar_triangles_formula():
    # 60 m up, 24 mm sensor, 12 mm lens, 4000 px wide -> 120 m swath / 4000 px.
    g = gsd_m_per_px(altitude_m=60.0, focal_mm=12.0,
                     sensor_width_mm=24.0, image_width_px=4000)
    assert abs(g - 0.03) < 1e-9


def test_gsd_scales_linearly_with_altitude():
    a = gsd_m_per_px(50.0, 12.0, 24.0, 4000)
    b = gsd_m_per_px(100.0, 12.0, 24.0, 4000)
    assert abs(b - 2 * a) < 1e-12


def test_gsd_rejects_nonsense_inputs():
    with pytest.raises(ValueError):
        gsd_m_per_px(0.0, 12.0, 24.0, 4000)


def test_drone_view_resolves_scale_from_intrinsics(cfg):
    cfg.set_path("view.profile", "drone_nadir")
    cfg.set_path("view.drone.altitude_m", 60.0)
    cfg.set_path("view.drone.camera.focal_mm", 12.0)
    cfg.set_path("view.drone.camera.sensor_width_mm", 24.0)

    view = resolve_view(cfg, 4000, 3000)
    assert view.has_scale
    assert abs(view.m_per_px - 0.03) < 1e-9
    # 100x100 px defect at 3 cm/px = 3 m x 3 m = 9 m².
    assert abs(view.px_to_m2(100 * 100) - 9.0) < 1e-6


def test_drone_view_without_intrinsics_has_no_scale(cfg):
    cfg.set_path("view.profile", "drone_nadir")
    view = resolve_view(cfg, 4000, 3000)
    assert not view.has_scale
    assert view.px_to_m2(1000) is None
    assert any("scale: unknown" in n for n in view.notes)


def test_explicit_gsd_overrides_intrinsics(cfg):
    cfg.set_path("view.profile", "drone_nadir")
    cfg.set_path("view.drone.gsd_m_per_px", 0.01)
    cfg.set_path("view.drone.altitude_m", 60.0)
    view = resolve_view(cfg, 4000, 3000)
    assert abs(view.m_per_px - 0.01) < 1e-12


# -- IPM area mapping ---------------------------------------------------------

def test_ipm_disabled_returns_none(cfg):
    assert build_transform(cfg, 1280, 720) is None


def _enable_ipm(cfg, extent=(6.0, 20.0)):
    cfg.set_path("preprocess.ipm.enabled", True)
    cfg.set_path("preprocess.ipm.ground_extent_m", list(extent))
    return cfg


def test_ipm_identity_homography_gives_uniform_known_scale(cfg):
    """With src == dst the map is affine, so ground area per pixel is constant."""
    _enable_ipm(cfg, extent=(6.4, 4.8))
    cfg.set_path("preprocess.ipm.out_width", 64)
    cfg.set_path("preprocess.ipm.out_height", 48)
    cfg.set_path("preprocess.ipm.src_points", [[0, 0], [1, 0], [1, 1], [0, 1]])
    cfg.set_path("preprocess.ipm.dst_points", [[0, 0], [1, 0], [1, 1], [0, 1]])

    t = build_transform(cfg, 64, 48)
    amap = t.area_map(64, 48)
    assert amap is not None
    # 6.4 m / 64 px = 0.1 m/px each way -> 0.01 m² per pixel, everywhere.
    assert np.allclose(amap, 0.01, rtol=1e-5), f"got {amap.min()}..{amap.max()}"

    full = PerPixelScaler(amap).area_m2(np.ones((48, 64), dtype=bool))
    assert abs(full - 6.4 * 4.8) < 1e-3


def test_ipm_perspective_scale_grows_with_distance(cfg):
    """The reason pixel area is not a defect size in a perspective view."""
    _enable_ipm(cfg)
    t = build_transform(cfg, 1280, 720)
    amap = t.area_map(1280, 720)
    assert amap is not None

    near = amap[700, 640]     # bottom of frame: close to the camera
    far = amap[420, 640]      # near the horizon
    assert far > near * 2, (
        f"far pixels must cover more ground than near ones ({far:.3g} vs {near:.3g})"
    )


def test_ipm_without_ground_extent_has_no_area_map(cfg):
    cfg.set_path("preprocess.ipm.enabled", True)
    t = build_transform(cfg, 1280, 720)
    assert t is not None and not t.has_scale
    assert t.area_map(1280, 720) is None


def test_degenerate_ipm_points_are_rejected(cfg):
    _enable_ipm(cfg)
    cfg.set_path("preprocess.ipm.src_points",
                 [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    with pytest.raises(ValueError):
        build_transform(cfg, 1280, 720)


def test_malformed_ipm_points_are_rejected(cfg):
    _enable_ipm(cfg)
    cfg.set_path("preprocess.ipm.src_points", [[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError):
        build_transform(cfg, 1280, 720)


# -- scaler selection ---------------------------------------------------------

def test_scaler_is_uniform_for_a_calibrated_drone(cfg):
    cfg.set_path("view.profile", "drone_nadir")
    cfg.set_path("view.drone.gsd_m_per_px", 0.02)
    view = resolve_view(cfg, 4000, 3000)
    scaler = build_area_scaler(cfg, view, 4000, 3000)
    assert isinstance(scaler, UniformScaler)
    assert abs(scaler.area_m2(np.ones((10, 10), dtype=bool)) - 100 * 0.0004) < 1e-9


def test_scaler_is_none_when_nothing_is_configured(cfg):
    view = resolve_view(cfg, 1280, 720)
    scaler = build_area_scaler(cfg, view, 1280, 720)
    assert isinstance(scaler, NoScale)
    assert scaler.area_m2(np.ones((4, 4), dtype=bool)) is None
    assert not scaler.has_scale


def test_scaler_is_perspective_when_ipm_is_calibrated(cfg):
    _enable_ipm(cfg)
    view = resolve_view(cfg, 1280, 720)
    scaler = build_area_scaler(cfg, view, 1280, 720)
    assert isinstance(scaler, PerPixelScaler)
    assert scaler.has_scale


def test_perspective_scaler_rejects_a_mismatched_mask(cfg):
    _enable_ipm(cfg)
    scaler = build_area_scaler(cfg, resolve_view(cfg, 1280, 720), 1280, 720)
    assert scaler.area_m2(np.ones((10, 10), dtype=bool)) is None


# -- GPS indexing -------------------------------------------------------------

def _track():
    # ~1 second apart along a line of latitude.
    return GpsTrack([GpsFix(t=float(i), lat=12.0 + i * 1e-4, lon=77.0)
                     for i in range(10)])


def test_nearest_fix_lookup_picks_the_closest_in_time():
    tr = _track()
    assert tr.at_time(0.0).t == 0.0
    assert tr.at_time(4.4).t == 4.0
    assert tr.at_time(4.6).t == 5.0
    assert tr.at_time(-100.0).t == 0.0, "clamp before the start"
    assert tr.at_time(1e6).t == 9.0, "clamp after the end"


def test_cumulative_distance_is_monotonic_and_cached():
    tr = _track()
    cum = tr.cumulative_distance_m()
    assert cum[0] == 0.0
    assert all(b >= a for a, b in zip(cum, cum[1:]))
    assert tr.cumulative_distance_m() == cum
    assert tr._cum is not None, "should be cached, not recomputed per call"


def test_distance_at_time_matches_the_cumulative_table():
    tr = _track()
    cum = tr.cumulative_distance_m()
    assert abs(tr.distance_at_time(5.0) - cum[5]) < 1e-9
    assert abs(tr.total_distance_m - cum[-1]) < 1e-9


def test_out_of_order_fixes_are_sorted_on_indexing():
    tr = GpsTrack([GpsFix(t=5.0, lat=12.0, lon=77.0),
                   GpsFix(t=1.0, lat=12.1, lon=77.0),
                   GpsFix(t=3.0, lat=12.2, lon=77.0)])
    assert tr.at_time(1.1).t == 1.0
    assert [f.t for f in tr.fixes] == [1.0, 3.0, 5.0]


def test_empty_track_is_safe():
    tr = GpsTrack()
    assert not tr.has_data
    assert tr.at_time(1.0) is None
    assert tr.cumulative_distance_m() == []
    assert tr.total_distance_m == 0.0
    assert tr.distance_at_time(1.0) is None


def test_haversine_against_a_known_separation():
    # 0.001 deg of latitude is ~111.2 m anywhere on the globe.
    assert abs(haversine_m(12.0, 77.0, 12.001, 77.0) - 111.19) < 0.5
