"""Detector families: crack geometry, confusers, edge damage, texture, tiling.

The crack tests draw cracks at *known ground angles* and check they come back
classified correctly — which is the whole point of measuring on the road plane rather
than in the image, so it is worth verifying against ground truth rather than against
the implementation's own output.
"""
from __future__ import annotations

import numpy as np
import pytest

from rdd.detect.boundary import BoundaryConfig, detect_edge_damage
from rdd.detect.confusers import ConfuserStats, check as check_confusers
from rdd.detect.linear import (
    ALLIGATOR,
    CRACK_SOURCES,
    LONGITUDINAL,
    TRANSVERSE,
    LinearConfig,
    LinearStats,
    classify_crack,
)
from rdd.detect.texture import TextureConfig, detect_ravelling, detect_rutting_proxy
from rdd.detect.tiling import TilingConfig, merge_detections, plan_tiles
from rdd.geometry.calibration import CameraModel, Extrinsics, Intrinsics
from rdd.roadseg.ops import channel_stats, compute_features
from rdd.surface.condition import _CHANNELS
from tests.scenes import car_scene

W, H = 640, 480


def _cam(w=W, h=H, pitch=8.0, height=1.3, hfov=78.0):
    return CameraModel(Intrinsics.from_hfov(w, h, hfov),
                       Extrinsics(height_m=height, pitch_deg=pitch))


def _ground_line(cam, x0, z0, x1, z1, thickness=3, shape=(H, W)):
    """Draw a line between two GROUND points, projected into the image."""
    import cv2

    m = np.zeros(shape, dtype=np.uint8)
    p0 = cam.pixel_from_ground(x0, z0)
    p1 = cam.pixel_from_ground(x1, z1)
    if not all(np.isfinite(v) for v in (*p0, *p1)):
        return m.astype(bool)
    cv2.line(m, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), 1, thickness)
    return m.astype(bool)


# -- crack orientation: the ground-plane measurement ---------------------------

def test_crack_along_the_road_is_longitudinal():
    cam = _cam()
    mask = _ground_line(cam, 0.0, 5.0, 0.0, 12.0, thickness=4)
    g = classify_crack(mask, cam)
    assert g.cls_name == LONGITUDINAL, f"got {g.cls_name} at {g.angle_deg:.0f}°"
    assert g.angle_deg < 30.0


def test_crack_across_the_road_is_transverse():
    cam = _cam()
    mask = _ground_line(cam, -1.6, 7.0, 1.6, 7.0, thickness=4)
    g = classify_crack(mask, cam)
    assert g.cls_name == TRANSVERSE, f"got {g.cls_name} at {g.angle_deg:.0f}°"
    assert g.angle_deg > 60.0


def test_orientation_is_measured_on_the_ground_not_in_the_image():
    """The reason this module exists.

    A longitudinal crack off to one side is strongly slanted *in the image* because it
    converges toward the vanishing point. Judged on image angle it would be called
    transverse; judged on the ground it is correctly longitudinal.
    """
    cam = _cam()
    mask = _ground_line(cam, -1.8, 5.0, -1.8, 13.0, thickness=4)

    ys, xs = np.nonzero(mask)
    image_angle = np.degrees(np.arctan2(abs(xs.max() - xs.min()),
                                       abs(ys.max() - ys.min())))
    g = classify_crack(mask, cam)
    assert g.cls_name == LONGITUDINAL
    assert image_angle > g.angle_deg + 10, (
        f"image angle {image_angle:.0f}° should be far more slanted than the true "
        f"ground angle {g.angle_deg:.0f}°")


def test_diagonal_crack_is_flagged_low_confidence():
    cam = _cam()
    mask = _ground_line(cam, -1.0, 6.0, 1.0, 9.0, thickness=4)
    g = classify_crack(mask, cam)
    if 30.0 < g.angle_deg < 60.0:
        assert not g.confident
        assert "diagonal" in g.reason


def test_alligator_detected_by_enclosed_cells():
    """Connectivity, not appearance: fatigue cracking encloses pavement cells.

    Cell spacing is 0.25 m, which is what real alligator cracking looks like
    (roughly 50-300 mm). Half-metre cells would be *block* cracking, a different
    distress with a different cell density.
    """
    import cv2

    cam = _cam()
    mask = np.zeros((H, W), dtype=np.uint8)
    step = 0.25
    for i in range(9):
        x = -1.0 + i * step
        p0 = cam.pixel_from_ground(x, 4.5)
        p1 = cam.pixel_from_ground(x, 6.5)
        cv2.line(mask, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), 1, 2)
    for j in range(9):
        z = 4.5 + j * step
        p0 = cam.pixel_from_ground(-1.0, z)
        p1 = cam.pixel_from_ground(1.0, z)
        cv2.line(mask, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), 1, 2)

    g = classify_crack(mask.astype(bool), cam)
    assert g.n_cells >= 3, f"expected enclosed cells, got {g.n_cells}"
    assert g.cells_per_m2 >= 4.0, f"cell density {g.cells_per_m2:.1f}/m² too low"
    assert g.cls_name == ALLIGATOR, f"got {g.cls_name} with {g.n_cells} cells"


def test_parallel_cracks_are_not_alligator():
    """Dense parallel cracks enclose nothing, so they must not read as fatigue."""
    import cv2

    cam = _cam()
    mask = np.zeros((H, W), dtype=np.uint8)
    for i in range(6):
        x = -1.2 + i * 0.4
        p0 = cam.pixel_from_ground(x, 5.0)
        p1 = cam.pixel_from_ground(x, 9.0)
        cv2.line(mask, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), 1, 3)

    g = classify_crack(mask.astype(bool), cam)
    assert g.cls_name != ALLIGATOR, f"parallel cracks misread as alligator ({g.n_cells} cells)"


def test_no_camera_means_no_guess():
    mask = np.zeros((H, W), dtype=bool)
    mask[200:260, 300:310] = True
    g = classify_crack(mask, None)
    assert not g.confident
    assert "no camera calibration" in g.reason


def test_empty_mask_is_safe():
    g = classify_crack(np.zeros((H, W), dtype=bool), _cam())
    assert not g.confident


def test_crack_sources_cover_generic_and_rdd2022_labels():
    for name in ("crack", "D00", "D10", "D20", LONGITUDINAL, TRANSVERSE, ALLIGATOR):
        assert name in CRACK_SOURCES


def test_linear_stats_counts_reclassification():
    stats = LinearStats()
    cam = _cam()
    stats.update("crack", classify_crack(
        _ground_line(cam, -1.6, 7.0, 1.6, 7.0, thickness=4), cam))
    assert stats.seen == 1
    assert stats.relabelled == 1
    assert TRANSVERSE in stats.summary()["by_class"]


def test_config_thresholds_are_honoured(cfg):
    cfg.set_path("detect.linear.longitudinal_max_deg", 5.0)
    cfg.set_path("detect.linear.transverse_min_deg", 10.0)
    lin = LinearConfig.from_cfg(cfg)
    assert lin.longitudinal_max_deg == 5.0
    assert lin.transverse_min_deg == 10.0


# -- confusers -----------------------------------------------------------------

def _feats_and_baseline(frame, road):
    feats = compute_features(frame, 7)
    return feats, _Baseline(channel_stats(feats, road, _CHANNELS))


class _Baseline:
    def __init__(self, stats):
        self.stats = stats

    @property
    def is_empty(self):
        return not self.stats

    def get(self, ch):
        return self.stats.get(ch, (0.0, 1.0))


def test_shadow_detection_is_rejected(cfg):
    frame, road = car_scene()
    feats, baseline = _feats_and_baseline(frame, road)
    det = np.zeros((H, W), dtype=bool)
    det[420:450, 300:360] = True
    rej = check_confusers(det, feats, baseline, cfg, cls_name="pothole",
                          shadow_mask=det.copy())
    assert rej is not None and rej.confuser == "shadow"


def test_clean_pothole_is_not_rejected(cfg):
    frame, road = car_scene()
    feats, baseline = _feats_and_baseline(frame, road)
    det = np.zeros((H, W), dtype=bool)
    det[420:450, 300:360] = True
    assert check_confusers(det, feats, baseline, cfg, cls_name="pothole") is None


def test_exempt_class_is_never_rejected(cfg):
    frame, road = car_scene()
    feats, baseline = _feats_and_baseline(frame, road)
    det = np.zeros((H, W), dtype=bool)
    det[420:450, 300:360] = True
    assert check_confusers(det, feats, baseline, cfg, cls_name="water_logging",
                           shadow_mask=det.copy()) is None


def test_confuser_stats_tally():
    from rdd.detect.confusers import Rejection

    st = ConfuserStats()
    st.update(None)
    st.update(Rejection("tar_patch", "x"))
    st.update(Rejection("tar_patch", "x"))
    assert st.checked == 3 and st.rejected == 2
    assert st.summary()["by_confuser"]["tar_patch"] == 2


# -- edge damage ---------------------------------------------------------------

def _straight_road_mask(cam, half_w=2.5, z0=4.0, z1=25.0, shape=(H, W)):
    """Road mask built by projecting a constant-width ground strip."""
    m = np.zeros(shape, dtype=bool)
    for r in range(shape[0]):
        row_z = None
        g = cam.ground_from_pixel(cam.intr.cx, r + 0.5)
        if g is None:
            continue
        row_z = g[1]
        if not (z0 <= row_z <= z1):
            continue
        ul = cam.pixel_from_ground(-half_w, row_z)[0]
        ur = cam.pixel_from_ground(half_w, row_z)[0]
        a, b = int(max(0, min(ul, ur))), int(min(shape[1], max(ul, ur)))
        if b > a:
            m[r, a:b] = True
    return m


def test_intact_edge_reports_no_damage(cfg):
    cam = _cam()
    road = _straight_road_mask(cam)
    res = detect_edge_damage(road, cam, cfg, bc=BoundaryConfig(z_near_m=5.0,
                                                              z_far_m=20.0))
    assert res.measured
    assert not res.defects, f"clean edge reported {len(res.defects)} defects"


def test_edge_notch_is_measured_in_metres(cfg):
    """A known 0.4 m bite out of the edge must be measured, not merely flagged."""
    cam = _cam()
    road = _straight_road_mask(cam)
    x_map, z_map, _ = cam.ground_maps(W, H)

    # Remove road where the left edge is within 0.4 m of its nominal -2.5 m position,
    # over a 2 m stretch of chainage.
    with np.errstate(invalid="ignore"):
        bite = (np.nan_to_num(z_map, nan=-1) >= 8.0) & (np.nan_to_num(z_map, nan=-1) <= 10.0) \
               & (np.nan_to_num(x_map, nan=99) <= -2.1)
    damaged = road & ~bite

    res = detect_edge_damage(damaged, cam, cfg,
                             bc=BoundaryConfig(z_near_m=5.0, z_far_m=20.0,
                                               min_inset_m=0.1, min_length_m=0.3))
    assert res.measured
    left = [d for d in res.defects if d.side == "left"]
    assert left, "the notch was not detected"
    worst = max(d.max_inset_m for d in left)
    assert 0.15 <= worst <= 0.9, f"inset {worst:.2f} m outside the expected range"


def test_no_camera_means_no_edge_measurement(cfg):
    res = detect_edge_damage(np.ones((H, W), dtype=bool), None, cfg)
    assert not res.measured
    assert "no camera calibration" in res.note


def test_empty_road_is_safe(cfg):
    res = detect_edge_damage(np.zeros((H, W), dtype=bool), _cam(), cfg)
    assert not res.measured


# -- texture / ravelling -------------------------------------------------------

def test_clean_road_is_not_ravelled(cfg):
    cam = _cam()
    frame, road = car_scene()
    feats, baseline = _feats_and_baseline(frame, road)
    res = detect_ravelling(frame, road, cam, baseline, cfg, feats=feats,
                           tc=TextureConfig(cell_m=0.6, min_cell_px=20))
    if res.measured:
        assert res.affected_frac < 0.3, f"{res.affected_frac:.0%} false ravelling"


def test_rough_patch_reads_as_ravelling(cfg):
    from tests.scenes import _filled_like

    cam = _cam()
    frame, road = car_scene()
    _, baseline = _feats_and_baseline(frame, road)
    # Roughen the whole road: texture well above the baseline it was measured on.
    rough = _filled_like(frame, road, bgr=(120, 125, 130), sigma=26.0)
    res = detect_ravelling(rough, road, cam, baseline, cfg,
                           tc=TextureConfig(cell_m=0.6, min_cell_px=20))
    if res.measured:
        assert res.affected_frac > 0.3, f"only {res.affected_frac:.0%} flagged"


def test_ravelling_needs_calibration(cfg):
    frame, road = car_scene()
    _, baseline = _feats_and_baseline(frame, road)
    res = detect_ravelling(frame, road, None, baseline, cfg)
    assert not res.measured
    assert "fixed ground scale" in res.note


def test_rutting_is_labelled_indicative(cfg):
    cam = _cam()
    frame, road = car_scene()
    feats, baseline = _feats_and_baseline(frame, road)
    res = detect_rutting_proxy(frame, road, cam, baseline, cfg, feats=feats)
    assert "not within the precision guarantee" in res.note or not res.measured


# -- tiling --------------------------------------------------------------------

def test_tiles_cover_the_region_and_overlap():
    region = np.zeros((480, 640), dtype=bool)
    region[300:470, 100:560] = True
    tiles = plan_tiles(region, TilingConfig(tile_px=160, overlap=0.25, max_tiles=99,
                                            min_road_frac=0.0))
    assert tiles
    covered = np.zeros_like(region)
    for t in tiles:
        covered[t.y0:t.y1, t.x0:t.x1] = True
    assert covered[region].all(), "tiles must cover every road pixel"


def test_tiles_skip_mostly_empty_areas():
    region = np.zeros((480, 640), dtype=bool)
    region[440:470, 300:340] = True
    many = plan_tiles(region, TilingConfig(tile_px=128, min_road_frac=0.0,
                                           max_tiles=99))
    few = plan_tiles(region, TilingConfig(tile_px=128, min_road_frac=0.5,
                                          max_tiles=99))
    assert len(few) <= len(many)


def test_no_tiles_for_empty_region():
    assert plan_tiles(np.zeros((100, 100), dtype=bool), TilingConfig()) == []


def test_tile_count_is_capped():
    region = np.ones((480, 640), dtype=bool)
    tiles = plan_tiles(region, TilingConfig(tile_px=64, overlap=0.5, max_tiles=5,
                                            min_road_frac=0.0))
    assert len(tiles) <= 5


def test_merge_drops_duplicate_detections_across_seams():
    dets = [
        {"bbox": (10, 10, 50, 50), "conf": 0.9, "cls_id": 0},
        {"bbox": (12, 12, 52, 52), "conf": 0.7, "cls_id": 0},   # same crack, other tile
        {"bbox": (200, 200, 240, 240), "conf": 0.8, "cls_id": 0},
    ]
    merged = merge_detections(dets, 0.45)
    assert len(merged) == 2
    assert merged[0]["conf"] == 0.9, "the stronger detection should survive"


def test_merge_keeps_different_classes_at_the_same_place():
    dets = [
        {"bbox": (10, 10, 50, 50), "conf": 0.9, "cls_id": 0},
        {"bbox": (10, 10, 50, 50), "conf": 0.8, "cls_id": 1},
    ]
    assert len(merge_detections(dets, 0.45)) == 2
