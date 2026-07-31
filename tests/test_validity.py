"""Assessment zones and the validity gates.

The gate tests are the ones that matter most for the stated requirement: when the
road is buried, unlocatable, or the vehicle is off the carriageway, the pipeline must
produce *no detections* and say why. Each gate is exercised in isolation with a
hand-built context so a failure names the responsible gate.
"""
from __future__ import annotations

import numpy as np
import pytest

from rdd.geometry.calibration import CameraModel, Extrinsics, Intrinsics
from rdd.geometry.zones import build_zones
from rdd.validity.egomotion import EgoMotion
from rdd.validity.gates import (
    FrameContext,
    gate_assessment_zone,
    gate_ego_motion,
    gate_image_condition,
    gate_off_track,
    gate_road_buried,
    gate_road_found,
    gate_surface_plausible,
    gate_traffic,
    gate_windscreen,
)
from rdd.validity.traffic import TrafficResult
from rdd.validity.verdict import Action, FrameVerdict, GateResult, ValidityStats

W, H = 640, 480


def _cam(pitch=5.0, height=1.3, w=W, h=H, hfov=78.0):
    return CameraModel(Intrinsics.from_hfov(w, h, hfov),
                       Extrinsics(height_m=height, pitch_deg=pitch))


class _Road:
    """Minimal RoadMask stand-in."""

    def __init__(self, mask, confidence=0.9, fell_back=False, baseline=None):
        self.mask = mask
        self.prior = mask.copy()
        self.confidence = confidence
        self.fell_back = fell_back
        self.baseline = baseline

    def coverage(self):
        return float(self.mask.sum()) / float(self.mask.size)


class _Surface:
    def __init__(self, occluded=0.0, water=0.0, mud=0.0, shape=(H, W)):
        self.occluded_frac = occluded
        self.water_frac = water
        self.mud_frac = mud
        self.occlusion = np.zeros(shape, dtype=bool)


class _Quality:
    def __init__(self, clipped_high=0.0, mean_luma=0.5, usable=True, reasons=()):
        self.clipped_high = clipped_high
        self.mean_luma = mean_luma
        self.usable = usable
        self.reasons = tuple(reasons)


def _good_road(w=W, h=H):
    """A centred trapezoid reaching the bottom of the frame."""
    import cv2

    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m, [np.array([
        [int(0.05 * w), h], [int(0.95 * w), h],
        [int(0.58 * w), int(0.55 * h)], [int(0.42 * w), int(0.55 * h)],
    ], dtype=np.int32)], 1)
    return m.astype(bool)


# -- assessment zones ---------------------------------------------------------

def test_tighter_gsd_requirement_gives_a_shorter_zone(cfg):
    cfg.set_path("model.classes", ["pothole", "longitudinal_crack"])
    zones = build_zones(cfg, _cam())
    pothole = zones.for_class("pothole")
    crack = zones.for_class("longitudinal_crack")
    assert crack.required_gsd_m < pothole.required_gsd_m
    assert crack.z_far_m < pothole.z_far_m, "cracks must be assessed closer in"


def test_zone_mask_covers_only_its_distance_band(cfg):
    cfg.set_path("model.classes", ["pothole"])
    cam = _cam()
    zones = build_zones(cfg, cam)
    zone = zones.for_class("pothole")
    mask = zones.mask("pothole", W, H)

    assert mask.any(), "zone mask should not be empty"
    _, z_map, valid = cam.ground_maps(W, H)
    inside = z_map[mask]
    assert inside.min() >= zone.z_near_m - 1e-6
    assert inside.max() <= zone.z_far_m + 1e-6
    assert not mask[: int(cam.horizon_row) - 5, :].any(), "sky is never in a zone"


def test_zone_mask_is_cached(cfg):
    cfg.set_path("model.classes", ["pothole"])
    zones = build_zones(cfg, _cam())
    assert zones.mask("pothole", W, H) is zones.mask("pothole", W, H)


def test_unachievable_class_is_flagged_not_silently_shrunk(cfg):
    """A budget this camera cannot meet anywhere must be reported, not fudged."""
    cfg.set_path("model.classes", ["hairline"])
    cfg.set_path("geometry.zones.required_gsd_m", {"hairline": 0.00002})
    zones = build_zones(cfg, _cam())
    assert not zones.for_class("hairline").achievable
    assert "hairline" in zones.unachievable()
    assert not zones.mask("hairline", W, H).any(), "no pixels for an impossible class"


def test_union_zone_spans_all_achievable_classes(cfg):
    cfg.set_path("model.classes", ["pothole", "longitudinal_crack"])
    zones = build_zones(cfg, _cam())
    union = zones.widest()
    assert union.z_far_m == max(z.z_far_m for z in zones.zones.values())


def test_higher_resolution_camera_extends_zones(cfg):
    cfg.set_path("model.classes", ["longitudinal_crack"])
    lo = build_zones(cfg, _cam(w=1280, h=720)).for_class("longitudinal_crack")
    hi = build_zones(cfg, _cam(w=3840, h=2160)).for_class("longitudinal_crack")
    assert hi.z_far_m > lo.z_far_m


# -- road availability gates --------------------------------------------------

def test_low_confidence_road_blocks(cfg):
    ctx = FrameContext(road=_Road(_good_road(), confidence=0.05))
    res = gate_road_found(ctx, cfg)
    assert res is not None and res.action is Action.BLOCK
    assert "not locatable" in res.reason


def test_missing_road_mask_blocks(cfg):
    res = gate_road_found(FrameContext(road=None), cfg)
    assert res.action is Action.BLOCK


def test_tiny_road_region_blocks(cfg):
    m = np.zeros((H, W), dtype=bool)
    m[-4:, :6] = True
    res = gate_road_found(FrameContext(road=_Road(m, confidence=0.9)), cfg)
    assert res is not None and res.action is Action.BLOCK
    assert "implausibly small" in res.reason


def test_prior_fallback_degrades_but_does_not_block(cfg):
    ctx = FrameContext(road=_Road(_good_road(), confidence=0.4, fell_back=True))
    res = gate_road_found(ctx, cfg)
    assert res is not None and res.action is Action.DEGRADE


def test_healthy_road_passes(cfg):
    assert gate_road_found(FrameContext(road=_Road(_good_road())), cfg) is None


def test_flooded_road_blocks(cfg):
    """The headline requirement: water over the road means no detection at all."""
    ctx = FrameContext(road=_Road(_good_road()),
                       surface=_Surface(occluded=0.85, water=0.85))
    res = gate_road_buried(ctx, cfg)
    assert res is not None and res.action is Action.BLOCK
    assert "cannot be inspected" in res.reason


def test_mud_covered_road_blocks(cfg):
    ctx = FrameContext(road=_Road(_good_road()),
                       surface=_Surface(occluded=0.7, mud=0.7))
    assert gate_road_buried(ctx, cfg).action is Action.BLOCK


def test_partial_occlusion_degrades(cfg):
    ctx = FrameContext(road=_Road(_good_road()),
                       surface=_Surface(occluded=0.35, water=0.35))
    assert gate_road_buried(ctx, cfg).action is Action.DEGRADE


def test_mostly_dry_road_passes(cfg):
    ctx = FrameContext(road=_Road(_good_road()), surface=_Surface(occluded=0.05))
    assert gate_road_buried(ctx, cfg) is None


# -- off-track ----------------------------------------------------------------

def test_road_not_reaching_the_vehicle_blocks(cfg):
    """If we are on a road, the ground just ahead is road. If not, we are off it."""
    m = _good_road()
    m[int(0.80 * H):, :] = False          # road stops well short of the bonnet
    res = gate_off_track(FrameContext(road=_Road(m)), cfg)
    assert res is not None and res.action is Action.BLOCK
    assert "off the carriageway" in res.reason


def test_badly_offset_road_blocks(cfg):
    m = np.zeros((H, W), dtype=bool)
    m[int(0.5 * H):, : int(0.16 * W)] = True   # road hugging the left edge
    res = gate_off_track(FrameContext(road=_Road(m)), cfg)
    assert res is not None and res.action is Action.BLOCK
    assert "off centre" in res.reason


def test_centred_road_passes_off_track(cfg):
    assert gate_off_track(FrameContext(road=_Road(_good_road())), cfg) is None


# -- ego-motion ---------------------------------------------------------------

def test_stationary_vehicle_blocks(cfg):
    ego = EgoMotion(valid=True, flow_px=0.05, radial=0.02)
    res = gate_ego_motion(FrameContext(ego=ego), cfg)
    assert res is not None and res.action is Action.BLOCK
    assert "stationary" in res.reason


def test_reversing_blocks(cfg):
    """Contracting flow means reverse; magnitude alone could not tell."""
    ego = EgoMotion(valid=True, flow_px=4.0, radial=-4.0)
    res = gate_ego_motion(FrameContext(ego=ego), cfg)
    assert res is not None and res.action is Action.BLOCK
    assert "reversing" in res.reason


def test_forward_motion_passes(cfg):
    ego = EgoMotion(valid=True, flow_px=6.0, radial=6.0, yaw_px=1.0,
                    pitch_px=0.5)
    assert gate_ego_motion(FrameContext(ego=ego), cfg) is None


def test_sharp_turn_degrades(cfg):
    ego = EgoMotion(valid=True, flow_px=6.0, radial=6.0, yaw_px=40.0)
    assert gate_ego_motion(FrameContext(ego=ego), cfg).action is Action.DEGRADE


def test_camera_pitching_degrades(cfg):
    ego = EgoMotion(valid=True, flow_px=6.0, radial=6.0, pitch_px=25.0)
    res = gate_ego_motion(FrameContext(ego=ego), cfg)
    assert res.action is Action.DEGRADE
    assert "pitching" in res.reason


def test_smooth_forward_motion_is_not_called_shaky(cfg):
    """Vibration must be measured at the horizon, not as the spatial spread of
    vertical flow — the latter is always large under plain forward motion."""
    ego = EgoMotion(valid=True, flow_px=12.0, radial=12.0, yaw_px=0.5, pitch_px=0.3)
    assert gate_ego_motion(FrameContext(ego=ego), cfg) is None


def test_invalid_egomotion_is_ignored(cfg):
    assert gate_ego_motion(FrameContext(ego=EgoMotion(valid=False)), cfg) is None


# -- traffic ------------------------------------------------------------------

def test_vehicle_masks_rather_than_blocks(cfg):
    """A car in one corner costs a region, not the whole frame."""
    m = np.zeros((H, W), dtype=bool)
    m[100:200, 100:200] = True
    tr = TrafficResult(available=True, n_detections=1, occluded_frac=0.1,
                       mask=m, labels=["car"])
    res = gate_traffic(FrameContext(traffic=tr), cfg)
    assert res is not None and res.action is Action.MASK
    assert res.mask is not None and res.mask.any()


def test_lane_filling_traffic_blocks(cfg):
    m = np.ones((H, W), dtype=bool)
    tr = TrafficResult(available=True, n_detections=2, occluded_frac=0.8,
                       mask=m, labels=["truck", "car"])
    res = gate_traffic(FrameContext(traffic=tr), cfg)
    assert res is not None and res.action is Action.BLOCK
    assert "assessment zone" in res.reason


def test_no_traffic_passes(cfg):
    assert gate_traffic(FrameContext(traffic=TrafficResult(available=True)), cfg) is None


def test_unavailable_detector_does_not_block(cfg):
    """A missing COCO model must degrade capability, not halt the survey."""
    assert gate_traffic(FrameContext(traffic=TrafficResult(available=False)), cfg) is None


# -- image conditions ---------------------------------------------------------

def test_sun_glare_blocks(cfg):
    res = gate_image_condition(FrameContext(quality=_Quality(clipped_high=0.4)), cfg)
    assert res is not None and res.action is Action.BLOCK
    assert "blown highlights" in res.reason


def test_night_blocks(cfg):
    res = gate_image_condition(FrameContext(quality=_Quality(mean_luma=0.03)), cfg)
    assert res is not None and res.action is Action.BLOCK
    assert "too dark" in res.reason


def test_dusk_degrades(cfg):
    res = gate_image_condition(FrameContext(quality=_Quality(mean_luma=0.14)), cfg)
    assert res is not None and res.action is Action.DEGRADE


def test_daylight_passes(cfg):
    assert gate_image_condition(FrameContext(quality=_Quality()), cfg) is None


def test_unusable_quality_blocks_with_its_own_reason(cfg):
    q = _Quality(usable=False, reasons=("blurry(sharpness 3.0 < 40.0)",))
    res = gate_image_condition(FrameContext(quality=q), cfg)
    assert res is not None and res.action is Action.BLOCK
    assert "blurry" in res.reason


# -- windscreen ---------------------------------------------------------------

def test_small_static_region_is_masked(cfg):
    sm = np.zeros((H, W), dtype=bool)
    sm[:40, :40] = True
    res = gate_windscreen(FrameContext(static_mask=sm), cfg)
    assert res is not None and res.action is Action.MASK


def test_heavy_static_occlusion_blocks(cfg):
    sm = np.zeros((H, W), dtype=bool)
    sm[: int(0.5 * H), :] = True
    res = gate_windscreen(FrameContext(static_mask=sm), cfg)
    assert res is not None and res.action is Action.BLOCK
    assert "static structure" in res.reason


def test_clean_windscreen_passes(cfg):
    assert gate_windscreen(FrameContext(static_mask=None), cfg) is None


# -- remaining assessable area ------------------------------------------------

def test_blocks_when_exclusions_leave_no_road(cfg):
    road = _Road(_good_road())
    zone = np.ones((H, W), dtype=bool)
    traffic = TrafficResult(available=True, n_detections=1, occluded_frac=0.9,
                            mask=np.ones((H, W), dtype=bool))
    res = gate_assessment_zone(
        FrameContext(road=road, zone_mask=zone, traffic=traffic), cfg)
    assert res is not None and res.action is Action.BLOCK
    assert "assessable road remain" in res.reason


def test_passes_when_plenty_of_road_remains(cfg):
    ctx = FrameContext(road=_Road(_good_road()), zone_mask=np.ones((H, W), dtype=bool))
    assert gate_assessment_zone(ctx, cfg) is None


# -- verdict aggregation ------------------------------------------------------

def test_block_wins_over_degrade():
    v = FrameVerdict(frame=1)
    v.results.append(GateResult("a", Action.DEGRADE, "meh"))
    v.results.append(GateResult("b", Action.BLOCK, "no road"))
    assert not v.assessable
    assert v.confidence() == 0.0
    assert v.block_gates == ("b",)
    assert "NOT ASSESSED" in v.banner()


def test_degrade_only_stays_assessable():
    v = FrameVerdict(frame=2)
    v.results.append(GateResult("a", Action.DEGRADE, "low light"))
    assert v.assessable and v.degraded
    assert 0.0 < v.confidence() < 1.0
    assert "LOW CONFIDENCE" in v.banner()


def test_clean_verdict_is_fully_confident():
    v = FrameVerdict(frame=3)
    assert v.assessable and not v.degraded
    assert v.confidence() == 1.0
    assert v.banner() == ""


def test_mask_action_does_not_block():
    v = FrameVerdict(frame=4)
    v.results.append(GateResult("traffic", Action.MASK, "1 car excluded"))
    assert v.assessable


# -- route statistics ---------------------------------------------------------

def _blocked(gate="road_buried"):
    v = FrameVerdict()
    v.results.append(GateResult(gate, Action.BLOCK, "flooded"))
    return v


def test_coverage_counts_frames_and_distance():
    stats = ValidityStats()
    for _ in range(6):
        stats.update(FrameVerdict(), distance_m=2.0)
    for _ in range(4):
        stats.update(_blocked(), distance_m=3.0)

    assert stats.frames == 10
    assert abs(stats.frame_coverage - 0.6) < 1e-9
    # Distance-weighted: 12 m assessed of 24 m driven, which is NOT the frame ratio.
    assert abs(stats.distance_coverage - 0.5) < 1e-9
    assert stats.blocked_by_gate == {"road_buried": 4}
    assert stats.dominant_reason() == "road_buried"


def test_distance_coverage_falls_back_to_frames_without_gps():
    stats = ValidityStats()
    stats.update(FrameVerdict())
    stats.update(_blocked())
    assert abs(stats.distance_coverage - stats.frame_coverage) < 1e-9


def test_longest_unassessed_run_is_tracked():
    stats = ValidityStats()
    for pattern in [True, False, False, False, True, False, True]:
        stats.update(FrameVerdict() if pattern else _blocked())
    assert stats.longest_unassessed_run == 3


def test_empty_stats_are_safe():
    stats = ValidityStats()
    assert stats.frame_coverage == 0.0
    assert stats.dominant_reason() is None
    assert stats.summary()["frames"] == 0


# -- surface plausibility (the fully-covered-road blind spot) -------------------

class _Plaus:
    def __init__(self, verdict, reason="because"):
        self.verdict = verdict
        self.reason = reason

    @property
    def is_road(self):
        return self.verdict in ("road", "unknown")


class _SurfaceP(_Surface):
    def __init__(self, verdict, **kw):
        super().__init__(**kw)
        self.plausibility = _Plaus(verdict)


def test_fully_flooded_road_blocks_even_when_relative_occlusion_reads_zero(cfg):
    """The blind spot this gate exists to close.

    When water covers the whole carriageway it becomes the appearance baseline, so
    the *relative* occlusion measurement reports 0% — a clean, dry road. The absolute
    plausibility check is the only thing that catches it.
    """
    surf = _SurfaceP("water", occluded=0.0)
    assert gate_road_buried(FrameContext(road=_Road(_good_road()), surface=surf),
                            cfg) is None, "relative gate is blind here, by design"

    res = gate_surface_plausible(FrameContext(surface=surf), cfg)
    assert res is not None and res.action is Action.BLOCK
    assert "entirely obscured" in res.reason


def test_fully_mud_covered_road_blocks(cfg):
    res = gate_surface_plausible(FrameContext(surface=_SurfaceP("mud")), cfg)
    assert res is not None and res.action is Action.BLOCK


def test_vegetation_surface_blocks_as_not_a_road(cfg):
    res = gate_surface_plausible(FrameContext(surface=_SurfaceP("vegetation")), cfg)
    assert res is not None and res.action is Action.BLOCK
    assert "not a road surface" in res.reason


def test_normal_road_passes_plausibility(cfg):
    assert gate_surface_plausible(FrameContext(surface=_SurfaceP("road")), cfg) is None
    assert gate_surface_plausible(FrameContext(surface=_SurfaceP("unknown")), cfg) is None


def test_plausibility_gate_can_be_disabled(cfg):
    cfg.set_path("validity.plausibility.enabled", False)
    assert gate_surface_plausible(FrameContext(surface=_SurfaceP("water")), cfg) is None


# -- plausibility classifier on real pixels ------------------------------------

def _classify(frame, road_mask, cfg):
    from rdd.roadseg.ops import channel_stats, compute_features
    from rdd.surface.condition import _CHANNELS, assess_plausibility

    feats = compute_features(frame, 7)
    stats = channel_stats(feats, road_mask, _CHANNELS)
    return assess_plausibility(feats, road_mask, stats, cfg)


def test_classifier_calls_a_gravel_road_a_road(cfg):
    from tests.scenes import car_scene

    frame, road = car_scene()
    assert _classify(frame, road, cfg).verdict == "road"


def test_classifier_detects_a_fully_submerged_road(cfg):
    from tests.scenes import _filled_like, car_scene

    frame, road = car_scene()
    frame = _filled_like(frame, road, bgr=(208, 203, 194), sigma=0.5)
    assert _classify(frame, road, cfg).verdict == "water"


def test_classifier_detects_a_fully_mud_covered_road(cfg):
    from tests.scenes import _filled_like, car_scene

    frame, road = car_scene()
    frame = _filled_like(frame, road, bgr=(42, 72, 118), sigma=2.0)
    assert _classify(frame, road, cfg).verdict == "mud"


def test_classifier_detects_vegetation_under_the_wheels(cfg):
    from tests.scenes import _filled_like, car_scene

    frame, road = car_scene()
    frame = _filled_like(frame, road, bgr=(40, 95, 45), sigma=26.0)
    assert _classify(frame, road, cfg).verdict == "vegetation"


def test_classifier_does_not_flag_a_merely_dark_road(cfg):
    """A shaded but textured gravel road is still a road."""
    from tests.scenes import _filled_like, car_scene

    frame, road = car_scene()
    frame = _filled_like(frame, road, bgr=(58, 61, 64), sigma=6.0)
    assert _classify(frame, road, cfg).is_road
