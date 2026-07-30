"""Surface condition: water and mud detected, shadow correctly rejected.

The shadow tests matter most. Without an explicit shadow rule, every patch of
shade reads as "mud", the unassessable fraction inflates, and the pipeline starts
abstaining on defects it can see perfectly well.
"""
from __future__ import annotations

import numpy as np

from tests.scenes import (
    add_mud,
    add_shadow,
    add_water,
    car_scene,
    make_road_mask,
    surface_map,
)
from rdd.roadseg.classical import ClassicalSegmenter
from rdd.roadseg.ops import overlap_fraction
from rdd.surface.condition import SurfaceStats, analyse_surface


def _segment(cfg, view, frame):
    return ClassicalSegmenter(cfg, view).segment(frame)


def test_water_is_detected_where_it_was_placed(cfg, car_view):
    frame, _ = car_scene()
    frame, water = add_water(frame)
    road = _segment(cfg, car_view, frame)

    sm = analyse_surface(frame, road, cfg)
    assert overlap_fraction(water, sm.water) > 0.6, "placed puddle not found"
    assert sm.water_frac > 0.0
    assert (sm.occlusion & sm.water).sum() == sm.water.sum(), "water must be occluding"


def test_mud_is_detected_where_it_was_placed(cfg, car_view):
    frame, _ = car_scene()
    frame, mud = add_mud(frame)
    road = _segment(cfg, car_view, frame)

    sm = analyse_surface(frame, road, cfg)
    assert overlap_fraction(mud, sm.mud) > 0.5, "placed mud not found"
    assert sm.mud_frac > 0.0


def test_shadow_is_not_reported_as_mud_or_water(cfg, car_view):
    """Darker, but textured and chromatically neutral — must stay 'dry'."""
    frame, _ = car_scene()
    frame, shadow = add_shadow(frame)
    road = _segment(cfg, car_view, frame)

    sm = analyse_surface(frame, road, cfg)
    mud_in_shadow = overlap_fraction(shadow, sm.mud)
    water_in_shadow = overlap_fraction(shadow, sm.water)
    assert mud_in_shadow < 0.15, f"shadow misread as mud ({mud_in_shadow:.0%})"
    assert water_in_shadow < 0.15, f"shadow misread as water ({water_in_shadow:.0%})"


def test_clean_road_reports_almost_nothing_occluded(cfg, car_view):
    frame, _ = car_scene()
    road = _segment(cfg, car_view, frame)
    sm = analyse_surface(frame, road, cfg)
    assert sm.occluded_frac < 0.05, f"false occlusion on a clean road: {sm.occluded_frac:.1%}"
    assert sm.dry_frac > 0.95


def test_water_and_mud_are_mutually_exclusive(cfg, car_view):
    frame, _ = car_scene()
    frame, _ = add_water(frame, centre=(300, 420), radii=(60, 26))
    frame, _ = add_mud(frame, centre=(340, 425), radii=(60, 26))
    road = _segment(cfg, car_view, frame)

    sm = analyse_surface(frame, road, cfg)
    assert not (sm.water & sm.mud).any(), "a pixel cannot be both water and mud"


def test_occlusion_is_confined_to_the_road(cfg, car_view):
    frame, _ = car_scene()
    frame, _ = add_water(frame)
    road = _segment(cfg, car_view, frame)
    sm = analyse_surface(frame, road, cfg)
    assert not (sm.occlusion & ~road.mask).any(), "occlusion leaked off the road"


def test_empty_road_mask_yields_empty_surface(cfg, car_view):
    frame, _ = car_scene()
    empty = make_road_mask(np.zeros(frame.shape[:2], dtype=bool))
    sm = analyse_surface(frame, empty, cfg)
    assert sm.road_area_px == 0.0
    assert sm.occluded_frac == 0.0
    assert not sm.occlusion.any()


def test_speckle_below_min_area_is_discarded(cfg, car_view):
    frame, _ = car_scene()
    frame, _ = add_water(frame, centre=(320, 430), radii=(2, 2))
    road = _segment(cfg, car_view, frame)
    cfg.set_path("surface.min_blob_area_frac", 0.05)
    sm = analyse_surface(frame, road, cfg)
    assert sm.water_frac == 0.0, "tiny speck should be filtered as noise"


# -- run-level aggregation ----------------------------------------------------

def test_stats_weight_by_road_area_not_by_frame():
    """A frame showing a sliver of road must not count as much as a full view.

    Averaging per-frame fractions would make a 100%-occluded 10 px sliver
    outweigh a clean full-width view; accumulating areas is the correct estimator.
    """
    stats = SurfaceStats()
    stats.update(surface_map(road_px=1000.0, water_px=0.0, mud_px=0.0))
    stats.update(surface_map(road_px=10.0, water_px=10.0, mud_px=0.0))

    # Area-weighted: 10 of 1010 px are water, not the naive (0% + 100%)/2 = 50%.
    assert abs(stats.unassessable_frac - 10.0 / 1010.0) < 1e-9
    assert stats.worst_frame_occluded_frac == 1.0
    assert stats.frames_with_occlusion == 1


def test_stats_empty_is_safe():
    stats = SurfaceStats()
    assert stats.unassessable_frac == 0.0
    assert stats.worst_frame_occluded_frac == 0.0
    assert stats.summary()["frames"] == 0
