"""Road segmentation: mask primitives, priors, and the classical segmenter."""
from __future__ import annotations

import numpy as np

from tests.scenes import (
    H,
    W,
    add_pothole,
    add_water,
    car_scene,
    drone_scene,
)
from rdd.roadseg.base import build_segmenter
from rdd.roadseg.classical import ClassicalSegmenter
from rdd.roadseg.geometric import GeometricSegmenter
from rdd.roadseg.ops import (
    fill_holes,
    keep_largest_component,
    overlap_fraction,
    principal_axis,
)


# -- primitives ---------------------------------------------------------------

def test_fill_holes_reclaims_enclosed_gap():
    """A hole inside the mask is filled; the outside is untouched.

    This is the behaviour the whole gating design rests on: potholes and puddles
    are appearance outliers that get carved out of the road mask, and they have
    to be put back or gating rejects the defects it exists to keep.
    """
    m = np.zeros((100, 100), dtype=bool)
    m[20:80, 20:80] = True
    m[40:60, 40:60] = False          # a "pothole" hole
    filled = fill_holes(m)

    assert filled[45:55, 45:55].all(), "enclosed hole should be filled"
    assert not filled[0:10, 0:10].any(), "outside must stay outside"
    assert filled.sum() == 60 * 60


def test_fill_holes_does_not_close_a_bay_open_to_the_edge():
    """A concavity connected to the image border is not enclosed, so it stays open."""
    m = np.zeros((100, 100), dtype=bool)
    m[20:80, 20:80] = True
    m[40:60, 20:50] = False          # notch open to the left edge of the shape
    m[40:60, 0:20] = False
    filled = fill_holes(m)
    assert not filled[45:55, 5:15].any()


def test_fill_holes_handles_mask_touching_all_borders():
    """Padding (not corner-seeding) is why a full-frame mask survives intact."""
    m = np.ones((50, 50), dtype=bool)
    m[20:30, 20:30] = False
    filled = fill_holes(m)
    assert filled.all()


def test_keep_largest_component_prefers_seed_overlap():
    m = np.zeros((100, 100), dtype=bool)
    m[0:10, 0:60] = True             # large blob, no seed overlap
    m[70:90, 70:90] = True           # smaller blob, contains the seed
    seed = np.zeros_like(m)
    seed[78:82, 78:82] = True

    kept = keep_largest_component(m, seed)
    assert kept[80, 80]
    assert not kept[5, 5], "component without seed overlap must be dropped"


def test_keep_largest_component_falls_back_to_area_without_seed_overlap():
    m = np.zeros((100, 100), dtype=bool)
    m[0:10, 0:60] = True
    m[70:75, 70:75] = True
    kept = keep_largest_component(m, np.zeros_like(m))
    assert kept[5, 5], "with no seed overlap anywhere, keep the biggest"


def test_overlap_fraction_empty_is_zero():
    a = np.zeros((10, 10), dtype=bool)
    b = np.ones((10, 10), dtype=bool)
    assert overlap_fraction(a, b) == 0.0
    assert overlap_fraction(b, b) == 1.0


def test_principal_axis_detects_orientation():
    vertical = np.zeros((100, 100), dtype=bool)
    vertical[:, 45:55] = True
    horizontal = np.zeros((100, 100), dtype=bool)
    horizontal[45:55, :] = True
    assert principal_axis(vertical) == "vertical"
    assert principal_axis(horizontal) == "horizontal"


# -- geometric prior ----------------------------------------------------------

def test_geometric_prior_is_a_bottom_heavy_trapezoid(cfg, car_view):
    seg = GeometricSegmenter(cfg, car_view)
    rm = seg.segment(car_scene()[0])

    bottom_row = rm.mask[H - 1].sum()
    mid_row = rm.mask[int(0.75 * H)].sum()
    assert bottom_row > mid_row > 0, "road should narrow toward the horizon"
    assert not rm.mask[: int(0.5 * H)].any(), "nothing above the configured horizon"
    assert rm.confidence < 0.5, "a prior is an assumption, not a measurement"


def test_geometric_none_backend_covers_whole_frame(cfg, car_view):
    cfg.set_path("roadseg.backend", "none")
    seg = build_segmenter(cfg, car_view)
    rm = seg.segment(car_scene()[0])
    assert rm.mask.all()


# -- classical segmenter ------------------------------------------------------

def test_classical_recovers_the_road_on_a_car_scene(cfg, car_view):
    frame, truth = car_scene()
    seg = ClassicalSegmenter(cfg, car_view)
    rm = seg.segment(frame)

    assert not rm.fell_back, "clean synthetic scene should not need the fallback"
    recall = overlap_fraction(truth, rm.mask)
    precision = overlap_fraction(rm.mask, truth)
    assert recall > 0.75, f"missed too much road (recall {recall:.2f})"
    assert precision > 0.75, f"leaked off-road (precision {precision:.2f})"


def test_classical_keeps_defects_inside_the_road_mask(cfg, car_view):
    """The point of hole-filling, end to end.

    A pothole and a puddle differ sharply from the road baseline, so similarity
    alone excludes them. They must still end up *on* the road.
    """
    frame, _ = car_scene()
    frame, pothole = add_pothole(frame)
    frame, water = add_water(frame)

    rm = ClassicalSegmenter(cfg, car_view).segment(frame)
    assert overlap_fraction(pothole, rm.mask) > 0.9, "pothole fell out of the road mask"
    assert overlap_fraction(water, rm.mask) > 0.9, "puddle fell out of the road mask"


def test_classical_falls_back_when_the_scene_is_featureless(cfg, car_view):
    """A uniform frame has no road/verge contrast; the segmenter must say so."""
    flat = np.full((H, W, 3), 128, dtype=np.uint8)
    seg = ClassicalSegmenter(cfg, car_view)
    rm = seg.segment(flat)
    assert rm.fell_back
    assert seg.fallback_rate == 1.0
    assert rm.mask.any(), "fallback still has to produce a usable road region"


def test_classical_resolves_drone_band_axis(cfg, drone_view):
    for axis in ("vertical", "horizontal"):
        frame, truth = drone_scene(axis=axis)
        seg = ClassicalSegmenter(cfg, drone_view)
        rm = seg.segment(frame)
        assert rm.axis == axis, f"expected {axis}, inferred {rm.axis}"
        assert overlap_fraction(truth, rm.mask) > 0.6


def test_axis_is_resolved_once_and_reused(cfg, drone_view):
    frame, _ = drone_scene(axis="horizontal")
    seg = ClassicalSegmenter(cfg, drone_view)
    first = seg.segment(frame).axis
    assert seg._axis_locked
    assert seg.segment(frame).axis == first


# -- temporal smoothing -------------------------------------------------------

def test_temporal_smoothing_suppresses_a_single_bad_frame(cfg, car_view):
    cfg.set_path("roadseg.temporal.alpha", 0.35)
    seg = build_segmenter(cfg, car_view)

    good, _ = car_scene()
    for _ in range(6):
        steady = seg.segment(good)

    flat = np.full((H, W, 3), 128, dtype=np.uint8)
    blipped = seg.segment(flat)
    # One anomalous frame cannot swing the mask far, because the EMA still
    # carries the six frames of agreement before it.
    assert overlap_fraction(steady.mask, blipped.mask) > 0.7


def test_temporal_never_returns_an_empty_mask(cfg, car_view):
    cfg.set_path("roadseg.temporal.alpha", 0.9)
    cfg.set_path("roadseg.temporal.threshold", 0.99)
    seg = build_segmenter(cfg, car_view)
    rm = seg.segment(car_scene()[0])
    assert rm.mask.any()


def test_reset_clears_temporal_state(cfg, car_view):
    cfg.set_path("roadseg.temporal.alpha", 0.3)
    seg = build_segmenter(cfg, car_view)
    seg.segment(car_scene()[0])
    seg.reset()
    assert seg._prob is None
