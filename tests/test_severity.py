"""Severity scoring: abstention under occlusion, and absolute vs relative basis."""
from __future__ import annotations

from rdd.depth.severity import score_tracks
from rdd.inference.counter import Track, TrackObservation, UniqueCounter


def _obs(frame, area=100.0, conf=0.9, occluded=0.0, area_m2=None):
    return TrackObservation(frame=frame, t=frame / 30.0, conf=conf,
                            mask_area_px=area, bbox=(0, 0, 10, 10),
                            occluded_frac=occluded, area_m2=area_m2)


def _track(tid, cls_id, cls_name, n=5, area=100.0, occluded=0.0, area_m2=None):
    t = Track(track_id=tid, cls_id=cls_id, cls_name=cls_name)
    for f in range(n):
        t.observations.append(_obs(f, area=area, occluded=occluded, area_m2=area_m2))
    return t


def _counter(occluders=("water_logging",)):
    return UniqueCounter(["pothole", "water_logging", "rut_erosion", "crack"],
                         min_track_len=3, occluder_classes=occluders)


# -- abstention ---------------------------------------------------------------

def test_defect_under_water_is_indeterminate_not_scored(cfg):
    """The core honesty requirement: no number for something we cannot see."""
    hidden = _track(1, 0, "pothole", area=500.0, occluded=0.9)
    report = score_tracks([hidden], cfg, counter=_counter())

    sev = report[1]
    assert sev.level == "indeterminate"
    assert sev.score is None, "an unobservable defect must not get a score"
    assert sev.basis == "abstained"
    assert "water/mud" in sev.reason
    assert report.n_indeterminate == 1


def test_visible_defect_is_scored_normally(cfg):
    visible = _track(1, 0, "pothole", area=500.0, occluded=0.0)
    sev = score_tracks([visible], cfg, counter=_counter())[1]
    assert sev.level in ("low", "medium", "high")
    assert sev.score is not None


def test_water_logging_is_never_occluded_by_itself(cfg):
    """Water is the occluder. Abstaining on it would abstain on every instance."""
    water = _track(1, 1, "water_logging", area=800.0, occluded=1.0)
    sev = score_tracks([water], cfg, counter=_counter())[1]
    assert sev.level != "indeterminate"
    assert sev.score is not None


def test_brief_occlusion_does_not_condemn_a_mostly_visible_defect(cfg):
    """Median, not max: one frame of spray or glare should not force abstention."""
    t = Track(track_id=1, cls_id=0, cls_name="pothole")
    for f in range(9):
        t.observations.append(_obs(f, area=400.0, occluded=0.0))
    t.observations.append(_obs(9, area=400.0, occluded=1.0))

    assert t.max_occluded_frac == 1.0
    assert not t.occluded, "a single occluded frame out of ten is not 'hidden'"
    assert score_tracks([t], cfg, counter=_counter())[1].level != "indeterminate"


def test_policy_flag_scores_occluded_defects_anyway(cfg):
    cfg.set_path("surface.occlusion_policy", "flag")
    hidden = _track(1, 0, "pothole", area=500.0, occluded=0.9)
    sev = score_tracks([hidden], cfg, counter=_counter())[1]
    assert sev.level != "indeterminate"
    assert sev.score is not None


# -- basis: absolute vs relative ----------------------------------------------

def test_absolute_basis_used_when_every_track_has_ground_area(cfg):
    small = _track(1, 0, "pothole", area=100.0, area_m2=0.02)
    mid = _track(2, 0, "pothole", area=400.0, area_m2=0.20)
    big = _track(3, 0, "pothole", area=900.0, area_m2=0.90)

    report = score_tracks([small, mid, big], cfg, counter=_counter())
    assert report.basis == "absolute_m2"
    assert report[1].level == "low"      # < 0.10 m²
    assert report[2].level == "medium"   # >= 0.10 m²
    assert report[3].level == "high"     # >= 0.50 m²


def test_absolute_severity_is_comparable_between_runs(cfg):
    """The same physical defect must score the same in a clip with no big ones."""
    lone = _track(1, 0, "pothole", area=100.0, area_m2=0.20)
    alone = score_tracks([lone], cfg, counter=_counter())[1]

    with_bigger = score_tracks(
        [_track(1, 0, "pothole", area=100.0, area_m2=0.20),
         _track(2, 0, "pothole", area=9000.0, area_m2=5.0)],
        cfg, counter=_counter(),
    )[1]
    assert alone.level == with_bigger.level == "medium"


def test_relative_basis_makes_the_biggest_defect_always_high(cfg):
    """Documents the weakness of the pixel fallback, which the report states."""
    tiny = _track(1, 0, "crack", area=10.0)
    small = _track(2, 0, "crack", area=20.0)
    report = score_tracks([tiny, small], cfg, counter=_counter())

    assert report.basis == "relative_px"
    assert report[2].level == "high", "min-max normalisation always tops out"
    assert "not comparable between videos" in report.scale_note


def test_mixed_scale_availability_falls_back_to_relative(cfg):
    scaled = _track(1, 0, "pothole", area=100.0, area_m2=0.2)
    unscaled = _track(2, 0, "pothole", area=400.0, area_m2=None)
    report = score_tracks([scaled, unscaled], cfg, counter=_counter())
    assert report.basis == "relative_px", "cannot mix m² and px bases"


def test_absolute_bins_are_configurable(cfg):
    cfg.set_path("severity.absolute_bins_m2", {"medium": 1.0, "high": 2.0})
    t = _track(1, 0, "pothole", area=100.0, area_m2=0.5)
    assert score_tracks([t], cfg, counter=_counter())[1].level == "low"


# -- report shape -------------------------------------------------------------

def test_level_counts_cover_all_levels(cfg):
    tracks = [
        _track(1, 0, "pothole", area=100.0, area_m2=0.02),
        _track(2, 0, "pothole", area=900.0, area_m2=0.90),
        _track(3, 0, "pothole", area=500.0, area_m2=0.30, occluded=0.9),
    ]
    counts = score_tracks(tracks, cfg, counter=_counter()).level_counts()
    assert counts["low"] == 1
    assert counts["high"] == 1
    assert counts["indeterminate"] == 1
    assert set(counts) == {"low", "medium", "high", "indeterminate"}


def test_empty_track_list_is_safe(cfg):
    report = score_tracks([], cfg, counter=_counter())
    assert len(report) == 0
    assert report.n_indeterminate == 0


def test_severity_supports_mapping_style_access(cfg):
    t = _track(1, 0, "pothole", area=100.0)
    sev = score_tracks([t], cfg, counter=_counter())[1]
    assert sev.get("level") == sev.level
    assert "area_px" in sev.as_dict()
