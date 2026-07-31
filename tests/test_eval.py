"""Precision measurement, threshold calibration, IRC banding and segment rollup."""
from __future__ import annotations

import pytest

from rdd.eval.precision import (
    GroundTruthDefect,
    calibrate_class,
    certify,
    load_ground_truth,
    match_tracks,
    sweep,
    wilson_interval,
)
from rdd.report.irc import INDETERMINATE, Segment, build_segments, grade_defect


class _Track:
    def __init__(self, tid, cls_name, first, last, conf=0.9):
        self.track_id = tid
        self.cls_name = cls_name
        self.first_frame = first
        self.last_frame = last
        self.peak_conf = conf
        self.max_area_m2 = None
        self.max_mask_area = 100.0
        self.n_frames = last - first + 1

    def representative(self):
        class _O:
            lat = lon = None
            frame = 0
            bbox = (0, 0, 10, 10)
            t = 0.0
        return _O()


def _gt(cls_name, first, last):
    return GroundTruthDefect(cls_name=cls_name, first_frame=first, last_frame=last)


# -- matching ------------------------------------------------------------------

def test_perfect_match():
    tracks = [_Track(1, "pothole", 10, 20)]
    m = match_tracks(tracks, [_gt("pothole", 12, 18)], "pothole")
    assert (m.n_tp, m.n_fp, m.n_fn) == (1, 0, 0)
    assert m.precision == 1.0 and m.recall == 1.0


def test_false_positive_and_negative():
    tracks = [_Track(1, "pothole", 100, 110)]      # nowhere near the truth
    m = match_tracks(tracks, [_gt("pothole", 10, 20)], "pothole")
    assert (m.n_tp, m.n_fp, m.n_fn) == (0, 1, 1)
    assert m.precision == 0.0 and m.recall == 0.0


def test_one_defect_seen_in_many_frames_counts_once():
    """The counting unit is the unique defect. A per-frame score would move with fps."""
    tracks = [_Track(1, "pothole", 5, 60)]
    m = match_tracks(tracks, [_gt("pothole", 10, 50)], "pothole")
    assert m.n_tp == 1


def test_duplicate_tracks_produce_one_tp_and_one_fp():
    """Two detections of the same defect: one is right, the other is a duplicate."""
    tracks = [_Track(1, "pothole", 10, 20, conf=0.9),
              _Track(2, "pothole", 11, 21, conf=0.5)]
    m = match_tracks(tracks, [_gt("pothole", 12, 18)], "pothole")
    assert (m.n_tp, m.n_fp) == (1, 1)
    assert m.tp[0][1].track_id == 1, "the higher-confidence track should claim the match"


def test_classes_do_not_cross_match():
    tracks = [_Track(1, "pothole", 10, 20)]
    m = match_tracks(tracks, [_gt("longitudinal_crack", 10, 20)], "pothole")
    assert (m.n_tp, m.n_fp) == (0, 1)


def test_slack_allows_slightly_offset_annotation():
    tracks = [_Track(1, "pothole", 21, 30)]
    assert match_tracks(tracks, [_gt("pothole", 10, 20)], "pothole", slack=0).n_tp == 0
    assert match_tracks(tracks, [_gt("pothole", 10, 20)], "pothole", slack=5).n_tp == 1


# -- Wilson interval -----------------------------------------------------------

def test_wilson_never_exceeds_one():
    """The reason for using Wilson rather than the normal approximation."""
    lo, hi = wilson_interval(20, 20)
    assert hi <= 1.0 and lo < 1.0


def test_wilson_narrows_with_more_data():
    small = wilson_interval(9, 10)
    large = wilson_interval(900, 1000)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_wilson_empty_is_maximally_uncertain():
    assert wilson_interval(0, 0) == (0.0, 1.0)


# -- calibration ---------------------------------------------------------------

def test_threshold_raised_to_meet_the_target():
    """Low-confidence false positives should be excluded by raising the threshold.

    Uses 40 instances because certifying 90% needs ~35 even with a perfect detector —
    see test_certifying_ninety_percent_needs_about_thirty_five_instances.
    """
    truth = [_gt("pothole", i * 100, i * 100 + 10) for i in range(40)]
    tracks = [_Track(i, "pothole", i * 100 + 2, i * 100 + 8, conf=0.95)
              for i in range(40)]
    tracks += [_Track(500 + i, "pothole", 9000 + i * 50, 9000 + i * 50 + 5, conf=0.3)
               for i in range(10)]

    cal = calibrate_class(tracks, truth, "pothole", target_precision=0.90)
    assert cal.target_met, cal.note
    assert cal.threshold > 0.3, "must exclude the weak false positives"
    assert cal.precision >= 0.90


def test_certifying_ninety_percent_needs_about_thirty_five_instances():
    """A hard statistical floor that sets the labelling budget.

    Requiring the confidence interval's lower bound to clear the target means a class
    cannot be certified from a handful of examples, however well the detector does.
    At 100% observed precision the Wilson lower bound only reaches 0.90 at n=35, so
    ~35 ground-truth instances PER CLASS is the minimum validation volume.
    """
    from rdd.eval.precision import _min_n_for_target

    assert _min_n_for_target(0.90) == 35
    assert wilson_interval(34, 34)[0] < 0.90
    assert wilson_interval(35, 35)[0] >= 0.90


def test_perfect_but_tiny_sample_is_blamed_on_labelling_not_the_model():
    """The diagnostic must distinguish "needs more labels" from "needs a better model"."""
    truth = [_gt("pothole", i * 100, i * 100 + 10) for i in range(8)]
    tracks = [_Track(i, "pothole", i * 100 + 2, i * 100 + 8, conf=0.95)
              for i in range(8)]
    cal = calibrate_class(tracks, truth, "pothole", target_precision=0.90,
                          min_support=1)
    assert not cal.target_met
    assert cal.precision == 1.0, "the detector was perfect on what it saw"
    assert "labelling-volume limit, not a model limit" in cal.note


def test_unreachable_target_is_reported_not_faked():
    """With false positives at every confidence, no threshold works — say so."""
    truth = [_gt("crack", 0, 10)]
    tracks = [_Track(1, "crack", 2, 8, conf=0.99)]
    tracks += [_Track(10 + i, "crack", 1000 + i * 50, 1000 + i * 50 + 5, conf=0.99)
               for i in range(10)]

    cal = calibrate_class(tracks, truth, "crack", target_precision=0.90)
    assert not cal.target_met
    assert "not reached at any threshold" in cal.note
    assert cal.precision < 0.90


def test_recall_is_reported_alongside_precision():
    """Precision alone can always be met by detecting almost nothing."""
    truth = [_gt("pothole", i * 100, i * 100 + 10) for i in range(10)]
    tracks = [_Track(0, "pothole", 2, 8, conf=0.99)]
    cal = calibrate_class(tracks, truth, "pothole", target_precision=0.90,
                          min_support=1)
    assert cal.precision == 1.0
    assert cal.recall <= 0.2, "one detection out of ten must show poor recall"


def test_thin_support_is_not_certified():
    """A single example cannot certify anything, whatever the point estimate says."""
    truth = [_gt("drainage_issue", 0, 10)]
    tracks = [_Track(1, "drainage_issue", 2, 8, conf=0.99)]
    cal = calibrate_class(tracks, truth, "drainage_issue", target_precision=0.90,
                          min_support=8)
    assert not cal.target_met
    assert "not enough evidence" in cal.note or "too few" in cal.note


def test_sweep_is_monotonic_in_recall():
    truth = [_gt("pothole", i * 100, i * 100 + 10) for i in range(6)]
    tracks = [_Track(i, "pothole", i * 100 + 2, i * 100 + 8, conf=0.1 + 0.15 * i)
              for i in range(6)]
    recalls = [m.recall for _, m in sweep(tracks, truth, "pothole")]
    assert recalls == sorted(recalls, reverse=True), "raising the bar cannot add recall"


# -- certification -------------------------------------------------------------

def test_certification_splits_certified_from_indicative(cfg):
    cfg.set_path("eval.min_support", 4)
    truth = [_gt("pothole", i * 100, i * 100 + 10) for i in range(40)]
    truth += [_gt("ravelling", 20000 + i * 100, 20000 + i * 100 + 10) for i in range(40)]

    tracks = [_Track(i, "pothole", i * 100 + 2, i * 100 + 8, conf=0.95)
              for i in range(40)]
    # Ravelling: mostly wrong at every confidence.
    tracks += [_Track(100, "ravelling", 20002, 20008, conf=0.9)]
    tracks += [_Track(200 + i, "ravelling", 50000 + i * 50, 50000 + i * 50 + 5,
                      conf=0.9) for i in range(9)]

    rep = certify(tracks, truth, cfg, route_coverage=0.83)
    assert "pothole" in rep.certified
    assert "ravelling" in rep.indicative
    assert rep.route_coverage == 0.83
    table = rep.table()
    assert "CERTIFIED" in table and "indicative" in table
    assert "83.0%" in table or "83%" in table


def test_excluded_class_is_never_certified(cfg):
    """Rutting is explicitly outside the guarantee."""
    cfg.set_path("eval.min_support", 1)
    truth = [_gt("rutting", i * 100, i * 100 + 10) for i in range(6)]
    tracks = [_Track(i, "rutting", i * 100 + 2, i * 100 + 8, conf=0.99)
              for i in range(6)]
    rep = certify(tracks, truth, cfg)
    assert "rutting" not in rep.certified
    assert "outside the precision guarantee" in rep.per_class["rutting"].note


def test_thresholds_are_emitted_per_class(cfg):
    rep = certify([], [], cfg)
    th = rep.thresholds()
    assert set(th) == set(cfg.get_path("model.classes"))


# -- ground truth I/O ----------------------------------------------------------

def test_load_ground_truth_csv(tmp_path):
    p = tmp_path / "gt.csv"
    p.write_text("class,first_frame,last_frame\npothole,10,20\ncrack,30,40\n",
                 encoding="utf-8")
    gts = load_ground_truth(p)
    assert len(gts) == 2 and gts[0].cls_name == "pothole"


def test_load_ground_truth_json(tmp_path):
    import json

    p = tmp_path / "gt.json"
    p.write_text(json.dumps([{"class": "pothole", "first_frame": 1,
                              "last_frame": 5, "lat": 12.9, "lon": 77.6}]),
                 encoding="utf-8")
    gts = load_ground_truth(p)
    assert gts[0].lat == 12.9


def test_bad_ground_truth_row_is_rejected(tmp_path):
    p = tmp_path / "gt.csv"
    p.write_text("class,first_frame\npothole,10\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_ground_truth(p)


def test_missing_ground_truth_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_ground_truth(tmp_path / "nope.csv")


# -- IRC banding ---------------------------------------------------------------

def test_crack_width_bands(cfg):
    assert grade_defect("longitudinal_crack", cfg, width_mm=1.0).level == "low"
    assert grade_defect("longitudinal_crack", cfg, width_mm=4.0).level == "medium"
    assert grade_defect("longitudinal_crack", cfg, width_mm=9.0).level == "high"


def test_pothole_area_bands(cfg):
    assert grade_defect("pothole", cfg, area_m2=0.02).level == "low"
    assert grade_defect("pothole", cfg, area_m2=0.20).level == "medium"
    assert grade_defect("pothole", cfg, area_m2=0.80).level == "high"


def test_occluded_defect_is_never_graded(cfg):
    g = grade_defect("pothole", cfg, area_m2=5.0, occluded=True)
    assert g.level == INDETERMINATE
    assert "not observable" in g.note


def test_missing_measurement_is_indeterminate_not_low(cfg):
    """No ground scale means no physical band — not a clean bill of health."""
    g = grade_defect("pothole", cfg, area_m2=None)
    assert g.level == INDETERMINATE
    assert "no ground scale" in g.note


def test_edge_damage_uses_inset(cfg):
    assert grade_defect("edge_damage", cfg, inset_m=0.05).level == "low"
    assert grade_defect("edge_damage", cfg, inset_m=0.40).level == "high"


def test_bands_are_configurable(cfg):
    cfg.set_path("report.irc.crack_width_mm", {"medium": 20.0, "high": 40.0})
    assert grade_defect("transverse_crack", cfg, width_mm=5.0).level == "low"


# -- segment rollup ------------------------------------------------------------

class _Counter:
    def __init__(self, tracks):
        self._tracks = tracks

    def confirmed_tracks(self):
        return self._tracks


class _Validity:
    def __init__(self, n, assessable):
        self.frames = n
        self._per_frame_assessable = assessable


def test_defects_land_in_the_right_segment(cfg):
    cfg.set_path("report.segments.length_m", 100.0)
    cfg.set_path("report.segments.assumed_speed_mps", 10.0)
    # 10 m/s at 10 fps -> 1 m per frame, so frame 250 is at chainage 250 m.
    tracks = [_Track(1, "pothole", 50, 60), _Track(2, "pothole", 250, 260)]
    segs = build_segments(_Counter(tracks), None, cfg, fps=10.0)
    assert segs[0].counts.get("pothole") == 1
    assert segs[2].counts.get("pothole") == 1


def test_unassessable_segment_is_indeterminate_not_sound(cfg):
    """A stretch nobody could see must not be reported as intact."""
    cfg.set_path("report.segments.length_m", 100.0)
    cfg.set_path("report.segments.assumed_speed_mps", 10.0)
    cfg.set_path("report.segments.min_coverage", 0.30)

    validity = _Validity(100, [False] * 100)      # nothing was assessable
    segs = build_segments(_Counter([]), None, cfg, fps=10.0, validity=validity)
    assert segs[0].grade(cfg) == INDETERMINATE, "no coverage must not grade as sound"


def test_clean_well_covered_segment_is_sound(cfg):
    cfg.set_path("report.segments.length_m", 100.0)
    cfg.set_path("report.segments.assumed_speed_mps", 10.0)
    validity = _Validity(100, [True] * 100)
    segs = build_segments(_Counter([]), None, cfg, fps=10.0, validity=validity)
    assert segs[0].grade(cfg) == "sound"


def test_segment_grade_takes_the_worst_severity(cfg):
    seg = Segment(index=0, start_m=0, end_m=100)
    seg.total_frames, seg.assessed_frames = 100, 100
    seg.counts = {"pothole": 2}
    seg.severity = {"low": 1, "high": 1}
    assert seg.grade(cfg) == "high"
