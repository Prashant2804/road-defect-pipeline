"""Core invariant: unique defects == confirmed track IDs, never per-frame counts.

Plus the occlusion bookkeeping that lets the report separate "no defect here" from
"could not see here".
"""
from rdd.inference.counter import TrackObservation, UniqueCounter


def _obs(frame, area=100.0, conf=0.9, occluded=0.0, road_overlap=1.0, area_m2=None):
    return TrackObservation(frame=frame, t=frame / 30.0, conf=conf,
                            mask_area_px=area, bbox=(0, 0, 10, 10),
                            occluded_frac=occluded, road_overlap=road_overlap,
                            area_m2=area_m2)


def test_unique_vs_raw():
    c = UniqueCounter(["pothole", "crack"], min_track_len=3)
    # track 1 (pothole) seen 5 frames -> confirmed
    for f in range(5):
        c.update(1, 0, _obs(f))
    # track 2 (crack) seen 4 frames -> confirmed
    for f in range(2, 6):
        c.update(2, 1, _obs(f))
    # track 3 (pothole) flickers 2 frames -> NOT confirmed (< min_track_len)
    for f in range(10, 12):
        c.update(3, 0, _obs(f))

    assert c.raw_detections == 5 + 4 + 2  # per-frame sum
    counts = c.unique_counts()
    assert counts == {"pothole": 1, "crack": 1}, counts  # NOT 3, NOT 11
    assert len(c.confirmed_tracks()) == 2


def test_running_total_monotonic():
    c = UniqueCounter(["pothole"], min_track_len=2)
    for f in range(3):
        c.update(1, 0, _obs(f))
    for f in range(5, 8):
        c.update(2, 0, _obs(f))
    # running total counts CONFIRMED tracks whose first_frame <= the given frame
    assert c.running_unique_total(0) == 1    # track1 (first_frame 0) confirmed
    assert c.running_unique_total(4) == 1    # track2 hasn't started yet (first_frame 5)
    assert c.running_unique_total(10) == 2   # both now present


def test_running_per_class_breakdown():
    c = UniqueCounter(["pothole", "crack"], min_track_len=2)
    for f in range(3):
        c.update(1, 0, _obs(f))
        c.update(2, 1, _obs(f))
    assert c.running_per_class(10) == {"pothole": 1, "crack": 1}


# -- occlusion accounting -----------------------------------------------------

def test_occluded_tracks_are_separated_from_assessable_ones():
    c = UniqueCounter(["pothole", "water_logging"], min_track_len=3,
                      occlusion_threshold=0.5, occluder_classes=("water_logging",))
    for f in range(5):
        c.update(1, 0, _obs(f, occluded=0.0))     # clearly visible
        c.update(2, 0, _obs(f, occluded=0.9))     # under water

    assert len(c.confirmed_tracks()) == 2
    assert [t.track_id for t in c.occluded_tracks()] == [2]
    assert [t.track_id for t in c.assessable_tracks()] == [1]
    assert c.occluded_counts() == {"pothole": 1}
    # Both are still real defects — occlusion changes measurability, not existence.
    assert c.unique_counts()["pothole"] == 2


def test_occluder_class_is_exempt_from_occlusion():
    c = UniqueCounter(["pothole", "water_logging"], min_track_len=3,
                      occluder_classes=("water_logging",))
    for f in range(5):
        c.update(1, 1, _obs(f, occluded=1.0))
    assert c.occluded_tracks() == []
    assert c.is_occluder("water_logging")
    assert not c.is_occluder("pothole")


def test_representative_prefers_a_visible_observation():
    """Crops should show the defect, not the puddle covering it."""
    c = UniqueCounter(["pothole"], min_track_len=2, occlusion_threshold=0.5)
    c.update(1, 0, _obs(0, area=1000.0, occluded=0.95))   # biggest but hidden
    c.update(1, 0, _obs(1, area=400.0, occluded=0.0))     # smaller, visible
    c.update(1, 0, _obs(2, area=300.0, occluded=0.0))

    rep = c.tracks[1].representative()
    assert rep.frame == 1, "should pick the largest *visible* observation"


def test_representative_falls_back_when_nothing_is_visible():
    c = UniqueCounter(["pothole"], min_track_len=2, occlusion_threshold=0.5)
    c.update(1, 0, _obs(0, area=1000.0, occluded=0.95))
    c.update(1, 0, _obs(1, area=400.0, occluded=0.99))
    assert c.tracks[1].representative().frame == 0


def test_ground_area_is_the_max_across_observations():
    c = UniqueCounter(["pothole"], min_track_len=2)
    c.update(1, 0, _obs(0, area_m2=0.1))
    c.update(1, 0, _obs(1, area_m2=0.4))
    c.update(1, 0, _obs(2, area_m2=None))
    assert c.tracks[1].max_area_m2 == 0.4


def test_ground_area_is_none_when_never_measured():
    c = UniqueCounter(["pothole"], min_track_len=1)
    c.update(1, 0, _obs(0, area_m2=None))
    assert c.tracks[1].max_area_m2 is None


def test_mean_road_overlap_is_tracked():
    c = UniqueCounter(["pothole"], min_track_len=1)
    c.update(1, 0, _obs(0, road_overlap=0.5))
    c.update(1, 0, _obs(1, road_overlap=1.0))
    assert abs(c.tracks[1].mean_road_overlap - 0.75) < 1e-9


def test_unmapped_class_id_is_named_conspicuously():
    """A class id outside model.classes means a checkpoint mismatch.

    It must not surface as a bare number in the report, which reads as though it
    were a real category.
    """
    c = UniqueCounter(["pothole"], min_track_len=1)
    c.update(1, 71, _obs(0))
    assert c.tracks[1].cls_name == "UNMAPPED_CLASS_71"


if __name__ == "__main__":
    test_unique_vs_raw()
    test_running_total_monotonic()
    print("all counter tests passed")
