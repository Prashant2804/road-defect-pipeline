"""Core invariant: unique defects == confirmed track IDs, never per-frame counts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rdd.inference.counter import TrackObservation, UniqueCounter  # noqa: E402


def _obs(frame, area=100.0, conf=0.9):
    return TrackObservation(frame=frame, t=frame / 30.0, conf=conf,
                            mask_area_px=area, bbox=(0, 0, 10, 10))


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


if __name__ == "__main__":
    test_unique_vs_raw()
    test_running_total_monotonic()
    print("all counter tests passed")
