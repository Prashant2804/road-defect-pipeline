"""Unique-defect accounting from tracker output.

The golden rule: one physical defect == one track ID. We NEVER count per-frame
detections as defects. A track is only promoted to a "unique defect" once it has
persisted for >= min_track_len frames (filters flicker/false positives).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrackObservation:
    frame: int
    t: float
    conf: float
    mask_area_px: float
    bbox: tuple[float, float, float, float]
    lat: float | None = None
    lon: float | None = None


@dataclass
class Track:
    track_id: int
    cls_id: int
    cls_name: str
    observations: list[TrackObservation] = field(default_factory=list)

    @property
    def first_frame(self) -> int:
        return self.observations[0].frame

    @property
    def last_frame(self) -> int:
        return self.observations[-1].frame

    @property
    def n_frames(self) -> int:
        return len(self.observations)

    @property
    def max_mask_area(self) -> float:
        return max((o.mask_area_px for o in self.observations), default=0.0)

    @property
    def peak_conf(self) -> float:
        return max((o.conf for o in self.observations), default=0.0)

    def representative(self) -> TrackObservation:
        """Observation with the largest mask (best crop / most visible)."""
        return max(self.observations, key=lambda o: o.mask_area_px)


class UniqueCounter:
    def __init__(self, class_names: list[str], min_track_len: int = 3):
        self.class_names = class_names
        self.min_track_len = min_track_len
        self.tracks: dict[int, Track] = {}
        self.raw_detections = 0  # per-frame detection count (for reporting both)

    def update(self, track_id: int, cls_id: int, obs: TrackObservation) -> None:
        self.raw_detections += 1
        tr = self.tracks.get(track_id)
        if tr is None:
            name = self.class_names[cls_id] if 0 <= cls_id < len(self.class_names) else str(cls_id)
            tr = Track(track_id=track_id, cls_id=cls_id, cls_name=name)
            self.tracks[track_id] = tr
        tr.observations.append(obs)

    def confirmed_tracks(self) -> list[Track]:
        return [t for t in self.tracks.values() if t.n_frames >= self.min_track_len]

    def unique_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in self.class_names}
        for t in self.confirmed_tracks():
            counts[t.cls_name] = counts.get(t.cls_name, 0) + 1
        return counts

    def running_unique_total(self, up_to_frame: int) -> int:
        """Unique confirmed tracks whose track started at or before this frame —
        used for the live HUD overlay."""
        return sum(
            1 for t in self.tracks.values()
            if t.n_frames >= self.min_track_len and t.first_frame <= up_to_frame
        )
