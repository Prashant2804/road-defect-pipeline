"""Unique-defect accounting from tracker output.

The golden rule: one physical defect == one track ID. We NEVER count per-frame
detections as defects. A track is only promoted to a "unique defect" once it has
persisted for >= min_track_len frames (filters flicker/false positives).

Each observation also carries what the road-segmentation and surface stages
concluded about it: how much of the detection sat on the road, and how much of it
sat under water or mud. That second number is what lets the report distinguish
"this road is fine" from "we could not see this road".
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
    road_overlap: float = 1.0       # fraction of the detection on the road surface
    occluded_frac: float = 0.0      # fraction under water/mud
    area_m2: float | None = None    # ground area, when scale is known


@dataclass
class Track:
    track_id: int
    cls_id: int
    cls_name: str
    observations: list[TrackObservation] = field(default_factory=list)
    occlusion_threshold: float = 0.5

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
    def max_area_m2(self) -> float | None:
        vals = [o.area_m2 for o in self.observations if o.area_m2 is not None]
        return max(vals) if vals else None

    @property
    def peak_conf(self) -> float:
        return max((o.conf for o in self.observations), default=0.0)

    @property
    def mean_road_overlap(self) -> float:
        if not self.observations:
            return 0.0
        return sum(o.road_overlap for o in self.observations) / len(self.observations)

    @property
    def max_occluded_frac(self) -> float:
        return max((o.occluded_frac for o in self.observations), default=0.0)

    @property
    def median_occluded_frac(self) -> float:
        vals = sorted(o.occluded_frac for o in self.observations)
        if not vals:
            return 0.0
        mid = len(vals) // 2
        if len(vals) % 2:
            return vals[mid]
        return (vals[mid - 1] + vals[mid]) / 2.0

    @property
    def occluded(self) -> bool:
        """True when the defect is mostly hidden under water/mud.

        Uses the median across observations, not the max: a single frame of glare
        or spray should not condemn a defect that was clearly visible for the
        rest of its track.
        """
        return self.median_occluded_frac >= self.occlusion_threshold

    def representative(self) -> TrackObservation:
        """Best observation for a crop: largest mask, preferring unoccluded views."""
        visible = [o for o in self.observations
                   if o.occluded_frac < self.occlusion_threshold]
        pool = visible or self.observations
        return max(pool, key=lambda o: o.mask_area_px)


class UniqueCounter:
    def __init__(self, class_names: list[str], min_track_len: int = 3,
                 occlusion_threshold: float = 0.5,
                 occluder_classes: tuple[str, ...] = ()):
        self.class_names = class_names
        self.min_track_len = min_track_len
        self.occlusion_threshold = occlusion_threshold
        # Water-logging is itself the thing doing the occluding, so it must never
        # be marked "hidden under water" — that would abstain on every instance
        # of a class we can see perfectly well.
        self.occluder_classes = tuple(occluder_classes)
        self.tracks: dict[int, Track] = {}
        self.raw_detections = 0     # per-frame detection count (reported alongside)
        self.rejected_off_road = 0  # detections dropped for not being on the road

    def is_occluder(self, cls_name: str) -> bool:
        return cls_name in self.occluder_classes

    def update(self, track_id: int, cls_id: int, obs: TrackObservation) -> None:
        self.raw_detections += 1
        tr = self.tracks.get(track_id)
        if tr is None:
            # An out-of-range id means the checkpoint's classes do not match
            # model.classes (see model.loader.check_class_alignment). Name it so
            # the report is obviously wrong rather than quietly wrong — a bare
            # "71" in a class column reads like it might mean something.
            name = (self.class_names[cls_id] if 0 <= cls_id < len(self.class_names)
                    else f"UNMAPPED_CLASS_{cls_id}")
            tr = Track(track_id=track_id, cls_id=cls_id, cls_name=name,
                       occlusion_threshold=self.occlusion_threshold)
            self.tracks[track_id] = tr
        tr.observations.append(obs)

    def confirmed_tracks(self) -> list[Track]:
        return [t for t in self.tracks.values() if t.n_frames >= self.min_track_len]

    def occluded_tracks(self) -> list[Track]:
        """Confirmed defects we could not properly see."""
        return [t for t in self.confirmed_tracks()
                if t.occluded and not self.is_occluder(t.cls_name)]

    def assessable_tracks(self) -> list[Track]:
        occluded = {t.track_id for t in self.occluded_tracks()}
        return [t for t in self.confirmed_tracks() if t.track_id not in occluded]

    def unique_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in self.class_names}
        for t in self.confirmed_tracks():
            counts[t.cls_name] = counts.get(t.cls_name, 0) + 1
        return counts

    def occluded_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in self.occluded_tracks():
            counts[t.cls_name] = counts.get(t.cls_name, 0) + 1
        return counts

    def running_unique_total(self, up_to_frame: int) -> int:
        """Unique confirmed tracks whose track started at or before this frame —
        used for the live HUD overlay."""
        return sum(
            1 for t in self.tracks.values()
            if t.n_frames >= self.min_track_len and t.first_frame <= up_to_frame
        )

    def running_per_class(self, up_to_frame: int) -> dict[str, int]:
        """Per-class confirmed counts so far — for the HUD breakdown."""
        pc: dict[str, int] = {}
        for t in self.tracks.values():
            if t.n_frames >= self.min_track_len and t.first_frame <= up_to_frame:
                pc[t.cls_name] = pc.get(t.cls_name, 0) + 1
        return pc
