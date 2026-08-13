"""Lightweight IoU tracker for unique defect timelines."""
from __future__ import annotations

from dataclasses import dataclass, field


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    class_id: int
    class_name: str
    bbox: tuple[float, float, float, float]
    conf: float
    frame_start: int
    frame_end: int
    t_start_s: float
    t_end_s: float
    hits: int = 1
    age: int = 0  # frames since last match (strided)
    conf_max: float = 0.0
    bbox_best: tuple[float, float, float, float] = field(default_factory=tuple)
    frame_best: int = -1
    t_best_s: float = 0.0

    def __post_init__(self) -> None:
        if not self.bbox_best:
            self.bbox_best = self.bbox
        if self.conf_max <= 0:
            self.conf_max = self.conf
        if self.frame_best < 0:
            self.frame_best = self.frame_start
        if self.t_best_s <= 0:
            self.t_best_s = self.t_start_s


@dataclass
class SimpleTracker:
    iou_match: float = 0.3
    max_age: int = 15
    _next_id: int = 1
    active: list[Track] = field(default_factory=list)
    finished: list[Track] = field(default_factory=list)

    def update(
        self,
        boxes,
        class_ids,
        confs,
        class_names: list[str],
        frame_i: int,
        t_s: float,
    ) -> list[Track]:
        """Match current detections; return list of tracks updated this frame."""
        for tr in self.active:
            tr.age += 1

        matched_track: set[int] = set()
        matched_det: set[int] = set()
        updated: list[Track] = []

        # Greedy IoU matching within same class
        pairs: list[tuple[float, int, int]] = []
        for ti, tr in enumerate(self.active):
            for di, box in enumerate(boxes):
                if class_ids is None:
                    continue
                if int(class_ids[di]) != tr.class_id:
                    continue
                pairs.append((_iou(tr.bbox, box), ti, di))
        pairs.sort(reverse=True)

        for iou, ti, di in pairs:
            if iou < self.iou_match:
                break
            if ti in matched_track or di in matched_det:
                continue
            tr = self.active[ti]
            conf = float(confs[di]) if confs is not None else 0.0
            box = tuple(float(x) for x in boxes[di])
            tr.bbox = box
            tr.conf = conf
            tr.frame_end = frame_i
            tr.t_end_s = t_s
            tr.hits += 1
            tr.age = 0
            if conf > tr.conf_max:
                tr.conf_max = conf
            # Nearest observation = lowest in the frame (largest y2).
            if (not tr.bbox_best) or box[3] >= tr.bbox_best[3]:
                tr.bbox_best = box
                tr.frame_best = frame_i
                tr.t_best_s = t_s
            matched_track.add(ti)
            matched_det.add(di)
            updated.append(tr)

        # New tracks
        for di, box in enumerate(boxes):
            if di in matched_det:
                continue
            cid = int(class_ids[di]) if class_ids is not None else -1
            name = (
                class_names[cid]
                if 0 <= cid < len(class_names)
                else str(cid)
            )
            conf = float(confs[di]) if confs is not None else 0.0
            box_t = tuple(float(x) for x in box)
            tr = Track(
                track_id=self._next_id,
                class_id=cid,
                class_name=name,
                bbox=box_t,
                conf=conf,
                frame_start=frame_i,
                frame_end=frame_i,
                t_start_s=t_s,
                t_end_s=t_s,
                conf_max=conf,
                bbox_best=box_t,
                frame_best=frame_i,
                t_best_s=t_s,
            )
            self._next_id += 1
            self.active.append(tr)
            updated.append(tr)

        # Age out
        still = []
        for tr in self.active:
            if tr.age > self.max_age:
                self.finished.append(tr)
            else:
                still.append(tr)
        self.active = still
        return updated

    def flush(self) -> list[Track]:
        self.finished.extend(self.active)
        self.active = []
        return list(self.finished)
