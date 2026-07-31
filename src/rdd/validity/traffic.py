"""Traffic occlusion: the road you cannot see because a vehicle is on it.

The most common occluder in dashcam footage is not water or mud — it is the vehicle
ahead. It hides the exact patch of road the camera is best positioned to inspect,
and it does so for long stretches when following traffic. Left unhandled this
produces two errors: the hidden road is scored as clean, and the vehicle itself
(dark tyres, shadow underneath, sharp body panel edges) generates pothole and crack
false positives.

Uniquely among everything in this pipeline, this needs **no training data**:
vehicles and people are COCO classes, so an off-the-shelf detector handles them.
That is why this gate is cheap and worth doing first.

Two actions, deliberately distinguished:

  * **MASK** — subtract the vehicle from the road region and carry on. A car in the
    oncoming lane costs a corner of the frame, not the frame.
  * **BLOCK** — when vehicles cover too much of the assessment zone, there is not
    enough road left to inspect, so refuse the frame.

Blanket-blocking any frame containing a vehicle would discard most of a real survey.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..utils.logging import get_logger

log = get_logger("rdd.validity.traffic")

# COCO ids for things that sit on a road and hide it.
_COCO_ROAD_USERS = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus",
    6: "train", 7: "truck",
}


@dataclass
class TrafficResult:
    available: bool = False
    n_detections: int = 0
    occluded_frac: float = 0.0      # of the assessment zone
    mask: object | None = None
    labels: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"available": self.available, "n": self.n_detections,
                "zone_occluded_frac": round(self.occluded_frac, 4),
                "labels": self.labels[:8]}


class TrafficDetector:
    """Lazily-loaded COCO detector used only to find road users.

    Kept separate from the defect model on purpose: the defect model is fine-tuned
    to road distress and has no vehicle class, so it cannot do this job. A nano
    checkpoint at reduced resolution is enough — we need "there is a truck roughly
    here", not a precise outline.
    """

    def __init__(self, cfg):
        tc = cfg.get_path("validity.traffic", {}) or {}
        self.enabled = bool(tc.get("enabled", True))
        self.weights = str(tc.get("model", "yolo11n.pt"))
        self.conf = float(tc.get("conf", 0.35))
        self.imgsz = int(tc.get("imgsz", 640))
        self.dilate_frac = float(tc.get("dilate_frac", 0.01))
        self._model = None
        self._failed = False

        from ..utils.device import resolve_device

        self.device = resolve_device(cfg.get_path("run.device", "auto"))

    def _load(self):
        if self._model is not None or self._failed:
            return self._model
        try:
            from ultralytics import YOLO

            self._model = YOLO(self.weights)
            log.info("Traffic detector: %s (COCO road users, no training needed)",
                     self.weights)
        except Exception as e:
            self._failed = True
            log.warning(
                "Traffic detector unavailable (%s). Vehicles will NOT be excluded "
                "from the road region, so expect false positives from vehicle "
                "bodies, tyres and under-car shadows.", e)
        return self._model

    def detect(self, frame, zone_mask=None) -> TrafficResult:
        """Mask of road users, and how much of the assessment zone they cover."""
        import numpy as np

        if not self.enabled:
            return TrafficResult(available=False)
        model = self._load()
        if model is None:
            return TrafficResult(available=False)

        h, w = frame.shape[:2]
        try:
            res = model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                                device=self.device, verbose=False,
                                classes=sorted(_COCO_ROAD_USERS))
        except Exception as e:
            log.warning("Traffic detection failed on a frame (%s) — continuing "
                        "without vehicle masking", e)
            return TrafficResult(available=False)

        mask = np.zeros((h, w), dtype=bool)
        labels: list[str] = []
        n = 0
        r = res[0] if res else None
        boxes = getattr(r, "boxes", None) if r is not None else None
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            clss = boxes.cls.int().cpu().tolist()
            for (x1, y1, x2, y2), cid in zip(xyxy, clss):
                labels.append(_COCO_ROAD_USERS.get(int(cid), str(cid)))
                xa, ya = max(0, int(x1)), max(0, int(y1))
                xb, yb = min(w, int(x2)), min(h, int(y2))
                if xb > xa and yb > ya:
                    mask[ya:yb, xa:xb] = True
                    n += 1

        if n and self.dilate_frac > 0:
            # Grow slightly: a box rarely covers the contact shadow under a vehicle,
            # which is itself a reliable source of spurious pothole detections.
            from ..roadseg.ops import dilate

            mask = dilate(mask, max(1, int(self.dilate_frac * w)))

        occluded = 0.0
        if zone_mask is not None and zone_mask.any():
            occluded = float((mask & zone_mask).sum()) / float(zone_mask.sum())

        return TrafficResult(available=True, n_detections=n, occluded_frac=occluded,
                             mask=mask, labels=labels)
