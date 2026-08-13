"""SAM-2 box prompts on RF-DETR tracks → masks → ground area.

One prompt per unique track, on the nearest (lowest in frame) observation.
Falls back to a filled rectangle if SAM is unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .camera import mask_area_m2, pothole_irc_band
from .track_simple import Track


@dataclass
class AreaMeasurement:
    defect_id: int
    class_name: str
    frame_best: int
    bbox: tuple[float, float, float, float]
    area_px: float
    area_m2: float | None
    near_frac: float
    area_source: str  # sam | box_fallback | none
    irc_band: str | None
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "defect_id": self.defect_id,
            "class": self.class_name,
            "frame_best": self.frame_best,
            "area_px": round(self.area_px, 1),
            "area_m2": None if self.area_m2 is None else round(self.area_m2, 4),
            "near_frac": round(self.near_frac, 3),
            "area_source": self.area_source,
            "irc_band": self.irc_band,
            "note": self.note,
        }


def _rect_mask(bbox, h: int, w: int) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    m = np.zeros((h, w), dtype=bool)
    xa, xb = int(max(0, x1)), int(min(w, x2))
    ya, yb = int(max(0, y1)), int(min(h, y2))
    if xb > xa and yb > ya:
        m[ya:yb, xa:xb] = True
    return m


def _clip_box(bbox, h: int, w: int) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [
        float(np.clip(x1, 0, w - 1)),
        float(np.clip(y1, 0, h - 1)),
        float(np.clip(x2, 1, w)),
        float(np.clip(y2, 1, h)),
    ]


class BoxFallbackSegmenter:
    """Filled AABB — overestimates; used only when SAM cannot run."""

    def mask(self, frame: np.ndarray, bbox) -> np.ndarray:
        h, w = frame.shape[:2]
        return _rect_mask(bbox, h, w)


class SamBoxSegmenter:
    def __init__(self, model_name: str = "sam2.1_b.pt"):
        from ultralytics import SAM

        self.model = SAM(model_name)

    def mask(self, frame: np.ndarray, bbox) -> np.ndarray:
        h, w = frame.shape[:2]
        box = _clip_box(bbox, h, w)
        if box[2] <= box[0] + 1 or box[3] <= box[1] + 1:
            return _rect_mask(bbox, h, w)
        result = self.model(frame, bboxes=[box], verbose=False)[0]
        masks = getattr(result, "masks", None)
        if masks is None or masks.data is None or len(masks.data) == 0:
            return _rect_mask(bbox, h, w)
        m = masks.data[0].detach().cpu().numpy()
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        return m.astype(bool)


def load_segmenter(sam_model: str | None, use_sam: bool = True):
    if not use_sam:
        return BoxFallbackSegmenter(), "box_fallback"
    name = sam_model or "sam2.1_b.pt"
    try:
        return SamBoxSegmenter(name), "sam"
    except Exception as e:
        print(f"SAM unavailable ({e}) — using filled-box area (overestimate)")
        return BoxFallbackSegmenter(), "box_fallback"


def apply_mask(
    frame: np.ndarray,
    bbox,
    near_mask: np.ndarray | None,
    area_map: np.ndarray | None,
    segmenter,
    source: str,
    track: Track,
) -> tuple[np.ndarray, AreaMeasurement]:
    h, w = frame.shape[:2]
    raw = segmenter.mask(frame, bbox)
    if near_mask is not None and near_mask.shape[:2] == raw.shape[:2]:
        geom = raw & near_mask
        if not geom.any():
            geom = raw
    else:
        geom = raw
    area_px = float(geom.sum())
    if near_mask is not None and np.asarray(near_mask).any():
        near_frac = float(geom.sum() / float(np.asarray(near_mask).sum()))
    else:
        near_frac = float(geom.mean()) if geom.size else 0.0
    area_m2 = mask_area_m2(geom, area_map)
    note = ""
    if source == "box_fallback":
        note = "filled bbox — overestimates true plan area"
    elif area_m2 is None:
        note = "mask ok but no camera scale — set --camera-height-m and --hfov-deg"
    band = pothole_irc_band(area_m2) if track.class_name == "pothole" else None
    if track.class_name == "rutting":
        note = (note + "; " if note else "") + (
            "rutting plan area is not a valid IRC quantity (indicative only)"
        )
    meas = AreaMeasurement(
        defect_id=track.track_id,
        class_name=track.class_name,
        frame_best=track.frame_best,
        bbox=tuple(float(x) for x in bbox),
        area_px=area_px,
        area_m2=area_m2,
        near_frac=near_frac,
        area_source=source,
        irc_band=band,
        note=note,
    )
    return geom, meas


def overlay_mask(frame: np.ndarray, mask: np.ndarray, bbox, label: str) -> np.ndarray:
    out = frame.copy()
    color = np.array([0, 220, 80], dtype=np.float32)
    sel = mask.astype(bool)
    if sel.any():
        blend = out[sel].astype(np.float32) * 0.45 + color * 0.55
        out[sel] = np.clip(blend, 0, 255).astype(np.uint8)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 180, 255), 2)
    cv2.putText(
        out, label, (x1, max(16, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return out


def measure_tracks_on_video(
    video_path: Path,
    tracks: list[Track],
    cfg,
    area_map: np.ndarray | None,
    segmenter,
    source: str,
    qa_dir: Path | None = None,
    undistort: tuple | None = None,
) -> list[AreaMeasurement]:
    """Second pass: seek each track's nearest frame, prompt SAM, measure."""
    from .near_field import build_near_field

    if not tracks:
        return []
    needed: dict[int, list[Track]] = {}
    for tr in tracks:
        needed.setdefault(int(tr.frame_best), []).append(tr)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not reopen video for area measurement: {video_path}")

    if qa_dir is not None:
        qa_dir.mkdir(parents=True, exist_ok=True)

    measurements: list[AreaMeasurement] = []
    frame_i = 0
    pending = set(needed)
    while pending:
        ok, frame = cap.read()
        if not ok:
            break
        if undistort is not None:
            frame = remap_undistort(frame, undistort)
        if frame_i in needed:
            nf = build_near_field(frame, cfg)
            for tr in needed[frame_i]:
                bbox = tr.bbox_best or tr.bbox
                geom, meas = apply_mask(
                    frame, bbox, nf.mask, area_map, segmenter, source, tr
                )
                measurements.append(meas)
                if qa_dir is not None:
                    label = f"id{tr.track_id} {tr.class_name}"
                    if meas.area_m2 is not None:
                        label += f" {meas.area_m2:.3f}m2"
                    vis = overlay_mask(frame, geom, bbox, label)
                    cv2.imwrite(str(qa_dir / f"defect_{tr.track_id:04d}.jpg"), vis)
            pending.discard(frame_i)
        frame_i += 1
    cap.release()

    if pending:
        print(f"WARNING: {len(pending)} track frames not found in video (seek miss)")
        have = {m.defect_id for m in measurements}
        for tr in tracks:
            if tr.track_id not in have:
                measurements.append(
                    AreaMeasurement(
                        defect_id=tr.track_id,
                        class_name=tr.class_name,
                        frame_best=tr.frame_best,
                        bbox=tuple(tr.bbox_best or tr.bbox),
                        area_px=0.0,
                        area_m2=None,
                        near_frac=0.0,
                        area_source="none",
                        irc_band=None,
                        note="best frame not in video",
                    )
                )
    measurements.sort(key=lambda m: m.defect_id)
    return measurements


def remap_undistort(frame, maps):
    from .camera import remap_frame

    return remap_frame(frame, maps[0], maps[1])
