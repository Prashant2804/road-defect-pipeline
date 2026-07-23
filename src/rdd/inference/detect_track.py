"""Detector + tracker over the full rectified video.

Runs ultralytics `.track()` (BoT-SORT or ByteTrack) so each physical defect gets
a stable track ID across frames. Builds a UniqueCounter, renders an annotated
video (masks + IDs + live unique-count HUD), and returns structured tracks for
the report stage. Reports BOTH raw per-frame detections and unique-track counts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..utils.device import resolve_device
from ..utils.geo import GpsTrack
from ..utils.logging import get_logger
from .counter import TrackObservation, UniqueCounter
from .render import draw_frame, draw_hud

log = get_logger("rdd.inference")


@dataclass
class InferenceResult:
    annotated_video: Path
    counter: UniqueCounter
    raw_detections: int
    unique_counts: dict[str, int]
    fps: float


def _tracker_yaml(cfg) -> str:
    custom = cfg.get_path("inference.tracker_cfg")
    if custom:
        return custom
    name = cfg.get_path("inference.tracker", "botsort")
    return "bytetrack.yaml" if name == "bytetrack" else "botsort.yaml"


def run_inference(video_path, model, cfg, gps: GpsTrack | None = None,
                  out_dir: Path | None = None) -> InferenceResult:
    import cv2
    import numpy as np

    video_path = Path(video_path)
    out_dir = Path(out_dir) if out_dir else video_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    gps = gps or GpsTrack()

    class_names = cfg.get_path("model.classes")
    ic = cfg.get_path("inference", {}) or {}
    device = resolve_device(cfg.get_path("run.device", "auto"))
    counter = UniqueCounter(class_names, min_track_len=int(ic.get("min_track_len", 3)))

    # Probe source fps/size via OpenCV for the writer.
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    out_fps = cfg.get_path("inference.render.fps") or src_fps

    annotated_path = out_dir / "annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(annotated_path), fourcc, out_fps, (W, H))

    log.info("Tracking with %s on %s (device=%s, %dx%d @ %.1ffps)",
             ic.get("tracker", "botsort"), video_path.name, device, W, H, src_fps)

    stream = model.track(
        source=str(video_path),
        stream=True,
        persist=True,
        tracker=_tracker_yaml(cfg),
        conf=float(ic.get("conf", 0.25)),
        iou=float(ic.get("iou", 0.5)),
        imgsz=int(ic.get("imgsz", 960)),
        device=device,
        verbose=False,
    )

    frame_idx = -1
    for result in stream:
        frame_idx += 1
        frame = result.orig_img
        if frame is None:
            continue
        fh, fw = frame.shape[:2]
        t = frame_idx / src_fps
        fix = gps.at_time(t) if gps.has_data else None

        detections: list[dict] = []
        boxes = result.boxes
        masks = result.masks
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.int().cpu().tolist()
            clss = boxes.cls.int().cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            xyxy = boxes.xyxy.cpu().numpy()
            mask_data = masks.data.cpu().numpy() if masks is not None else None

            for i, (tid, cid, cf) in enumerate(zip(ids, clss, confs)):
                mask_bool = None
                area = 0.0
                if mask_data is not None and i < len(mask_data):
                    m = mask_data[i]
                    if m.shape != (fh, fw):
                        m = cv2.resize(m, (fw, fh), interpolation=cv2.INTER_NEAREST)
                    mask_bool = m.astype(bool)
                    area = float(mask_bool.sum())
                bbox = tuple(xyxy[i].tolist())
                obs = TrackObservation(
                    frame=frame_idx, t=round(t, 3), conf=float(cf),
                    mask_area_px=area, bbox=bbox,
                    lat=fix.lat if fix else None, lon=fix.lon if fix else None,
                )
                counter.update(tid, cid, obs)
                detections.append(
                    {"track_id": tid, "cls_id": cid, "conf": float(cf),
                     "mask": mask_bool, "bbox": bbox}
                )

        frame = draw_frame(frame, detections, class_names, cfg)
        if ic.get("render", {}).get("running_count_overlay", True):
            running = counter.running_unique_total(frame_idx)
            # per-class running breakdown
            pc = {}
            for tr in counter.tracks.values():
                if tr.n_frames >= counter.min_track_len and tr.first_frame <= frame_idx:
                    pc[tr.cls_name] = pc.get(tr.cls_name, 0) + 1
            frame = draw_hud(frame, running, pc, frame_idx)

        writer.write(frame)

    writer.release()
    unique = counter.unique_counts()
    log.info("Inference done: %d frames, %d raw detections, %d unique defects %s",
             frame_idx + 1, counter.raw_detections, sum(unique.values()), unique)
    return InferenceResult(
        annotated_video=annotated_path,
        counter=counter,
        raw_detections=counter.raw_detections,
        unique_counts=unique,
        fps=src_fps,
    )
