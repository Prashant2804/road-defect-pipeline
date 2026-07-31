"""Detector + tracker over the rectified video, constrained to the road surface.

Order of operations per frame, and why:

  1. **Quality check.** An unusable frame is not detected on at all. It is still
     written to the annotated video (with a banner) so the output stays a
     complete, timeline-faithful record of the survey rather than a silently
     edited one.
  2. **Enhance.** The same `EnhanceSpec` used when the labeling frames were
     written, so the detector sees the distribution it was trained on.
  3. **Segment the road.** Everything after this is restricted to the drivable
     surface.
  4. **Classify the surface.** Water and mud become an occlusion mask.
  5. **Detect and track**, then gate: a detection that is not on the road is
     discarded, and one sitting under water/mud is flagged as unobservable.

Frames are read and fed to the tracker one at a time rather than handing
ultralytics the file path. That costs a little throughput but is required —
enhancement and optional road masking have to happen *between* decode and
detection, which the built-in dataloader gives no hook for.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..detect.aggregate import ConditionAggregator
from ..detect.boundary import BoundaryConfig, detect_edge_damage
from ..detect.confusers import ConfuserStats
from ..detect.confusers import check as check_confusers
from ..detect.linear import CRACK_SOURCES, LinearConfig, LinearStats, classify_crack
from ..detect.texture import (
    TextureConfig,
    detect_drainage,
    detect_ravelling,
    detect_rutting_proxy,
)
from ..quality.enhance import EnhanceSpec, enhance_frame
from ..quality.metrics import QualityProfile, judge, measure_frame
from ..roadseg.base import build_segmenter
from ..surface.condition import SurfaceStats, analyse_surface
from ..utils.device import resolve_device
from ..utils.geo import GpsTrack
from ..utils.logging import get_logger
from ..validity.checker import ValidityChecker
from ..validity.verdict import ValidityStats
from .counter import TrackObservation, UniqueCounter
from .render import draw_frame, draw_hud, draw_quality_banner

log = get_logger("rdd.inference")


@dataclass
class RoadSegStats:
    frames: int = 0
    fell_back: int = 0
    coverage_sum: float = 0.0
    confidence_sum: float = 0.0
    axis: str | None = None

    def update(self, rm) -> None:
        self.frames += 1
        self.fell_back += int(rm.fell_back)
        self.coverage_sum += rm.coverage()
        self.confidence_sum += rm.confidence
        self.axis = rm.axis or self.axis

    def _avg(self, total: float) -> float:
        return total / self.frames if self.frames else 0.0

    def summary(self) -> dict:
        return {
            "frames": self.frames,
            "mean_road_coverage": round(self._avg(self.coverage_sum), 4),
            "mean_confidence": round(self._avg(self.confidence_sum), 4),
            "fallback_frames": self.fell_back,
            "fallback_rate": round(self._avg(float(self.fell_back)), 4),
            "band_axis": self.axis,
        }


@dataclass
class InferenceResult:
    annotated_video: Path
    counter: UniqueCounter
    raw_detections: int
    unique_counts: dict[str, int]
    fps: float
    frames_total: int = 0
    frames_detected: int = 0
    frames_skipped_quality: int = 0
    quality_skip_reasons: dict[str, int] = field(default_factory=dict)
    surface: SurfaceStats = field(default_factory=SurfaceStats)
    roadseg: RoadSegStats = field(default_factory=RoadSegStats)
    validity: ValidityStats = field(default_factory=ValidityStats)
    conditions: ConditionAggregator = field(default_factory=ConditionAggregator)
    linear: LinearStats = field(default_factory=LinearStats)
    confusers: ConfuserStats = field(default_factory=ConfuserStats)
    scale_note: str = ""
    enhance_fingerprint: str = ""
    gating_mode: str = "gate"
    calibration: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "frames_total": self.frames_total,
            "frames_detected_on": self.frames_detected,
            "frames_skipped_quality": self.frames_skipped_quality,
            "quality_skip_reasons": dict(self.quality_skip_reasons),
            "detections_rejected_off_road": self.counter.rejected_off_road,
            "detections_rejected_out_of_zone": self.counter.rejected_out_of_zone,
            "gating_mode": self.gating_mode,
            "enhance_fingerprint": self.enhance_fingerprint,
            "ground_scale": self.scale_note,
            "roadseg": self.roadseg.summary(),
            "surface": self.surface.summary(),
            "validity": self.validity.summary(),
            "crack_classification": self.linear.summary(),
            "confuser_rejections": self.confusers.summary(),
            "surface_conditions": self.conditions.summary(),
            "calibration": self.calibration,
        }


def _tracker_yaml(cfg) -> str:
    custom = cfg.get_path("inference.tracker_cfg")
    if custom:
        return custom
    name = cfg.get_path("inference.tracker", "botsort")
    return "bytetrack.yaml" if name == "bytetrack" else "botsort.yaml"


def _rect_mask(bbox, h: int, w: int):
    """Fallback 'mask' for a box-only detection (non-seg model)."""
    import numpy as np

    m = np.zeros((h, w), dtype=bool)
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, max(x1 + 1, x2)), min(h, max(y1 + 1, y2))
    m[y1:y2, x1:x2] = True
    return m


def _overlap(det_mask, other) -> float:
    total = float(det_mask.sum())
    if total <= 0:
        return 0.0
    return float((det_mask & other).sum()) / total


def _without(road, exclude):
    """A copy of the road mask with an excluded region removed.

    Returns a shallow copy so the segmenter's own (temporally smoothed) mask is not
    mutated — otherwise a vehicle in one frame would carve a permanent hole in the
    road mask for the rest of the clip.
    """
    import copy

    trimmed = copy.copy(road)
    trimmed.mask = road.mask & ~exclude
    return trimmed


def run_inference(video_path, model, cfg, gps: GpsTrack | None = None,
                  out_dir: Path | None = None, view=None,
                  profile: QualityProfile | None = None,
                  spec: EnhanceSpec | None = None,
                  scaler=None, calibration=None) -> InferenceResult:
    import cv2

    from ..utils.ffmpeg import VideoWriter

    video_path = Path(video_path)
    out_dir = Path(out_dir) if out_dir else video_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    gps = gps or GpsTrack()
    spec = spec or EnhanceSpec(enabled=False)

    class_names = list(cfg.get_path("model.classes") or [])
    ic = cfg.get_path("inference", {}) or {}
    rc = cfg.get_path("inference.render", {}) or {}
    gc = cfg.get_path("roadseg.gating", {}) or {}
    device = resolve_device(cfg.get_path("run.device", "auto"))

    gating_mode = str(gc.get("mode", "gate"))
    min_road_overlap = float(gc.get("min_road_overlap", 0.3))
    min_gate_confidence = float(gc.get("min_confidence", 0.2))
    occlusion_threshold = float(cfg.get_path("surface.occlusion_threshold", 0.5))
    occluders = tuple(cfg.get_path("surface.occluder_classes") or ())
    surface_enabled = bool(cfg.get_path("surface.enabled", True))
    roadseg_stride = max(1, int(cfg.get_path("roadseg.stride", 1)))
    surface_stride = max(1, int(cfg.get_path("surface.stride", 1)))
    drop_unusable = (
        bool(cfg.get_path("quality.assess.drop_unusable", True))
        and profile is not None and profile.enabled
    )

    counter = UniqueCounter(
        class_names,
        min_track_len=int(ic.get("min_track_len", 3)),
        occlusion_threshold=occlusion_threshold,
        occluder_classes=occluders,
    )
    segmenter = build_segmenter(cfg, view)
    segmenter.reset()

    camera = getattr(calibration, "camera", None)
    zones = getattr(calibration, "zones", None)
    validity = ValidityChecker(cfg, camera=camera, zones=zones)

    lin_cfg = LinearConfig.from_cfg(cfg)
    tex_cfg = TextureConfig.from_cfg(cfg)
    bnd_cfg = BoundaryConfig.from_cfg(cfg)
    conditions_stride = max(1, int(cfg.get_path("detect.conditions_stride", 5)))
    do_conditions = bool(cfg.get_path("detect.conditions_enabled", True))
    do_confusers = bool(cfg.get_path("detect.confusers.enabled", True))
    texture_ksize = int(cfg.get_path("surface.texture_ksize", 7))
    ground_maps = None      # (x_map, z_map); depends only on calibration + frame size

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for inference: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps = float(rc.get("fps") or src_fps)

    annotated_path = out_dir / "annotated.mp4"
    result = InferenceResult(
        annotated_video=annotated_path, counter=counter, raw_detections=0,
        unique_counts={}, fps=src_fps, gating_mode=gating_mode,
        enhance_fingerprint=spec.fingerprint(),
        scale_note=scaler.describe() if scaler is not None else "pixel areas only",
    )

    log.info("Tracking with %s (device=%s), road gating=%s, enhancement=%s",
             ic.get("tracker", "botsort"), device, gating_mode, spec.describe())

    writer = None
    road = None
    surf = None
    frame_idx = -1

    try:
        while True:
            ok, raw = cap.read()
            if not ok:
                break
            frame_idx += 1
            result.frames_total += 1

            quality = judge(measure_frame(raw, frame_idx), profile) \
                if drop_unusable else None

            frame = enhance_frame(raw, spec) if spec.enabled else raw
            fh, fw = frame.shape[:2]
            if writer is None:
                writer = VideoWriter(
                    annotated_path, fw, fh, out_fps,
                    crf=int(rc.get("crf", 18)), preset=str(rc.get("preset", "medium")),
                )

            t = frame_idx / src_fps
            if road is None or frame_idx % roadseg_stride == 0:
                road = segmenter.segment(frame)
            result.roadseg.update(road)

            if surface_enabled and (surf is None or frame_idx % surface_stride == 0):
                surf = analyse_surface(frame, road, cfg)
            if surf is not None:
                result.surface.update(surf)

            # The validity gate decides whether this frame may be assessed at all.
            # It runs after road/surface because it consumes their outputs, and
            # before detection because a blocked frame must produce no detections.
            step_m = 0.0
            if gps.has_data:
                d_now = gps.distance_at_time(t)
                d_prev = gps.distance_at_time((frame_idx - 1) / src_fps)
                if d_now is not None and d_prev is not None:
                    step_m = max(0.0, d_now - d_prev)
            verdict = validity.check(frame_idx, t, frame, road=road, surface=surf,
                                     quality=quality, distance_m=step_m)

            if not verdict.assessable:
                result.frames_skipped_quality += 1
                for gate in verdict.block_gates:
                    result.quality_skip_reasons[gate] = \
                        result.quality_skip_reasons.get(gate, 0) + 1
                out = draw_frame(frame.copy(), [], class_names, cfg,
                                 road=road, surface=surf)
                writer.write(draw_quality_banner(out, verdict.block_reasons))
                continue

            # Regions the gate wants excluded (vehicles, windscreen dirt) are
            # removed from the road mask, so they cannot host a detection.
            if verdict.exclude_mask is not None:
                road = _without(road, verdict.exclude_mask)

            if camera is not None and ground_maps is None:
                gx, gz, _ = camera.ground_maps(fw, fh)
                ground_maps = (gx, gz)

            # Area and boundary conditions (ravelling, rutting proxy, edge damage,
            # drainage) are properties of a stretch of road rather than objects, so
            # they are accumulated on a stride instead of tracked.
            if do_conditions and camera is not None and frame_idx % conditions_stride == 0:
                baseline = getattr(road, "baseline", None)
                try:
                    result.conditions.update_ravelling(detect_ravelling(
                        frame, road.mask, camera, baseline, cfg, zones=zones,
                        x_map=ground_maps[0], z_map=ground_maps[1], tc=tex_cfg))
                    result.conditions.update_rutting(detect_rutting_proxy(
                        frame, road.mask, camera, baseline, cfg, zones=zones,
                        x_map=ground_maps[0], z_map=ground_maps[1], tc=tex_cfg))
                    result.conditions.update_boundary(
                        detect_edge_damage(road.mask, camera, cfg, zones=zones,
                                           x_map=ground_maps[0], z_map=ground_maps[1],
                                           bc=bnd_cfg),
                        chainage_m=(gps.distance_at_time(t) or 0.0) if gps.has_data else 0.0)
                    result.conditions.update_drainage(detect_drainage(
                        surf, road.mask, camera, cfg, zones=zones,
                        x_map=ground_maps[0], z_map=ground_maps[1]))
                except Exception as e:
                    log.warning("Surface-condition stage failed on frame %d (%s) — "
                                "continuing without it", frame_idx, e)

            frame_feats = None
            if do_confusers and getattr(road, "baseline", None) is not None:
                from ..roadseg.ops import compute_features

                frame_feats = compute_features(frame, texture_ksize)

            det_input = frame
            if gating_mode == "mask":
                import numpy as np

                det_input = np.where(road.mask[..., None], frame, 0)

            results = model.track(
                source=det_input, persist=True, tracker=_tracker_yaml(cfg),
                conf=float(ic.get("conf", 0.25)), iou=float(ic.get("iou", 0.5)),
                imgsz=int(ic.get("imgsz", 960)), device=device, verbose=False,
            )
            result.frames_detected += 1
            r = results[0] if results else None

            fix = gps.at_time(t) if gps.has_data else None
            detections = _collect(
                r, fh, fw, road, surf, counter, scaler, class_names,
                frame_idx, t, fix, gating_mode, min_road_overlap,
                min_gate_confidence, occlusion_threshold, zones,
                camera=camera, ground_maps=ground_maps, cfg=cfg, lin_cfg=lin_cfg,
                linear_stats=result.linear, confuser_stats=result.confusers,
                feats=frame_feats,
            )

            out = draw_frame(frame.copy(), detections, class_names, cfg,
                             road=road, surface=surf)
            if rc.get("running_count_overlay", True):
                out = draw_hud(
                    out, counter.running_unique_total(frame_idx),
                    counter.running_per_class(frame_idx), frame_idx,
                    occluded_frac=(surf.occluded_frac if surf else 0.0),
                    road_conf=road.confidence if road else 0.0,
                )
            if verdict.degraded:
                out = draw_quality_banner(out, verdict.degrade_reasons)
            writer.write(out)
    finally:
        cap.release()
        if writer is not None:
            writer.close()

    result.validity = validity.stats
    result.conditions.log_summary()
    if result.linear.seen:
        log.info("Crack classification (ground-plane geometry): %d measured, %d "
                 "reclassified from the model's label -> %s", result.linear.seen,
                 result.linear.relabelled, result.linear.summary()["by_class"])
    if result.confusers.rejected:
        log.info("Confusers rejected %d detections: %s", result.confusers.rejected,
                 result.confusers.summary()["by_confuser"])
    if calibration is not None:
        result.calibration = calibration.summary()
    validity.log_summary()

    result.raw_detections = counter.raw_detections
    result.unique_counts = counter.unique_counts()
    unique_total = sum(result.unique_counts.values())

    log.info("Inference done: %d frames (%d detected on, %d skipped on quality), "
             "%d raw detections, %d rejected off-road, %d unique defects %s",
             result.frames_total, result.frames_detected,
             result.frames_skipped_quality, counter.raw_detections,
             counter.rejected_off_road, unique_total, result.unique_counts)
    if result.surface.frames:
        log.info("Road surface: %.1f%% water, %.1f%% mud -> %.1f%% of the road "
                 "could not be assessed",
                 100 * result.surface.water_frac, 100 * result.surface.mud_frac,
                 100 * result.surface.unassessable_frac)
    occluded = counter.occluded_counts()
    if occluded:
        log.warning("%d confirmed defects are hidden under water/mud and will be "
                    "reported as indeterminate: %s",
                    sum(occluded.values()), occluded)
    return result


def _collect(r, fh: int, fw: int, road, surf, counter: UniqueCounter, scaler,
             class_names, frame_idx: int, t: float, fix, gating_mode: str,
             min_road_overlap: float, min_gate_confidence: float,
             occlusion_threshold: float, zones=None, camera=None,
             ground_maps=None, cfg=None, lin_cfg=None, linear_stats=None,
             confuser_stats=None, feats=None) -> list[dict]:
    """Turn one frame's tracker output into gated, annotated observations."""
    import cv2

    detections: list[dict] = []
    if r is None:
        return detections
    boxes = getattr(r, "boxes", None)
    if boxes is None or boxes.id is None:
        return detections

    ids = boxes.id.int().cpu().tolist()
    clss = boxes.cls.int().cpu().tolist()
    confs = boxes.conf.cpu().tolist()
    xyxy = boxes.xyxy.cpu().numpy()
    masks = getattr(r, "masks", None)
    mask_data = masks.data.cpu().numpy() if masks is not None and masks.data is not None else None

    # A low-confidence road mask (usually the geometric fallback) is an
    # assumption, not a measurement — gating on it would discard real defects
    # for failing to match a guess.
    gate_active = (
        gating_mode == "gate" and road is not None
        and road.confidence >= min_gate_confidence
    )

    for i, (tid, cid, cf) in enumerate(zip(ids, clss, confs)):
        bbox = tuple(float(v) for v in xyxy[i].tolist())

        mask_bool = None
        if mask_data is not None and i < len(mask_data):
            m = mask_data[i]
            if m.shape != (fh, fw):
                m = cv2.resize(m.astype("float32"), (fw, fh),
                               interpolation=cv2.INTER_NEAREST)
            mask_bool = m.astype(bool)
        geom = mask_bool if mask_bool is not None else _rect_mask(bbox, fh, fw)
        area_px = float(geom.sum())

        road_overlap = _overlap(geom, road.mask) if road is not None else 1.0
        if gate_active and road_overlap < min_road_overlap:
            counter.rejected_off_road += 1
            continue

        cls_name = (class_names[cid] if 0 <= cid < len(class_names)
                    else f"UNMAPPED_CLASS_{cid}")

        # Cracks are named by GEOMETRY, not by the model. Orientation is measured on
        # the road plane, where "along the road" vs "across it" is a direct
        # measurement instead of a perspective-confounded appearance cue, and
        # alligator cracking is identified by enclosed cells rather than texture.
        crack_geom = None
        if cls_name in CRACK_SOURCES and camera is not None and ground_maps is not None:
            crack_geom = classify_crack(
                geom, camera, scaler=scaler, cfg=cfg, lin=lin_cfg,
                x_map=ground_maps[0], z_map=ground_maps[1])
            if linear_stats is not None:
                linear_stats.update(cls_name, crack_geom)
            cls_name = crack_geom.cls_name
            cid = class_names.index(cls_name) if cls_name in class_names else cid

        # Reject the recurring look-alikes (shadow, tar patch, road marking, manhole,
        # joint) before they can become tracked defects.
        if confuser_stats is not None and cfg is not None and feats is not None:
            rejection = check_confusers(
                geom, feats, getattr(road, "baseline", None), cfg, cls_name=cls_name,
                shadow_mask=getattr(surf, "shadow", None) if surf is not None else None,
                angle_deg=crack_geom.angle_deg if crack_geom is not None else None,
            )
            confuser_stats.update(rejection)
            if rejection is not None:
                continue

        # Outside this class's assessment zone the camera cannot resolve it, so any
        # detection there is far more likely to be aliasing or compression noise
        # than a real defect. Rejecting it raises precision; the zone is reported so
        # the gap reads as "not assessed" rather than "clean".
        if zones is not None:
            zone_mask = zones.mask(cls_name, fw, fh)
            if zone_mask is not None and zone_mask.any():
                if _overlap(geom, zone_mask) < 0.5:
                    counter.rejected_out_of_zone += 1
                    continue

        occluded_frac = _overlap(geom, surf.occlusion) if surf is not None else 0.0
        if counter.is_occluder(cls_name):
            # This class *is* the water/mud. It cannot be occluded by itself.
            occluded_frac = 0.0

        # Measure only the on-road part of the defect: a detection straddling the
        # verge should not have the verge counted into its size.
        on_road = (geom & road.mask) if road is not None else geom
        area_m2 = scaler.area_m2(on_road) if scaler is not None else None

        counter.update(tid, cid, TrackObservation(
            frame=frame_idx, t=round(t, 3), conf=float(cf),
            mask_area_px=float(on_road.sum()) or area_px, bbox=bbox,
            lat=fix.lat if fix else None, lon=fix.lon if fix else None,
            road_overlap=road_overlap, occluded_frac=occluded_frac,
            area_m2=area_m2,
        ))
        detections.append({
            "track_id": tid, "cls_id": cid, "conf": float(cf),
            "mask": mask_bool, "bbox": bbox,
            "road_overlap": road_overlap, "occluded_frac": occluded_frac,
            "occluded": occluded_frac >= occlusion_threshold,
            "area_m2": area_m2,
        })
    return detections
