"""Frame sampling by distance traveled (GPS) or optical-flow odometry.

Rural survey footage is hugely redundant frame-to-frame. Sampling by *distance*
(a frame every N metres) gives even spatial coverage for labeling/training and
cuts near-duplicate leakage. Falls back to optical-flow odometry when GPS is
absent, and to fixed-N as a last resort.

Two things happen here besides sampling, both deliberate:

**Unusable frames are skipped.** A motion-blurred or blown-out frame is worse
than no frame: label it and you teach the model on mush, feed it to the detector
and you get confident nonsense. Counts of what was dropped and why are returned,
never silently discarded.

**Saved frames are enhanced with the run's spec.** These frames become the
labeling set, so they must look exactly like what the detector will see at
inference time. Writing raw frames here and enhancing only at inference is a
silent train/serve skew, so both paths call the same `enhance_frame`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..quality.enhance import EnhanceSpec, enhance_frame
from ..quality.metrics import QualityProfile, judge, measure_frame
from ..utils.geo import GpsTrack
from ..utils.logging import get_logger

log = get_logger("rdd.preprocess.sampling")


@dataclass
class SampledFrame:
    index: int          # frame index in the (flat) video
    t: float            # seconds
    path: str
    lat: float | None = None
    lon: float | None = None
    dist_m: float | None = None
    sharpness: float | None = None


@dataclass
class SamplingResult:
    frames: list[SampledFrame] = field(default_factory=list)
    frames_dir: str = ""
    mode: str = ""
    frames_read: int = 0
    skipped_unusable: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)
    enhanced: bool = False

    def summary(self) -> dict:
        return {
            "sampled": len(self.frames),
            "mode": self.mode,
            "frames_read": self.frames_read,
            "skipped_unusable": self.skipped_unusable,
            "skip_reasons": dict(self.skip_reasons),
            "enhanced": self.enhanced,
        }


def _optical_flow_step(prev_gray, gray) -> float:
    """Pseudo-distance for this frame step from mean optical-flow magnitude.

    Not metric — used only to *space out* samples evenly when GPS is absent. The
    scale is arbitrary but monotonic with real motion.

    Run on downscaled frames: this is dense Farneback, whose cost is proportional to
    pixel count, and it runs on every frame. At 1920x1080 it took 12 minutes over a
    30-second clip. A ~480px working width is over 15x cheaper and makes no difference
    to a measure that is only ever compared against itself.
    """
    import cv2
    import numpy as np

    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return float(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean())


def _flow_gray(frame, work_width: int):
    """Greyscale, downscaled copy for optical flow."""
    import cv2

    h, w = frame.shape[:2]
    if work_width and w > work_width:
        scale = work_width / float(w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def sample_frames(video_path: Path, out_dir: Path, gps: GpsTrack, cfg,
                  profile: QualityProfile | None = None,
                  spec: EnhanceSpec | None = None) -> SamplingResult:
    import cv2

    sc = cfg.get_path("preprocess.sampling", {}) or {}
    if not sc.get("enabled", True):
        # Sampling exists to build a LABELING set. A pure inference run does not need
        # it, and on GPS-less footage it is the single most expensive stage because
        # distance has to be estimated by optical flow on every frame.
        log.info("Frame sampling disabled — no labeling frames will be written")
        return SamplingResult(frames_dir=str(out_dir), mode="disabled")
    mode = sc.get("mode", "distance")
    out_dir = Path(out_dir)
    save = bool(sc.get("save_frames", True))
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for sampling: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    use_gps = mode == "distance" and gps.has_data
    use_flow = (
        mode == "distance" and not gps.has_data
        and sc.get("odometry_fallback", "optical_flow") == "optical_flow"
    )
    if mode == "distance" and not use_gps and not use_flow:
        log.warning("distance sampling requested but no GPS/odometry — using every_n")
        mode = "every_n"

    target_dist = float(sc.get("distance_m", 2.0))
    flow_scale = float(sc.get("flow_units_per_m", 5.0))
    time_step = float(sc.get("time_s", 0.5))
    every_n = max(1, int(sc.get("every_n", 15)))
    flow_width = int(sc.get("flow_work_width", 480))
    drop_unusable = bool(cfg.get_path("quality.assess.drop_unusable", True)) \
        and profile is not None and profile.enabled

    result = SamplingResult(
        frames_dir=str(out_dir),
        mode="distance(flow)" if use_flow else mode,
        enhanced=bool(spec and spec.enabled and save),
    )

    idx = -1
    last_saved_dist = 0.0
    accum_flow = 0.0
    last_t_saved = -1e9
    prev_gray = None
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        result.frames_read += 1
        t = idx / fps
        keep = False

        if use_gps:
            d = gps.distance_at_time(t) or 0.0
            if saved == 0 or (d - last_saved_dist) >= target_dist:
                keep = True
        elif use_flow:
            gray = _flow_gray(frame, flow_width)
            if prev_gray is not None:
                accum_flow += _optical_flow_step(prev_gray, gray)
            prev_gray = gray
            if saved == 0 or accum_flow >= target_dist * flow_scale:
                keep = True
        elif mode == "time":
            if saved == 0 or (t - last_t_saved) >= time_step:
                keep = True
        else:  # every_n
            keep = idx % every_n == 0

        if not keep:
            continue

        if drop_unusable:
            q = judge(measure_frame(frame, idx), profile)
            if not q.usable:
                result.skipped_unusable += 1
                for r in q.reasons:
                    key = r.split("(")[0]
                    result.skip_reasons[key] = result.skip_reasons.get(key, 0) + 1
                # Do not advance the sampling cursor: we still owe a frame for
                # this stretch of road, so the next usable one is taken instead.
                continue
            sharpness = q.sharpness
        else:
            sharpness = None

        if use_gps:
            last_saved_dist = gps.distance_at_time(t) or 0.0
        elif use_flow:
            accum_flow = 0.0
        elif mode == "time":
            last_t_saved = t

        fix = gps.at_time(t) if gps.has_data else None
        fpath = ""
        if save:
            out = enhance_frame(frame, spec) if (spec and spec.enabled) else frame
            fpath = str(out_dir / f"frame_{idx:07d}.jpg")
            cv2.imwrite(fpath, out, [cv2.IMWRITE_JPEG_QUALITY,
                                     int(sc.get("jpeg_quality", 95))])
        result.frames.append(
            SampledFrame(
                index=idx, t=round(t, 3), path=fpath,
                lat=fix.lat if fix else None, lon=fix.lon if fix else None,
                dist_m=round(last_saved_dist, 2) if use_gps else None,
                sharpness=round(sharpness, 2) if sharpness is not None else None,
            )
        )
        saved += 1

    cap.release()
    log.info("Sampled %d frames from %d read (mode=%s) -> %s",
             saved, result.frames_read, result.mode, out_dir)
    if result.skipped_unusable:
        log.warning("Skipped %d candidate frames on quality: %s",
                    result.skipped_unusable, result.skip_reasons)
    if result.enhanced:
        log.info("Saved frames were enhanced with spec %s — label these, not the raw "
                 "video, so training matches inference", spec.fingerprint())
    return result
