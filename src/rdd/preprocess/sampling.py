"""Frame sampling by distance traveled (GPS) or optical-flow odometry.

Rural survey footage is hugely redundant frame-to-frame. Sampling by *distance*
(a frame every N metres) gives even spatial coverage for labeling/training and
cuts near-duplicate leakage. Falls back to optical-flow odometry when GPS is
absent, and to fixed-N as a last resort.

Writes selected frames to frames_dir and returns a manifest list.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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


@dataclass
class SamplingResult:
    frames: list[SampledFrame] = field(default_factory=list)
    frames_dir: str = ""
    mode: str = ""


def _optical_flow_step_m(prev_gray, gray) -> float:
    """Return a pseudo-distance for this frame step from mean optical-flow
    magnitude. Not metric; used only to *space out* samples evenly when GPS is
    absent. Scale is arbitrary but monotonic with real motion."""
    import cv2
    import numpy as np

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return float(mag.mean())


def sample_frames(video_path: Path, out_dir: Path, gps: GpsTrack, cfg) -> SamplingResult:
    import cv2

    sc = cfg.get_path("preprocess.sampling", {}) or {}
    mode = sc.get("mode", "distance")
    out_dir = Path(out_dir)
    if sc.get("save_frames", True):
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
    time_step = float(sc.get("time_s", 0.5))
    every_n = int(sc.get("every_n", 15))

    result = SamplingResult(frames_dir=str(out_dir), mode=mode if not use_flow else "distance(flow)")
    idx = -1
    accum = 0.0          # accumulated distance/flow since last save
    last_t_saved = -1e9
    prev_gray = None
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        t = idx / fps
        keep = False

        if use_gps:
            cum = gps.cumulative_distance_m()
            # nearest fix -> cumulative distance; save when we've advanced target_dist
            fix = gps.at_time(t)
            if fix is not None:
                nearest_i = min(range(len(gps.fixes)), key=lambda i: abs(gps.fixes[i].t - t))
                d = cum[nearest_i]
                if d - accum >= target_dist or saved == 0:
                    keep = True
                    accum = d
        elif use_flow:
            import cv2 as _cv2

            gray = _cv2.cvtColor(frame, _cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                accum += _optical_flow_step_m(prev_gray, gray)
            prev_gray = gray
            # threshold in flow units; ~ target_dist scaled. Save first frame too.
            if saved == 0 or accum >= target_dist * 5.0:
                keep = True
                accum = 0.0
        elif mode == "time":
            if t - last_t_saved >= time_step or saved == 0:
                keep = True
                last_t_saved = t
        else:  # every_n
            if idx % every_n == 0:
                keep = True

        if keep:
            fix = gps.at_time(t) if gps.has_data else None
            fpath = ""
            if sc.get("save_frames", True):
                fpath = str(out_dir / f"frame_{idx:07d}.jpg")
                cv2.imwrite(fpath, frame)
            result.frames.append(
                SampledFrame(
                    index=idx, t=round(t, 3), path=fpath,
                    lat=fix.lat if fix else None,
                    lon=fix.lon if fix else None,
                )
            )
            saved += 1

    cap.release()
    log.info("Sampled %d frames (mode=%s) -> %s", saved, result.mode, out_dir)
    return result
