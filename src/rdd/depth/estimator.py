"""Optional depth estimation. Toggled by depth.enabled in config.

Two backends (both optional heavy deps):
  * video_depth_anything : temporally-consistent metric video depth
  * yolo_depth           : ultralytics depth task (if available in the build)

If the backend isn't installed we log and return None so severity falls back to
mask-area-only. This stage NEVER blocks the core pipeline.
"""
from __future__ import annotations

from pathlib import Path

from ..utils.logging import get_logger

log = get_logger("rdd.depth")


def estimate_track_depths(video_path: Path, counter, cfg) -> dict[int, float] | None:
    """Return {track_id: representative_depth} or None if disabled/unavailable.

    Representative depth = median depth sampled inside the track's largest mask,
    at that mask's frame. Implemented against Video-Depth-Anything; guarded so a
    missing install degrades gracefully.
    """
    if not cfg.get_path("depth.enabled", False):
        log.info("Depth stage disabled — skipping (severity uses mask area only)")
        return None

    backend = cfg.get_path("depth.backend", "video_depth_anything")
    try:
        if backend == "video_depth_anything":
            return _video_depth_anything(video_path, counter, cfg)
        if backend == "yolo_depth":
            return _yolo_depth(video_path, counter, cfg)
        log.warning("Unknown depth backend %r — skipping", backend)
    except ImportError as e:
        log.warning("Depth backend %s not installed (%s) — skipping", backend, e)
    except Exception as e:
        log.warning("Depth estimation failed (%s) — skipping", e)
    return None


def _video_depth_anything(video_path, counter, cfg) -> dict[int, float]:
    # Import guarded: package installed from source per README.
    from video_depth_anything.video_depth import VideoDepthAnything  # type: ignore

    raise NotImplementedError(
        "Video-Depth-Anything hook is stubbed. Wire the model call here once the "
        "package is installed; sample depth inside each track's representative "
        "mask and return {track_id: median_depth_m}."
    )


def _yolo_depth(video_path, counter, cfg) -> dict[int, float]:
    raise NotImplementedError(
        "YOLO depth-task hook is stubbed pending availability in the ultralytics build."
    )
