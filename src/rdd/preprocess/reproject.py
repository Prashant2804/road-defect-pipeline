"""360 equirectangular -> flat rectilinear "road camera" via FFmpeg v360.

Produces one rectified .mp4 (the whole clip, virtual camera pitched down at the
road). The rest of the pipeline (sampling, inference) operates on this flat
video. This is a single FFmpeg pass — fast and deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..utils import ffmpeg
from ..utils.logging import get_logger

log = get_logger("rdd.preprocess.reproject")


@dataclass
class ReprojectResult:
    video_path: Path
    width: int
    height: int


def _v360_flat_filter(rc: dict) -> str:
    """Build the v360 filter string for equirect -> flat.

    v360 angle convention: yaw (left/right), pitch (up/down; negative looks
    down), roll. output=flat is a gnomonic/rectilinear projection.
    """
    return (
        "v360="
        "input=equirect:output=flat:"
        f"yaw={rc.get('yaw_deg', 0.0)}:"
        f"pitch={rc.get('pitch_deg', -30.0)}:"
        f"roll={rc.get('roll_deg', 0.0)}:"
        f"h_fov={rc.get('h_fov_deg', 110.0)}:"
        f"v_fov={rc.get('v_fov_deg', 70.0)}:"
        f"w={rc.get('out_width', 1280)}:"
        f"h={rc.get('out_height', 720)}:"
        f"interp={rc.get('interp', 'lanczos')}"
    )


def reproject_video(equirect_path: Path, out_dir: Path, cfg) -> ReprojectResult:
    rc = cfg.get_path("preprocess.reproject", {}) or {}
    if not rc.get("enabled", True):
        log.info("Reprojection disabled — using input as flat view: %s", equirect_path)
        # Caller passed a video already treated as flat.
        return ReprojectResult(Path(equirect_path), rc.get("out_width", 0), rc.get("out_height", 0))

    if not ffmpeg.has_v360():
        raise RuntimeError(
            "FFmpeg is present but the v360 filter is unavailable. Install a full "
            "FFmpeg build (gyan.dev on Windows) that includes v360."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "rectified.mp4"
    vf = _v360_flat_filter(rc)
    ffmpeg.run(
        [
            "-i", str(equirect_path),
            "-vf", vf,
            "-c:v", "libx264", "-crf", "20", "-preset", "medium",
            "-an",
            str(dst),
        ],
        desc="equirect -> flat road view",
    )
    log.info("Rectified video: %s", dst)
    return ReprojectResult(dst, rc.get("out_width", 1280), rc.get("out_height", 720))
