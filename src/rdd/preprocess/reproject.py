"""360 equirectangular -> flat rectilinear "road camera" via FFmpeg v360.

Produces one rectified .mp4 (virtual camera pitched down at the road) that the
rest of the pipeline operates on. Single FFmpeg pass — fast and deterministic.

Two things here decide how much detail survives to the detector, and both used
to be silently wrong:

**Output resolution.** An equirectangular frame spreads 360° across its width, so
a view of `h_fov` degrees is carved out of only `src_width * h_fov/360` pixels of
real angular detail. Asking for a smaller output than that throws detail away
before the detector ever runs; asking for a much larger one just interpolates and
costs time. With `out_width: auto` we match the source's angular resolution.

**Aspect ratio.** A rectilinear (gnomonic) projection has square pixels only when
`w/h == tan(h_fov/2) / tan(v_fov/2)`. Any other pair stretches the image — which
distorts defect shape and, worse, does so invisibly. `preserve_aspect` derives
`out_height` from the FOVs instead of trusting a hand-typed pair.
"""
from __future__ import annotations

import math
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
    source_width: int | None = None
    source_height: int | None = None
    resolution_note: str = ""


def _even(n: float) -> int:
    """Round to a positive even integer — H.264 requires even dimensions."""
    v = int(round(n / 2.0)) * 2
    return max(2, v)


def native_angular_width(src_w: int, h_fov_deg: float) -> int:
    """Pixels of genuine angular detail an equirect source has across h_fov."""
    return int(round(src_w * (float(h_fov_deg) / 360.0)))


def plan_output_size(rc: dict, src_w: int | None, src_h: int | None) -> tuple[int, int, str]:
    """Resolve (out_width, out_height, explanation) for the flat view."""
    h_fov = float(rc.get("h_fov_deg", 110.0))
    v_fov = float(rc.get("v_fov_deg", 70.0))
    want = rc.get("out_width", "auto")
    preserve = bool(rc.get("preserve_aspect", True))
    notes: list[str] = []

    if want in (None, "auto"):
        if not src_w:
            out_w = 1920
            notes.append("source width unknown (no ffprobe) — defaulting to 1920 px wide")
        else:
            native = native_angular_width(src_w, h_fov)
            lo = int(rc.get("min_width", 960))
            hi = int(rc.get("max_width", 3840))
            out_w = max(lo, min(hi, native))
            notes.append(
                f"auto width {out_w} px from {src_w}px equirect across {h_fov:g}° "
                f"(native angular detail {native} px"
                + (f", clamped to [{lo},{hi}]" if out_w != native else "")
                + ")"
            )
    else:
        out_w = int(want)
        if src_w:
            native = native_angular_width(src_w, h_fov)
            if out_w < 0.9 * native:
                notes.append(
                    f"WARNING out_width {out_w} discards detail: the source carries "
                    f"~{native} px across {h_fov:g}°. Use 'auto' to keep it."
                )
            elif out_w > 1.5 * native:
                notes.append(
                    f"out_width {out_w} exceeds the source's ~{native} px of real "
                    f"detail across {h_fov:g}° — upsampling, not extra information."
                )

    aspect = math.tan(math.radians(h_fov) / 2.0) / math.tan(math.radians(v_fov) / 2.0)
    if preserve or rc.get("out_height", "auto") in (None, "auto"):
        out_h = _even(out_w / aspect)
        notes.append(f"height {out_h} derived from h_fov/v_fov for square pixels")
    else:
        out_h = _even(int(rc.get("out_height", 720)))

    return _even(out_w), out_h, "; ".join(notes)


def _v360_flat_filter(rc: dict, out_w: int, out_h: int) -> str:
    """Build the v360 filter string for equirect -> flat.

    v360 angles: yaw (left/right), pitch (up/down; negative looks down), roll.
    output=flat is the gnomonic/rectilinear projection.
    """
    return (
        "v360="
        "input=equirect:output=flat:"
        f"yaw={rc.get('yaw_deg', 0.0)}:"
        f"pitch={rc.get('pitch_deg', -30.0)}:"
        f"roll={rc.get('roll_deg', 0.0)}:"
        f"h_fov={rc.get('h_fov_deg', 110.0)}:"
        f"v_fov={rc.get('v_fov_deg', 70.0)}:"
        f"w={out_w}:h={out_h}:"
        f"interp={rc.get('interp', 'lanczos')}"
    )


def _probe_size(path: Path) -> tuple[int | None, int | None]:
    info = ffmpeg.probe(path)
    for s in info.get("streams", []):
        if s.get("codec_type") == "video":
            return s.get("width"), s.get("height")
    return None, None


def reproject_video(equirect_path: Path, out_dir: Path, cfg,
                    view=None) -> ReprojectResult:
    """Flatten an equirect video to a virtual road camera.

    When the viewpoint is already rectilinear (drone nadir, flat car camera) this
    is a no-op passthrough: reprojecting an already-flat image would only
    resample it and lose sharpness.
    """
    rc = dict(cfg.get_path("preprocess.reproject", {}) or {})
    src_w, src_h = _probe_size(Path(equirect_path))

    if view is not None and not view.needs_reprojection:
        log.info("Viewpoint '%s' is already rectilinear — skipping 360 reprojection "
                 "(no resampling, no quality loss)", view.name)
        return ReprojectResult(Path(equirect_path), src_w or 0, src_h or 0,
                               src_w, src_h, "passthrough (rectilinear source)")

    if not rc.get("enabled", True):
        log.info("Reprojection disabled — treating input as already flat: %s", equirect_path)
        return ReprojectResult(Path(equirect_path), src_w or 0, src_h or 0,
                               src_w, src_h, "passthrough (reprojection disabled)")

    if not ffmpeg.has_v360():
        raise RuntimeError(
            "FFmpeg is present but the v360 filter is unavailable. Install a full "
            "FFmpeg build (gyan.dev on Windows) that includes v360."
        )

    out_w, out_h, note = plan_output_size(rc, src_w, src_h)
    for line in note.split("; "):
        (log.warning if line.startswith("WARNING") else log.info)("reproject: %s", line)

    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "rectified.mp4"
    lossless = bool(rc.get("lossless", False))
    if lossless:
        log.info("reproject: lossless intermediate requested — expect a large file")

    ffmpeg.run(
        [
            "-i", str(equirect_path),
            "-vf", _v360_flat_filter(rc, out_w, out_h),
            *ffmpeg.encode_args(
                crf=int(rc.get("crf", 18)),
                preset=str(rc.get("preset", "medium")),
                lossless=lossless,
            ),
            "-an",
            str(dst),
        ],
        desc="equirect -> flat road view",
    )
    log.info("Rectified video: %s (%dx%d)", dst, out_w, out_h)
    return ReprojectResult(dst, out_w, out_h, src_w, src_h, note)
