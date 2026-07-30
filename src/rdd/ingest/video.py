"""Video ingest: detect format, convert Insta360 dual-fisheye -> equirectangular.

Outcomes:
  * .mp4 already equirectangular  -> pass through (returned as-is)
  * .insv / .insp dual-fisheye    -> convert via ffmpeg v360 (if auto_convert)
                                     else raise with Insta360 Studio instructions

We can't reliably read Insta360's proprietary lens calibration from the raw
container, so the v360 dual-fisheye conversion is a *best-effort* geometric
remap. For production-grade stitching, export from Insta360 Studio. This is
documented in the README and surfaced in the exception message.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..utils import ffmpeg
from ..utils.logging import get_logger

log = get_logger("rdd.ingest.video")

EQUIRECT_EXTS = {".mp4", ".mov", ".mkv"}
INSTA360_EXTS = {".insv", ".insp"}

_STUDIO_INSTRUCTIONS = (
    "Input is an Insta360 dual-fisheye file ({ext}).\n"
    "Best quality path: open it in Insta360 Studio and export as an\n"
    "EQUIRECTANGULAR .mp4 (2:1 aspect, e.g. 5760x2880), then re-run the\n"
    "pipeline on that .mp4. To attempt an automatic (lower-quality) FFmpeg\n"
    "conversion instead, set ingest.auto_convert_insv: true in config.yaml."
)


@dataclass
class IngestResult:
    video_path: Path        # equirectangular video ready for preprocess
    source_path: Path       # original input
    was_converted: bool
    projection: str         # "equirect"
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    duration_s: float | None = None


def _probe_dims(path: Path) -> tuple[int | None, int | None, float | None, float | None]:
    info = ffmpeg.probe(path)
    if not info:
        return None, None, None, None
    vstreams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    if not vstreams:
        return None, None, None, None
    v = vstreams[0]
    w = v.get("width")
    h = v.get("height")
    fps = None
    if v.get("avg_frame_rate", "0/0") not in ("0/0", None):
        num, _, den = v["avg_frame_rate"].partition("/")
        try:
            fps = float(num) / float(den) if float(den) else None
        except (ValueError, ZeroDivisionError):
            fps = None
    dur = None
    try:
        dur = float(info.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        dur = None
    return w, h, fps, dur


def _convert_dual_fisheye(src: Path, dst: Path, out_w: int = 5760, out_h: int = 2880) -> None:
    """Best-effort dual-fisheye -> equirectangular via ffmpeg v360.

    Assumes the two circular fisheyes are packed side-by-side (Insta360 X-series
    raw layout). ih_fov/iv_fov ~193 matches typical 200-deg-ish lenses with
    overlap. Tune in config if your camera differs.
    """
    ffmpeg.run(
        [
            "-i", str(src),
            "-vf",
            (
                f"v360=input=dfisheye:output=equirect:"
                f"ih_fov=193:iv_fov=193:w={out_w}:h={out_h}:interp=lanczos"
            ),
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-an",
            str(dst),
        ],
        desc="dual-fisheye -> equirect",
    )


def ingest_video(input_path: str | Path, cfg) -> IngestResult:
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Input video not found: {src}")

    ext = src.suffix.lower()

    if ext in EQUIRECT_EXTS:
        w, h, fps, dur = _probe_dims(src)
        profile = cfg.get_path("view.profile", "car_360")
        # Only 360 footage should be ~2:1. Warning about it for a drone or flat
        # car camera would be noise, and staying silent for car_360 would let a
        # non-equirect input be reprojected into nonsense.
        if profile == "car_360" and w and h and abs(w - 2 * h) > 0.1 * w:
            log.warning(
                "Input %s is %dx%d — not ~2:1, but view.profile is 'car_360' which "
                "expects equirectangular. If this is a normal (rectilinear) camera, "
                "set view.profile to 'car_flat' or 'drone_nadir'; otherwise "
                "reprojection will be wrong.", src.name, w, h,
            )
        log.info("Video passthrough: %s (%sx%s @ %sfps, view.profile=%s)",
                 src.name, w, h, fps, profile)
        projection = "equirect" if profile == "car_360" else "rectilinear"
        return IngestResult(src, src, False, projection, w, h, fps, dur)

    if ext in INSTA360_EXTS:
        if not cfg.get_path("ingest.auto_convert_insv", False):
            raise RuntimeError(_STUDIO_INSTRUCTIONS.format(ext=ext))
        cache_dir = Path(cfg.get_path("ingest.equirect_cache", "data/raw"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        dst = cache_dir / (src.stem + "_equirect.mp4")
        if dst.exists():
            log.info("Reusing cached equirect conversion: %s", dst)
        else:
            log.info("Converting %s -> equirectangular (best-effort v360)...", src.name)
            _convert_dual_fisheye(src, dst)
        w, h, fps, dur = _probe_dims(dst)
        return IngestResult(dst, src, True, "equirect", w, h, fps, dur)

    raise ValueError(
        f"Unsupported input extension {ext!r}. Supported: "
        f"{sorted(EQUIRECT_EXTS | INSTA360_EXTS)}"
    )
