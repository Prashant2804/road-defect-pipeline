"""Telemetry / GPS extraction. Entirely optional — pipeline runs GPS-less.

Search order (config: ingest.telemetry.sources):
  * sidecar_gpx : a .gpx next to the video (or ingest.telemetry.gpx_path)
  * sidecar_srt : a .srt with GPS lines (common for drone/action-cam exports)
  * embedded    : GPS track muxed into the container (read via ffprobe)

Returns a GpsTrack (possibly empty). Never raises on missing GPS.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from ..utils import ffmpeg
from ..utils.geo import GpsFix, GpsTrack
from ..utils.logging import get_logger

log = get_logger("rdd.ingest.telemetry")


def _parse_gpx(path: Path) -> GpsTrack:
    try:
        import gpxpy
    except ImportError:
        log.warning("gpxpy not installed; cannot parse %s", path)
        return GpsTrack()
    with path.open("r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)
    fixes: list[GpsFix] = []
    t0: datetime | None = None
    for track in gpx.tracks:
        for seg in track.segments:
            for pt in seg.points:
                if pt.time is None:
                    continue
                if t0 is None:
                    t0 = pt.time
                fixes.append(
                    GpsFix(
                        t=(pt.time - t0).total_seconds(),
                        lat=pt.latitude,
                        lon=pt.longitude,
                        ele=pt.elevation,
                    )
                )
    log.info("Parsed %d GPS fixes from %s", len(fixes), path.name)
    return GpsTrack(fixes)


# SRT GPS lines vary by vendor; match "lat: 12.34 lon: 56.78" style and DJI-style.
_LATLON_RE = re.compile(
    r"(?:lat[:\s]+)(?P<lat>-?\d+\.\d+).*?(?:lon[g]?[:\s]+)(?P<lon>-?\d+\.\d+)",
    re.IGNORECASE | re.DOTALL,
)
_TS_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->")


def _parse_srt(path: Path) -> GpsTrack:
    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\s*\n", text)
    fixes: list[GpsFix] = []
    for b in blocks:
        m_ts = _TS_RE.search(b)
        m_ll = _LATLON_RE.search(b)
        if not (m_ts and m_ll):
            continue
        h, mn, s, ms = map(int, m_ts.groups())
        t = h * 3600 + mn * 60 + s + ms / 1000.0
        fixes.append(GpsFix(t=t, lat=float(m_ll["lat"]), lon=float(m_ll["lon"])))
    log.info("Parsed %d GPS fixes from SRT %s", len(fixes), path.name)
    return GpsTrack(fixes)


def _parse_embedded(video: Path) -> GpsTrack:
    """Try to read GPS from container tags. Coverage is vendor-dependent; this
    reads global location tags if present. Frame-accurate GoPro GPMF / Insta360
    telemetry needs a dedicated extractor (documented in README as a TODO)."""
    info = ffmpeg.probe(video)
    if not info:
        return GpsTrack()
    tags = {**info.get("format", {}).get("tags", {})}
    loc = tags.get("location") or tags.get("com.apple.quicktime.location.ISO6709")
    if loc:
        m = re.match(r"([+-]\d+\.\d+)([+-]\d+\.\d+)", loc)
        if m:
            log.info("Found single embedded location tag (no per-frame track)")
            return GpsTrack([GpsFix(t=0.0, lat=float(m[1]), lon=float(m[2]))])
    return GpsTrack()


def extract_telemetry(video_path: Path, source_path: Path, cfg) -> GpsTrack:
    tcfg = cfg.get_path("ingest.telemetry", {}) or {}
    if not tcfg.get("enabled", True):
        return GpsTrack()

    sources = tcfg.get("sources", ["sidecar_gpx", "sidecar_srt", "embedded"])
    # Look for sidecars next to the *original* source, not the converted mp4.
    stem_dir = source_path.parent
    stem = source_path.stem

    for src in sources:
        try:
            if src == "sidecar_gpx":
                p = Path(tcfg["gpx_path"]) if tcfg.get("gpx_path") else stem_dir / f"{stem}.gpx"
                if p.exists():
                    track = _parse_gpx(p)
                    if track.has_data:
                        return track
            elif src == "sidecar_srt":
                p = Path(tcfg["srt_path"]) if tcfg.get("srt_path") else stem_dir / f"{stem}.srt"
                if p.exists():
                    track = _parse_srt(p)
                    if track.has_data:
                        return track
            elif src == "embedded":
                track = _parse_embedded(video_path)
                if track.has_data:
                    return track
        except Exception as e:  # telemetry is best-effort; never fail the run
            log.warning("Telemetry source %s failed: %s", src, e)

    log.info("No usable GPS found — proceeding GPS-less (timestamps as location proxy)")
    return GpsTrack()
