"""SRT / GPS loading for defect timelines (reuses rdd helpers when available)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

from .config import repo_root

# Prefer shared pipeline helpers
_SRC = repo_root() / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    from rdd.utils.geo import GpsFix, GpsTrack  # type: ignore
except Exception:  # minimal fallback
    from dataclasses import dataclass, field
    import bisect
    import math

    def _haversine_m(lat1, lon1, lat2, lon2) -> float:
        r = 6371000.0
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dp = math.radians(lat2 - lat1)
        dl = math.radians(lon2 - lon1)
        a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 2 * r * math.asin(math.sqrt(a))

    @dataclass
    class GpsFix:
        t: float
        lat: float
        lon: float
        ele: float | None = None

    @dataclass
    class GpsTrack:
        fixes: list[GpsFix] = field(default_factory=list)
        _times: list[float] | None = field(default=None, repr=False, compare=False)
        _cum: list[float] | None = field(default=None, repr=False, compare=False)

        @property
        def has_data(self) -> bool:
            return len(self.fixes) >= 2

        def __len__(self) -> int:
            return len(self.fixes)

        def _ensure_index(self) -> None:
            if self._times is not None and len(self._times) == len(self.fixes):
                return
            self.fixes.sort(key=lambda f: f.t)
            self._times = [f.t for f in self.fixes]
            cum = [0.0]
            for a, b in zip(self.fixes, self.fixes[1:]):
                cum.append(cum[-1] + _haversine_m(a.lat, a.lon, b.lat, b.lon))
            self._cum = cum

        def index_at_time(self, t: float) -> int | None:
            if not self.fixes:
                return None
            self._ensure_index()
            times = self._times or []
            i = bisect.bisect_left(times, t)
            if i == 0:
                return 0
            if i >= len(times):
                return len(times) - 1
            return i if (times[i] - t) < (t - times[i - 1]) else i - 1

        def at_time(self, t: float) -> GpsFix | None:
            i = self.index_at_time(t)
            return None if i is None else self.fixes[i]

        def distance_at_time(self, t: float) -> float | None:
            i = self.index_at_time(t)
            if i is None:
                return None
            self._ensure_index()
            return float((self._cum or [0.0])[i])


_TS_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->")

# Multiple dashcam / GoPro / DJI SRT layouts
_LATLON_PATTERNS = [
    re.compile(
        r"(?:lat(?:itude)?[:\s]+)(?P<lat>-?\d+\.\d+).*?"
        r"(?:lon(?:gitude)?[:\s]+)(?P<lon>-?\d+\.\d+)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"GPS\s*\(\s*(?P<lat>-?\d+\.\d+)\s*,\s*(?P<lon>-?\d+\.\d+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\[latitude[:\s:]*(?P<lat>-?\d+\.\d+)\].*?\[longitude[:\s:]*(?P<lon>-?\d+\.\d+)\]",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"(?P<lat>-?\d{1,2}\.\d{4,})\s*[,/\s]\s*(?P<lon>-?\d{1,3}\.\d{4,})",
    ),
]


def _parse_lat_lon(block: str) -> tuple[float, float] | None:
    for pat in _LATLON_PATTERNS:
        m = pat.search(block)
        if not m:
            continue
        lat, lon = float(m.group("lat")), float(m.group("lon"))
        # Sanity: reject clearly swapped / non-geo pairs
        if abs(lat) <= 90 and abs(lon) <= 180 and not (abs(lat) < 1e-5 and abs(lon) < 1e-5):
            return lat, lon
    return None


def parse_srt(path: Path) -> GpsTrack:
    try:
        from rdd.ingest.telemetry import _parse_srt  # type: ignore

        track = _parse_srt(path)
        if len(track) >= 1:
            return track
    except Exception:
        pass

    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\s*\n", text)
    fixes: list[GpsFix] = []
    for b in blocks:
        m_ts = _TS_RE.search(b)
        ll = _parse_lat_lon(b)
        if not (m_ts and ll):
            continue
        h, mn, s, ms = map(int, m_ts.groups())
        t = h * 3600 + mn * 60 + s + ms / 1000.0
        lat, lon = ll
        fixes.append(GpsFix(t=t, lat=lat, lon=lon))
    return GpsTrack(fixes)


def load_gps(video: Path, srt: Path | None = None) -> GpsTrack:
    """Load GPS from explicit SRT or sidecar next to the video."""
    candidates: list[Path] = []
    if srt is not None:
        candidates.append(Path(srt))
    stem = video.stem
    parent = video.parent
    candidates.extend(
        [
            parent / f"{stem}.srt",
            parent / f"{stem}.SRT",
            parent / f"{stem}.gpx",
        ]
    )
    # Also scan sibling .srt files in the same folder (Drive downloads)
    if parent.is_dir():
        for p in sorted(parent.glob("*.srt")) + sorted(parent.glob("*.SRT")):
            if p not in candidates:
                candidates.append(p)

    for p in candidates:
        if not p.exists():
            continue
        if p.suffix.lower() == ".srt":
            track = parse_srt(p)
            if len(track) >= 1:
                print(f"GPS: loaded {len(track)} fixes from {p}")
                return track
            print(f"GPS: SRT present but no lat/lon parsed: {p}")
        if p.suffix.lower() == ".gpx":
            try:
                from rdd.ingest.telemetry import _parse_gpx  # type: ignore

                track = _parse_gpx(p)
                if track.has_data:
                    print(f"GPS: loaded {len(track)} fixes from {p}")
                    return track
            except Exception as e:
                print(f"GPS: failed to parse {p}: {e}")
    print("GPS: none — timestamps only (map trail will note no GPS)")
    return GpsTrack()


def route_from_track(gps: GpsTrack, *, dt: float = 0.5) -> list[dict]:
    """Downsample a GPS track to route points for the dashboard."""
    if not gps.fixes:
        return []
    out: list[dict] = []
    for fix in gps.fixes:
        if out and fix.t - out[-1]["t"] < dt:
            continue
        chainage = gps.distance_at_time(fix.t)
        out.append(
            {
                "lat": fix.lat,
                "lon": fix.lon,
                "t": round(float(fix.t), 3),
                "chainage_m": None if chainage is None else round(float(chainage), 2),
            }
        )
    return out
