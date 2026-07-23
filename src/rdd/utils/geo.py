"""GPS/geo helpers: haversine distance and a track sampled by cumulative distance."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class GpsFix:
    t: float          # seconds from video start
    lat: float
    lon: float
    ele: float | None = None


@dataclass
class GpsTrack:
    fixes: list[GpsFix] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.fixes)

    @property
    def has_data(self) -> bool:
        return len(self.fixes) >= 2

    def at_time(self, t: float) -> GpsFix | None:
        """Nearest fix to time t (linear scan; tracks are small)."""
        if not self.fixes:
            return None
        return min(self.fixes, key=lambda f: abs(f.t - t))

    def cumulative_distance_m(self) -> list[float]:
        """Cumulative metres traveled at each fix."""
        out = [0.0]
        for a, b in zip(self.fixes, self.fixes[1:]):
            out.append(out[-1] + haversine_m(a.lat, a.lon, b.lat, b.lon))
        return out


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
