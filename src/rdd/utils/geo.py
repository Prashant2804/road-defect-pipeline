"""GPS/geo helpers: haversine distance and a track indexed by time and distance.

Lookups are cached and binary-searched. The naive version — a linear scan per
frame plus a full cumulative-distance recomputation per frame — is O(frames x
fixes), which on a 20-minute survey with 1 Hz GPS is tens of millions of
haversine calls spent rediscovering the same numbers.
"""
from __future__ import annotations

import bisect
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
    _times: list[float] | None = field(default=None, repr=False, compare=False)
    _cum: list[float] | None = field(default=None, repr=False, compare=False)

    def __len__(self) -> int:
        return len(self.fixes)

    @property
    def has_data(self) -> bool:
        return len(self.fixes) >= 2

    def _ensure_index(self) -> None:
        """Sort by time once, then cache the time and cumulative-distance arrays."""
        if self._times is not None and len(self._times) == len(self.fixes):
            return
        self.fixes.sort(key=lambda f: f.t)
        self._times = [f.t for f in self.fixes]
        cum = [0.0]
        for a, b in zip(self.fixes, self.fixes[1:]):
            cum.append(cum[-1] + haversine_m(a.lat, a.lon, b.lat, b.lon))
        self._cum = cum

    def invalidate(self) -> None:
        """Call after mutating `fixes` directly."""
        self._times = None
        self._cum = None

    def index_at_time(self, t: float) -> int | None:
        """Index of the fix nearest in time to `t`, via binary search."""
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
        """Nearest fix to time t."""
        i = self.index_at_time(t)
        return None if i is None else self.fixes[i]

    def cumulative_distance_m(self) -> list[float]:
        """Cumulative metres traveled at each fix (cached)."""
        if not self.fixes:
            return []
        self._ensure_index()
        return list(self._cum or [])

    def distance_at_time(self, t: float) -> float | None:
        """Cumulative distance travelled by time `t`, in metres."""
        i = self.index_at_time(t)
        if i is None:
            return None
        self._ensure_index()
        return (self._cum or [0.0])[i]

    @property
    def total_distance_m(self) -> float:
        if not self.has_data:
            return 0.0
        self._ensure_index()
        return (self._cum or [0.0])[-1]


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
