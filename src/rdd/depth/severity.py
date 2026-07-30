"""Severity scoring — with an explicit "we could not tell" outcome.

Two deliberate departures from the obvious implementation.

**Abstention.** A defect sitting under standing water or mud is not scored. Its
extent is hidden, its depth is unknowable under a reflective surface, and a
number produced anyway would be a guess wearing the costume of a measurement. It
gets `level="indeterminate"` and a stated reason. Water-logging itself is exempt:
it is the occluder, not a victim of one.

**Absolute units when we have them.** With ground scale available (drone GSD or an
IPM homography) severity is binned against fixed physical thresholds in m². Only
without scale do we fall back to min-max normalisation across the run — and that
fallback is reported, because it has a property users need to know about: it is
*relative to this clip*, so the largest defect present is always "high" even if
it is trivial, and the same road shot twice at different framing can score
differently.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..utils.logging import get_logger

log = get_logger("rdd.severity")

LEVELS = ("low", "medium", "high", "indeterminate")


@dataclass
class Severity:
    level: str
    score: float | None = None      # 0..1 for scored defects, None when abstained
    area_px: float = 0.0
    area_m2: float | None = None
    depth: float | None = None
    basis: str = "relative_px"      # absolute_m2 | relative_px | abstained
    reason: str = ""

    @property
    def is_indeterminate(self) -> bool:
        return self.level == "indeterminate"

    def as_dict(self) -> dict:
        return {
            "level": self.level,
            "score": self.score,
            "area_px": self.area_px,
            "area_m2": self.area_m2,
            "depth": self.depth,
            "basis": self.basis,
            "reason": self.reason,
        }

    # Mapping-style access so existing consumers keep working.
    def get(self, key: str, default=None):
        return self.as_dict().get(key, default)


@dataclass
class SeverityReport:
    """Per-track severities plus how they were derived."""

    by_track: dict[int, Severity] = field(default_factory=dict)
    basis: str = "relative_px"
    scale_note: str = ""
    n_indeterminate: int = 0

    def get(self, track_id: int, default=None):
        return self.by_track.get(track_id, default)

    def __getitem__(self, track_id: int) -> Severity:
        return self.by_track[track_id]

    def __contains__(self, track_id: object) -> bool:
        return track_id in self.by_track

    def __len__(self) -> int:
        return len(self.by_track)

    def items(self):
        return self.by_track.items()

    def level_counts(self) -> dict[str, int]:
        counts = {lv: 0 for lv in LEVELS}
        for s in self.by_track.values():
            counts[s.level] = counts.get(s.level, 0) + 1
        return counts


def _severity_cfg(cfg) -> dict:
    """Read `severity:`, falling back to the older `depth.severity:` location."""
    top = cfg.get_path("severity")
    if isinstance(top, dict) and top:
        return dict(top)
    return dict(cfg.get_path("depth.severity", {}) or {})


def _normalize(values: list[float]) -> dict[int, float]:
    if not values:
        return {}
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    return {i: (v - lo) / span for i, v in enumerate(values)}


def _level_from_score(score: float, bins: dict) -> str:
    med = float(bins.get("medium", 0.66))
    low = float(bins.get("low", 0.33))
    return "high" if score >= med else "medium" if score >= low else "low"


def _level_from_area_m2(area_m2: float, bins: dict) -> tuple[str, float]:
    """Absolute physical bins, plus a 0..1 score scaled to the 'high' threshold."""
    high = float(bins.get("high", 0.5))
    medium = float(bins.get("medium", 0.1))
    level = "high" if area_m2 >= high else "medium" if area_m2 >= medium else "low"
    return level, max(0.0, min(1.0, area_m2 / high if high > 0 else 0.0))


def score_tracks(tracks, cfg, depths: dict[int, float] | None = None,
                 counter=None) -> SeverityReport:
    """Score confirmed tracks. `counter` supplies occluder-class knowledge."""
    sc = _severity_cfg(cfg)
    bins = sc.get("bins", {"low": 0.33, "medium": 0.66})
    abs_bins = sc.get("absolute_bins_m2", {"medium": 0.1, "high": 0.5})
    w_area = float(sc.get("w_area", 0.5))
    w_depth = float(sc.get("w_depth", 0.5))
    depth_enabled = bool(cfg.get_path("depth.enabled", False)) and depths is not None
    policy = cfg.get_path("surface.occlusion_policy", "abstain")

    tracks = list(tracks)
    use_absolute = bool(tracks) and all(t.max_area_m2 is not None for t in tracks)

    report = SeverityReport(basis="absolute_m2" if use_absolute else "relative_px")
    if use_absolute:
        report.scale_note = (
            f"Severity from absolute ground area: >={abs_bins.get('high', 0.5)} m² high, "
            f">={abs_bins.get('medium', 0.1)} m² medium."
        )
    else:
        report.scale_note = (
            "No ground scale available, so severity is min-max normalised across "
            "THIS RUN only. The largest defect present is always 'high' — these "
            "levels are not comparable between videos. Set drone altitude/optics "
            "or preprocess.ipm.ground_extent_m for absolute m² severity."
        )
        if tracks:
            log.warning("severity: %s", report.scale_note)

    areas = [t.max_mask_area for t in tracks]
    area_norm = _normalize(areas)
    depth_norm = {}
    if depth_enabled:
        depth_norm = _normalize([depths.get(t.track_id, 0.0) for t in tracks])

    for i, t in enumerate(tracks):
        is_occluder = counter.is_occluder(t.cls_name) if counter is not None else False

        if policy == "abstain" and t.occluded and not is_occluder:
            report.by_track[t.track_id] = Severity(
                level="indeterminate", score=None,
                area_px=t.max_mask_area, area_m2=t.max_area_m2, depth=None,
                basis="abstained",
                reason=(f"{t.median_occluded_frac:.0%} of the defect lies under "
                        f"water/mud — extent and depth are not observable"),
            )
            report.n_indeterminate += 1
            continue

        if use_absolute:
            level, score = _level_from_area_m2(float(t.max_area_m2), abs_bins)
            basis = "absolute_m2"
            if depth_enabled:
                d = depth_norm.get(i, 0.0)
                score = (w_area * score + w_depth * d) / ((w_area + w_depth) or 1.0)
                level = _level_from_score(score, bins)
                basis = "absolute_m2+depth"
        else:
            score = area_norm.get(i, 0.0)
            basis = "relative_px"
            if depth_enabled:
                score = (w_area * score + w_depth * depth_norm.get(i, 0.0))
                score /= (w_area + w_depth) or 1.0
                basis = "relative_px+depth"
            level = _level_from_score(score, bins)

        report.by_track[t.track_id] = Severity(
            level=level, score=round(float(score), 4),
            area_px=t.max_mask_area, area_m2=t.max_area_m2,
            depth=depths.get(t.track_id) if depth_enabled else None,
            basis=basis,
            reason=("partially occluded" if t.max_occluded_frac > 0.1 and not is_occluder
                    else ""),
        )

    if report.n_indeterminate:
        log.info("Severity: %d of %d confirmed defects abstained on (hidden under "
                 "water/mud)", report.n_indeterminate, len(tracks))
    return report
