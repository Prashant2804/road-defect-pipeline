"""IRC/PMGSY severity banding and per-100 m segment rollup.

Two changes to how results are expressed, both driven by what a road authority actually
consumes.

**Severity in physical bands, not a normalised score.** IRC grades distress on measured
quantities — crack width in millimetres, pothole area and depth, percentage of surface
affected. A 0-1 score normalised across whatever happened to be in one clip is not
convertible to those bands, and is not comparable between surveys. Where ground scale
is available these bands apply directly; where it is not, the report says so rather than
implying a physical grade.

**Reporting per 100 m rather than per defect.** A list of nine hundred individual
defects is not actionable. Maintenance is planned per stretch, so distress is
aggregated into chainage segments with a condition grade each, while the per-defect CSV
remains available underneath for verification.

The band values below follow the IRC/PMGSY structure for rural road condition
assessment. They are config-driven because specifications are revised and different
authorities adopt variants — treat `report.irc` as the place to align them to whatever
document a given contract cites.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..utils.logging import get_logger

log = get_logger("rdd.report.irc")

LOW, MEDIUM, HIGH = "low", "medium", "high"
INDETERMINATE = "indeterminate"

# Crack width bands in millimetres (IRC-style hairline / narrow / wide).
_CRACK_WIDTH_MM = {"medium": 3.0, "high": 6.0}
# Pothole plan area in m².
_POTHOLE_AREA_M2 = {"medium": 0.10, "high": 0.50}
# Percentage of surface affected, for area distresses.
_AREA_PERCENT = {"medium": 10.0, "high": 30.0}
# Edge loss inward from the intended edge, in metres.
_EDGE_INSET_M = {"medium": 0.15, "high": 0.30}


def _band(value: float, bands: dict, keys=("medium", "high")) -> str:
    med, high = float(bands.get(keys[0])), float(bands.get(keys[1]))
    if value >= high:
        return HIGH
    if value >= med:
        return MEDIUM
    return LOW


@dataclass
class IrcGrade:
    level: str
    basis: str                 # what quantity produced the grade
    value: float | None = None
    unit: str = ""
    note: str = ""

    def as_dict(self) -> dict:
        return {"level": self.level, "basis": self.basis,
                "value": (round(self.value, 4) if self.value is not None else None),
                "unit": self.unit, "note": self.note}


def grade_defect(cls_name: str, cfg, area_m2: float | None = None,
                 width_mm: float | None = None, percent_area: float | None = None,
                 inset_m: float | None = None, occluded: bool = False) -> IrcGrade:
    """Assign an IRC severity band from whichever measurement applies to the class."""
    ic = cfg.get_path("report.irc", {}) or {}
    if occluded:
        return IrcGrade(INDETERMINATE, "occluded",
                        note="hidden under water/mud — extent not observable")

    if cls_name in ("longitudinal_crack", "transverse_crack", "alligator_crack"):
        if cls_name == "alligator_crack" and percent_area is not None:
            bands = ic.get("area_percent", _AREA_PERCENT)
            return IrcGrade(_band(percent_area, bands), "surface affected",
                            percent_area, "%")
        if width_mm is None:
            return IrcGrade(INDETERMINATE, "crack width",
                            note="no ground scale — crack width not measurable in mm")
        bands = ic.get("crack_width_mm", _CRACK_WIDTH_MM)
        return IrcGrade(_band(width_mm, bands), "crack width", width_mm, "mm")

    if cls_name == "pothole":
        if area_m2 is None:
            return IrcGrade(INDETERMINATE, "pothole area",
                            note="no ground scale — area not measurable in m²")
        bands = ic.get("pothole_area_m2", _POTHOLE_AREA_M2)
        return IrcGrade(_band(area_m2, bands), "pothole area", area_m2, "m²")

    if cls_name == "edge_damage":
        if inset_m is None:
            return IrcGrade(INDETERMINATE, "edge loss",
                            note="no ground scale — edge loss not measurable")
        bands = ic.get("edge_inset_m", _EDGE_INSET_M)
        return IrcGrade(_band(inset_m, bands), "edge loss", inset_m, "m")

    if cls_name in ("ravelling", "rutting", "water_logging", "drainage_issue"):
        if percent_area is None:
            return IrcGrade(INDETERMINATE, "surface affected",
                            note="extent not quantified")
        bands = ic.get("area_percent", _AREA_PERCENT)
        return IrcGrade(_band(percent_area, bands), "surface affected",
                        percent_area, "%")

    if area_m2 is not None:
        bands = ic.get("pothole_area_m2", _POTHOLE_AREA_M2)
        return IrcGrade(_band(area_m2, bands), "area", area_m2, "m²")
    return IrcGrade(INDETERMINATE, "unknown", note="no applicable measurement")


# -- segment rollup ------------------------------------------------------------

@dataclass
class Segment:
    """One chainage segment of the surveyed route."""

    index: int
    start_m: float
    end_m: float
    counts: dict = field(default_factory=dict)
    severity: dict = field(default_factory=dict)
    assessed_frames: int = 0
    total_frames: int = 0
    indeterminate: int = 0

    @property
    def length_m(self) -> float:
        return self.end_m - self.start_m

    @property
    def coverage(self) -> float:
        return (self.assessed_frames / self.total_frames) if self.total_frames else 0.0

    @property
    def total_defects(self) -> int:
        return sum(self.counts.values())

    def grade(self, cfg) -> str:
        """Worst-case segment grade, with coverage honesty.

        A segment nobody could see is not a good segment. If coverage is too low the
        grade is `indeterminate` regardless of how few defects were found there —
        otherwise an unassessable stretch would be indistinguishable from a sound one,
        which is the exact confusion this pipeline exists to avoid.
        """
        sc = cfg.get_path("report.segments", {}) or {}
        if self.total_frames and self.coverage < float(sc.get("min_coverage", 0.30)):
            return INDETERMINATE
        if self.severity.get(HIGH):
            return HIGH
        if self.severity.get(MEDIUM):
            return MEDIUM
        if self.total_defects or self.severity.get(LOW):
            return LOW
        return "sound"

    def as_dict(self, cfg) -> dict:
        return {
            "segment": self.index,
            "chainage_m": [round(self.start_m, 1), round(self.end_m, 1)],
            "grade": self.grade(cfg),
            "coverage": round(self.coverage, 3),
            "defects": dict(sorted(self.counts.items())),
            "severity": dict(sorted(self.severity.items())),
            "indeterminate": self.indeterminate,
        }


def _distance_for_frame(frame: int, fps: float, gps, speed_mps: float) -> float:
    """Chainage of a frame: GPS distance when available, else time x assumed speed."""
    t = frame / fps if fps else 0.0
    if gps is not None and getattr(gps, "has_data", False):
        d = gps.distance_at_time(t)
        if d is not None:
            return float(d)
    return t * speed_mps


def build_segments(counter, severity, cfg, fps: float = 30.0, gps=None,
                   validity=None) -> list[Segment]:
    """Roll confirmed defects up into fixed-length chainage segments."""
    sc = cfg.get_path("report.segments", {}) or {}
    seg_len = float(sc.get("length_m", 100.0))
    assumed_speed = float(sc.get("assumed_speed_mps", 8.0))
    if seg_len <= 0:
        return []

    have_gps = gps is not None and getattr(gps, "has_data", False)
    if not have_gps:
        log.info("Segment rollup without GPS: chainage estimated from elapsed time at "
                 "an assumed %.1f m/s. Distances are indicative only.", assumed_speed)

    tracks = counter.confirmed_tracks()
    max_frame = max((t.last_frame for t in tracks), default=0)
    if validity is not None and getattr(validity, "frames", 0):
        max_frame = max(max_frame, validity.frames - 1)
    route_m = _distance_for_frame(max_frame, fps, gps, assumed_speed)
    n_segments = max(1, int(route_m // seg_len) + 1)

    segments = [Segment(index=i, start_m=i * seg_len, end_m=(i + 1) * seg_len)
                for i in range(n_segments)]

    # Frame coverage per segment, so a segment can report how much of it was seen.
    if validity is not None and getattr(validity, "frames", 0):
        per_frame = getattr(validity, "_per_frame_assessable", None)
        for f in range(validity.frames):
            d = _distance_for_frame(f, fps, gps, assumed_speed)
            i = min(n_segments - 1, int(d // seg_len))
            segments[i].total_frames += 1
            if per_frame is None or (f < len(per_frame) and per_frame[f]):
                segments[i].assessed_frames += 1

    for tr in tracks:
        d = _distance_for_frame(tr.first_frame, fps, gps, assumed_speed)
        i = min(n_segments - 1, int(d // seg_len))
        seg = segments[i]
        seg.counts[tr.cls_name] = seg.counts.get(tr.cls_name, 0) + 1
        sev = severity.get(tr.track_id) if severity is not None else None
        level = sev.level if sev is not None else INDETERMINATE
        seg.severity[level] = seg.severity.get(level, 0) + 1
        if level == INDETERMINATE:
            seg.indeterminate += 1

    return segments


def segments_summary(segments: list[Segment], cfg) -> dict:
    grades: dict[str, int] = {}
    for s in segments:
        g = s.grade(cfg)
        grades[g] = grades.get(g, 0) + 1
    return {
        "n_segments": len(segments),
        "segment_length_m": round(segments[0].length_m, 1) if segments else 0.0,
        "grades": dict(sorted(grades.items())),
        "worst_segments": [s.index for s in segments if s.grade(cfg) == HIGH][:10],
    }
