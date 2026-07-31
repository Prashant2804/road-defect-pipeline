"""Measuring precision per unique defect, and calibrating thresholds to hit a target.

This is where the ≥90% precision target is proven or disproven, and it is deliberately
scheduled before any bulk labelling so that labelling effort goes only where
measurement says it must.

Three things here matter more than the arithmetic.

**The counting unit is a unique defect, not a per-frame detection.** A defect seen in
forty frames is one defect. Scoring per frame would let a single flickering false
positive count forty times against precision while a stable true positive counts forty
times for it — a number that moves with frame rate is not a quality measure.

**Precision is an operating point, not a property.** Any detector traces a
precision-recall curve; "90% precision" means choosing a confidence threshold on that
curve. So the honest deliverable is per-class thresholds *plus the recall paid for
them*, which is why `calibrate` reports both. A 90% precision figure with unstated
recall can always be met by detecting almost nothing.

**Classes that miss the target are reported, not hidden.** `certify` splits classes
into certified and indicative with their measured numbers and Wilson confidence
intervals, so a shortfall is visible and quantified. With public-data-only training,
several classes are expected to land here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..utils.logging import get_logger

log = get_logger("rdd.eval")


@dataclass
class GroundTruthDefect:
    """One real defect a human marked in a clip."""

    cls_name: str
    first_frame: int
    last_frame: int
    id: str = ""
    lat: float | None = None
    lon: float | None = None

    def overlaps(self, first: int, last: int, slack: int = 0) -> int:
        """Frames of overlap with a predicted track's span."""
        return max(0, min(self.last_frame + slack, last)
                   - max(self.first_frame - slack, first) + 1)


@dataclass
class MatchResult:
    tp: list[tuple] = field(default_factory=list)     # (gt, track)
    fp: list = field(default_factory=list)            # unmatched tracks
    fn: list = field(default_factory=list)            # unmatched ground truth

    @property
    def n_tp(self) -> int:
        return len(self.tp)

    @property
    def n_fp(self) -> int:
        return len(self.fp)

    @property
    def n_fn(self) -> int:
        return len(self.fn)

    @property
    def precision(self) -> float:
        denom = self.n_tp + self.n_fp
        return (self.n_tp / denom) if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.n_tp + self.n_fn
        return (self.n_tp / denom) if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r / (p + r)) if (p + r) else 0.0


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Used instead of the textbook normal approximation because validation sets here are
    small (a few hundred frames, tens of defects per class). At n=20 the normal
    interval can extend past 1.0, which would let a class appear to clear 90% on the
    strength of an interval that is not even a valid probability.
    """
    if total <= 0:
        return 0.0, 1.0
    p = successes / total
    d = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    half = (z / d) * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return max(0.0, centre - half), min(1.0, centre + half)


def match_tracks(tracks, truth: list[GroundTruthDefect], cls_name: str,
                 min_overlap_frames: int = 1, slack: int = 5) -> MatchResult:
    """Greedily match confirmed tracks to ground truth within one class.

    Matching is on temporal overlap rather than mask IoU. Ground truth for a survey is
    naturally recorded as "there is a pothole here, from about this frame to that one";
    demanding per-pixel agreement would measure annotation precision rather than
    detection quality, and would punish a correct detection for having a slightly
    different outline than the annotator drew.
    """
    gts = [g for g in truth if g.cls_name == cls_name]
    preds = [t for t in tracks if t.cls_name == cls_name]

    result = MatchResult()
    used_gt: set[int] = set()
    # Highest-confidence predictions claim their match first, so a weak duplicate
    # cannot steal the ground truth from the detection that actually found it.
    for tr in sorted(preds, key=lambda t: -t.peak_conf):
        best_i, best_overlap = -1, 0
        for i, g in enumerate(gts):
            if i in used_gt:
                continue
            ov = g.overlaps(tr.first_frame, tr.last_frame, slack)
            if ov > best_overlap:
                best_i, best_overlap = i, ov
        if best_i >= 0 and best_overlap >= min_overlap_frames:
            used_gt.add(best_i)
            result.tp.append((gts[best_i], tr))
        else:
            result.fp.append(tr)

    result.fn = [g for i, g in enumerate(gts) if i not in used_gt]
    return result


@dataclass
class ClassCalibration:
    """The chosen operating point for one class."""

    cls_name: str
    threshold: float
    precision: float
    recall: float
    n_tp: int
    n_fp: int
    n_fn: int
    precision_lo: float
    precision_hi: float
    target_met: bool
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "class": self.cls_name,
            "threshold": round(self.threshold, 3),
            "precision": round(self.precision, 4),
            "precision_ci": [round(self.precision_lo, 4), round(self.precision_hi, 4)],
            "recall": round(self.recall, 4),
            "tp": self.n_tp, "fp": self.n_fp, "fn": self.n_fn,
            "target_met": self.target_met,
            "note": self.note,
        }


def sweep(tracks, truth: list[GroundTruthDefect], cls_name: str,
          thresholds=None, slack: int = 5) -> list[tuple[float, MatchResult]]:
    """Precision/recall at a range of confidence thresholds."""
    if thresholds is None:
        thresholds = [i / 20.0 for i in range(1, 20)]
    out = []
    for th in thresholds:
        kept = [t for t in tracks if t.peak_conf >= th]
        out.append((th, match_tracks(kept, truth, cls_name, slack=slack)))
    return out


def calibrate_class(tracks, truth: list[GroundTruthDefect], cls_name: str,
                    target_precision: float = 0.90, min_support: int = 8,
                    require_lower_bound: bool = True,
                    slack: int = 5) -> ClassCalibration:
    """Pick the threshold that meets the precision target at the best recall.

    Among thresholds that hit the target, the one with the highest recall is chosen —
    raising the threshold further only discards true positives. When
    `require_lower_bound` is set, the *lower* end of the confidence interval must clear
    the target, so a class does not get certified on the strength of 9 out of 10.
    """
    curve = sweep(tracks, truth, cls_name, slack=slack)
    if not curve:
        return ClassCalibration(cls_name, 1.0, 0.0, 0.0, 0, 0, 0, 0.0, 1.0, False,
                                "no data")

    best: ClassCalibration | None = None
    best_point: ClassCalibration | None = None   # met on the point estimate alone
    for th, m in curve:
        total = m.n_tp + m.n_fp
        lo, hi = wilson_interval(m.n_tp, total)
        met = total >= 1 and (lo if require_lower_bound else m.precision) >= target_precision
        cand = ClassCalibration(cls_name, th, m.precision, m.recall,
                                m.n_tp, m.n_fp, m.n_fn, lo, hi, met)
        if met and (best is None or cand.recall > best.recall):
            best = cand
        if total >= 1 and m.precision >= target_precision:
            if best_point is None or cand.recall > best_point.recall:
                best_point = cand

    if best is not None:
        support = best.n_tp + best.n_fn
        if support < min_support:
            best.target_met = False
            best.note = (f"only {support} ground-truth instances — too few to certify "
                         f"even though the point estimate clears the target")
        return best

    # The point estimate clears the target but the interval does not. That is a
    # *sample-size* limit, not a detector limit, and the two need different responses:
    # more validation labels versus more training data. Saying which is which is the
    # difference between an actionable result and a shrug.
    if best_point is not None:
        n = best_point.n_tp + best_point.n_fp
        needed = _min_n_for_target(target_precision)
        best_point.target_met = False
        best_point.note = (
            f"precision {best_point.precision:.0%} on {n} prediction(s), but the 95% "
            f"lower bound is {best_point.precision_lo:.0%} — not enough evidence to "
            f"certify {target_precision:.0%}. Even a perfect detector needs about "
            f"{needed} instances to clear this bar. This is a labelling-volume limit, "
            f"not a model limit."
        )
        return best_point

    th, m = max(curve, key=lambda kv: kv[1].precision)
    lo, hi = wilson_interval(m.n_tp, m.n_tp + m.n_fp)
    return ClassCalibration(
        cls_name, th, m.precision, m.recall, m.n_tp, m.n_fp, m.n_fn, lo, hi, False,
        f"target {target_precision:.0%} not reached at any threshold; best is "
        f"{m.precision:.0%} at conf {th:.2f}",
    )


def _min_n_for_target(target: float, max_n: int = 400) -> int:
    """Smallest sample size whose Wilson lower bound clears `target` at 100% precision.

    Reported in diagnostics so the labelling budget can be planned against the actual
    statistical requirement rather than guessed at.
    """
    for n in range(1, max_n + 1):
        if wilson_interval(n, n)[0] >= target:
            return n
    return max_n


@dataclass
class CertificationReport:
    """Which classes may be quoted as meeting the target, and which may not."""

    target_precision: float = 0.90
    per_class: dict = field(default_factory=dict)
    route_coverage: float | None = None

    @property
    def certified(self) -> list[str]:
        return sorted(c for c, v in self.per_class.items() if v.target_met)

    @property
    def indicative(self) -> list[str]:
        return sorted(c for c, v in self.per_class.items() if not v.target_met)

    def thresholds(self) -> dict:
        return {c: round(v.threshold, 3) for c, v in sorted(self.per_class.items())}

    def summary(self) -> dict:
        return {
            "target_precision": self.target_precision,
            "certified": self.certified,
            "indicative": self.indicative,
            "route_coverage": (round(self.route_coverage, 4)
                               if self.route_coverage is not None else None),
            "per_class": {c: v.as_dict() for c, v in sorted(self.per_class.items())},
        }

    def table(self) -> str:
        lines = [
            f"{'class':<22}{'thresh':>7}{'prec':>7}{'CI low':>8}{'recall':>8}"
            f"{'TP':>5}{'FP':>5}{'FN':>5}  status",
            "-" * 88,
        ]
        for cls in sorted(self.per_class):
            v = self.per_class[cls]
            status = "CERTIFIED" if v.target_met else "indicative"
            lines.append(
                f"{cls:<22}{v.threshold:>7.2f}{v.precision:>7.2f}{v.precision_lo:>8.2f}"
                f"{v.recall:>8.2f}{v.n_tp:>5}{v.n_fp:>5}{v.n_fn:>5}  {status}"
            )
        if self.route_coverage is not None:
            lines.append("")
            lines.append(f"Measured over {self.route_coverage:.1%} of frames "
                         f"(the assessable subset).")
        return "\n".join(lines)


def certify(tracks, truth: list[GroundTruthDefect], cfg,
            route_coverage: float | None = None) -> CertificationReport:
    """Calibrate every configured class and split certified from indicative."""
    ec = cfg.get_path("eval", {}) or {}
    target = float(ec.get("target_precision", 0.90))
    min_support = int(ec.get("min_support", 8))
    require_lb = bool(ec.get("require_ci_lower_bound", True))
    slack = int(ec.get("match_slack_frames", 5))

    classes = [str(c) for c in (cfg.get_path("model.classes") or [])]
    exclude = set(ec.get("exclude_from_target") or ())

    report = CertificationReport(target_precision=target,
                                 route_coverage=route_coverage)
    tracks = list(tracks)
    for cls in classes:
        cal = calibrate_class(tracks, truth, cls, target_precision=target,
                              min_support=min_support,
                              require_lower_bound=require_lb, slack=slack)
        if cls in exclude:
            cal.target_met = False
            cal.note = (cal.note + "; " if cal.note else "") + \
                       "explicitly outside the precision guarantee"
        report.per_class[cls] = cal

    log.info("Certification at %.0f%% precision: %d certified, %d indicative",
             100 * target, len(report.certified), len(report.indicative))
    for cls in report.indicative:
        v = report.per_class[cls]
        log.warning("  %s NOT certified: precision %.0f%% (CI %.0f-%.0f%%) — %s",
                    cls, 100 * v.precision, 100 * v.precision_lo,
                    100 * v.precision_hi, v.note or "below target")
    return report


def load_ground_truth(path) -> list[GroundTruthDefect]:
    """Read ground truth from CSV or JSON.

    CSV columns: class, first_frame, last_frame[, id, lat, lon].
    JSON: a list of objects with the same keys.
    """
    import json
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Ground truth not found: {p}")

    if p.suffix.lower() == ".json":
        rows = json.loads(p.read_text(encoding="utf-8"))
    else:
        import csv

        with p.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    out = []
    for i, r in enumerate(rows):
        try:
            out.append(GroundTruthDefect(
                cls_name=str(r["class"]).strip(),
                first_frame=int(r["first_frame"]),
                last_frame=int(r["last_frame"]),
                id=str(r.get("id") or f"gt{i}"),
                lat=float(r["lat"]) if r.get("lat") not in (None, "") else None,
                lon=float(r["lon"]) if r.get("lon") not in (None, "") else None,
            ))
        except (KeyError, ValueError) as e:
            raise ValueError(f"Bad ground-truth row {i} in {p}: {r} ({e})") from e
    log.info("Loaded %d ground-truth defects from %s", len(out), p)
    return out
