"""Frame verdicts: deciding whether a frame may be assessed at all.

The requirement this implements: when the road is buried under water or mud, when
the road cannot be located, when the vehicle has left the carriageway, or in similar
degraded conditions — **do not detect anything**. Producing a defect list from a
frame where the road is not visible is worse than producing nothing, because a
downstream reader cannot tell the difference between "inspected and clean" and
"never inspected".

There is a second, less obvious reason this stage exists. Almost all false positives
come from degraded frames: glare, motion blur, a car filling the lane, mud that
looks like a pothole. So refusing to assess them is the cheapest available precision
lever, and the only one that needs no training data. That makes gating a *tunable
dial*: tighten it and precision rises while route coverage falls. Both numbers are
reported, because a precision figure quoted over an undisclosed subset of the route
is not a meaningful figure.

Design notes:

* A verdict carries **reasons**, never a bare boolean. "Frame 812 excluded" is not
  auditable; "frame 812 excluded: road 87% under water" is.
* Gates are *independent* and all of them run, so the report can say which
  conditions dominated a route rather than only the first one that tripped.
* Severity is graded. `BLOCK` refuses the frame; `DEGRADE` allows detection but
  marks results low-confidence; `MASK` removes a region (a vehicle ahead) while
  keeping the rest of the frame usable. Throwing away a whole frame because one car
  occupies a corner of it would waste most of a real survey.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Action(str, Enum):
    """What a gate wants done about the condition it found."""

    BLOCK = "block"      # do not assess this frame at all
    DEGRADE = "degrade"  # assess, but flag results as low-confidence
    MASK = "mask"        # exclude a region; the rest of the frame is fine


@dataclass
class GateResult:
    """One gate's finding for one frame."""

    gate: str
    action: Action
    reason: str
    value: float | None = None      # the measured quantity that tripped it
    threshold: float | None = None
    mask: object | None = None      # region to exclude, for Action.MASK

    def as_dict(self) -> dict:
        d = {"gate": self.gate, "action": self.action.value, "reason": self.reason}
        if self.value is not None:
            d["value"] = round(float(self.value), 4)
        if self.threshold is not None:
            d["threshold"] = round(float(self.threshold), 4)
        return d


@dataclass
class FrameVerdict:
    """The combined decision for one frame."""

    frame: int = 0
    t: float = 0.0
    results: list[GateResult] = field(default_factory=list)
    exclude_mask: object | None = None   # union of MASK regions

    @property
    def blocked(self) -> bool:
        return any(r.action is Action.BLOCK for r in self.results)

    @property
    def assessable(self) -> bool:
        return not self.blocked

    @property
    def degraded(self) -> bool:
        return any(r.action is Action.DEGRADE for r in self.results)

    @property
    def block_reasons(self) -> tuple[str, ...]:
        return tuple(r.reason for r in self.results if r.action is Action.BLOCK)

    @property
    def block_gates(self) -> tuple[str, ...]:
        return tuple(r.gate for r in self.results if r.action is Action.BLOCK)

    @property
    def degrade_reasons(self) -> tuple[str, ...]:
        return tuple(r.reason for r in self.results if r.action is Action.DEGRADE)

    def confidence(self) -> float:
        """Coarse 0..1 trust in this frame. 0 when blocked."""
        if self.blocked:
            return 0.0
        return max(0.0, 1.0 - 0.25 * sum(1 for r in self.results
                                         if r.action is Action.DEGRADE))

    def banner(self) -> str:
        if self.blocked:
            return "NOT ASSESSED: " + "; ".join(self.block_reasons)
        if self.degraded:
            return "LOW CONFIDENCE: " + "; ".join(self.degrade_reasons)
        return ""

    def as_dict(self) -> dict:
        return {
            "frame": self.frame,
            "t": round(self.t, 3),
            "assessable": self.assessable,
            "degraded": self.degraded,
            "confidence": round(self.confidence(), 3),
            "gates": [r.as_dict() for r in self.results],
        }


@dataclass
class ValidityStats:
    """Route-level assessability, accumulated over a clip.

    Distance-weighted where GPS is available: excluding 200 frames while stopped at
    a junction is not the same as excluding 200 frames over 2 km of road, and a
    frame count cannot tell those apart. Frame counts are kept too, since GPS is
    often absent.
    """

    frames: int = 0
    assessable: int = 0
    degraded: int = 0
    distance_total_m: float = 0.0
    distance_assessable_m: float = 0.0
    blocked_by_gate: dict[str, int] = field(default_factory=dict)
    degraded_by_gate: dict[str, int] = field(default_factory=dict)
    _longest_gap: int = 0
    _current_gap: int = 0
    # Kept so the segment rollup can report coverage per chainage segment: a stretch
    # nobody could see must not be graded "sound".
    _per_frame_assessable: list[bool] = field(default_factory=list)

    def update(self, verdict: FrameVerdict, distance_m: float = 0.0) -> None:
        self.frames += 1
        self.distance_total_m += distance_m
        for r in verdict.results:
            if r.action is Action.BLOCK:
                self.blocked_by_gate[r.gate] = self.blocked_by_gate.get(r.gate, 0) + 1
            elif r.action is Action.DEGRADE:
                self.degraded_by_gate[r.gate] = self.degraded_by_gate.get(r.gate, 0) + 1

        self._per_frame_assessable.append(verdict.assessable)
        if verdict.assessable:
            self.assessable += 1
            self.distance_assessable_m += distance_m
            self._current_gap = 0
        else:
            self._current_gap += 1
            self._longest_gap = max(self._longest_gap, self._current_gap)
        if verdict.degraded:
            self.degraded += 1

    @property
    def frame_coverage(self) -> float:
        return (self.assessable / self.frames) if self.frames else 0.0

    @property
    def distance_coverage(self) -> float:
        if self.distance_total_m <= 0:
            return self.frame_coverage
        return self.distance_assessable_m / self.distance_total_m

    @property
    def longest_unassessed_run(self) -> int:
        return max(self._longest_gap, self._current_gap)

    def dominant_reason(self) -> str | None:
        if not self.blocked_by_gate:
            return None
        return max(self.blocked_by_gate.items(), key=lambda kv: kv[1])[0]

    def summary(self) -> dict:
        return {
            "frames": self.frames,
            "frames_assessable": self.assessable,
            "frames_degraded": self.degraded,
            "frame_coverage": round(self.frame_coverage, 4),
            "distance_total_m": round(self.distance_total_m, 1),
            "distance_assessable_m": round(self.distance_assessable_m, 1),
            "distance_coverage": round(self.distance_coverage, 4),
            "blocked_by_gate": dict(sorted(self.blocked_by_gate.items(),
                                           key=lambda kv: -kv[1])),
            "degraded_by_gate": dict(sorted(self.degraded_by_gate.items(),
                                            key=lambda kv: -kv[1])),
            "dominant_exclusion": self.dominant_reason(),
            "longest_unassessed_run_frames": self.longest_unassessed_run,
        }
