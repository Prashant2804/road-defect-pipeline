"""Run-level aggregation of the area and boundary conditions.

Potholes and cracks are *objects*: they get tracked, counted once each, and listed
individually. Ravelling, rutting, edge damage and drainage are **conditions of a
stretch of road** — "37% of the surface is ravelled" is the meaningful statement, not
"there are 412 ravelling instances". Forcing them through the object tracker would
produce a defect count that is really a function of grid size.

So they are accumulated here as extents, area-weighted, and reported as percentages
and worst-cases per class.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..utils.logging import get_logger

log = get_logger("rdd.detect.aggregate")


@dataclass
class ConditionAggregator:
    """Accumulates per-frame area/boundary conditions over a clip."""

    frames: int = 0

    # Ravelling: area-weighted so a frame showing more road counts for more.
    ravelling_frames: int = 0
    ravelling_cells: int = 0
    ravelling_affected_cells: int = 0
    ravelling_area_m2: float = 0.0
    ravelling_worst_z: float = 0.0

    # Rutting proxy — indicative only, never a depth.
    rutting_frames: int = 0
    rutting_index_sum: float = 0.0
    rutting_worst: float = 0.0

    # Edge damage: distinct stretches, deduplicated by chainage.
    edge_frames: int = 0
    edge_defect_frames: int = 0
    edge_worst_inset_m: float = 0.0
    edge_raggedness_sum: float = 0.0
    edge_stretches: list = field(default_factory=list)

    # Drainage symptoms.
    drainage_frames: int = 0
    drainage_pool_frames: int = 0
    drainage_worst_pool_m2: float = 0.0
    drainage_edge_water_sum: float = 0.0

    def update_ravelling(self, r) -> None:
        if r is None or not r.measured:
            return
        self.ravelling_frames += 1
        self.ravelling_cells += r.cells_total
        self.ravelling_affected_cells += r.cells_affected
        self.ravelling_area_m2 += r.affected_area_m2
        self.ravelling_worst_z = max(self.ravelling_worst_z, r.worst_texture_z)

    def update_rutting(self, r) -> None:
        if r is None or not r.measured:
            return
        self.rutting_frames += 1
        self.rutting_index_sum += r.wheelpath_index
        self.rutting_worst = max(self.rutting_worst, r.wheelpath_index)

    def update_boundary(self, b, chainage_m: float = 0.0) -> None:
        if b is None or not b.measured:
            return
        self.edge_frames += 1
        self.edge_raggedness_sum += 0.5 * (b.left_raggedness_m + b.right_raggedness_m)
        if not b.defects:
            return
        self.edge_defect_frames += 1
        for d in b.defects:
            self.edge_worst_inset_m = max(self.edge_worst_inset_m, d.max_inset_m)
            # Deduplicate: the same eroded stretch is seen over many frames as the
            # vehicle approaches it. Merge by absolute chainage rather than counting
            # once per frame, which would multiply one defect by the frame rate.
            key = (d.side, round((chainage_m + d.z_start_m) / 2.0) * 2.0)
            existing = next((s for s in self.edge_stretches
                             if s["side"] == d.side and abs(s["chainage_m"] - key[1]) < 3.0),
                            None)
            if existing is None:
                self.edge_stretches.append({
                    "side": d.side, "chainage_m": key[1],
                    "max_inset_m": d.max_inset_m, "length_m": d.length_m,
                })
            else:
                existing["max_inset_m"] = max(existing["max_inset_m"], d.max_inset_m)
                existing["length_m"] = max(existing["length_m"], d.length_m)

    def update_drainage(self, d: dict | None) -> None:
        if not d or not d.get("measured"):
            return
        self.drainage_frames += 1
        self.drainage_edge_water_sum += float(d.get("edge_water_frac", 0.0))
        if d.get("n_pools", 0) > 0:
            self.drainage_pool_frames += 1
            self.drainage_worst_pool_m2 = max(self.drainage_worst_pool_m2,
                                              float(d.get("worst_pool_m2", 0.0)))

    # -- derived ----------------------------------------------------------
    @property
    def ravelling_percent(self) -> float:
        if self.ravelling_cells <= 0:
            return 0.0
        return 100.0 * self.ravelling_affected_cells / self.ravelling_cells

    @property
    def mean_rutting_index(self) -> float:
        return (self.rutting_index_sum / self.rutting_frames) if self.rutting_frames else 0.0

    @property
    def edge_damage_percent(self) -> float:
        if self.edge_frames <= 0:
            return 0.0
        return 100.0 * self.edge_defect_frames / self.edge_frames

    @property
    def mean_edge_raggedness_m(self) -> float:
        return (self.edge_raggedness_sum / self.edge_frames) if self.edge_frames else 0.0

    @property
    def drainage_percent(self) -> float:
        if self.drainage_frames <= 0:
            return 0.0
        return 100.0 * self.drainage_pool_frames / self.drainage_frames

    def summary(self) -> dict:
        return {
            "ravelling": {
                "frames_measured": self.ravelling_frames,
                "percent_surface_affected": round(self.ravelling_percent, 2),
                "affected_area_m2": round(self.ravelling_area_m2, 2),
                "worst_texture_z": round(self.ravelling_worst_z, 2),
                "basis": "indicative — texture anomaly, no labelled training data",
            },
            "rutting": {
                "frames_measured": self.rutting_frames,
                "mean_wheelpath_index": round(self.mean_rutting_index, 3),
                "worst_wheelpath_index": round(self.rutting_worst, 3),
                "basis": "indicative proxy — a monocular camera cannot measure rut "
                         "depth; explicitly outside the precision guarantee",
            },
            "edge_damage": {
                "frames_measured": self.edge_frames,
                "percent_frames_with_damage": round(self.edge_damage_percent, 2),
                "n_distinct_stretches": len(self.edge_stretches),
                "worst_inset_m": round(self.edge_worst_inset_m, 3),
                "mean_raggedness_m": round(self.mean_edge_raggedness_m, 4),
                "basis": "geometric — measured from the road-mask boundary, label-free",
            },
            "drainage": {
                "frames_measured": self.drainage_frames,
                "percent_frames_with_edge_pooling": round(self.drainage_percent, 2),
                "worst_pool_m2": round(self.drainage_worst_pool_m2, 3),
                "basis": "symptom only — culvert identification needs labelled examples",
            },
        }

    def log_summary(self) -> None:
        s = self.summary()
        if self.ravelling_frames:
            log.info("Ravelling: %.1f%% of graded surface (worst texture %.1f sigma)",
                     s["ravelling"]["percent_surface_affected"],
                     s["ravelling"]["worst_texture_z"])
        if self.edge_frames:
            log.info("Edge damage: %d distinct stretches, worst inset %.2f m",
                     len(self.edge_stretches), self.edge_worst_inset_m)
        if self.drainage_pool_frames:
            log.info("Drainage: edge pooling on %.0f%% of frames, worst pool %.2f m²",
                     s["drainage"]["percent_frames_with_edge_pooling"],
                     self.drainage_worst_pool_m2)
        if self.rutting_frames:
            log.info("Rutting proxy: mean wheel-path index %.2f (indicative only)",
                     self.mean_rutting_index)
