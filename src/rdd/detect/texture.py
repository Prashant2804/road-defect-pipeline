"""Ravelling and the rutting proxy: surface condition on a fixed-ground-resolution grid.

Both are *area* conditions rather than objects, and neither has usable public training
data, so both are computed statistically instead of learned.

**Ravelling** is loss of surface aggregate, which shows up as elevated texture relative
to intact pavement. The measurement must be made at a **fixed ground resolution** or it
is meaningless: the same surface imaged at 5 m and at 15 m yields completely different
texture statistics purely because of sampling, so a fixed-pixel-size window would grade
the near field as rough and the far field as smooth on every road ever surveyed. Cells
here are a fixed size *in metres*, so a grade is comparable across the frame and across
clips.

**Rutting** is a 3-D deformation and is largely invisible to a single forward camera —
it is explicitly outside the ≥90% precision scope. What is detectable is its
*correlates*: wheel paths sit at predictable lateral offsets, and where ruts form the
surface is polished, discoloured, or holds water differently from the crown between
them. That comparison is reported as an indicative index, never as a rut depth, because
a depth figure from a monocular camera would be a fabrication.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..utils.logging import get_logger

log = get_logger("rdd.detect.texture")


@dataclass
class TextureCell:
    z_m: float
    x_m: float
    texture_z: float           # z-score of relative texture vs the road baseline
    area_m2: float


@dataclass
class RavellingResult:
    cells_total: int = 0
    cells_affected: int = 0
    affected_area_m2: float = 0.0
    road_area_m2: float = 0.0
    worst_texture_z: float = 0.0
    cells: list[TextureCell] = field(default_factory=list)
    measured: bool = False
    note: str = ""

    @property
    def affected_frac(self) -> float:
        if self.cells_total <= 0:
            return 0.0
        return self.cells_affected / self.cells_total

    def summary(self) -> dict:
        return {
            "measured": self.measured,
            "cells": self.cells_total,
            "cells_affected": self.cells_affected,
            "affected_frac": round(self.affected_frac, 4),
            "affected_area_m2": round(self.affected_area_m2, 3),
            "worst_texture_z": round(self.worst_texture_z, 2),
            "note": self.note,
        }


@dataclass
class RuttingResult:
    """Indicative only — never a rut depth."""

    wheelpath_index: float = 0.0     # >0 means wheel paths differ from the crown
    left_z: float = 0.0
    right_z: float = 0.0
    crown_z: float = 0.0
    measured: bool = False
    note: str = "indicative proxy only; not within the precision guarantee"

    def summary(self) -> dict:
        return {
            "measured": self.measured,
            "wheelpath_index": round(self.wheelpath_index, 3),
            "left_z": round(self.left_z, 3),
            "right_z": round(self.right_z, 3),
            "crown_z": round(self.crown_z, 3),
            "note": self.note,
        }


@dataclass
class TextureConfig:
    cell_m: float = 0.5              # ground size of a grid cell
    min_cell_px: int = 40            # too few pixels -> no reliable statistic
    ravelling_min_z: float = 1.5     # texture above baseline to call ravelling
    min_cells: int = 4
    wheelpath_offset_m: float = 0.85
    wheelpath_half_width_m: float = 0.35

    @classmethod
    def from_cfg(cls, cfg) -> "TextureConfig":
        tc = cfg.get_path("detect.texture", {}) or {}
        return cls(
            cell_m=float(tc.get("cell_m", 0.5)),
            min_cell_px=int(tc.get("min_cell_px", 40)),
            ravelling_min_z=float(tc.get("ravelling_min_z", 1.5)),
            min_cells=int(tc.get("min_cells", 4)),
            wheelpath_offset_m=float(tc.get("wheelpath_offset_m", 0.85)),
            wheelpath_half_width_m=float(tc.get("wheelpath_half_width_m", 0.35)),
        )


def _zone_bounds(cfg, camera, zones, cls_name: str):
    zone = zones.for_class(cls_name) if zones is not None else None
    if zone is not None and zone.achievable:
        return zone.z_near_m, zone.z_far_m
    return camera.visible_range()


def detect_ravelling(frame, road_mask, camera, baseline, cfg, zones=None,
                     feats=None, x_map=None, z_map=None,
                     tc: TextureConfig | None = None) -> RavellingResult:
    """Grade surface texture on a fixed-metre grid over the road."""
    import numpy as np

    from ..roadseg.ops import compute_features

    tc = tc or TextureConfig.from_cfg(cfg)
    if camera is None:
        return RavellingResult(note="no camera calibration — texture cannot be "
                                    "compared at a fixed ground scale")
    if baseline is None or getattr(baseline, "is_empty", True):
        return RavellingResult(note="no road appearance baseline")
    if road_mask is None or not road_mask.any():
        return RavellingResult(note="no road mask")

    h, w = road_mask.shape[:2]
    if feats is None:
        feats = compute_features(frame, int(cfg.get_path("surface.texture_ksize", 7)))
    if x_map is None or z_map is None:
        x_map, z_map, _ = camera.ground_maps(w, h)

    z_near, z_far = _zone_bounds(cfg, camera, zones, "ravelling")
    med, sigma = baseline.get("rtex")
    rtex = feats.channels()["rtex"]

    # Texture is computed with a box filter, so any pixel within half a kernel of the
    # road boundary mixes in whatever is beyond it — grass, gravel shoulder, kerb. Those
    # pixels read as extremely rough and would be graded as ravelling on every road
    # ever surveyed. In the far field, where the road is only a few pixels wide, that
    # contamination is most of it. So the boundary band is excluded outright.
    from ..roadseg.ops import erode

    interior = erode(road_mask, int(cfg.get_path("surface.texture_ksize", 7)) // 2 + 1)
    if not interior.any():
        interior = road_mask

    valid = interior & np.isfinite(z_map) & (z_map >= z_near) & (z_map <= z_far)
    if valid.sum() < tc.min_cell_px:
        return RavellingResult(note=f"no road inside {z_near:.1f}-{z_far:.1f} m")

    # Integer cell indices in GROUND space, which is what makes cells a constant
    # physical size regardless of where they fall in the image.
    with np.errstate(invalid="ignore"):
        cz = np.where(valid, np.floor(z_map / tc.cell_m), 0.0)
        cx = np.where(valid, np.floor(x_map / tc.cell_m), 0.0)
    # Pack the 2-D cell index into one integer so np.unique can group by cell.
    keys = (cz.astype(np.int64) * 1_000_003 + cx.astype(np.int64))[valid]
    rvals = rtex[valid]
    zvals = z_map[valid]
    xvals = x_map[valid]

    result = RavellingResult(measured=True)
    cell_area = tc.cell_m * tc.cell_m
    uniq, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    for i, n_px in enumerate(counts):
        if n_px < tc.min_cell_px:
            continue
        sel = inverse == i
        tz = float((np.mean(rvals[sel]) - med) / max(sigma, 1e-6))
        result.cells_total += 1
        result.worst_texture_z = max(result.worst_texture_z, tz)
        if tz >= tc.ravelling_min_z:
            result.cells_affected += 1
            result.affected_area_m2 += cell_area
            result.cells.append(TextureCell(
                z_m=float(np.mean(zvals[sel])), x_m=float(np.mean(xvals[sel])),
                texture_z=tz, area_m2=cell_area))

    result.road_area_m2 = result.cells_total * cell_area
    if result.cells_total < tc.min_cells:
        result.measured = False
        result.note = (f"only {result.cells_total} usable cells — too little road "
                       f"at a workable resolution to grade texture")
    return result


def detect_rutting_proxy(frame, road_mask, camera, baseline, cfg, zones=None,
                         feats=None, x_map=None, z_map=None,
                         tc: TextureConfig | None = None) -> RuttingResult:
    """Compare wheel paths against the crown between them.

    Indicative only. A monocular forward camera cannot measure a transverse profile,
    so this reports *whether the wheel paths look different from the crown*, which is
    a correlate of rutting, not a depth.
    """
    import numpy as np

    from ..roadseg.ops import compute_features

    tc = tc or TextureConfig.from_cfg(cfg)
    if camera is None or baseline is None or getattr(baseline, "is_empty", True):
        return RuttingResult(note="needs camera calibration and a road baseline")
    if road_mask is None or not road_mask.any():
        return RuttingResult(note="no road mask")

    h, w = road_mask.shape[:2]
    if feats is None:
        feats = compute_features(frame, int(cfg.get_path("surface.texture_ksize", 7)))
    if x_map is None or z_map is None:
        x_map, z_map, _ = camera.ground_maps(w, h)

    z_near, z_far = _zone_bounds(cfg, camera, zones, "rutting")
    from ..roadseg.ops import erode

    interior = erode(road_mask, int(cfg.get_path("surface.texture_ksize", 7)) // 2 + 1)
    if not interior.any():
        interior = road_mask
    band = interior & np.isfinite(z_map) & (z_map >= z_near) & (z_map <= z_far)
    if band.sum() < 200:
        return RuttingResult(note="too little road in the rutting zone")

    med, sigma = baseline.get("rtex")
    rtex = feats.channels()["rtex"]
    off, hw = tc.wheelpath_offset_m, tc.wheelpath_half_width_m

    def _z(sel):
        if sel.sum() < 60:
            return None
        return float((np.mean(rtex[sel]) - med) / max(sigma, 1e-6))

    left = _z(band & (np.abs(x_map + off) <= hw))
    right = _z(band & (np.abs(x_map - off) <= hw))
    crown = _z(band & (np.abs(x_map) <= hw * 0.8))
    if left is None or right is None or crown is None:
        return RuttingResult(note="wheel paths and crown not all visible")

    return RuttingResult(
        wheelpath_index=float(0.5 * (left + right) - crown),
        left_z=left, right_z=right, crown_z=crown, measured=True,
    )


def detect_drainage(surface, road_mask, camera, cfg, zones=None,
                    x_map=None, z_map=None) -> dict:
    """Drainage symptoms: water accumulating at the carriageway edge.

    Identifying a culvert as a structure needs labelled examples and is deferred. The
    *symptom* of a choked one does not: water pooling against the edge of the road
    instead of draining away from it. That is reported as an observation, clearly
    separate from a confirmed culvert defect.
    """
    import numpy as np

    dc = cfg.get_path("detect.drainage", {}) or {}
    out = {"measured": False, "n_pools": 0, "edge_water_frac": 0.0,
           "worst_pool_m2": 0.0, "note": ""}
    if surface is None or road_mask is None or not road_mask.any():
        out["note"] = "no surface/road data"
        return out
    water = getattr(surface, "water", None)
    if water is None or not water.any():
        out["measured"] = True
        return out
    if camera is None:
        out["note"] = "no calibration — cannot locate water relative to the edge"
        return out

    h, w = road_mask.shape[:2]
    if x_map is None or z_map is None:
        x_map, z_map, _ = camera.ground_maps(w, h)

    # "Edge" band: the outer fraction of the road half-width on each side.
    from ..roadseg.ops import erode

    core = erode(road_mask, max(1, int(float(dc.get("edge_erode_frac", 0.10)) * w)))
    edge_band = road_mask & ~core

    edge_water = water & edge_band
    total_water = float(water.sum())
    out["measured"] = True
    out["edge_water_frac"] = (float(edge_water.sum()) / total_water) if total_water else 0.0

    import cv2

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        edge_water.astype(np.uint8), connectivity=8)
    min_px = int(dc.get("min_pool_px", 250))
    pools = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_px]
    out["n_pools"] = len(pools)

    if pools:
        # Convert the largest pool to ground area where a scaler is unavailable by
        # summing per-pixel ground area from the maps.
        gx = np.abs(np.gradient(np.nan_to_num(x_map, nan=0.0), axis=1))
        gz = np.abs(np.gradient(np.nan_to_num(z_map, nan=0.0), axis=0))
        per_px = gx * gz
        biggest = max(pools, key=lambda i: stats[i, cv2.CC_STAT_AREA])
        m = labels == biggest
        out["worst_pool_m2"] = float(np.nansum(per_px[m]))
    return out
