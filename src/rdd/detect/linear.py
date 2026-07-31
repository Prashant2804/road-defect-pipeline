"""Crack classification: longitudinal, transverse, or alligator — decided by geometry.

The key idea is to **not** ask the network to make this distinction. Detect cracks
class-agnostically, then classify each one by measuring it on the road plane.

Why. In a forward-facing view, a crack's apparent orientation is confounded by
perspective: a longitudinal crack converges toward the vanishing point, so its image
angle depends entirely on where in the frame it happens to be. A network can learn
that, but it has to learn the camera geometry along with the defect, from labels that
are themselves inconsistent — RDD2022's own D00/D10 split is camera-dependent and
noisy. Projected onto the ground plane the question stops being a classification
problem and becomes a direct measurement: is this line along the road, or across it?

Alligator cracking is separated by **connectivity, not appearance**. Fatigue cracking
forms interconnected closed cells, so the crack mask has many enclosed holes; several
parallel cracks, however dense, have none. Counting enclosed cells per m² therefore
distinguishes them deterministically, with no labels and no texture model.

Crack *width* comes from a distance transform converted to ground units, which is
what IRC severity bands are defined on.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..utils.logging import get_logger

log = get_logger("rdd.detect.linear")

LONGITUDINAL = "longitudinal_crack"
TRANSVERSE = "transverse_crack"
ALLIGATOR = "alligator_crack"

# Any of these labels, however the model produced them, get re-measured on the ground
# plane. Both a generic "crack" model and an RDD2022-style D00/D10/D20 model are
# handled, because the model's own longitudinal-vs-transverse split is
# perspective-confounded and is deliberately not trusted.
CRACK_SOURCES = frozenset({
    "crack", "cracks", "linear_crack",
    LONGITUDINAL, TRANSVERSE, ALLIGATOR,
    "D00", "D10", "D20",
})


@dataclass
class CrackGeometry:
    """Ground-plane measurements of one crack detection."""

    cls_name: str = LONGITUDINAL
    angle_deg: float = 0.0          # 0 = along the road, 90 = across it
    length_m: float = 0.0
    width_m: float = 0.0            # representative (median) width
    max_width_m: float = 0.0
    elongation: float = 1.0         # major/minor extent ratio
    cells_per_m2: float = 0.0       # enclosed cells — the alligator signature
    n_cells: int = 0
    area_m2: float = 0.0
    reason: str = ""
    confident: bool = True

    def as_dict(self) -> dict:
        return {
            "class": self.cls_name,
            "angle_deg": round(self.angle_deg, 1),
            "length_m": round(self.length_m, 3),
            "width_mm": round(1000 * self.width_m, 1),
            "max_width_mm": round(1000 * self.max_width_m, 1),
            "elongation": round(self.elongation, 2),
            "cells_per_m2": round(self.cells_per_m2, 2),
            "n_cells": self.n_cells,
            "area_m2": round(self.area_m2, 4),
            "reason": self.reason,
        }


@dataclass
class LinearConfig:
    longitudinal_max_deg: float = 30.0
    transverse_min_deg: float = 60.0
    alligator_min_cells_per_m2: float = 4.0
    alligator_min_cells: int = 3
    alligator_max_elongation: float = 4.0
    min_length_m: float = 0.15
    min_points: int = 30

    @classmethod
    def from_cfg(cls, cfg) -> "LinearConfig":
        lc = cfg.get_path("detect.linear", {}) or {}
        return cls(
            longitudinal_max_deg=float(lc.get("longitudinal_max_deg", 30.0)),
            transverse_min_deg=float(lc.get("transverse_min_deg", 60.0)),
            alligator_min_cells_per_m2=float(lc.get("alligator_min_cells_per_m2", 4.0)),
            alligator_min_cells=int(lc.get("alligator_min_cells", 3)),
            alligator_max_elongation=float(lc.get("alligator_max_elongation", 4.0)),
            min_length_m=float(lc.get("min_length_m", 0.15)),
            min_points=int(lc.get("min_points", 30)),
        )


def _ground_points(mask, x_map, z_map):
    """Ground coordinates of the mask's pixels, dropping any without a ground point."""
    import numpy as np

    xs = x_map[mask]
    zs = z_map[mask]
    ok = np.isfinite(xs) & np.isfinite(zs)
    return xs[ok], zs[ok]


def _principal_orientation(xs, zs) -> tuple[float, float, float, float]:
    """(angle_deg_from_road_axis, length_m, minor_extent_m, elongation).

    PCA on the ground points. `z` is along the road and `x` across it, so an angle of
    0 means the crack runs with traffic and 90 means it crosses the carriageway.
    """
    import numpy as np

    pts = np.stack([xs, zs], axis=1)
    pts = pts - pts.mean(axis=0)
    if len(pts) < 3:
        return 0.0, 0.0, 0.0, 1.0

    cov = np.cov(pts, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    major = vecs[:, 0]

    # Projected extents give a robust length that a single outlier cannot inflate.
    proj_major = pts @ major
    proj_minor = pts @ vecs[:, 1]
    length = float(proj_major.max() - proj_major.min())
    minor = float(proj_minor.max() - proj_minor.min())

    # major = (dx, dz). atan2(|dx|, |dz|): 0 -> along road, 90 -> across.
    angle = math.degrees(math.atan2(abs(float(major[0])), abs(float(major[1]))))
    elong = length / minor if minor > 1e-6 else float("inf")
    return angle, length, minor, elong


def _width_m(mask, x_map, z_map):
    """(median, max) crack width in metres, from a distance transform.

    The distance transform gives each interior pixel its distance to the nearest
    edge, so twice the ridge value is the local width. Converting with the ground
    scale *at each pixel* matters: the same crack is several times more pixels wide
    near the vehicle than at the far end of its zone.
    """
    import cv2
    import numpy as np

    m = mask.astype(np.uint8)
    if not m.any():
        return 0.0, 0.0
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)

    # Lateral ground scale per pixel, from the horizontal gradient of the x map.
    gx = np.abs(np.gradient(np.nan_to_num(x_map, nan=0.0), axis=1))
    ridge = dist > 0.5 * dist.max()
    sel = ridge & mask
    if not sel.any():
        sel = mask
    widths = 2.0 * dist[sel] * gx[sel]
    widths = widths[np.isfinite(widths) & (widths > 0)]
    if widths.size == 0:
        return 0.0, 0.0
    return float(np.median(widths)), float(widths.max())


def _enclosed_cells(mask, x_map, z_map, area_m2: float):
    """(n_cells, cells_per_m2) — enclosed regions surrounded by crack.

    The alligator signature. Interconnected fatigue cracking encloses pavement
    fragments; parallel cracks enclose nothing. Cells smaller than a few pixels are
    ignored as skeleton noise.
    """
    import cv2
    import numpy as np

    from ..roadseg.ops import fill_holes

    holes = fill_holes(mask) & ~mask
    if not holes.any():
        return 0, 0.0

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        holes.astype(np.uint8), connectivity=8)
    kept = sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 6)
    if kept == 0 or area_m2 <= 1e-6:
        return kept, 0.0
    return kept, kept / area_m2


def classify_crack(mask, camera, scaler=None, cfg=None,
                   lin: LinearConfig | None = None,
                   x_map=None, z_map=None) -> CrackGeometry:
    """Measure a crack mask on the road plane and name it.

    `mask` is a boolean image mask for one detection. `camera` supplies the
    pixel->ground mapping; without it no ground measurement is possible and the
    detection is returned unclassified rather than guessed at.
    """
    import numpy as np

    lin = lin or (LinearConfig.from_cfg(cfg) if cfg is not None else LinearConfig())
    if camera is None:
        return CrackGeometry(cls_name=LONGITUDINAL, confident=False,
                             reason="no camera calibration — orientation not measurable")
    if not mask.any():
        return CrackGeometry(confident=False, reason="empty mask")

    h, w = mask.shape[:2]
    if x_map is None or z_map is None:
        x_map, z_map, _ = camera.ground_maps(w, h)

    xs, zs = _ground_points(mask, x_map, z_map)
    if xs.size < lin.min_points:
        return CrackGeometry(confident=False,
                             reason=f"only {xs.size} px with a ground point")

    angle, length, minor, elong = _principal_orientation(xs, zs)
    area_m2 = float(scaler.area_m2(mask) or 0.0) if scaler is not None else 0.0
    if area_m2 <= 0:
        # Fall back to a bounding-box estimate on the ground plane.
        area_m2 = max(1e-6, length * max(minor, 1e-3))

    n_cells, cells_per_m2 = _enclosed_cells(mask, x_map, z_map, area_m2)
    width_m, max_width_m = _width_m(mask, x_map, z_map)

    geom = CrackGeometry(
        angle_deg=angle, length_m=length, width_m=width_m, max_width_m=max_width_m,
        elongation=elong if math.isfinite(elong) else 999.0,
        cells_per_m2=cells_per_m2, n_cells=n_cells, area_m2=area_m2,
    )

    # Alligator first: interconnected cells outrank orientation, because a patch of
    # fatigue cracking has no single meaningful direction.
    if (n_cells >= lin.alligator_min_cells
            and cells_per_m2 >= lin.alligator_min_cells_per_m2
            and geom.elongation <= lin.alligator_max_elongation):
        geom.cls_name = ALLIGATOR
        geom.reason = (f"{n_cells} enclosed cells ({cells_per_m2:.1f}/m²) — "
                       f"interconnected fatigue cracking")
        return geom

    if length < lin.min_length_m:
        geom.confident = False
        geom.reason = f"only {length * 100:.0f} cm long on the ground"

    if angle <= lin.longitudinal_max_deg:
        geom.cls_name = LONGITUDINAL
        geom.reason = geom.reason or f"{angle:.0f}° to the road axis — runs with traffic"
    elif angle >= lin.transverse_min_deg:
        geom.cls_name = TRANSVERSE
        geom.reason = geom.reason or f"{angle:.0f}° to the road axis — crosses the road"
    else:
        # Diagonal: assign to the nearer class but flag the ambiguity rather than
        # pretending the measurement was clean.
        mid = 0.5 * (lin.longitudinal_max_deg + lin.transverse_min_deg)
        geom.cls_name = LONGITUDINAL if angle < mid else TRANSVERSE
        geom.confident = False
        geom.reason = f"diagonal at {angle:.0f}° — between the L and T bands"
    return geom


@dataclass
class LinearStats:
    """Run-level tally of how cracks were reclassified."""

    seen: int = 0
    relabelled: int = 0
    by_class: dict = field(default_factory=dict)
    unconfident: int = 0

    def update(self, source: str, geom: CrackGeometry) -> None:
        self.seen += 1
        self.by_class[geom.cls_name] = self.by_class.get(geom.cls_name, 0) + 1
        if geom.cls_name != source:
            self.relabelled += 1
        if not geom.confident:
            self.unconfident += 1

    def summary(self) -> dict:
        return {
            "cracks_measured": self.seen,
            "reclassified": self.relabelled,
            "by_class": dict(sorted(self.by_class.items())),
            "low_confidence": self.unconfident,
        }
