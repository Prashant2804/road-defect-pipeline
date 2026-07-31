"""Edge damage and shoulder erosion, measured from the road mask's own boundary.

This class needs no detector and no labels, which makes it the cheapest of the seven
to add: the road segmentation already produces the exact feature involved. Edge damage
*is* an irregularity of the carriageway boundary.

The method is to fit what the edge *should* look like and measure how far reality
departs from it. An intact edge, in ground coordinates, is a smooth curve — straight on
a straight road, gently curving on a bend. Break-up, ravelled shoulders and erosion
make it locally ragged and pull it inward, because material is missing.

Working in ground coordinates rather than pixels is essential. In the image, a
perfectly straight edge converges toward the vanishing point, so "deviation from
straight" in pixels is dominated by perspective and says nothing about the road. On the
ground plane the deviation is a distance in metres, which is directly what a condition
survey reports and what IRC severity bands are defined on.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..utils.logging import get_logger

log = get_logger("rdd.detect.boundary")


@dataclass
class EdgeDefect:
    """One stretch of damaged carriageway edge."""

    side: str                  # left | right
    z_start_m: float
    z_end_m: float
    max_inset_m: float         # deepest loss of surface, metres
    mean_inset_m: float
    raggedness_m: float        # high-frequency deviation energy
    length_m: float

    def as_dict(self) -> dict:
        return {
            "side": self.side,
            "z_range_m": [round(self.z_start_m, 2), round(self.z_end_m, 2)],
            "length_m": round(self.length_m, 2),
            "max_inset_m": round(self.max_inset_m, 3),
            "mean_inset_m": round(self.mean_inset_m, 3),
            "raggedness_m": round(self.raggedness_m, 4),
        }


@dataclass
class BoundaryResult:
    defects: list[EdgeDefect] = field(default_factory=list)
    left_raggedness_m: float = 0.0
    right_raggedness_m: float = 0.0
    measured: bool = False
    note: str = ""

    def summary(self) -> dict:
        return {
            "measured": self.measured,
            "n_edge_defects": len(self.defects),
            "left_raggedness_m": round(self.left_raggedness_m, 4),
            "right_raggedness_m": round(self.right_raggedness_m, 4),
            "worst_inset_m": round(max((d.max_inset_m for d in self.defects),
                                       default=0.0), 3),
            "note": self.note,
        }


@dataclass
class BoundaryConfig:
    min_inset_m: float = 0.10        # loss below this is measurement noise
    min_length_m: float = 0.40       # a defect must persist along the road
    poly_order: int = 2              # straight or gently curving
    n_samples: int = 40
    z_near_m: float = 0.0            # 0 -> use the class's assessment zone
    z_far_m: float = 0.0
    max_residual_m: float = 1.2      # beyond this the fit itself is untrustworthy

    @classmethod
    def from_cfg(cls, cfg) -> "BoundaryConfig":
        bc = cfg.get_path("detect.boundary", {}) or {}
        return cls(
            min_inset_m=float(bc.get("min_inset_m", 0.10)),
            min_length_m=float(bc.get("min_length_m", 0.40)),
            poly_order=int(bc.get("poly_order", 2)),
            n_samples=int(bc.get("n_samples", 40)),
            z_near_m=float(bc.get("z_near_m", 0.0)),
            z_far_m=float(bc.get("z_far_m", 0.0)),
            max_residual_m=float(bc.get("max_residual_m", 1.2)),
        )


def _sample_edges(road_mask, x_map, z_map, z_near: float, z_far: float, n: int):
    """Ground-coordinate (z, x_left, x_right) samples of the carriageway edges."""
    import numpy as np

    h, w = road_mask.shape[:2]
    rows = np.linspace(0, h - 1, num=min(h, max(8, n * 3))).astype(int)
    zs, lefts, rights = [], [], []

    for r in rows:
        cols = np.nonzero(road_mask[r])[0]
        if cols.size < 8:
            continue
        z = z_map[r, cols[cols.size // 2]]
        if not np.isfinite(z) or not (z_near <= z <= z_far):
            continue
        xl, xr = x_map[r, cols[0]], x_map[r, cols[-1]]
        if not (np.isfinite(xl) and np.isfinite(xr)):
            continue
        zs.append(float(z))
        lefts.append(float(xl))
        rights.append(float(xr))

    if not zs:
        return None
    order = np.argsort(zs)
    return (np.array(zs)[order], np.array(lefts)[order], np.array(rights)[order])


def _fit_and_deviate(z, x, order: int):
    """Robust polynomial fit of edge position against range; returns (fit, residual).

    Robust because the damage itself is the outlier we are measuring — a plain
    least-squares fit would be dragged into the eroded section and under-report it.
    One reweighted pass is enough at these sample counts.
    """
    import numpy as np

    order = max(1, min(order, 3))
    if len(z) < order + 3:
        return None, None
    coef = np.polyfit(z, x, order)
    resid = x - np.polyval(coef, z)
    keep = np.abs(resid) <= max(0.05, 2.0 * float(np.median(np.abs(resid))))
    if keep.sum() >= order + 3:
        coef = np.polyfit(z[keep], x[keep], order)
        resid = x - np.polyval(coef, z)
    return coef, resid


def _runs(flags, z, inset):
    """Contiguous stretches where the edge is inset beyond threshold."""
    import numpy as np

    out = []
    i = 0
    n = len(flags)
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flags[j + 1]:
            j += 1
        seg = inset[i:j + 1]
        out.append((float(z[i]), float(z[j]), float(np.max(seg)), float(np.mean(seg))))
        i = j + 1
    return out


def detect_edge_damage(road_mask, camera, cfg, zones=None,
                       x_map=None, z_map=None,
                       bc: BoundaryConfig | None = None) -> BoundaryResult:
    """Measure carriageway-edge loss on the ground plane."""
    import numpy as np

    bc = bc or BoundaryConfig.from_cfg(cfg)
    if camera is None:
        return BoundaryResult(note="no camera calibration — edge loss is not "
                                   "measurable in metres")
    if road_mask is None or not road_mask.any():
        return BoundaryResult(note="no road mask")

    h, w = road_mask.shape[:2]
    if x_map is None or z_map is None:
        x_map, z_map, _ = camera.ground_maps(w, h)

    z_near, z_far = bc.z_near_m, bc.z_far_m
    if z_far <= z_near:
        zone = zones.for_class("edge_damage") if zones is not None else None
        if zone is not None and zone.achievable:
            z_near, z_far = zone.z_near_m, zone.z_far_m
        else:
            z_near, z_far = camera.visible_range()

    sampled = _sample_edges(road_mask, x_map, z_map, z_near, z_far, bc.n_samples)
    if sampled is None:
        return BoundaryResult(note=f"no road rows inside {z_near:.1f}-{z_far:.1f} m")
    z, left, right = sampled

    result = BoundaryResult(measured=True)
    for side, x, sign in (("left", left, +1.0), ("right", right, -1.0)):
        coef, resid = _fit_and_deviate(z, x, bc.poly_order)
        if coef is None:
            result.note = "too few edge samples to fit a baseline"
            continue

        rag = float(np.sqrt(np.mean(resid ** 2)))
        if side == "left":
            result.left_raggedness_m = rag
        else:
            result.right_raggedness_m = rag

        if rag > bc.max_residual_m:
            # The "edge" is not following any smooth curve, so the mask is probably
            # wrong rather than the road being catastrophically broken.
            result.note = (f"{side} edge does not follow a smooth curve "
                           f"(RMS {rag:.2f} m) — treating the mask as unreliable")
            continue

        # Inset = the edge has moved *toward* the centreline, i.e. surface is missing.
        # Sign differs per side: left edge insets by increasing x, right by decreasing.
        inset = sign * resid
        flags = inset >= bc.min_inset_m
        for z0, z1, mx, mean in _runs(flags, z, inset):
            if (z1 - z0) < bc.min_length_m:
                continue
            result.defects.append(EdgeDefect(
                side=side, z_start_m=z0, z_end_m=z1, max_inset_m=mx,
                mean_inset_m=mean, raggedness_m=rag, length_m=z1 - z0))

    return result
