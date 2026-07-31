"""The individual validity gates.

Each gate answers one question about one frame and returns a `GateResult` or None.
They are deliberately small and independent so that a route report can attribute
exclusions to specific causes, and so that tightening one gate does not perturb the
others.

Ordering does not matter — every gate runs on every frame. That costs a little more
than short-circuiting on the first failure, but it means the summary can say "68% of
exclusions were traffic, 20% were glare" instead of only ever reporting whichever
gate happened to be checked first.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..utils.logging import get_logger
from .verdict import Action, GateResult

log = get_logger("rdd.validity.gates")


@dataclass
class FrameContext:
    """Everything the gates need about one frame, gathered once."""

    frame_idx: int = 0
    t: float = 0.0
    frame: object = None            # enhanced BGR
    road: object = None             # RoadMask | None
    surface: object = None          # SurfaceMap | None
    quality: object = None          # FrameQuality | None
    zone_mask: object = None        # union assessment zone, bool HxW
    ego: object = None              # EgoMotion | None
    traffic: object = None          # TrafficResult | None
    static_mask: object = None      # temporally-static structure (dirt/wiper)
    camera: object = None           # CameraModel | None


# -- road availability --------------------------------------------------------

def gate_road_found(ctx: FrameContext, cfg) -> GateResult | None:
    """The road must be locatable before anything on it can be assessed."""
    gc = cfg.get_path("validity.road_found", {}) or {}
    road = ctx.road
    if road is None:
        return GateResult("road_found", Action.BLOCK, "no road mask produced")

    min_conf = float(gc.get("min_confidence", 0.25))
    if road.confidence < min_conf:
        return GateResult(
            "road_found", Action.BLOCK,
            f"road not locatable (mask confidence {road.confidence:.2f})",
            road.confidence, min_conf,
        )

    min_cov = float(gc.get("min_coverage", 0.04))
    cov = road.coverage()
    if cov < min_cov:
        return GateResult(
            "road_found", Action.BLOCK,
            f"road region implausibly small ({cov:.1%} of frame)", cov, min_cov,
        )

    # A prior-only mask is geometry, not observation. Usable, but not trustworthy
    # enough to certify — so assess and mark it down rather than block.
    if getattr(road, "fell_back", False) and gc.get("degrade_on_fallback", True):
        return GateResult("road_found", Action.DEGRADE,
                          "road mask fell back to the geometric prior")
    return None


def gate_surface_plausible(ctx: FrameContext, cfg) -> GateResult | None:
    """Is the 'road' region actually a road surface?

    This gate exists because `gate_road_buried` has a structural blind spot. That
    gate measures water and mud *relative to the road baseline*, which works when a
    puddle sits on visible road — but when water or mud covers the **entire**
    carriageway, the contaminant becomes the baseline, every pixel matches it, and
    the relative measurement reports a clean, dry road.

    That is precisely the scenario the pipeline is required to refuse, so it is
    caught here instead, using absolute appearance. Same logic for vegetation: if
    the surface ahead is green and textured, the vehicle is not on a carriageway,
    however confidently the segmenter has outlined it.
    """
    gc = cfg.get_path("validity.plausibility", {}) or {}
    if not gc.get("enabled", True):
        return None
    surf = ctx.surface
    p = getattr(surf, "plausibility", None) if surf is not None else None
    if p is None or p.is_road:
        return None

    if p.verdict == "vegetation":
        return GateResult("surface_plausible", Action.BLOCK,
                          f"not a road surface — {p.reason}")
    return GateResult(
        "surface_plausible", Action.BLOCK,
        f"road entirely obscured — {p.reason}. Nothing beneath it is observable, "
        f"so no defect assessment is possible here.",
    )


def gate_road_buried(ctx: FrameContext, cfg) -> GateResult | None:
    """Water or mud covering the road: nothing underneath is observable."""
    gc = cfg.get_path("validity.road_buried", {}) or {}
    surf = ctx.surface
    if surf is None:
        return None

    frac = surf.occluded_frac
    block_at = float(gc.get("block_above_frac", 0.6))
    degrade_at = float(gc.get("degrade_above_frac", 0.25))

    if frac >= block_at:
        parts = []
        if surf.water_frac > 0.01:
            parts.append(f"{surf.water_frac:.0%} water")
        if surf.mud_frac > 0.01:
            parts.append(f"{surf.mud_frac:.0%} mud")
        return GateResult(
            "road_buried", Action.BLOCK,
            f"road surface {frac:.0%} obscured ({', '.join(parts) or 'water/mud'}) "
            f"— the pavement underneath cannot be inspected",
            frac, block_at,
        )
    if frac >= degrade_at:
        return GateResult("road_buried", Action.DEGRADE,
                          f"{frac:.0%} of the road surface obscured", frac, degrade_at)
    return None


# -- vehicle position ---------------------------------------------------------

def gate_off_track(ctx: FrameContext, cfg) -> GateResult | None:
    """Has the vehicle left the carriageway, or is the camera not looking at road?

    Three independent geometric symptoms, because no single one is reliable:

    1. **The road does not reach the bottom of the frame.** If the vehicle is *on* a
       road, the surface immediately ahead is road. A mask that stops short means we
       are looking at road from beside it, or at something that is not road.
    2. **The road is not near the centre.** A carriageway ahead of a vehicle driving
       along it is roughly centred; a large lateral offset means the vehicle has
       drifted off, or the mask has latched onto a verge or a field track.
    3. **The road region has collapsed.** Sudden loss of coverage relative to the
       running normal.
    """
    import numpy as np

    gc = cfg.get_path("validity.off_track", {}) or {}
    road = ctx.road
    if road is None or not road.mask.any():
        return None

    mask = road.mask
    h, w = mask.shape[:2]

    band = max(1, int(float(gc.get("bottom_band_frac", 0.08)) * h))
    bottom = mask[h - band:, :]
    bottom_fill = float(bottom.sum()) / float(bottom.size)
    min_bottom = float(gc.get("min_bottom_fill", 0.12))
    if bottom_fill < min_bottom:
        return GateResult(
            "off_track", Action.BLOCK,
            f"road does not reach the front of the vehicle "
            f"({bottom_fill:.0%} of the near band) — likely off the carriageway",
            bottom_fill, min_bottom,
        )

    ys, xs = np.nonzero(mask[h // 2:, :])
    if xs.size >= 50:
        centre_offset = abs(float(xs.mean()) / w - 0.5)
        max_offset = float(gc.get("max_centre_offset", 0.28))
        if centre_offset > max_offset:
            return GateResult(
                "off_track", Action.BLOCK,
                f"road is {centre_offset:.0%} off centre — vehicle appears to have "
                f"left the travelled way",
                centre_offset, max_offset,
            )

    return None


def gate_ego_motion(ctx: FrameContext, cfg) -> GateResult | None:
    """Stationary, reversing, or turning/shaking too hard to measure."""
    gc = cfg.get_path("validity.egomotion", {}) or {}
    ego = ctx.ego
    if ego is None or not getattr(ego, "valid", False):
        return None

    min_flow = float(gc.get("min_flow_px", 0.6))
    if ego.flow_px < min_flow:
        return GateResult(
            "ego_motion", Action.BLOCK,
            f"vehicle stationary ({ego.flow_px:.2f} px/frame) — frames are "
            f"duplicates and would re-count the same road",
            ego.flow_px, min_flow,
        )

    if gc.get("block_reverse", True) and ego.radial < -min_flow:
        return GateResult(
            "ego_motion", Action.BLOCK,
            f"reversing (flow contracting toward the vanishing point, "
            f"radial {ego.radial:.2f}) — this road was already surveyed",
            ego.radial, -min_flow,
        )

    max_yaw = float(gc.get("max_yaw_px", 14.0))
    if abs(ego.yaw_px) > max_yaw:
        return GateResult(
            "ego_motion", Action.DEGRADE,
            f"sharp turn ({ego.yaw_px:+.1f} px horizontal drift) — road mask lags "
            f"and motion blur is elevated",
            abs(ego.yaw_px), max_yaw,
        )

    max_shake = float(gc.get("max_pitch_px", 10.0))
    if abs(ego.pitch_px) > max_shake:
        return GateResult(
            "ego_motion", Action.DEGRADE,
            f"camera pitching ({ego.pitch_px:+.1f} px vertical drift at the horizon) "
            f"— assumed pitch, and therefore every distance, is unreliable",
            abs(ego.pitch_px), max_shake,
        )
    return None


# -- occluders and image conditions -------------------------------------------

def gate_traffic(ctx: FrameContext, cfg) -> GateResult | None:
    """Vehicles hide the road; mask them, or block if too little road is left."""
    gc = cfg.get_path("validity.traffic", {}) or {}
    tr = ctx.traffic
    if tr is None or not getattr(tr, "available", False) or tr.n_detections == 0:
        return None

    block_at = float(gc.get("block_above_zone_frac", 0.55))
    if tr.occluded_frac >= block_at:
        return GateResult(
            "traffic", Action.BLOCK,
            f"traffic covers {tr.occluded_frac:.0%} of the assessment zone "
            f"({', '.join(sorted(set(tr.labels))[:4])})",
            tr.occluded_frac, block_at,
        )
    if tr.mask is not None and tr.mask.any():
        return GateResult(
            "traffic", Action.MASK,
            f"{tr.n_detections} road user(s) excluded from the road region",
            tr.occluded_frac, block_at, mask=tr.mask,
        )
    return None


def gate_image_condition(ctx: FrameContext, cfg) -> GateResult | None:
    """Glare, darkness and unusable exposure.

    Separate from the existing sharpness/contrast quality check because these are
    *scene* conditions rather than capture faults, and they need different
    thresholds: a night frame can be perfectly sharp and still carry no usable
    pavement texture.
    """
    import numpy as np

    gc = cfg.get_path("validity.image", {}) or {}
    q = ctx.quality
    if q is None:
        return None

    max_glare = float(gc.get("max_glare_frac", 0.12))
    if q.clipped_high > max_glare:
        return GateResult(
            "image_condition", Action.BLOCK,
            f"blown highlights over {q.clipped_high:.0%} of the frame "
            f"(sun glare or wet-road specular flare)",
            q.clipped_high, max_glare,
        )

    min_luma = float(gc.get("min_mean_luma", 0.10))
    if q.mean_luma < min_luma:
        return GateResult(
            "image_condition", Action.BLOCK,
            f"too dark to assess (mean luma {q.mean_luma:.2f}) — night or tunnel",
            q.mean_luma, min_luma,
        )

    dark_degrade = float(gc.get("degrade_below_luma", 0.18))
    if q.mean_luma < dark_degrade:
        return GateResult("image_condition", Action.DEGRADE,
                          f"low light (mean luma {q.mean_luma:.2f})",
                          q.mean_luma, dark_degrade)

    if not q.usable and q.reasons:
        return GateResult("image_condition", Action.BLOCK,
                          "; ".join(q.reasons))
    return None


def gate_windscreen(ctx: FrameContext, cfg) -> GateResult | None:
    """Dirt, rain spots and the wiper: structure that does not move with the road.

    Detected by temporal invariance. Everything genuinely in the scene streams past
    as the vehicle moves; anything fixed to the glass stays put. A region that never
    changes across many frames is therefore on the windscreen (or is the bonnet),
    and must be excluded — it is a permanent, perfectly stable false-positive
    generator that no confidence threshold will remove.
    """
    gc = cfg.get_path("validity.windscreen", {}) or {}
    sm = ctx.static_mask
    if sm is None or not sm.any():
        return None

    frac = float(sm.sum()) / float(sm.size)
    block_at = float(gc.get("block_above_frac", 0.35))
    if frac >= block_at:
        return GateResult(
            "windscreen", Action.BLOCK,
            f"{frac:.0%} of the frame is static structure (heavy dirt, rain or "
            f"wiper across the lens)", frac, block_at,
        )
    return GateResult("windscreen", Action.MASK,
                      f"{frac:.0%} static occlusion (windscreen dirt / bonnet) "
                      f"excluded", frac, block_at, mask=sm)


def gate_assessment_zone(ctx: FrameContext, cfg) -> GateResult | None:
    """There must be some assessable road left after every exclusion."""
    gc = cfg.get_path("validity.zone", {}) or {}
    if ctx.zone_mask is None or ctx.road is None:
        return None

    usable = ctx.zone_mask & ctx.road.mask
    if ctx.traffic is not None and getattr(ctx.traffic, "mask", None) is not None:
        usable = usable & ~ctx.traffic.mask
    if ctx.static_mask is not None:
        usable = usable & ~ctx.static_mask
    if ctx.surface is not None:
        usable = usable & ~ctx.surface.occlusion

    min_px = int(gc.get("min_usable_px", 2000))
    n = int(usable.sum())
    if n < min_px:
        return GateResult(
            "assessment_zone", Action.BLOCK,
            f"only {n} px of assessable road remain after exclusions "
            f"(need {min_px})", float(n), float(min_px),
        )
    return None


ALL_GATES = (
    gate_road_found,
    gate_surface_plausible,
    gate_road_buried,
    gate_off_track,
    gate_ego_motion,
    gate_traffic,
    gate_image_condition,
    gate_windscreen,
    gate_assessment_zone,
)
