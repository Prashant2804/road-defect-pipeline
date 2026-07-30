"""Road-surface condition: which parts of the road are water, mud, or clear.

The problem this solves is not "detect puddles" — it is that **you cannot inspect
a surface you cannot see**. Standing water and mud sit *on top of* the road and
hide whatever is underneath. A pothole under 5 cm of muddy water is invisible; a
detector that reports "no defect there" is not observing an intact road, it is
failing to observe anything at all. Treating that as a clean reading is how a
survey ends up certifying a road it never actually inspected.

So this stage produces an **occlusion mask**, and the pipeline uses it to
*abstain*: defects overlapping it are marked indeterminate rather than scored,
and the report states what fraction of the road could not be assessed.

Everything is judged relative to the road's own measured appearance (the
`RoadBaseline` from segmentation) rather than against absolute colour values.
Laterite, gravel and dried mud look nothing like each other, and all of them
change under cloud, so absolute thresholds do not survive contact with real
footage.

Physical cues used, in z-scores against the road baseline:

  water : *smoothness is mandatory*. A water surface is specular — it destroys
          the high-frequency texture of gravel. Supporting evidence is either
          unusual brightness (sky/sun reflection) or unusual darkness (depth).
  mud   : a warm chroma shift (toward red/yellow in LAB), usually darker and
          somewhat smoother than dry surface.
  shadow: darker, but texture is *preserved* and chroma barely shifts. Detected
          explicitly and excluded, because otherwise every shadow becomes "mud"
          and the unassessable fraction is nonsense.

Known limitation, stated plainly: if the *entire* road is uniformly covered, the
covering becomes the baseline and relative detection finds nothing. That case is
detected heuristically and warned about, but it cannot be resolved from relative
statistics alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..roadseg.ops import channel_stats, compute_features, keep_components_touching
from ..utils.logging import get_logger

log = get_logger("rdd.surface")

_CHANNELS = ("l", "a", "b", "s", "tex", "rtex", "cr", "cg", "v")


@dataclass
class SurfaceMap:
    """Per-frame surface condition, restricted to the road mask."""

    water: "object"
    mud: "object"
    dry: "object"
    occlusion: "object"          # water | mud — the unassessable region
    road_area_px: float = 0.0
    water_px: float = 0.0
    mud_px: float = 0.0
    baseline_source: str = "segmenter"

    def _frac(self, px: float) -> float:
        return (px / self.road_area_px) if self.road_area_px > 0 else 0.0

    @property
    def water_frac(self) -> float:
        return self._frac(self.water_px)

    @property
    def mud_frac(self) -> float:
        return self._frac(self.mud_px)

    @property
    def occluded_frac(self) -> float:
        return self._frac(self.water_px + self.mud_px)

    @property
    def dry_frac(self) -> float:
        return max(0.0, 1.0 - self.occluded_frac)


@dataclass
class SurfaceStats:
    """Run-level aggregate over frames.

    Accumulated as areas, not as a mean of per-frame fractions: road area varies
    frame to frame (bends, occlusion of the view), and averaging fractions would
    weight a frame showing 2 m² of road the same as one showing 40 m².
    """

    frames: int = 0
    road_px: float = 0.0
    water_px: float = 0.0
    mud_px: float = 0.0
    frames_with_occlusion: int = 0
    _per_frame_occluded: list[float] = field(default_factory=list)

    def update(self, sm: SurfaceMap) -> None:
        self.frames += 1
        self.road_px += sm.road_area_px
        self.water_px += sm.water_px
        self.mud_px += sm.mud_px
        occ = sm.occluded_frac
        self._per_frame_occluded.append(occ)
        if occ > 0.01:
            self.frames_with_occlusion += 1

    def _frac(self, px: float) -> float:
        return (px / self.road_px) if self.road_px > 0 else 0.0

    @property
    def water_frac(self) -> float:
        return self._frac(self.water_px)

    @property
    def mud_frac(self) -> float:
        return self._frac(self.mud_px)

    @property
    def unassessable_frac(self) -> float:
        return self._frac(self.water_px + self.mud_px)

    @property
    def worst_frame_occluded_frac(self) -> float:
        return max(self._per_frame_occluded, default=0.0)

    def summary(self) -> dict:
        return {
            "frames": self.frames,
            "road_surface_water_frac": round(self.water_frac, 5),
            "road_surface_mud_frac": round(self.mud_frac, 5),
            "road_surface_unassessable_frac": round(self.unassessable_frac, 5),
            "frames_with_occlusion": self.frames_with_occlusion,
            "worst_frame_unassessable_frac": round(self.worst_frame_occluded_frac, 5),
        }


def _zmaps(feats, stats: dict, smooth_ksize: int = 21) -> dict:
    """Signed z-score maps per channel, spatially smoothed.

    Smoothing is essential here, not cosmetic. The baseline sigma is a *per-pixel*
    spread, so on a perfectly clean road roughly a third of pixels sit beyond one
    sigma from sensor noise alone — thresholding raw per-pixel z-scores flags most
    of the road as contaminated. Water and mud are *regions*, so averaging over a
    neighbourhood suppresses noise by roughly the kernel width while a genuine
    area-wide shift passes through intact. A threshold of 1.0 then means "this
    region really is a sigma away from normal road", which is what config implies.
    """
    import cv2

    channels = feats.channels()
    k = (max(3, int(smooth_ksize) | 1),) * 2
    z = {}
    for ch in _CHANNELS:
        arr = channels.get(ch)
        if arr is None:
            arr = getattr(feats, ch)
        med, sigma = stats.get(ch, (0.0, 1.0))
        z[ch] = cv2.boxFilter((arr - med) / max(sigma, 1e-6), -1, k, normalize=True)
    return z


def _baseline_stats(feats, road, baseline) -> tuple[dict, str]:
    """Use the segmenter's baseline when available, else measure the road region.

    The segmenter's baseline is preferred because it is measured on a small
    high-confidence seed. Falling back to the whole road mask lets heavy
    contamination drag the median toward the mud, after which the mud reads as
    perfectly normal road.
    """
    if baseline is not None and not baseline.is_empty:
        if all(c in baseline.stats for c in _CHANNELS):
            return dict(baseline.stats), "segmenter"
    if not road.any():
        return {c: (0.0, 1.0) for c in _CHANNELS}, "empty"
    return channel_stats(feats, road, _CHANNELS), "road-region"


def _warn_uniform_contamination(stats: dict, cfg) -> None:
    """Flag the case where the whole road already looks like mud."""
    a_med = stats.get("a", (0.0, 1.0))[0]
    b_med = stats.get("b", (0.0, 1.0))[0]
    tex_med = stats.get("tex", (0.0, 1.0))[0]
    ac = cfg.get_path("surface.absolute_mud_hint", {}) or {}
    if (a_med >= float(ac.get("min_a", 140.0))
            and b_med >= float(ac.get("min_b", 140.0))
            and tex_med <= float(ac.get("max_texture", 4.0))):
        log.warning(
            "Road baseline itself looks mud-like (LAB a=%.0f b=%.0f, texture=%.1f). "
            "If the whole surface is covered, relative detection cannot separate "
            "mud from road — the unassessable fraction will read low. Inspect the "
            "annotated video before trusting the condition breakdown.",
            a_med, b_med, tex_med,
        )


def _warn_featureless_baseline(stats: dict) -> None:
    """Flag a road surface with no measurable texture at all.

    Usually video compression, which smooths fine gravel detail away entirely.
    It matters because texture is the primary water cue: with no baseline texture
    to lose, water becomes much harder to distinguish from a pale dry patch.
    """
    tex_med = stats.get("tex", (0.0, 1.0))[0]
    if tex_med < 0.5:
        log.warning(
            "Road surface has almost no measurable texture (median %.2f). The "
            "source is probably compressed enough to erase gravel detail, which "
            "weakens water detection — it relies on losing that texture. Prefer a "
            "higher-bitrate source, or raise preprocess.reproject.crf quality.",
            tex_med,
        )


def _classify(z: dict, sc: dict, scale: float = 1.0):
    """Apply the water/mud/shadow rules to a set of z-maps.

    `scale` multiplies every threshold, so the same rules serve both the strict
    detection pass and the relaxed extent pass of the hysteresis.
    """
    import numpy as np

    # Every cue below is built from an illumination-invariant quantity where one
    # exists, so that shade cannot masquerade as contamination.
    #   smoother     : relative texture — invariant under a shadow, collapses on water
    #   warmer       : linear-RGB chromaticity — invariant under a shadow, shifts on mud
    #   chroma_shift : total chromaticity change; ~0 for a true shadow
    smoother = np.maximum(0.0, -z["rtex"])
    darker = np.maximum(0.0, -z["l"])
    brighter = np.maximum(0.0, z["v"])
    desat = np.maximum(0.0, -z["s"])
    warmer = np.maximum(0.0, z["cr"])
    chroma_shift = np.abs(z["cr"]) + np.abs(z["cg"])

    wc = sc.get("water", {}) or {}
    mc = sc.get("mud", {}) or {}
    shc = sc.get("shadow", {}) or {}

    def thr(d: dict, key: str, default: float) -> float:
        return float(d.get(key, default)) * scale

    # Shadow: darker, with chromaticity and relative texture both intact — the
    # signature of less light rather than a different surface. Excluded from both
    # classes, otherwise every patch of shade becomes "mud" and the unassessable
    # fraction is meaningless. The smoothness clause keeps a *shadowed puddle*
    # detectable as water rather than being written off as shade.
    shadow = (
        (darker >= thr(shc, "min_darker_z", 1.0))
        & (smoother < float(shc.get("max_smoother_z", 0.6)))
        & (chroma_shift < float(shc.get("max_chroma_shift_z", 1.5)))
    )

    # Water: smoothness is mandatory, then any one supporting cue.
    water = (
        (smoother >= thr(wc, "min_smoother_z", 1.0))
        & (
            (brighter >= thr(wc, "min_brighter_z", 1.0))
            | (desat >= thr(wc, "min_desat_z", 1.0))
            | (darker >= thr(wc, "min_darker_z", 1.5))
        )
        & ~shadow
    )

    # Mud: a warm chroma shift is mandatory — that is what separates wet soil
    # from mere shade — plus darkness or smoothness as support.
    mud = (
        (warmer >= thr(mc, "min_warmer_z", 2.5))
        & (
            (darker >= thr(mc, "min_darker_z", 0.6))
            | (smoother >= thr(mc, "min_smoother_z", 0.5))
        )
        & ~shadow
    )

    # Where both fire, the stronger evidence wins so the classes stay exclusive.
    both = water & mud
    if both.any():
        water_strength = smoother + brighter + desat
        mud_strength = warmer + darker
        water = water & ~(both & (mud_strength > water_strength))
        mud = mud & ~(both & (water_strength >= mud_strength))
    return water, mud


def _clean(mask, road, min_area_frac: float, close_px: int):
    """Restrict to road, bridge gaps, drop speckle below a minimum area."""
    import cv2
    import numpy as np

    m = (mask & road).astype(np.uint8)
    if not m.any():
        return m.astype(bool)
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1,) * 2)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    road_area = float(road.sum())
    min_area = max(1.0, min_area_frac * road_area)
    n, labels, stats_cc, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m, dtype=bool)
    for i in range(1, n):
        if stats_cc[i, cv2.CC_STAT_AREA] >= min_area:
            out |= labels == i
    return out & road


def analyse_surface(frame, road_mask, cfg) -> SurfaceMap:
    """Classify the road surface in one frame into water / mud / dry."""
    import numpy as np

    sc = cfg.get_path("surface", {}) or {}
    road = road_mask.mask
    if not road.any():
        empty = np.zeros(frame.shape[:2], dtype=bool)
        return SurfaceMap(water=empty, mud=empty.copy(), dry=empty.copy(),
                          occlusion=empty.copy(), road_area_px=0.0)

    feats = compute_features(frame, int(sc.get("texture_ksize", 7)))
    stats, source = _baseline_stats(feats, road, road_mask.baseline)
    _warn_uniform_contamination(stats, cfg)
    _warn_featureless_baseline(stats)
    # Hysteresis: detect on a heavily smoothed map (noise-robust, but it erodes
    # the edges of small patches), then recover true extent by growing into a
    # lightly smoothed map at a relaxed threshold. Detection confidence and
    # measured area are different jobs, and one kernel cannot do both — and the
    # measured area is what the "% unassessable" headline depends on.
    strong_k = int(sc.get("region_ksize", 21))
    weak_k = int(sc.get("extent_ksize", 7))
    weak_scale = float(sc.get("extent_threshold_scale", 0.6))

    z_strong = _zmaps(feats, stats, strong_k)
    water_core, mud_core = _classify(z_strong, sc, 1.0)

    if weak_k != strong_k or weak_scale != 1.0:
        z_weak = _zmaps(feats, stats, weak_k)
        water_ext, mud_ext = _classify(z_weak, sc, weak_scale)
        # A permissive region survives only if it contains a confident core, so
        # the relaxed threshold cannot invent contamination on a clean road.
        water = keep_components_touching(water_ext | water_core, water_core)
        mud = keep_components_touching(mud_ext | mud_core, mud_core)
    else:
        water, mud = water_core, mud_core

    min_area_frac = float(sc.get("min_blob_area_frac", 0.002))
    close_px = int(sc.get("close_px", 3))
    water = _clean(water, road, min_area_frac, close_px)
    mud = _clean(mud, road, min_area_frac, close_px)
    mud &= ~water

    occlusion = water | mud
    return SurfaceMap(
        water=water, mud=mud, dry=road & ~occlusion, occlusion=occlusion,
        road_area_px=float(road.sum()), water_px=float(water.sum()),
        mud_px=float(mud.sum()), baseline_source=source,
    )
