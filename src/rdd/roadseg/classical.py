"""Classical road-surface segmentation: colour + texture similarity, seeded by geometry.

Chosen as the default because it runs comfortably in real time on CPU and needs
no extra dependencies, while handling the thing that defeats naive approaches on
unpaved rural roads: there is no fixed "road colour". Laterite, gravel, dried
mud and dust all differ, and all change with light. So nothing is compared
against an absolute threshold. Instead:

  1. Take a small, high-confidence region where the road almost certainly is
     (an eroded geometric prior) and measure what the road looks like *here*.
  2. Grow to every nearby pixel that resembles that measurement.
  3. Fill enclosed holes, because the potholes, puddles and mud patches are
     precisely the pixels that failed step 2 — and they belong to the road.

Statistics are median/MAD rather than mean/std throughout: the seed region
deliberately contains the outliers we are hunting for, and a mean would be
dragged toward them.

Limits worth knowing: it assumes the road is one contiguous surface that is
locally more uniform than its surroundings. Deep shade across half the road, or
a verge made of the same material as the road, will confuse it. When the result
is implausible it says so (`fell_back=True`) rather than returning nonsense.
"""
from __future__ import annotations

from ..utils.logging import get_logger
from .base import RoadBaseline, RoadMask
from .geometric import GeometricSegmenter
from .ops import (
    channel_stats,
    compute_features,
    dilate,
    erode,
    fill_holes,
    keep_largest_component,
    morph_clean,
    overlap_fraction,
    polygon_mask,
    resize_mask,
)

log = get_logger("rdd.roadseg.classical")

# rtex/cr/cg are measured and stored for the surface stage but carry no weight in
# the segmentation distance (default weight 0), so they do not alter this stage.
_BASELINE_CHANNELS = ("l", "a", "b", "s", "tex", "rtex", "cr", "cg", "v")
_DEFAULT_WEIGHTS = {"l": 1.0, "a": 1.0, "b": 1.0, "s": 0.7, "tex": 1.3}


class ClassicalSegmenter:
    name = "classical(colour+texture)"

    def __init__(self, cfg, view):
        self.cfg = cfg
        self.view = view
        rc = cfg.get_path("roadseg.classical", {}) or {}

        self.work_width = max(160, int(rc.get("work_width", 480)))
        self.tau = float(rc.get("distance_tau", 2.5))
        self.weights = {**_DEFAULT_WEIGHTS, **dict(rc.get("weights", {}) or {})}
        self.texture_ksize = int(rc.get("texture_ksize", 7))
        # Slow: the road material does not change within a clip, so the baseline
        # should resist a transient puddle rather than chase it.
        self.baseline_alpha = min(1.0, max(0.01, float(rc.get("baseline_alpha", 0.12))))
        self.seed_erode_frac = float(rc.get("seed_erode_frac", 0.025))
        self.search_dilate_frac = float(rc.get("search_dilate_frac", 0.06))
        self.open_frac = float(rc.get("open_frac", 0.006))
        self.close_frac = float(rc.get("close_frac", 0.02))
        self.do_fill_holes = bool(rc.get("fill_holes", True))
        self.min_coverage = float(rc.get("min_coverage", 0.02))
        self.max_coverage = float(rc.get("max_coverage", 0.95))
        self.min_seed_retention = float(rc.get("min_seed_retention", 0.25))
        self.max_search_fill = float(rc.get("max_search_fill", 0.98))

        self._fallback = GeometricSegmenter(cfg, view)
        self._axis: str | None = None
        self._axis_locked = False
        self._baseline: dict | None = None
        self._fallback_frames = 0
        self._total_frames = 0

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        self._axis = None
        self._axis_locked = False
        self._baseline = None
        self._fallback_frames = 0
        self._total_frames = 0

    @property
    def fallback_rate(self) -> float:
        if not self._total_frames:
            return 0.0
        return self._fallback_frames / float(self._total_frames)

    # -- geometry helpers --------------------------------------------------
    def _priors(self, w: int, h: int, axis: str | None):
        """(prior, seed, search) masks at working resolution."""
        prior = polygon_mask(self.view.road_prior.polygon(w, h, axis=axis), w, h)
        seed = erode(prior, int(round(self.seed_erode_frac * w)))
        if not seed.any():
            seed = prior
        search = dilate(prior, int(round(self.search_dilate_frac * w)))
        return prior, seed, search

    def _resolve_axis(self, small, feats) -> str | None:
        """For a nadir band prior with axis 'auto', decide which way the road runs.

        The obvious score — "which band is most filled by seed-like pixels" —
        does not work, and the reason is worth recording: the appearance model is
        *learned from the seed*, so a band laid across vegetation faithfully
        matches vegetation and scores just as well as the true road band.

        What actually distinguishes them is texture. A road surface is smoother
        than the vegetation, gravel shoulders and crop rows around it, so the
        correct axis is the one whose seed has the lower texture energy. Band fill
        is kept only as a tie-breaker. Resolved once per clip; a flight line does
        not change mid-shot.
        """
        import numpy as np

        prior_cfg = self.view.road_prior
        if prior_cfg.kind != "band":
            return None
        if prior_cfg.axis != "auto":
            return prior_cfg.axis

        h, w = small.shape[:2]
        scored = []
        for axis in ("vertical", "horizontal"):
            prior, seed, search = self._priors(w, h, axis)
            tex_med = float(np.median(feats.tex[seed])) if seed.any() else 1e9
            cand = self._similar(feats, seed, search)[0]
            fill = overlap_fraction(prior, cand)
            scored.append((-tex_med, fill, axis))
            log.debug("axis %s: seed texture %.2f, band fill %.3f", axis, tex_med, fill)

        scored.sort(reverse=True)
        best = scored[0][2]
        log.info("Drone band axis resolved to '%s' (seed texture %.2f, fill %.2f)",
                 best, -scored[0][0], scored[0][1])
        return best

    # -- appearance model --------------------------------------------------
    @staticmethod
    def _frame_stats(feats, seed) -> dict:
        return channel_stats(feats, seed, _BASELINE_CHANNELS)

    def _blend_baseline(self, frame_stats: dict) -> dict:
        """Blend this frame's seed statistics into a running clip baseline.

        Necessary because the seed is a *geometric* guess at where road is, and it
        can be dominated by the very things we are looking for. A puddle filling
        the near field — common, since that is where water collects and where the
        camera looks — makes the seed median describe water, after which the dry
        road becomes the outlier and gets flagged as contamination. Median/MAD
        tolerates outliers up to half the sample; this blows straight past that.

        Road *material* is a property of the clip, not of one frame, while
        contamination is transient and moves through. A slow EMA therefore
        converges on the actual surface and dilutes any single frame's
        contamination. Lighting drifts slowly enough for the EMA to follow it.
        """
        if self._baseline is None:
            self._baseline = dict(frame_stats)
            return self._baseline
        a = self.baseline_alpha
        self._baseline = {
            ch: (a * med + (1 - a) * self._baseline[ch][0],
                 a * sigma + (1 - a) * self._baseline[ch][1])
            for ch, (med, sigma) in frame_stats.items()
            if ch in self._baseline
        }
        return self._baseline

    def _mask_from(self, feats, stats: dict, search):
        """Pixels resembling the given appearance baseline."""
        import numpy as np

        channels = feats.channels()
        num = None
        den = 0.0
        for ch, (med, sigma) in stats.items():
            weight = float(self.weights.get(ch, 0.0))
            arr = channels.get(ch)
            if weight <= 0 or arr is None:
                continue
            z = (arr - med) / max(sigma, 1e-6)
            term = weight * (z * z)
            num = term if num is None else num + term
            den += weight

        if num is None or den <= 0:
            return search.copy()
        dist = np.sqrt(num / den)
        return (dist < self.tau) & search

    def _similar(self, feats, seed, search):
        """Single-frame convenience: (mask, frame_stats). Does not touch the EMA."""
        stats = self._frame_stats(feats, seed)
        return self._mask_from(feats, stats, search), stats

    # -- main --------------------------------------------------------------
    def segment(self, frame) -> RoadMask:
        import cv2

        H, W = frame.shape[:2]
        self._total_frames += 1

        scale = min(1.0, self.work_width / float(W))
        if scale < 1.0:
            sw, sh = max(1, int(round(W * scale))), max(1, int(round(H * scale)))
            small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        else:
            small = frame
        sh, sw = small.shape[:2]

        feats = compute_features(small, self.texture_ksize)

        if not self._axis_locked:
            self._axis = self._resolve_axis(small, feats)
            self._axis_locked = True
            self._fallback._axis = self._axis

        prior, seed, search = self._priors(sw, sh, self._axis)
        if not seed.any():
            return self._degenerate(frame, "empty geometric seed")

        # Measure this frame, blend it into the clip baseline, then segment against
        # the baseline rather than the frame — see _blend_baseline for why.
        stats = self._blend_baseline(self._frame_stats(feats, seed))
        cand = self._mask_from(feats, stats, search)

        # If the similarity test accepted essentially everything it was allowed
        # to accept, it discriminated nothing — the result is the search region,
        # i.e. pure geometry wearing a measurement's confidence. Say so instead.
        search_fill = overlap_fraction(search, cand)
        if search_fill >= self.max_search_fill:
            return self._degenerate(
                frame, f"appearance model separated nothing "
                       f"({search_fill:.0%} of the search region accepted)",
                prior_small=prior)

        cand = morph_clean(
            cand,
            open_px=int(round(self.open_frac * sw)),
            close_px=int(round(self.close_frac * sw)),
        )
        cand = keep_largest_component(cand, seed)
        if self.do_fill_holes:
            cand = fill_holes(cand)

        coverage = float(cand.sum()) / float(sw * sh)
        retention = overlap_fraction(seed, cand)

        if not (self.min_coverage <= coverage <= self.max_coverage):
            return self._degenerate(
                frame, f"implausible coverage {coverage:.1%}", prior_small=prior)
        if retention < self.min_seed_retention:
            return self._degenerate(
                frame, f"seed retention {retention:.1%} too low", prior_small=prior)

        mask = resize_mask(cand, W, H)
        prior_full = resize_mask(prior, W, H)
        # Confidence blends "did we keep the region we were sure about" with a
        # penalty for coverage near the plausibility limits.
        margin = min(coverage - self.min_coverage, self.max_coverage - coverage)
        span = max(self.max_coverage - self.min_coverage, 1e-6)
        confidence = max(0.0, min(1.0, retention * (0.5 + 0.5 * min(1.0, margin / (0.25 * span)))))

        return RoadMask(
            mask=mask, prior=prior_full, backend=self.name,
            confidence=confidence, fell_back=False,
            baseline=RoadBaseline(stats={k: stats[k] for k in _BASELINE_CHANNELS}),
            axis=self._axis,
        )

    def _degenerate(self, frame, why: str, prior_small=None) -> RoadMask:
        """Fall back to the geometric prior for this frame and say so."""
        self._fallback_frames += 1
        if self._fallback_frames <= 3 or self._fallback_frames % 50 == 0:
            log.warning("Classical road seg fell back to prior (%s) [%d/%d frames]",
                        why, self._fallback_frames, self._total_frames)
        rm = self._fallback.segment(frame)
        rm.fell_back = True
        rm.backend = f"{self.name}->prior"
        rm.axis = self._axis
        return rm
