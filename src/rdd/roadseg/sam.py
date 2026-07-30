"""Optional SAM backend — prompt a promptable segmenter with the geometric prior.

SAM is class-agnostic: it will not tell you "this is road", it segments whatever
you point at. That is enough here, because the viewpoint profile already tells us
*where* the road must be. We sample positive prompts inside the eroded prior and
negative prompts outside the dilated one, and let SAM find the actual boundary —
which it does far more accurately than colour similarity, especially where the
road meets a similarly-coloured verge.

The catch is cost. On CPU this is seconds per frame, not milliseconds, so it is
opt-in and best used on sampled frames (labeling, spot checks) rather than every
frame of a long survey. Any failure degrades to the classical backend.
"""
from __future__ import annotations

from ..utils.logging import get_logger
from .base import RoadBaseline, RoadMask
from .classical import ClassicalSegmenter
from .ops import (
    compute_features,
    dilate,
    erode,
    fill_holes,
    keep_largest_component,
    morph_clean,
    overlap_fraction,
    polygon_mask,
    robust_stats,
)

log = get_logger("rdd.roadseg.sam")

_BASELINE_CHANNELS = ("l", "a", "b", "s", "tex", "v")


def build_sam_segmenter(cfg, view):
    """Return a SamSegmenter, or the classical one if SAM cannot be loaded."""
    try:
        return SamSegmenter(cfg, view)
    except Exception as e:
        log.warning("SAM backend unavailable (%s) — using classical road segmentation", e)
        return ClassicalSegmenter(cfg, view)


def _sample_points(mask, n: int, axis: str = "vertical"):
    """Pick ~n points spread along the mask's long dimension."""
    import numpy as np

    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return []
    order = np.argsort(ys if axis == "vertical" else xs)
    ys, xs = ys[order], xs[order]
    idx = np.linspace(0, len(xs) - 1, num=min(n, len(xs))).astype(int)
    return [[int(xs[i]), int(ys[i])] for i in idx]


class SamSegmenter:
    def __init__(self, cfg, view):
        from ultralytics import SAM

        self.cfg = cfg
        self.view = view
        sc = cfg.get_path("roadseg.sam", {}) or {}
        self.checkpoint = sc.get("model", cfg.get_path("annotate.sam_model", "sam2.1_b.pt"))
        self.n_positive = int(sc.get("n_positive", 5))
        self.n_negative = int(sc.get("n_negative", 4))
        self.seed_erode_frac = float(sc.get("seed_erode_frac", 0.05))
        self.search_dilate_frac = float(sc.get("search_dilate_frac", 0.06))
        self.texture_ksize = int(sc.get("texture_ksize", 7))
        self.min_coverage = float(sc.get("min_coverage", 0.02))
        self.max_coverage = float(sc.get("max_coverage", 0.95))

        from ..utils.device import resolve_device

        self.device = resolve_device(cfg.get_path("run.device", "auto"))
        if self.device == "cpu":
            log.warning(
                "SAM road segmentation on CPU is very slow (seconds per frame). "
                "Consider roadseg.backend: classical for full-video runs."
            )
        self.model = SAM(self.checkpoint)
        self.name = f"sam({self.checkpoint})"
        self._fallback = ClassicalSegmenter(cfg, view)
        self._axis: str | None = None
        self._axis_locked = False
        self._fallback_frames = 0
        self._total_frames = 0

    def reset(self) -> None:
        self._axis = None
        self._axis_locked = False
        self._fallback_frames = 0
        self._total_frames = 0
        self._fallback.reset()

    @property
    def fallback_rate(self) -> float:
        if not self._total_frames:
            return 0.0
        return self._fallback_frames / float(self._total_frames)

    def segment(self, frame) -> RoadMask:
        import numpy as np

        self._total_frames += 1
        H, W = frame.shape[:2]

        if not self._axis_locked:
            # Reuse the classical backend's cheap axis estimate.
            probe = self._fallback.segment(frame)
            self._axis = probe.axis
            self._axis_locked = True

        prior = polygon_mask(self.view.road_prior.polygon(W, H, axis=self._axis), W, H)
        seed = erode(prior, int(round(self.seed_erode_frac * W)))
        if not seed.any():
            seed = prior
        search = dilate(prior, int(round(self.search_dilate_frac * W)))

        axis = self._axis or "vertical"
        pos = _sample_points(seed, self.n_positive, axis)
        neg = _sample_points(~search, self.n_negative, axis)
        if not pos:
            return self._degrade(frame, "no positive prompt points")

        points = pos + neg
        labels = [1] * len(pos) + [0] * len(neg)

        try:
            res = self.model(frame, points=[points], labels=[labels],
                             device=self.device, verbose=False)
        except Exception as e:
            return self._degrade(frame, f"SAM inference failed: {e}")

        mask = self._union_masks(res, H, W)
        if mask is None:
            return self._degrade(frame, "SAM returned no masks")

        mask &= search
        mask = morph_clean(mask, open_px=max(1, int(0.004 * W)),
                           close_px=max(1, int(0.012 * W)))
        mask = keep_largest_component(mask, seed)
        mask = fill_holes(mask)

        coverage = float(mask.sum()) / float(H * W)
        if not (self.min_coverage <= coverage <= self.max_coverage):
            return self._degrade(frame, f"implausible coverage {coverage:.1%}")

        retention = overlap_fraction(seed, mask)
        feats = compute_features(frame, self.texture_ksize)
        channels = feats.channels()
        stats = {}
        for ch in _BASELINE_CHANNELS:
            arr = channels.get(ch) if ch in channels else getattr(feats, ch)
            stats[ch] = robust_stats(arr[mask] if mask.any() else arr[seed])

        return RoadMask(
            mask=mask, prior=prior, backend=self.name,
            confidence=max(0.0, min(1.0, 0.6 + 0.4 * retention)),
            fell_back=False, baseline=RoadBaseline(stats=stats), axis=self._axis,
        )

    @staticmethod
    def _union_masks(res, H: int, W: int):
        import cv2
        import numpy as np

        if not res:
            return None
        r = res[0]
        if getattr(r, "masks", None) is None or r.masks.data is None:
            return None
        data = r.masks.data.cpu().numpy()
        if data.size == 0:
            return None
        acc = np.zeros((H, W), dtype=bool)
        for m in data:
            if m.shape != (H, W):
                m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            acc |= m.astype(bool)
        return acc

    def _degrade(self, frame, why: str) -> RoadMask:
        self._fallback_frames += 1
        if self._fallback_frames <= 3:
            log.warning("SAM road seg degraded to classical (%s)", why)
        rm = self._fallback.segment(frame)
        rm.fell_back = True
        rm.backend = f"{self.name}->classical"
        return rm
