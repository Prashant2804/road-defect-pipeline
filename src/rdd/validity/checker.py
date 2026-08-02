"""Orchestrates the validity gates into one per-frame verdict.

Owns the stateful detectors (ego-motion, traffic, static structure), runs every gate,
merges MASK regions, and accumulates route-level assessability.
"""
from __future__ import annotations

from ..utils.logging import get_logger
from .egomotion import EgoMotionEstimator
from .gates import ALL_GATES, FrameContext
from .traffic import TrafficDetector
from .verdict import Action, FrameVerdict, ValidityStats

log = get_logger("rdd.validity")


class StaticStructureDetector:
    """Finds image regions that never change — i.e. are attached to the camera.

    Windscreen dirt, rain spots, a smeared wiper arc and the vehicle's own bonnet
    are all perfectly stable false-positive generators. No confidence threshold
    removes them, because they look the same in every frame and a tracker will
    happily follow one for the entire clip as a single confident "defect".

    The discriminator is temporal variance under motion: real scene content streams
    past, so its pixels change constantly; anything fixed to the glass does not.
    Accumulation only runs while the vehicle is actually moving — sitting at a
    junction makes the whole frame static and would otherwise mark the entire image
    as dirt.
    """

    def __init__(self, cfg):
        wc = cfg.get_path("validity.windscreen", {}) or {}
        self.enabled = bool(wc.get("enabled", True))
        self.min_frames = int(wc.get("min_frames", 30))
        self.std_threshold = float(wc.get("max_temporal_std", 5.0))
        self.alpha = float(wc.get("alpha", 0.08))
        self.work_width = int(wc.get("work_width", 320))
        self._mean = None
        self._var = None
        self._n = 0

    def reset(self) -> None:
        self._mean = None
        self._var = None
        self._n = 0

    def update(self, frame, moving: bool, horizon_row: float | None = None):
        """Accumulate, and return the static mask once enough motion has been seen."""
        import cv2
        import numpy as np

        if not self.enabled:
            return None

        h, w = frame.shape[:2]
        scale = min(1.0, self.work_width / float(w))
        sw, sh = max(8, int(w * scale)), max(8, int(h * scale))
        small = cv2.cvtColor(
            cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        ).astype(np.float32)

        if self._mean is None or self._mean.shape != small.shape:
            self._mean = small.copy()
            self._var = np.zeros_like(small)
            self._n = 0
            return None

        if moving:
            # Exponential mean/variance: cheap, bounded memory, and naturally
            # forgets the distant past as conditions change.
            delta = small - self._mean
            self._mean += self.alpha * delta
            self._var = (1 - self.alpha) * self._var + self.alpha * (delta * delta)
            self._n += 1

        if self._n < self.min_frames:
            return None

        static = np.sqrt(np.maximum(self._var, 0.0)) < self.std_threshold
        static = cv2.morphologyEx(
            static.astype(np.uint8), cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        ).astype(bool)

        full = cv2.resize(static.astype(np.uint8), (w, h),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        if horizon_row is not None:
            # Sky is genuinely low-variance and is not windscreen dirt. It also sits
            # outside every assessment zone, so counting it would only inflate the
            # "static fraction" and block good frames.
            cut = max(0, min(h - 1, int(horizon_row)))
            full[:cut, :] = False
        return full if full.any() else None


class ValidityChecker:
    """Per-frame assessability decisions plus route-level coverage."""

    def __init__(self, cfg, camera=None, zones=None,
                 vanishing_point: tuple[float, float] | None = None):
        self.cfg = cfg
        self.camera = camera
        self.zones = zones
        self.enabled = bool(cfg.get_path("validity.enabled", True))

        vp = vanishing_point
        if vp is None and camera is not None:
            vp = camera.vanishing_point()
        self.ego = EgoMotionEstimator(cfg, vanishing_point=vp)
        self.traffic = TrafficDetector(cfg)
        self.static = StaticStructureDetector(cfg)

        self.traffic_stride = max(1, int(cfg.get_path("validity.traffic.stride", 3)))
        self.stats = ValidityStats()
        self._last_traffic = None
        self._zone_mask = None

        if not self.enabled:
            log.warning(
                "validity.enabled is false — frames will be assessed even when the "
                "road is buried, unlocatable, or the vehicle is off the carriageway. "
                "Expect false positives and an unquantified coverage claim."
            )

    def reset(self) -> None:
        self.ego.reset()
        self.static.reset()
        self.stats = ValidityStats()
        self._last_traffic = None
        self._zone_mask = None

    def zone_mask(self, width: int, height: int):
        """Union assessment zone — the only region worth detecting in."""
        import numpy as np

        if self._zone_mask is not None and self._zone_mask.shape == (height, width):
            return self._zone_mask
        if self.zones is None or self.camera is None:
            self._zone_mask = np.ones((height, width), dtype=bool)
            return self._zone_mask

        union = self.zones.widest()
        if union is None:
            self._zone_mask = np.zeros((height, width), dtype=bool)
            return self._zone_mask

        _, z_map, _ = self.camera.ground_maps(width, height)
        with np.errstate(invalid="ignore"):
            m = (z_map >= union.z_near_m) & (z_map <= union.z_far_m)
        self._zone_mask = np.nan_to_num(m, nan=False).astype(bool)
        return self._zone_mask

    def check(self, frame_idx: int, t: float, frame, road=None, surface=None,
              quality=None, distance_m: float = 0.0, gap: int = 1) -> FrameVerdict:
        """Run every gate and combine the results."""
        import numpy as np

        verdict = FrameVerdict(frame=frame_idx, t=t)
        if not self.enabled:
            self.stats.update(verdict, distance_m)
            return verdict

        h, w = frame.shape[:2]
        zone = self.zone_mask(w, h)

        ego = self.ego.update(frame, gap=gap)
        moving = bool(getattr(ego, "valid", False)) and ego.flow_px > 0.6
        horizon = self.camera.horizon_row if self.camera is not None else None
        static_mask = self.static.update(frame, moving, horizon)

        # Traffic detection is the expensive gate; stride it and reuse. Vehicles do
        # not appear and vanish between adjacent frames.
        if frame_idx % self.traffic_stride == 0 or self._last_traffic is None:
            self._last_traffic = self.traffic.detect(frame, zone)
        traffic = self._last_traffic

        ctx = FrameContext(
            frame_idx=frame_idx, t=t, frame=frame, road=road, surface=surface,
            quality=quality, zone_mask=zone, ego=ego, traffic=traffic,
            static_mask=static_mask, camera=self.camera,
        )

        for gate in ALL_GATES:
            try:
                res = gate(ctx, self.cfg)
            except Exception as e:      # a gate must never break the pipeline
                log.warning("Validity gate %s failed on frame %d (%s) — skipping",
                            getattr(gate, "__name__", gate), frame_idx, e)
                continue
            if res is not None:
                verdict.results.append(res)

        masks = [r.mask for r in verdict.results
                 if r.action is Action.MASK and r.mask is not None]
        if masks:
            acc = masks[0].copy()
            for m in masks[1:]:
                acc |= m
            verdict.exclude_mask = acc

        self.stats.update(verdict, distance_m)
        if verdict.blocked and self.stats.frames <= 2000:
            log.debug("frame %d not assessed: %s", frame_idx,
                      "; ".join(verdict.block_reasons))
        return verdict

    def log_summary(self) -> None:
        s = self.stats.summary()
        log.info("Route assessability: %.1f%% of frames (%d/%d)",
                 100 * s["frame_coverage"], s["frames_assessable"], s["frames"])
        if s["distance_total_m"] > 0:
            log.info("  by distance: %.1f%% of %.0f m",
                     100 * s["distance_coverage"], s["distance_total_m"])
        if s["blocked_by_gate"]:
            log.info("  excluded by: %s", ", ".join(
                f"{k} x{v}" for k, v in s["blocked_by_gate"].items()))
        if s["degraded_by_gate"]:
            log.info("  degraded by: %s", ", ".join(
                f"{k} x{v}" for k, v in s["degraded_by_gate"].items()))
        run = s["longest_unassessed_run_frames"]
        if run >= 30:
            log.warning("  longest unbroken unassessed stretch: %d frames — a "
                        "section of this route has no coverage at all", run)
