"""Ego-motion from sparse optical flow: is the vehicle moving, turning, or shaking?

Cheap and label-free, using the fact that a forward-facing camera on a moving
vehicle produces a very characteristic flow field: features stream *outward* from
the vanishing point. That single property answers several questions at once.

  * **Stationary** — near-zero flow. Frames at a junction or in traffic are
    near-duplicates of each other; assessing them re-detects the same metre of road
    hundreds of times, which corrupts any per-distance statistic and wastes compute.
  * **Reversing** — flow contracts *toward* the vanishing point instead of expanding
    away from it. Reversing footage breaks distance-based sampling and re-surveys
    road already covered, so it should not be assessed.
  * **Turning** — a strong horizontal flow component near the horizon. Sharp turns
    are where the road mask lags most and where motion blur is worst.
  * **Vibration** — vertical flow scatter. On rough unpaved roads this is the main
    source of motion blur, and it also invalidates the assumed camera pitch, which
    every metric measurement depends on.

The radial expansion test is the important one: it distinguishes forward from
reverse motion, which flow *magnitude* alone cannot do.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..utils.logging import get_logger

log = get_logger("rdd.validity.egomotion")


@dataclass
class EgoMotion:
    """Per-frame motion state estimated from the previous frame."""

    valid: bool = False
    flow_px: float = 0.0     # median flow magnitude, pixels/frame
    radial: float = 0.0      # >0 expanding (forward), <0 contracting (reverse)
    yaw_px: float = 0.0      # horizontal drift near the horizon (turn proxy)
    pitch_px: float = 0.0    # vertical drift near the horizon (bounce/vibration proxy)
    n_tracks: int = 0

    @property
    def forward(self) -> bool:
        return self.radial > 0

    def as_dict(self) -> dict:
        return {
            "valid": self.valid,
            "flow_px": round(self.flow_px, 3),
            "radial": round(self.radial, 3),
            "yaw_px": round(self.yaw_px, 3),
            "pitch_px": round(self.pitch_px, 3),
            "n_tracks": self.n_tracks,
        }


class EgoMotionEstimator:
    """Sparse Lucas-Kanade tracker, re-seeded when tracks run out.

    Sparse rather than dense: a few hundred corners give a robust median and cost a
    fraction of Farneback, and this runs on every frame.
    """

    def __init__(self, cfg, vanishing_point: tuple[float, float] | None = None):
        ec = cfg.get_path("validity.egomotion", {}) or {}
        self.max_corners = int(ec.get("max_corners", 300))
        self.quality_level = float(ec.get("quality_level", 0.01))
        self.min_distance = int(ec.get("min_distance", 12))
        self.work_width = int(ec.get("work_width", 480))
        self.vp = vanishing_point
        self._prev = None
        self._prev_pts = None
        self._scale = 1.0

    def reset(self) -> None:
        self._prev = None
        self._prev_pts = None

    def _seed(self, gray):
        import cv2

        return cv2.goodFeaturesToTrack(
            gray, maxCorners=self.max_corners, qualityLevel=self.quality_level,
            minDistance=self.min_distance, blockSize=7,
        )

    def update(self, frame) -> EgoMotion:
        import cv2
        import numpy as np

        h, w = frame.shape[:2]
        self._scale = min(1.0, self.work_width / float(w))
        if self._scale < 1.0:
            small = cv2.resize(frame, (int(w * self._scale), int(h * self._scale)),
                               interpolation=cv2.INTER_AREA)
        else:
            small = frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if self._prev is None or self._prev_pts is None or len(self._prev_pts) < 12:
            self._prev, self._prev_pts = gray, self._seed(gray)
            return EgoMotion(valid=False)

        nxt, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev, gray, self._prev_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if nxt is None or status is None:
            self._prev, self._prev_pts = gray, self._seed(gray)
            return EgoMotion(valid=False)

        ok = status.reshape(-1) == 1
        p0 = self._prev_pts.reshape(-1, 2)[ok]
        p1 = nxt.reshape(-1, 2)[ok]
        self._prev = gray
        # Re-seed when tracks thin out, otherwise keep them for temporal stability.
        self._prev_pts = p1.reshape(-1, 1, 2) if len(p1) >= 40 else self._seed(gray)

        if len(p0) < 12:
            return EgoMotion(valid=False, n_tracks=len(p0))

        d = p1 - p0
        mag = np.linalg.norm(d, axis=1)
        flow_px = float(np.median(mag)) / self._scale

        # Radial component about the vanishing point: forward motion expands the
        # field outward, reverse contracts it. Magnitude alone cannot tell them apart.
        if self.vp is not None:
            cx, cy = self.vp[0] * self._scale, self.vp[1] * self._scale
        else:
            cx, cy = small.shape[1] / 2.0, small.shape[0] * 0.45
        r = p0 - np.array([cx, cy], dtype=np.float32)
        rn = np.linalg.norm(r, axis=1)
        keep = rn > 5.0
        if keep.sum() >= 8:
            radial = float(np.median(np.sum(d[keep] * (r[keep] / rn[keep, None]), axis=1)))
        else:
            radial = 0.0

        # Rotation proxies, both measured on features near the horizon. Distant
        # features barely move under pure translation, so whatever motion they do
        # show is dominated by camera *rotation* — yaw for horizontal, pitch for
        # vertical.
        #
        # Measuring vibration as the spatial spread of vertical flow across the whole
        # frame does not work: under a perfectly smooth ride, forward motion makes
        # vertical flow large at the bottom of the frame and near zero at the horizon,
        # so the spread is always large. That conflates driving forwards with hitting
        # a pothole, and flagged every frame of smooth synthetic footage as shaky.
        upper = p0[:, 1] < small.shape[0] * 0.55
        if upper.sum() >= 8:
            yaw_px = float(np.median(d[upper, 0]))
            pitch_px = float(np.median(d[upper, 1]))
        else:
            yaw_px = pitch_px = 0.0

        return EgoMotion(
            valid=True,
            flow_px=flow_px,
            radial=radial / self._scale,
            yaw_px=yaw_px / self._scale,
            pitch_px=pitch_px / self._scale,
            n_tracks=int(len(p0)),
        )
