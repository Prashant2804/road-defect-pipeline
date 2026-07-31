"""Deterministic rejection of the things that look like defects but aren't.

On dashcam footage most false positives are not subtle model failures — they are a
short list of recurring look-alikes. Each has a physical signature that separates it
from real damage without any training data, so rejecting them is the cheapest
precision available after the validity gate.

  shadow          darker, but chromaticity and brightness-relative texture intact
  wet patch       specular: texture collapses
  tar / sealant   dark AND smooth AND chromatically distinct from the surface
  road marking    bright, straight, elongated, aligned with the road axis
  manhole / joint | strongly circular, or a perfect full-width transverse line

The rules are intentionally conservative. Rejecting a genuine pothole to avoid a tar
patch is a bad trade, so each rule requires several cues to agree, and every rejection
is counted and reported by reason — a silent filter that quietly removes real defects
would be worse than no filter at all.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..utils.logging import get_logger

log = get_logger("rdd.detect.confusers")


@dataclass
class Rejection:
    confuser: str
    reason: str


@dataclass
class ConfuserStats:
    checked: int = 0
    rejected: int = 0
    by_confuser: dict = field(default_factory=dict)

    def update(self, rej: Rejection | None) -> None:
        self.checked += 1
        if rej is not None:
            self.rejected += 1
            self.by_confuser[rej.confuser] = self.by_confuser.get(rej.confuser, 0) + 1

    def summary(self) -> dict:
        return {"checked": self.checked, "rejected": self.rejected,
                "by_confuser": dict(sorted(self.by_confuser.items(),
                                           key=lambda kv: -kv[1]))}


def _zstats(feats, mask, baseline):
    """Mean z-score per channel inside the mask, against the road baseline."""
    import numpy as np

    if baseline is None or baseline.is_empty or not mask.any():
        return {}
    channels = feats.channels()
    out = {}
    for ch, (med, sigma) in baseline.stats.items():
        arr = channels.get(ch)
        if arr is None:
            continue
        vals = arr[mask]
        if vals.size == 0:
            continue
        out[ch] = float((np.mean(vals) - med) / max(sigma, 1e-6))
    return out


def _shape(mask):
    """(circularity, elongation, fill) descriptors of the mask's largest contour."""
    import cv2
    import numpy as np

    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 1.0, 0.0
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    perim = float(cv2.arcLength(c, True))
    if area <= 1.0 or perim <= 1e-6:
        return 0.0, 1.0, 0.0

    circularity = 4.0 * math.pi * area / (perim * perim)
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    long_side, short_side = max(rw, rh), max(min(rw, rh), 1e-6)
    elongation = long_side / short_side
    fill = area / max(long_side * short_side, 1e-6)
    return circularity, elongation, fill


def check(mask, feats, baseline, cfg, cls_name: str = "",
          shadow_mask=None, angle_deg: float | None = None) -> Rejection | None:
    """Return a Rejection if this detection matches a known look-alike."""
    import numpy as np

    cc = cfg.get_path("detect.confusers", {}) or {}
    if not cc.get("enabled", True) or not mask.any():
        return None
    if cls_name in set(cc.get("exempt_classes") or ()):
        return None

    # Shadow: reuse the surface stage's shadow mask, which is already built from
    # illumination invariants (chromaticity and relative texture both preserved).
    if shadow_mask is not None and shadow_mask.any():
        overlap = float((mask & shadow_mask).sum()) / float(mask.sum())
        if overlap >= float(cc.get("shadow_overlap", 0.7)):
            return Rejection("shadow",
                             f"{overlap:.0%} of the detection lies in shade with its "
                             f"texture and chromaticity intact")

    z = _zstats(feats, mask, baseline)
    if not z:
        return None
    darker = -z.get("l", 0.0)
    smoother = -z.get("rtex", 0.0)
    brighter = z.get("v", 0.0)
    chroma = abs(z.get("cr", 0.0)) + abs(z.get("cg", 0.0))
    circularity, elongation, fill = _shape(mask)

    # Tar patch / sealed crack: as dark as a pothole, but smooth and a different
    # material. A real pothole is a broken surface, so it stays rough.
    if (darker >= float(cc.get("tar_min_darker_z", 1.2))
            and smoother >= float(cc.get("tar_min_smoother_z", 1.5))
            and chroma >= float(cc.get("tar_min_chroma_z", 1.0))):
        return Rejection("tar_patch",
                         f"dark but smooth and chromatically distinct "
                         f"(darker {darker:.1f}σ, smoother {smoother:.1f}σ) — "
                         f"bitumen patch or sealed crack, not broken surface")

    # Road marking: bright, straight, thin, and running with the road.
    if (brighter >= float(cc.get("marking_min_brighter_z", 1.5))
            and elongation >= float(cc.get("marking_min_elongation", 6.0))
            and fill >= float(cc.get("marking_min_fill", 0.55))
            and (angle_deg is None
                 or angle_deg <= float(cc.get("marking_max_angle_deg", 25.0)))):
        return Rejection("road_marking",
                         f"bright straight line ({elongation:.0f}:1, "
                         f"{brighter:.1f}σ brighter) aligned with the road")

    # Manhole cover / utility fitting: near-circular and unusually regular.
    if (circularity >= float(cc.get("manhole_min_circularity", 0.80))
            and smoother >= float(cc.get("manhole_min_smoother_z", 0.8))):
        return Rejection("manhole",
                         f"near-circular ({circularity:.2f}) and smooth — a cover "
                         f"or fitting rather than a pothole")

    # Expansion / construction joint: a ruler-straight full-width transverse line.
    if (angle_deg is not None and angle_deg >= float(cc.get("joint_min_angle_deg", 75.0))
            and elongation >= float(cc.get("joint_min_elongation", 12.0))
            and fill >= float(cc.get("joint_min_fill", 0.7))
            and abs(darker) < float(cc.get("joint_max_darker_z", 3.0))):
        return Rejection("joint",
                         f"ruler-straight transverse line ({elongation:.0f}:1 at "
                         f"{angle_deg:.0f}°) — construction or expansion joint")
    return None
