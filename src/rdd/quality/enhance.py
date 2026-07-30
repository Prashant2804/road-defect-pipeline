"""Frame enhancement — one deterministic function, shared by labeling and inference.

This module exists to solve a specific trap. If you enhance frames before
labeling but not before inference (or tune the settings between the two), the
model is served a different image distribution than it learnt, and accuracy
drops for reasons that look like a modelling problem. So: enhancement is a pure
function of `(frame, EnhanceSpec)`, the spec is resolved once per run, and its
fingerprint goes into the manifest. A mismatch becomes visible instead of
mysterious.

Operation order is deliberate:

  white balance -> denoise -> CLAHE -> upscale -> unsharp

Denoise precedes CLAHE because local contrast enhancement amplifies whatever
noise it is given. Sharpening comes last so it also recovers the softness that
any resampling introduces.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from ..utils.logging import get_logger

log = get_logger("rdd.quality.enhance")

_DENOISE_MODES = ("none", "bilateral", "nlmeans")


@dataclass(frozen=True)
class EnhanceSpec:
    """Resolved, immutable enhancement settings."""

    enabled: bool = True
    white_balance: bool = False
    denoise: str = "none"
    denoise_strength: float = 0.4       # 0..1, scaled into filter params
    clahe_clip: float = 2.0             # 0 disables CLAHE
    clahe_grid: int = 8
    unsharp_amount: float = 0.6         # 0 disables sharpening
    unsharp_sigma: float = 1.2
    unsharp_threshold: int = 3          # ignore detail below this (noise guard)
    min_width: int = 0                  # upscale if narrower; 0 = never
    max_upscale: float = 2.0

    def fingerprint(self) -> str:
        """Short stable hash — recorded in the manifest to catch train/infer drift."""
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]

    def describe(self) -> str:
        if not self.enabled:
            return "disabled"
        parts = []
        if self.white_balance:
            parts.append("grey-world WB")
        if self.denoise != "none":
            parts.append(f"{self.denoise} denoise @{self.denoise_strength:.2f}")
        if self.clahe_clip > 0:
            parts.append(f"CLAHE clip={self.clahe_clip:.2f} grid={self.clahe_grid}")
        if self.min_width:
            parts.append(f"upscale to >={self.min_width}px")
        if self.unsharp_amount > 0:
            parts.append(f"unsharp {self.unsharp_amount:.2f}/sigma{self.unsharp_sigma:.2f}")
        return ", ".join(parts) or "no-op"


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def resolve_spec(cfg, profile=None) -> EnhanceSpec:
    """Build the spec from config, optionally adapted to a measured QualityProfile.

    Adaptation is bounded: config values are the baseline and the measured clip
    can only push them within a fixed multiple, so a pathological clip cannot
    produce absurd settings.
    """
    ec = cfg.get_path("quality.enhance", {}) or {}
    enabled = bool(ec.get("enabled", True))

    denoise = str(ec.get("denoise", "none"))
    if denoise not in _DENOISE_MODES:
        log.warning("Unknown quality.enhance.denoise %r — using 'none'", denoise)
        denoise = "none"

    spec = EnhanceSpec(
        enabled=enabled,
        white_balance=bool(ec.get("white_balance", False)),
        denoise=denoise,
        denoise_strength=float(ec.get("denoise_strength", 0.4)),
        clahe_clip=float(ec.get("clahe_clip", 2.0)),
        clahe_grid=int(ec.get("clahe_grid", 8)),
        unsharp_amount=float(ec.get("unsharp_amount", 0.6)),
        unsharp_sigma=float(ec.get("unsharp_sigma", 1.2)),
        unsharp_threshold=int(ec.get("unsharp_threshold", 3)),
        min_width=int(ec.get("min_width", 0)),
        max_upscale=float(ec.get("max_upscale", 2.0)),
    )

    if not (enabled and bool(ec.get("adaptive", True)) and profile is not None):
        return spec

    clip = spec.clahe_clip
    if spec.clahe_clip > 0 and profile.contrast_median > 0:
        # Low-contrast footage (haze, overcast, dust) needs a stronger clip limit.
        boost = _clamp(0.10 / profile.contrast_median, 1.0, 2.5)
        clip = _clamp(spec.clahe_clip * boost, spec.clahe_clip, 2.5 * spec.clahe_clip)

    dn, dn_strength = spec.denoise, spec.denoise_strength
    amount = spec.unsharp_amount
    if profile.noise_median > 3.0:
        # Noisy source: denoise even if config said none, and sharpen less so we
        # do not re-amplify what we just removed.
        if dn == "none":
            dn = "bilateral"
        dn_strength = _clamp(dn_strength * (profile.noise_median / 3.0), dn_strength, 1.0)
        amount = _clamp(amount * 0.6, 0.0, amount)

    adapted = EnhanceSpec(
        **{**asdict(spec), "clahe_clip": round(clip, 4), "denoise": dn,
           "denoise_strength": round(dn_strength, 4),
           "unsharp_amount": round(amount, 4)}
    )
    if adapted != spec:
        log.info("Enhancement adapted to clip: %s", adapted.describe())
    return adapted


def _grey_world(frame):
    import numpy as np

    means = frame.reshape(-1, 3).mean(axis=0)
    if float(means.min()) <= 1e-6:
        return frame
    gain = means.mean() / means
    gain = np.clip(gain, 0.5, 2.0)          # never invent extreme colour shifts
    return np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def _denoise(frame, mode: str, strength: float):
    import cv2

    s = _clamp(strength, 0.0, 1.0)
    if s <= 0:
        return frame
    if mode == "bilateral":
        # Edge-preserving: keeps crack boundaries while flattening sensor grain.
        return cv2.bilateralFilter(frame, d=5, sigmaColor=int(20 + 60 * s),
                                   sigmaSpace=int(20 + 60 * s))
    if mode == "nlmeans":
        h = float(3 + 7 * s)
        return cv2.fastNlMeansDenoisingColored(frame, None, h, h, 7, 21)
    return frame


def _clahe(frame, clip: float, grid: int):
    import cv2

    if clip <= 0:
        return frame
    grid = max(1, int(grid))
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def _upscale(frame, min_width: int, max_upscale: float):
    import cv2

    h, w = frame.shape[:2]
    if min_width <= 0 or w >= min_width:
        return frame
    factor = min(min_width / float(w), max(1.0, max_upscale))
    if factor <= 1.0:
        return frame
    return cv2.resize(frame, (int(round(w * factor)), int(round(h * factor))),
                      interpolation=cv2.INTER_LANCZOS4)


def _unsharp(frame, amount: float, sigma: float, threshold: int):
    import cv2
    import numpy as np

    if amount <= 0 or sigma <= 0:
        return frame
    blur = cv2.GaussianBlur(frame, (0, 0), float(sigma))
    detail = frame.astype(np.int16) - blur.astype(np.int16)
    if threshold > 0:
        # Only sharpen genuine structure; leave flat, noisy areas alone.
        detail = np.where(np.abs(detail) >= int(threshold), detail, 0)
    return np.clip(frame.astype(np.int16) + (amount * detail).astype(np.int16),
                   0, 255).astype(np.uint8)


def enhance_frame(frame, spec: EnhanceSpec):
    """Apply `spec` to one BGR frame. Pure; returns a new array."""
    if not spec.enabled:
        return frame
    out = frame
    if spec.white_balance:
        out = _grey_world(out)
    out = _denoise(out, spec.denoise, spec.denoise_strength)
    out = _clahe(out, spec.clahe_clip, spec.clahe_grid)
    out = _upscale(out, spec.min_width, spec.max_upscale)
    out = _unsharp(out, spec.unsharp_amount, spec.unsharp_sigma, spec.unsharp_threshold)
    return out
