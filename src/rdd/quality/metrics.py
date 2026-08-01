"""Per-frame image quality measurement, with thresholds learned per clip.

Why not fixed thresholds: variance-of-Laplacian (the usual sharpness proxy)
scales with resolution, texture and contrast. A cutoff tuned on one camera is
meaningless on another — a sharp 720p frame can score below a blurry 4K one. So
we sample the clip, learn its own distribution, and judge each frame *relative
to that clip* with an absolute floor as a backstop.

The point of measuring is triage, not vanity metrics: a motion-blurred or
blown-out frame produces confident nonsense from the detector, so it is better
dropped than analysed. Every dropped frame is counted and reported — silently
discarding input would misrepresent coverage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..utils.logging import get_logger

log = get_logger("rdd.quality.metrics")


@dataclass
class FrameQuality:
    index: int = 0
    sharpness: float = 0.0       # variance of Laplacian (content-dependent)
    tenengrad: float = 0.0       # mean squared Sobel gradient
    contrast: float = 0.0        # RMS contrast of L, 0..1
    clipped_low: float = 0.0     # fraction of crushed-black pixels
    clipped_high: float = 0.0    # fraction of blown-white pixels
    mean_luma: float = 0.0       # 0..1
    noise_sigma: float = 0.0     # Immerkaer estimate, 0..255 scale
    usable: bool = True
    reasons: tuple[str, ...] = ()

    def as_row(self) -> dict:
        return {
            "index": self.index,
            "sharpness": round(self.sharpness, 3),
            "tenengrad": round(self.tenengrad, 3),
            "contrast": round(self.contrast, 5),
            "clipped_low": round(self.clipped_low, 5),
            "clipped_high": round(self.clipped_high, 5),
            "mean_luma": round(self.mean_luma, 4),
            "noise_sigma": round(self.noise_sigma, 3),
            "usable": self.usable,
            "reasons": ";".join(self.reasons),
        }


@dataclass
class QualityProfile:
    """Thresholds + clip statistics, learned from a sample of frames."""

    n_sampled: int = 0
    sharpness_median: float = 0.0
    sharpness_thresh: float = 0.0
    contrast_median: float = 0.0
    min_contrast: float = 0.0
    max_clipped_high: float = 1.0
    max_clipped_low: float = 1.0
    noise_median: float = 0.0
    enabled: bool = True
    samples: list[FrameQuality] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "frames_sampled": self.n_sampled,
            "sharpness_median": round(self.sharpness_median, 3),
            "sharpness_threshold": round(self.sharpness_thresh, 3),
            "contrast_median": round(self.contrast_median, 5),
            "noise_sigma_median": round(self.noise_median, 3),
        }


def _percentile(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def estimate_noise_sigma(gray) -> float:
    """Immerkaer's fast noise estimator (single 3x3 convolution).

    The kernel annihilates locally-linear image content, so what survives is
    dominated by noise rather than by edges.
    """
    import cv2
    import numpy as np

    h, w = gray.shape[:2]
    if h < 3 or w < 3:
        return 0.0
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    conv = cv2.filter2D(gray.astype(np.float32), -1, k)
    inner = conv[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2.0) * np.abs(inner).mean() / 6.0)


def measure_frame(frame, index: int = 0) -> FrameQuality:
    """Compute quality metrics for one BGR frame."""
    import cv2
    import numpy as np

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g32 = gray.astype(np.float32)

    lap = cv2.Laplacian(g32, cv2.CV_32F, ksize=3)
    sharpness = float(lap.var())

    gx = cv2.Sobel(g32, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g32, cv2.CV_32F, 0, 1, ksize=3)
    tenengrad = float((gx * gx + gy * gy).mean())

    contrast = float(g32.std() / 255.0)
    mean_luma = float(g32.mean() / 255.0)
    total = float(gray.size) or 1.0
    clipped_low = float((gray <= 2).sum() / total)
    clipped_high = float((gray >= 253).sum() / total)

    return FrameQuality(
        index=index, sharpness=sharpness, tenengrad=tenengrad, contrast=contrast,
        clipped_low=clipped_low, clipped_high=clipped_high, mean_luma=mean_luma,
        noise_sigma=estimate_noise_sigma(gray),
    )


def judge(m: FrameQuality, profile: QualityProfile) -> FrameQuality:
    """Mark a measured frame usable/unusable against a learned profile."""
    reasons: list[str] = []
    if not profile.enabled:
        m.usable, m.reasons = True, ()
        return m
    if m.sharpness < profile.sharpness_thresh:
        reasons.append(f"blurry(sharpness {m.sharpness:.1f} < {profile.sharpness_thresh:.1f})")
    if m.contrast < profile.min_contrast:
        reasons.append(f"flat(contrast {m.contrast:.3f} < {profile.min_contrast:.3f})")
    if m.clipped_high > profile.max_clipped_high:
        reasons.append(f"blown({m.clipped_high:.0%} clipped white)")
    if m.clipped_low > profile.max_clipped_low:
        reasons.append(f"crushed({m.clipped_low:.0%} clipped black)")
    m.reasons = tuple(reasons)
    m.usable = not reasons
    return m


def build_profile(cfg, samples: list[FrameQuality]) -> QualityProfile:
    """Turn sampled measurements into thresholds."""
    qc = cfg.get_path("quality.assess", {}) or {}
    enabled = bool(qc.get("enabled", True))

    sharps = [s.sharpness for s in samples]
    contrasts = [s.contrast for s in samples]
    noises = [s.noise_sigma for s in samples]

    sharp_med = _percentile(sharps, 0.5)
    rel_floor = float(qc.get("sharpness_rel_floor", 0.35))
    abs_floor = float(qc.get("sharpness_abs_floor", 8.0))
    # Relative floor catches "much blurrier than this clip normally is";
    # absolute floor catches a clip that is uniformly out of focus.
    thresh = max(rel_floor * sharp_med, abs_floor)

    profile = QualityProfile(
        n_sampled=len(samples),
        sharpness_median=sharp_med,
        sharpness_thresh=thresh,
        contrast_median=_percentile(contrasts, 0.5),
        min_contrast=float(qc.get("min_contrast", 0.04)),
        max_clipped_high=float(qc.get("max_clipped_high", 0.25)),
        max_clipped_low=float(qc.get("max_clipped_low", 0.35)),
        noise_median=_percentile(noises, 0.5),
        enabled=enabled,
        samples=samples,
    )
    if enabled:
        log.info(
            "Quality profile from %d frames: sharpness median %.1f -> drop below "
            "%.1f; contrast median %.3f; noise sigma %.2f",
            profile.n_sampled, sharp_med, thresh, profile.contrast_median,
            profile.noise_median,
        )
        if sharp_med < abs_floor:
            log.warning(
                "Whole clip is soft (median sharpness %.1f below the absolute floor "
                "%.1f). Every frame would be dropped, so the relative test is doing "
                "the work — check focus/resolution at the source.", sharp_med, abs_floor,
            )
    return profile


def assess_video(video_path: str | Path, cfg) -> QualityProfile:
    """Sample frames evenly across the clip and learn its quality distribution.

    Sampling seeks rather than decoding the whole file. Which frame we get is not
    important — we want a representative spread, not frame N — and on a long clip the
    difference is decoding 60 frames instead of 30,000. Doing it the other way made a
    quality pass over 4K dashcam footage take minutes.
    """
    from ..utils.video import iter_sampled_frames

    qc = cfg.get_path("quality.assess", {}) or {}
    n_target = max(2, int(qc.get("sample_frames", 60)))

    samples = [measure_frame(frame, idx)
               for idx, frame in iter_sampled_frames(video_path, n_target)]

    if not samples:
        raise RuntimeError(f"No frames readable from {video_path}")
    return build_profile(cfg, samples)
