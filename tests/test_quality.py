"""Video quality measurement and enhancement."""
from __future__ import annotations

import numpy as np

from rdd.quality.enhance import EnhanceSpec, enhance_frame, resolve_spec
from rdd.quality.metrics import (
    FrameQuality,
    build_profile,
    estimate_noise_sigma,
    judge,
    measure_frame,
)
from tests.scenes import H, W, blur, car_scene, overexpose


def test_blur_lowers_the_sharpness_metric():
    frame, _ = car_scene()
    sharp = measure_frame(frame).sharpness
    soft = measure_frame(blur(frame, 4.0)).sharpness
    assert soft < sharp * 0.5, f"blur not reflected ({soft:.1f} vs {sharp:.1f})"


def test_overexposure_is_detected_as_clipping():
    frame, _ = car_scene()
    assert measure_frame(overexpose(frame, 3.0)).clipped_high > 0.2
    assert measure_frame(frame).clipped_high < 0.05


def test_noise_estimator_tracks_added_noise():
    rng = np.random.default_rng(0)
    flat = np.full((200, 200), 128, dtype=np.uint8)
    clean = estimate_noise_sigma(flat)
    noisy = estimate_noise_sigma(
        np.clip(flat + rng.normal(0, 12, flat.shape), 0, 255).astype(np.uint8)
    )
    assert clean < 1.0
    assert noisy > 8.0, f"expected ~12, measured {noisy:.1f}"


def test_thresholds_are_relative_to_the_clip(cfg):
    """A soft clip is judged on its own distribution, not an absolute cutoff.

    The whole point: a fixed sharpness threshold tuned on one camera silently
    rejects every frame from a softer one.
    """
    soft = [FrameQuality(index=i, sharpness=20.0, contrast=0.2) for i in range(20)]
    sharp = [FrameQuality(index=i, sharpness=900.0, contrast=0.2) for i in range(20)]

    soft_profile = build_profile(cfg, soft)
    sharp_profile = build_profile(cfg, sharp)
    assert sharp_profile.sharpness_thresh > soft_profile.sharpness_thresh

    # A 20-sharpness frame is normal in the soft clip and terrible in the sharp one.
    probe = FrameQuality(index=99, sharpness=20.0, contrast=0.2)
    assert judge(FrameQuality(**vars(probe)), soft_profile).usable
    assert not judge(FrameQuality(**vars(probe)), sharp_profile).usable


def test_judge_reports_a_reason_for_every_rejection(cfg):
    profile = build_profile(cfg, [FrameQuality(sharpness=500.0, contrast=0.2)])
    bad = judge(FrameQuality(sharpness=1.0, contrast=0.001, clipped_high=0.9), profile)
    assert not bad.usable
    assert bad.reasons, "a rejection with no stated reason is not auditable"
    assert any("blurry" in r for r in bad.reasons)


def test_disabled_assessment_passes_everything(cfg):
    cfg.set_path("quality.assess.enabled", False)
    profile = build_profile(cfg, [FrameQuality(sharpness=500.0, contrast=0.2)])
    assert judge(FrameQuality(sharpness=0.0, contrast=0.0), profile).usable


# -- enhancement --------------------------------------------------------------

def test_enhancement_is_deterministic():
    frame, _ = car_scene()
    spec = EnhanceSpec(clahe_clip=2.0, unsharp_amount=0.6)
    assert np.array_equal(enhance_frame(frame, spec), enhance_frame(frame, spec))


def test_disabled_spec_is_an_identity():
    frame, _ = car_scene()
    out = enhance_frame(frame, EnhanceSpec(enabled=False))
    assert np.array_equal(out, frame)


def test_clahe_raises_local_contrast_on_dull_footage():
    dull = np.clip(np.full((H, W, 3), 120, dtype=np.float32)
                   + np.random.default_rng(3).normal(0, 4, (H, W, 3)), 0, 255).astype(np.uint8)
    before = measure_frame(dull).contrast
    after = measure_frame(enhance_frame(dull, EnhanceSpec(clahe_clip=3.0,
                                                          unsharp_amount=0.0))).contrast
    assert after > before * 1.5, f"CLAHE did not lift contrast ({before:.4f}->{after:.4f})"


def test_unsharp_recovers_sharpness_after_blur():
    frame, _ = car_scene()
    soft = blur(frame, 2.0)
    resharpened = enhance_frame(soft, EnhanceSpec(clahe_clip=0.0, unsharp_amount=1.2,
                                                  unsharp_sigma=2.0, unsharp_threshold=0))
    assert measure_frame(resharpened).sharpness > measure_frame(soft).sharpness


def test_fingerprint_changes_with_settings_and_is_stable():
    a = EnhanceSpec(clahe_clip=2.0)
    b = EnhanceSpec(clahe_clip=2.5)
    assert a.fingerprint() == EnhanceSpec(clahe_clip=2.0).fingerprint()
    assert a.fingerprint() != b.fingerprint(), \
        "a fingerprint that ignores settings cannot detect train/serve drift"


def test_upscale_respects_the_cap():
    small = np.zeros((100, 200, 3), dtype=np.uint8)
    out = enhance_frame(small, EnhanceSpec(clahe_clip=0.0, unsharp_amount=0.0,
                                           min_width=2000, max_upscale=2.0))
    assert out.shape[1] == 400, "max_upscale should cap the blow-up at 2x"


def test_adaptive_strengthens_clahe_on_low_contrast(cfg):
    from rdd.quality.metrics import QualityProfile

    cfg.set_path("quality.enhance.adaptive", True)
    baseline = resolve_spec(cfg, QualityProfile(contrast_median=0.20, noise_median=0.5))
    dull = resolve_spec(cfg, QualityProfile(contrast_median=0.02, noise_median=0.5))
    assert dull.clahe_clip > baseline.clahe_clip


def test_adaptive_denoises_and_sharpens_less_on_noisy_footage(cfg):
    from rdd.quality.metrics import QualityProfile

    cfg.set_path("quality.enhance.adaptive", True)
    cfg.set_path("quality.enhance.denoise", "none")
    noisy = resolve_spec(cfg, QualityProfile(contrast_median=0.15, noise_median=9.0))
    assert noisy.denoise == "bilateral", "noisy clip should switch denoising on"
    clean = resolve_spec(cfg, QualityProfile(contrast_median=0.15, noise_median=0.5))
    assert noisy.unsharp_amount < clean.unsharp_amount, \
        "sharpening noise back in defeats the denoise"
