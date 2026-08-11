"""Augmentation presets for custom Stage-2 Medium fine-tune (anti-overfit)."""
from __future__ import annotations

from typing import Any

# Five families (train online via rfdetr aug_config + offline offline expand):
# 1 brightness/contrast  2 blur  3 coarse dropout  4 hflip  5 HSV + noise
CUSTOM_ROAD_AUG: dict[str, dict[str, Any]] = {
    "HorizontalFlip": {"p": 0.5},
    "RandomBrightnessContrast": {
        "brightness_limit": 0.25,
        "contrast_limit": 0.25,
        "p": 0.5,
    },
    "GaussianBlur": {"blur_limit": (3, 7), "p": 0.3},
    "MotionBlur": {"blur_limit": 7, "p": 0.15},
    "CoarseDropout": {
        "max_holes": 4,
        "max_height": 48,
        "max_width": 48,
        "min_holes": 1,
        "fill_value": 0,
        "p": 0.3,
    },
    "HueSaturationValue": {
        "hue_shift_limit": 12,
        "sat_shift_limit": 25,
        "val_shift_limit": 20,
        "p": 0.3,
    },
    "GaussNoise": {"std_range": (0.02, 0.08), "p": 0.25},
}

# Milder stress views for qualitative val/test (not used for early-stop metric)
STRESS_LIGHT_AUG: dict[str, dict[str, Any]] = {
    "HorizontalFlip": {"p": 0.5},
    "RandomBrightnessContrast": {
        "brightness_limit": 0.15,
        "contrast_limit": 0.15,
        "p": 0.35,
    },
    "GaussianBlur": {"blur_limit": (3, 5), "p": 0.2},
    "HueSaturationValue": {
        "hue_shift_limit": 8,
        "sat_shift_limit": 15,
        "val_shift_limit": 12,
        "p": 0.25,
    },
}

AUG_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "custom_road": CUSTOM_ROAD_AUG,
    "stress_light": STRESS_LIGHT_AUG,
    "none": {},
}


def get_aug_preset(name: str) -> dict[str, dict[str, Any]]:
    key = (name or "custom_road").strip().lower()
    if key not in AUG_PRESETS:
        raise SystemExit(
            f"Unknown --aug-preset {name!r}. Choose from: {sorted(AUG_PRESETS)}"
        )
    return dict(AUG_PRESETS[key])
