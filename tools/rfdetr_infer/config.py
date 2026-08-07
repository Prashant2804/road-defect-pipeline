"""Defaults for RF-DETR near-field dashcam inference."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class InferConfig:
    video: Path | None = None
    weights: Path | None = None
    srt: Path | None = None
    out_dir: Path = field(
        default_factory=lambda: repo_root() / "runs" / "rfdetr_infer" / "latest"
    )

    # Detector
    conf: float = 0.25
    frame_stride: int = 3
    max_frames: int = 0  # 0 = whole video

    # Near-field trapezoid (normalized image coords, Colab-style)
    z_near_m: float = 0.5
    z_far_m: float = 5.0
    road_bottom_y: float = 1.0
    road_top_y: float = 0.55  # ~5 m ahead proxy when no camera model
    road_bottom_half_w: float = 0.55
    road_top_half_w: float = 0.28
    road_center_x: float = 0.5
    use_classical_road: bool = True
    min_overlap: float = 0.25

    # Optional metric camera (metres). If unset, trapezoid top is the far edge.
    camera_height_m: float | None = None
    camera_pitch_deg: float | None = None
    vfov_deg: float | None = None

    # Tracker
    iou_match: float = 0.3
    max_age: int = 15  # strided frames without match before closing track

    # Encode quality (H.264 CRF; lower = sharper / larger)
    crf: int = 18
