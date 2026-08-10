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

    # Detector — RF-DETR Medium recall defaults (RT-DETR uses stricter via CLI/wrapper)
    conf: float = 0.15
    frame_stride: int = 3
    max_frames: int = 0  # 0 = whole video

    # Near-field trapezoid — wide enough for both lanes on typical GoPro dashcam
    z_near_m: float = 0.5
    z_far_m: float = 5.0
    road_bottom_y: float = 1.0
    road_top_y: float = 0.52  # ~5 m ahead proxy when no camera model
    road_bottom_half_w: float = 0.78
    road_top_half_w: float = 0.50
    road_center_x: float = 0.52  # slight right bias (vehicle often left of road center)
    # Classical grow often drops cracked / rutted asphalt — off by default for gating
    use_classical_road: bool = False
    # Soft gate for Medium recall; RT-DETR wrapper raises this
    min_overlap: float = 0.15
    require_center: bool = False
    clip_to_mask: bool = False
    # Overlay: green wash lives in the assess polygon; far corridor tint is optional
    near_wash_alpha: float = 0.28
    far_wash_alpha: float = 0.0

    # Optional metric camera (metres). If unset, trapezoid top is the far edge.
    camera_height_m: float | None = None
    camera_pitch_deg: float | None = None
    vfov_deg: float | None = None

    # Tracker
    iou_match: float = 0.3
    max_age: int = 15  # strided frames without match before closing track

    # Cross-class NMS (0 = disabled; 0.5 collapses stacked same-patch boxes)
    nms_iou: float = 0.5

    # Encode quality (H.264 CRF; lower = sharper / larger). 23 ≈ Drive-friendly size.
    crf: int = 23

    # Detector backend: "rfdetr" (Roboflow RF-DETR) or "rtdetr" (Ultralytics RT-DETR)
    backend: str = "rfdetr"
