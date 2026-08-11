"""Extended near-field ROI infer — taller trapezoid, Stage-1 Medium @ conf 0.30.

Does NOT modify default InferConfig / run.py / near_field.py. Prior POC runs
(e.g. ROAD-1-Gopro-medium-stage1-c030) keep their original geometry.

Defaults (override via CLI):
  road_top_y=0.32  (was 0.52) — assess covers ~bottom 68% of frame
  z_far_m=8.0      (was 5.0)  — HUD/label for farther horizon
  conf=0.30, nms_iou=0.5
  weights=runs/rfdetr_stage1/checkpoint_best_total.pth
  out=runs/rfdetr_infer/ROAD-1-Gopro-medium-stage1-c030-extroi

  python -m tools.rfdetr_infer.run_extended_nearfield
  # or: ./scripts/run_rfdetr_infer_extended_roi.sh
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import InferConfig, repo_root
from .run import run_inference

DEFAULT_DRIVE_FOLDER = (
    "https://drive.google.com/drive/folders/1rhnvLoPFv87-vecmMhN-G2FJMbqYpJbj"
)
DEFAULT_OUT_NAME = "ROAD-1-Gopro-medium-stage1-c030-extroi"

# Extended ROI (image-2 style: taller assess zone; widths unchanged)
DEFAULT_ROAD_TOP_Y = 0.32
DEFAULT_Z_FAR_M = 8.0
DEFAULT_CONF = 0.30
DEFAULT_NMS_IOU = 0.5


def _default_weights(root: Path) -> Path:
    total = root / "runs" / "rfdetr_stage1" / "checkpoint_best_total.pth"
    if total.is_file():
        return total.resolve()
    ema = root / "runs" / "rfdetr_stage1" / "checkpoint_best_ema.pth"
    if ema.is_file():
        return ema.resolve()
    return total  # resolved later; will fail with clear error if missing


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    p = argparse.ArgumentParser(
        description=(
            "RF-DETR Medium Stage-1 near-field infer with EXTENDED ROI "
            "(taller trapezoid). Leaves default run.py defaults untouched."
        )
    )
    p.add_argument(
        "--video",
        type=str,
        default=DEFAULT_DRIVE_FOLDER,
        help="Local path, gs://, https, or Drive folder (default: ROAD-1 folder)",
    )
    p.add_argument(
        "--srt",
        type=str,
        default=DEFAULT_DRIVE_FOLDER,
        help="SRT path/URL/folder (default: same Drive folder as video)",
    )
    p.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Stage-1 Medium checkpoint (default: checkpoint_best_total.pth)",
    )
    p.add_argument(
        "--media-dir",
        type=Path,
        default=None,
        help="Downloaded media cache (default: data/rfdetr/infer_media)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Default: runs/rfdetr_infer/{DEFAULT_OUT_NAME}",
    )
    p.add_argument("--conf", type=float, default=DEFAULT_CONF)
    p.add_argument("--stride", type=int, default=3, dest="frame_stride")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--z-near", type=float, default=0.5, dest="z_near_m")
    p.add_argument("--z-far", type=float, default=DEFAULT_Z_FAR_M, dest="z_far_m")
    p.add_argument("--road-top-y", type=float, default=DEFAULT_ROAD_TOP_Y)
    p.add_argument("--road-bottom-y", type=float, default=1.0)
    p.add_argument("--road-bottom-half-w", type=float, default=0.78)
    p.add_argument("--road-top-half-w", type=float, default=0.50)
    p.add_argument("--road-center-x", type=float, default=0.52)
    p.add_argument("--min-overlap", type=float, default=0.15)
    p.add_argument("--nms-iou", type=float, default=DEFAULT_NMS_IOU, dest="nms_iou")
    p.add_argument("--near-wash-alpha", type=float, default=0.28)
    p.add_argument("--far-wash-alpha", type=float, default=0.0)
    p.add_argument("--crf", type=int, default=23)
    return p


def main(argv: list[str] | None = None) -> int:
    from .media_fetch import resolve_video_and_srt

    args = build_parser().parse_args(argv)
    root = repo_root()

    weights = args.weights
    if weights is None:
        weights = _default_weights(root)
    else:
        weights = Path(weights).expanduser()
    if not weights.is_file():
        raise SystemExit(
            f"--weights is not a file: {weights}\n"
            "Expected Stage-1 Medium, e.g. runs/rfdetr_stage1/checkpoint_best_total.pth"
        )
    weights = weights.resolve()

    media_dir = args.media_dir or (root / "data" / "rfdetr" / "infer_media")
    video, srt_path = resolve_video_and_srt(args.video, args.srt, media_dir)

    out = args.out_dir
    if out is None:
        out = root / "runs" / "rfdetr_infer" / DEFAULT_OUT_NAME

    print(
        f"EXTENDED ROI road_top_y={args.road_top_y} z_far={args.z_far_m} "
        f"(defaults unchanged elsewhere)"
    )
    print(f"  weights={weights}")
    print(f"  conf={args.conf} nms_iou={args.nms_iou}")
    print(f"  out_dir={out}")

    cfg = InferConfig(
        video=video,
        weights=weights,
        srt=srt_path,
        out_dir=out,
        conf=args.conf,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        z_near_m=args.z_near_m,
        z_far_m=args.z_far_m,
        road_top_y=args.road_top_y,
        road_bottom_y=args.road_bottom_y,
        road_bottom_half_w=args.road_bottom_half_w,
        road_top_half_w=args.road_top_half_w,
        road_center_x=args.road_center_x,
        use_classical_road=False,
        min_overlap=args.min_overlap,
        require_center=False,
        clip_to_mask=False,
        nms_iou=args.nms_iou,
        near_wash_alpha=args.near_wash_alpha,
        far_wash_alpha=args.far_wash_alpha,
        crf=args.crf,
        backend="rfdetr",
    )
    run_inference(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
