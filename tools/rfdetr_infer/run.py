"""CLI: RF-DETR near-field dashcam inference → video + CSV + map trail."""
from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tools.rfdetr_train.taxonomy import CLASS_NAMES

from .config import InferConfig, repo_root
from .export_out import tracks_to_rows, write_defects_csv, write_defects_json, write_summary
from .gate import gate_boxes
from .gps_io import load_gps
from .map_trail import write_map_trail
from .near_field import build_near_field
from .render import draw_boxes, draw_hud, draw_near_field
from .track_simple import SimpleTracker


def _predictions_to_arrays(detections):
    """Normalize rfdetr / supervision outputs to numpy arrays."""
    if detections is None:
        return (
            np.zeros((0, 4), dtype=np.float32),
            None,
            None,
        )
    # supervision Detections
    if hasattr(detections, "xyxy"):
        xyxy = np.asarray(detections.xyxy, dtype=np.float32)
        if xyxy.size == 0:
            return np.zeros((0, 4), dtype=np.float32), None, None
        cid = (
            np.asarray(detections.class_id)
            if getattr(detections, "class_id", None) is not None
            else None
        )
        conf = (
            np.asarray(detections.confidence, dtype=np.float32)
            if getattr(detections, "confidence", None) is not None
            else None
        )
        return xyxy, cid, conf
    return np.zeros((0, 4), dtype=np.float32), None, None


def run_inference(cfg: InferConfig) -> dict:
    if cfg.video is None or not Path(cfg.video).exists():
        raise SystemExit(f"Missing video: {cfg.video}")
    if cfg.weights is None or not Path(cfg.weights).exists():
        raise SystemExit(
            f"Missing weights: {cfg.weights}\n"
            "Pass --weights runs/rfdetr_stage1/checkpoint_best_total.pth"
        )

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gps = load_gps(Path(cfg.video), Path(cfg.srt) if cfg.srt else None)

    from rfdetr import RFDETRMedium

    print(f"Loading RFDETRMedium from {cfg.weights}")
    model = RFDETRMedium(pretrain_weights=str(cfg.weights))

    cap = cv2.VideoCapture(str(cfg.video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {cfg.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    annotated_path = out_dir / "annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(annotated_path), fourcc, fps, (w, h))

    tracker = SimpleTracker(iou_match=cfg.iou_match, max_age=cfg.max_age)
    unique_counts: Counter[str] = Counter()
    gated_away = 0
    raw_dets = 0
    frame_i = 0
    processed = 0
    route_samples: list[dict] = []
    t0 = time.time()

    print(
        f"Video {w}x{h} @ {fps:.1f}fps  stride={cfg.frame_stride}  "
        f"z_far={cfg.z_far_m}m  GPS={'yes' if gps.has_data else 'no'}"
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if cfg.max_frames and frame_i >= cfg.max_frames:
            break

        t_s = frame_i / max(fps, 1e-6)
        chainage = gps.distance_at_time(t_s) if gps.has_data else None
        fix = gps.at_time(t_s) if len(gps) else None

        # Sample route ~2 Hz for map polyline
        if fix is not None and (
            not route_samples or t_s - route_samples[-1]["t"] >= 0.5
        ):
            route_samples.append(
                {
                    "lat": fix.lat,
                    "lon": fix.lon,
                    "t": round(t_s, 3),
                    "chainage_m": None if chainage is None else round(float(chainage), 2),
                }
            )

        do_detect = frame_i % max(cfg.frame_stride, 1) == 0
        boxes = np.zeros((0, 4), dtype=np.float32)
        cids = None
        confs = None

        nf = build_near_field(frame, cfg)

        if do_detect:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            dets = model.predict(pil, threshold=cfg.conf)
            try:
                import supervision as sv

                if not isinstance(dets, sv.Detections):
                    try:
                        dets = sv.Detections.from_inference(dets)
                    except Exception:
                        pass
            except ImportError:
                pass

            boxes, cids, confs = _predictions_to_arrays(dets)
            raw_dets += len(boxes)
            boxes, cids, confs, n_drop = gate_boxes(
                boxes, cids, confs, nf.mask, min_overlap=cfg.min_overlap
            )
            gated_away += n_drop
            tracker.update(boxes, cids, confs, CLASS_NAMES, frame_i, t_s)
            unique_counts = Counter(
                tr.class_name for tr in (tracker.active + tracker.finished)
            )
            processed += 1

        annotated = draw_near_field(frame, nf)
        annotated = draw_boxes(annotated, boxes, cids, confs, CLASS_NAMES)
        annotated = draw_hud(
            annotated,
            counts=dict(unique_counts),
            chainage_m=chainage,
            t_s=t_s,
            z_far_m=cfg.z_far_m,
            gps_ok=gps.has_data,
        )
        writer.write(annotated)

        if frame_i % 100 == 0:
            print(
                f"  frame {frame_i}/{n_frames or '?'}  "
                f"unique={dict(unique_counts)}  gated_away={gated_away}"
            )
        frame_i += 1

    cap.release()
    writer.release()

    all_tracks = tracker.flush()
    rows = tracks_to_rows(all_tracks, gps)
    write_defects_csv(out_dir / "defects.csv", rows)
    write_defects_json(out_dir / "defects.json", rows)
    write_map_trail(
        out_dir / "map_trail.html",
        route=route_samples,
        defects=rows,
        title="RF-DETR near-field defects",
    )

    elapsed = time.time() - t0
    summary = {
        "video": str(cfg.video),
        "weights": str(cfg.weights),
        "out_dir": str(out_dir),
        "frames_total": frame_i,
        "frames_detected": processed,
        "frame_stride": cfg.frame_stride,
        "raw_detections": raw_dets,
        "gated_away": gated_away,
        "unique_defects": len(rows),
        "counts": dict(Counter(r["class"] for r in rows)),
        "gps": gps.has_data,
        "n_gps_fixes": len(gps),
        "z_near_m": cfg.z_near_m,
        "z_far_m": cfg.z_far_m,
        "annotated_video": str(annotated_path),
        "elapsed_s": round(elapsed, 1),
        "phase2_note": (
            "Phase 2 (later): set inference.backend=rfdetr in config.yaml and "
            "reuse detect_track + IRC report; optional full 3-panel dashboard."
        ),
    }
    write_summary(out_dir / "summary.json", summary)

    print(f"\nDone in {elapsed:.1f}s → {out_dir}")
    print(f"  annotated: {annotated_path}")
    print(f"  defects:   {out_dir / 'defects.csv'} ({len(rows)} unique)")
    print(f"  map:       {out_dir / 'map_trail.html'}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RF-DETR near-field dashcam inference (boxes + SRT map trail)."
    )
    p.add_argument(
        "--video",
        type=str,
        required=True,
        help="Local path, gs:// URI, or https:// URL to the dashcam video",
    )
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument(
        "--srt",
        type=str,
        default=None,
        help="Local path, gs:// URI, or https:// URL to the SRT (optional)",
    )
    p.add_argument(
        "--media-dir",
        type=Path,
        default=None,
        help="Where to cache downloaded media (default: data/rfdetr/infer_media)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: runs/rfdetr_infer/<video_stem>",
    )
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--stride", type=int, default=3, dest="frame_stride")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--z-near", type=float, default=0.5, dest="z_near_m")
    p.add_argument("--z-far", type=float, default=5.0, dest="z_far_m")
    p.add_argument("--road-top-y", type=float, default=0.55)
    p.add_argument("--road-bottom-y", type=float, default=1.0)
    p.add_argument("--road-bottom-half-w", type=float, default=0.48)
    p.add_argument("--road-top-half-w", type=float, default=0.12)
    p.add_argument("--no-classical-road", action="store_true")
    p.add_argument("--min-overlap", type=float, default=0.25)
    p.add_argument("--camera-height-m", type=float, default=None)
    p.add_argument("--camera-pitch-deg", type=float, default=None)
    p.add_argument("--vfov-deg", type=float, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    from .media_fetch import fetch_media

    args = build_parser().parse_args(argv)
    media_dir = args.media_dir or (repo_root() / "data" / "rfdetr" / "infer_media")

    print("==> Resolving video")
    video = fetch_media(args.video, media_dir, default_name="input.mp4")
    srt_path = None
    if args.srt:
        print("==> Resolving SRT")
        srt_path = fetch_media(args.srt, media_dir, default_name=f"{video.stem}.srt")

    out = args.out_dir
    if out is None:
        out = repo_root() / "runs" / "rfdetr_infer" / video.stem
    cfg = InferConfig(
        video=video,
        weights=args.weights,
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
        use_classical_road=not args.no_classical_road,
        min_overlap=args.min_overlap,
        camera_height_m=args.camera_height_m,
        camera_pitch_deg=args.camera_pitch_deg,
        vfov_deg=args.vfov_deg,
    )
    run_inference(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
