"""CLI: near-field dashcam inference (RF-DETR or Ultralytics RT-DETR)."""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tools.rfdetr_train.taxonomy import CLASS_NAMES

from .config import InferConfig, repo_root
from .export_out import tracks_to_rows, write_defects_csv, write_defects_json, write_summary
from .gate import gate_boxes, nms_boxes
from .gps_io import load_gps
from .map_trail import write_map_trail
from .near_field import build_near_field
from .render import draw_boxes, draw_hud, draw_near_field
from .track_simple import SimpleTracker


def _class_names_for_model(model, backend: str) -> list[str]:
    """Prefer names embedded in the checkpoint; fall back to repo 6-class taxonomy."""
    candidates = []
    for attr in ("class_names", "names", "CLASS_NAMES"):
        if not hasattr(model, attr):
            continue
        val = getattr(model, attr)
        if callable(val):
            try:
                val = val()
            except TypeError:
                continue
        candidates.append(val)
    # Ultralytics
    if hasattr(model, "model") and hasattr(model.model, "names"):
        candidates.append(model.model.names)

    for val in candidates:
        if isinstance(val, dict) and val:
            try:
                return [str(val[i]) for i in sorted(val)]
            except Exception:
                continue
        if isinstance(val, (list, tuple)) and len(val) > 0:
            return [str(x) for x in val]
    return list(CLASS_NAMES)


def _predictions_to_arrays(detections):
    """Normalize rfdetr / supervision / ultralytics outputs to numpy arrays."""
    if detections is None:
        return (
            np.zeros((0, 4), dtype=np.float32),
            None,
            None,
        )
    # Ultralytics Results
    if hasattr(detections, "boxes"):
        boxes = detections.boxes
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32), None, None
        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        cid = boxes.cls.detach().cpu().numpy().astype(np.int64)
        conf = boxes.conf.detach().cpu().numpy().astype(np.float32)
        return xyxy, cid, conf
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


def _load_model(cfg: InferConfig):
    backend = (cfg.backend or "rfdetr").lower().strip()
    if cfg.weights is None or not str(cfg.weights).strip():
        raise SystemExit(
            "Missing --weights. Example:\n"
            "  --weights runs/rfdetr_stage1/checkpoint_best_total.pth"
        )
    weights = Path(cfg.weights).expanduser()
    if not weights.is_absolute():
        weights = (repo_root() / weights).resolve()
    else:
        weights = weights.resolve()
    if weights.is_dir():
        raise SystemExit(
            f"Weights path is a directory, expected a .pth/.pt file: {weights}\n"
            "Did --weights get an empty $WEIGHTS? Pass the file explicitly, e.g.:\n"
            "  --weights runs/rfdetr_stage1/checkpoint_best_total.pth"
        )
    if not weights.is_file():
        raise SystemExit(f"Missing weights file: {weights}")

    if backend == "rtdetr":
        from ultralytics import RTDETR

        print(f"Loading Ultralytics RTDETR from {weights}")
        model = RTDETR(str(weights))
        return "rtdetr", model

    if backend != "rfdetr":
        raise SystemExit(f"Unknown --backend {backend!r} (use rfdetr or rtdetr)")

    return "rfdetr", _load_rfdetr(weights)


def _load_rfdetr(weights: Path):
    """Load a local RF-DETR train checkpoint for inference.

    Newer rfdetr builds resolve bare/COCO names under ``~/.roboflow/models``.
    Fine-tuned ``checkpoint_best_*.pth`` files must be loaded via
    ``from_checkpoint(..., trust_checkpoint=True)`` (training pickles are not
    ``weights_only``-safe).
    """
    w = str(weights.resolve())
    print(f"Loading RF-DETR checkpoint: {w}")

    # 1) Preferred: package-level / class from_checkpoint (auto-detects Medium/Large)
    try:
        from rfdetr import from_checkpoint as rf_from_checkpoint

        try:
            model = rf_from_checkpoint(w, trust_checkpoint=True)
        except TypeError:
            model = rf_from_checkpoint(w)
        print(f"  via rfdetr.from_checkpoint → {type(model).__name__}")
        return model
    except Exception as e:
        print(f"  rfdetr.from_checkpoint unavailable/failed ({e})")

    # 2) Variant-specific from_checkpoint
    name = weights.name.lower()
    parent = str(weights.parent).lower()
    prefer_large = ("large" in name) or (
        "stage2" in parent and "medium" not in parent and "medium" not in name
    )
    variants = []
    if prefer_large:
        variants.append("RFDETRLarge")
    variants.extend(["RFDETRMedium", "RFDETRLarge", "RFDETRBase", "RFDETRSmall"])

    import rfdetr as rf_mod

    last_err: Exception | None = None
    for cls_name in variants:
        cls = getattr(rf_mod, cls_name, None)
        if cls is None:
            continue
        if hasattr(cls, "from_checkpoint"):
            try:
                try:
                    model = cls.from_checkpoint(w, trust_checkpoint=True)
                except TypeError:
                    model = cls.from_checkpoint(w)
                print(f"  via {cls_name}.from_checkpoint")
                return model
            except Exception as e:
                last_err = e
                print(f"  {cls_name}.from_checkpoint failed: {e}")

        # 3) Legacy constructor — must pass absolute file + trust_checkpoint
        try:
            try:
                model = cls(pretrain_weights=w, trust_checkpoint=True)
            except TypeError:
                model = cls(pretrain_weights=w)
            print(f"  via {cls_name}(pretrain_weights=..., trust_checkpoint=True)")
            return model
        except Exception as e:
            last_err = e
            print(f"  {cls_name}(pretrain_weights=...) failed: {e}")

    raise SystemExit(
        f"Could not load RF-DETR weights: {w}\n"
        f"Last error: {last_err}\n"
        "Ensure the file is a training checkpoint (.pth) from this repo, "
        "not a directory under ~/.roboflow/models."
    )


def _predict(backend: str, model, frame_bgr: np.ndarray, conf: float):
    if backend == "rtdetr":
        results = model.predict(frame_bgr, conf=conf, verbose=False)
        return results[0] if results else None

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    dets = model.predict(pil, threshold=conf)
    try:
        import supervision as sv

        if not isinstance(dets, sv.Detections):
            try:
                dets = sv.Detections.from_inference(dets)
            except Exception:
                pass
    except ImportError:
        pass
    return dets


def run_inference(cfg: InferConfig) -> dict:
    if cfg.video is None or not Path(cfg.video).exists():
        raise SystemExit(f"Missing video: {cfg.video}")
    if cfg.weights is None or not str(cfg.weights).strip():
        raise SystemExit(
            "Missing --weights. Example:\n"
            "  --weights runs/rfdetr_stage1/checkpoint_best_total.pth"
        )
    weights_path = Path(cfg.weights).expanduser()
    if not weights_path.is_file():
        # Empty Path("") resolves to cwd — catch that early
        raise SystemExit(
            f"Missing weights file: {cfg.weights}\n"
            "Pass --weights runs/rfdetr_stage1/checkpoint_best_total.pth  (or RF-DETR .pth)"
        )


    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gps = load_gps(Path(cfg.video), Path(cfg.srt) if cfg.srt else None)
    backend, model = _load_model(cfg)
    class_names = _class_names_for_model(model, backend)
    print(f"Class names ({len(class_names)}): {class_names}")

    cap = cv2.VideoCapture(str(cfg.video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {cfg.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    annotated_path = out_dir / "annotated.mp4"
    # Proven full-length path: OpenCV mp4v during infer, then H.264 CRF compress.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(annotated_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"Could not open VideoWriter for {annotated_path}")
    print(
        f"Writing annotated video (OpenCV mp4v → H.264 crf={cfg.crf}) "
        f"backend={backend} → {annotated_path}"
    )

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
            dets = _predict(backend, model, frame, cfg.conf)
            boxes, cids, confs = _predictions_to_arrays(dets)
            raw_dets += len(boxes)
            boxes, cids, confs = nms_boxes(
                boxes, cids, confs, iou_thresh=cfg.nms_iou
            )
            boxes, cids, confs, n_drop = gate_boxes(
                boxes,
                cids,
                confs,
                nf.mask,
                min_overlap=cfg.min_overlap,
                require_center=cfg.require_center,
                clip_to_mask=cfg.clip_to_mask,
            )
            gated_away += n_drop
            tracker.update(boxes, cids, confs, class_names, frame_i, t_s)
            unique_counts = Counter(
                tr.class_name for tr in (tracker.active + tracker.finished)
            )
            processed += 1

        annotated = draw_near_field(
            frame,
            nf,
            far_alpha=cfg.far_wash_alpha,
            near_alpha=cfg.near_wash_alpha,
        )
        annotated = draw_boxes(annotated, boxes, cids, confs, class_names)
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

    try:
        from .compress_video import compress_mp4

        h264 = compress_mp4(
            annotated_path, crf=cfg.crf, preset="medium", replace_src=True
        )
        annotated_path = h264
        print(f"Annotated video compressed in place: {annotated_path}")
    except Exception as e:
        print(
            f"WARNING: H.264 compress skipped ({e}). "
            "Install ffmpeg for Drive-friendly uploads."
        )

    all_tracks = tracker.flush()
    rows = tracks_to_rows(all_tracks, gps)
    write_defects_csv(out_dir / "defects.csv", rows)
    write_defects_json(out_dir / "defects.json", rows)
    (out_dir / "route.json").write_text(
        json.dumps(route_samples, indent=2), encoding="utf-8"
    )
    maps_key = os.environ.get("GOOGLE_MAPS_API_KEY") or None
    dash_dir = out_dir / "dashboard"
    dash_dir.mkdir(parents=True, exist_ok=True)
    title = (
        "RT-DETR near-field defects"
        if backend == "rtdetr"
        else "RF-DETR near-field defects"
    )
    write_map_trail(
        dash_dir / "index.html",
        route=route_samples,
        defects=rows,
        title=title,
        video_src="../annotated.mp4",
        z_far_m=cfg.z_far_m,
        maps_api_key=maps_key,
    )

    elapsed = time.time() - t0
    expected_s = frame_i / max(fps, 1e-6)
    out_s = None
    try:
        out_cap = cv2.VideoCapture(str(annotated_path))
        out_n = int(out_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        out_fps = float(out_cap.get(cv2.CAP_PROP_FPS) or fps)
        out_cap.release()
        out_s = out_n / max(out_fps, 1e-6)
        if expected_s > 30 and out_s < expected_s * 0.5:
            print(
                f"WARNING: annotated video looks truncated "
                f"({out_s:.1f}s vs expected ~{expected_s:.1f}s). Re-run inference."
            )
    except Exception:
        pass

    summary = {
        "video": str(cfg.video),
        "weights": str(cfg.weights),
        "backend": backend,
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
        "expected_duration_s": round(expected_s, 1),
        "annotated_duration_s": None if out_s is None else round(out_s, 1),
        "elapsed_s": round(elapsed, 1),
        "phase2_note": (
            "Phase 2 (later): set inference.backend in config.yaml and "
            "reuse detect_track + IRC report; optional full 3-panel dashboard."
        ),
    }
    write_summary(out_dir / "summary.json", summary)

    print(f"\nDone in {elapsed:.1f}s → {out_dir}")
    print(f"  backend:   {backend}")
    print(f"  annotated: {annotated_path}")
    print(f"  defects:   {out_dir / 'defects.csv'} ({len(rows)} unique)")
    print(f"  map:       {out_dir / 'dashboard' / 'index.html'}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Near-field dashcam inference (RF-DETR or Ultralytics RT-DETR)."
    )
    p.add_argument(
        "--video",
        type=str,
        required=True,
        help="Local path, gs:// URI, or https:// URL to the dashcam video",
    )
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument(
        "--backend",
        type=str,
        default="rfdetr",
        choices=("rfdetr", "rtdetr"),
        help="Detector backend (default: rfdetr)",
    )
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
    p.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Detection confidence (default: 0.15 rfdetr / 0.5 rtdetr)",
    )
    p.add_argument("--stride", type=int, default=3, dest="frame_stride")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--z-near", type=float, default=0.5, dest="z_near_m")
    p.add_argument("--z-far", type=float, default=5.0, dest="z_far_m")
    p.add_argument("--road-top-y", type=float, default=0.52)
    p.add_argument("--road-bottom-y", type=float, default=1.0)
    p.add_argument("--road-bottom-half-w", type=float, default=0.78)
    p.add_argument("--road-top-half-w", type=float, default=0.50)
    p.add_argument("--road-center-x", type=float, default=0.52)
    p.add_argument(
        "--classical-road",
        action="store_true",
        help="Enable classical color/texture road grow (often drops cracked asphalt)",
    )
    p.add_argument(
        "--min-overlap",
        type=float,
        default=None,
        help="Min box/mask overlap (default: 0.15 rfdetr / 0.50 rtdetr)",
    )
    p.add_argument(
        "--nms-iou",
        type=float,
        default=None,
        dest="nms_iou",
        help="Cross-class NMS IoU (0=off; default 0.5 for both backends)",
    )
    p.add_argument("--near-wash-alpha", type=float, default=0.28)
    p.add_argument(
        "--far-wash-alpha",
        type=float,
        default=0.0,
        help="Green tint beyond assess band (0=off; wash is in the polygon by default)",
    )
    p.add_argument("--camera-height-m", type=float, default=None)
    p.add_argument("--camera-pitch-deg", type=float, default=None)
    p.add_argument("--vfov-deg", type=float, default=None)
    p.add_argument(
        "--crf",
        type=int,
        default=23,
        help="H.264 CRF (lower=sharper/larger; 23 default — Drive-friendly)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    from .media_fetch import resolve_video_and_srt

    args = build_parser().parse_args(argv)

    # Empty --weights "" becomes Path('.') with argparse type=Path — reject early.
    warg = args.weights
    if warg is None or str(warg).strip() in {"", "."}:
        raise SystemExit(
            "Missing/empty --weights. Use the Stage-1 Medium file explicitly:\n"
            "  --weights runs/rfdetr_stage1/checkpoint_best_total.pth"
        )
    wpath = Path(warg).expanduser()
    if not wpath.is_file():
        raise SystemExit(
            f"--weights is not a file: {warg}\n"
            "Expected e.g. runs/rfdetr_stage1/checkpoint_best_total.pth"
        )
    args.weights = wpath.resolve()

    media_dir = args.media_dir or (repo_root() / "data" / "rfdetr" / "infer_media")

    video, srt_path = resolve_video_and_srt(args.video, args.srt, media_dir)

    out = args.out_dir
    if out is None:
        out = repo_root() / "runs" / "rfdetr_infer" / video.stem

    # Backend-aware defaults: RF-DETR Medium = recall; RT-DETR = stricter gates
    if args.backend == "rtdetr":
        conf = 0.5 if args.conf is None else args.conf
        min_overlap = 0.50 if args.min_overlap is None else args.min_overlap
        nms_iou = 0.5 if args.nms_iou is None else args.nms_iou
        require_center = True
        clip_to_mask = True
    else:
        conf = 0.15 if args.conf is None else args.conf
        min_overlap = 0.15 if args.min_overlap is None else args.min_overlap
        nms_iou = 0.5 if args.nms_iou is None else args.nms_iou
        require_center = False
        clip_to_mask = False

    cfg = InferConfig(
        video=video,
        weights=args.weights,
        srt=srt_path,
        out_dir=out,
        conf=conf,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        z_near_m=args.z_near_m,
        z_far_m=args.z_far_m,
        road_top_y=args.road_top_y,
        road_bottom_y=args.road_bottom_y,
        road_bottom_half_w=args.road_bottom_half_w,
        road_top_half_w=args.road_top_half_w,
        road_center_x=args.road_center_x,
        use_classical_road=args.classical_road,
        min_overlap=min_overlap,
        require_center=require_center,
        clip_to_mask=clip_to_mask,
        nms_iou=nms_iou,
        near_wash_alpha=args.near_wash_alpha,
        far_wash_alpha=args.far_wash_alpha,
        camera_height_m=args.camera_height_m,
        camera_pitch_deg=args.camera_pitch_deg,
        vfov_deg=args.vfov_deg,
        crf=args.crf,
        backend=args.backend,
    )
    run_inference(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
