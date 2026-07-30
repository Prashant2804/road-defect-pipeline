#!/usr/bin/env python
"""Road Defect Detection Pipeline — single CLI entrypoint.

End-to-end (default):
    python run.py --input <video> --config config.yaml --output out/

Individual stages (each runnable on its own):
    python run.py preprocess --input <video> --config config.yaml
    python run.py quality    --input <video> --config config.yaml
    python run.py roadseg    --input <video> --config config.yaml   # mask preview
    python run.py annotate   --frames data/rectified --config config.yaml
    python run.py train      --labels data/labels --config config.yaml
    python run.py infer      --input <rectified.mp4> --config config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `src/` importable without installation.
sys.path.insert(0, str(Path(__file__).parent / "src"))


def _overrides(args) -> dict:
    """CLI flags as dotted config overrides."""
    import yaml

    out = {}
    if getattr(args, "view", None):
        out["view.profile"] = args.view
    if getattr(args, "device", None):
        out["run.device"] = args.device
    for item in getattr(args, "set", None) or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, _, raw = item.partition("=")
        # Parse through YAML so numbers, booleans and lists keep their types.
        out[key.strip()] = yaml.safe_load(raw)
    return out


def _load(args):
    """Load config, applying CLI overrides."""
    from rdd.config import load_config

    cfg = load_config(args.config)
    for dotted, value in _overrides(args).items():
        cfg.set_path(dotted, value)
    return cfg


def _cmd_run(args):
    from rdd.pipeline import run_pipeline

    out = run_pipeline(args.input, args.config, args.output,
                       overrides=_overrides(args))
    print("\n=== RESULT ===")
    print(f"Unique defects        : {out['unique_counts']}")
    print(f"Indeterminate         : {out['indeterminate']} (hidden under water/mud)")
    print(f"Road unassessable     : {out['unassessable_road_frac'] * 100:.1f}%")
    print(f"Severity basis        : {out['severity_basis']}")
    print(f"Artifacts             : {out['run_dir']}")


def _cmd_preprocess(args):
    from rdd.ingest.telemetry import extract_telemetry
    from rdd.ingest.video import ingest_video
    from rdd.preprocess.reproject import reproject_video
    from rdd.preprocess.sampling import sample_frames
    from rdd.quality.enhance import resolve_spec
    from rdd.quality.metrics import assess_video
    from rdd.utils.logging import setup_logging
    from rdd.viewpoint import resolve_view

    setup_logging()
    cfg = _load(args)
    ing = ingest_video(args.input, cfg)
    gps = extract_telemetry(ing.video_path, ing.source_path, cfg)
    view = resolve_view(cfg, ing.width, ing.height)
    out_dir = Path(args.output or "out") / cfg.get_path("run.name", "default") / "preprocess"
    rect = reproject_video(ing.video_path, out_dir, cfg, view=view)

    quality = assess_video(rect.video_path, cfg)
    spec = resolve_spec(cfg, quality)
    sampled = sample_frames(rect.video_path,
                            cfg.get_path("preprocess.sampling.frames_dir"),
                            gps, cfg, profile=quality, spec=spec)

    print(f"Rectified   : {rect.video_path} ({rect.width}x{rect.height})")
    print(f"  {rect.resolution_note}")
    print(f"Quality     : {quality.summary()}")
    print(f"Enhancement : {spec.describe()}  [fingerprint {spec.fingerprint()}]")
    print(f"Sampled     : {len(sampled.frames)} frames ({sampled.mode}), "
          f"{sampled.skipped_unusable} skipped on quality")


def _cmd_quality(args):
    from rdd.quality.enhance import resolve_spec
    from rdd.quality.metrics import assess_video, judge
    from rdd.utils.logging import setup_logging

    setup_logging()
    cfg = _load(args)
    profile = assess_video(args.input, cfg)
    judged = [judge(s, profile) for s in profile.samples]
    unusable = [s for s in judged if not s.usable]

    print(f"\nSampled {profile.n_sampled} frames from {args.input}")
    print(f"  sharpness  median {profile.sharpness_median:8.1f}  "
          f"drop below {profile.sharpness_thresh:.1f}")
    print(f"  contrast   median {profile.contrast_median:8.4f}")
    print(f"  noise      median {profile.noise_median:8.2f}")
    print(f"  unusable   {len(unusable)}/{len(judged)} sampled frames")
    for s in unusable[:12]:
        print(f"    frame {s.index:7d}: {', '.join(s.reasons)}")
    if len(unusable) > 12:
        print(f"    ... and {len(unusable) - 12} more")
    print(f"\nEnhancement that would be applied: {resolve_spec(cfg, profile).describe()}")

    if args.csv:
        import pandas as pd

        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([s.as_row() for s in judged]).to_csv(path, index=False)
        print(f"Per-frame metrics -> {path}")


def _cmd_roadseg(args):
    """Render road + surface masks on a few frames so thresholds can be tuned by eye."""
    import cv2

    from rdd.inference.render import draw_road, draw_surface
    from rdd.quality.enhance import resolve_spec
    from rdd.quality.metrics import assess_video
    from rdd.roadseg.base import build_segmenter
    from rdd.surface.condition import analyse_surface
    from rdd.utils.logging import setup_logging
    from rdd.viewpoint import resolve_view

    setup_logging()
    cfg = _load(args)
    quality = assess_video(args.input, cfg)
    spec = resolve_spec(cfg, quality)

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.input}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    view = resolve_view(cfg, w, h)
    seg = build_segmenter(cfg, view)
    out_dir = Path(args.output or "out") / "roadseg_preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    stride = max(1, (total // max(1, args.n)) if total else 1)
    from rdd.quality.enhance import enhance_frame

    idx, saved = -1, 0
    print(f"\nviewpoint {view.name}, prior {view.road_prior.kind}")
    while saved < args.n:
        ok, raw = cap.read()
        if not ok:
            break
        idx += 1
        if idx % stride:
            continue
        frame = enhance_frame(raw, spec) if spec.enabled else raw
        road = seg.segment(frame)
        surf = analyse_surface(frame, road, cfg)

        vis = draw_surface(draw_road(frame.copy(), road, cfg), surf, cfg)
        path = out_dir / f"preview_{idx:07d}.jpg"
        cv2.imwrite(str(path), vis)
        print(f"  frame {idx:6d}  road {road.coverage() * 100:5.1f}% of frame  "
              f"conf {road.confidence:.2f}  "
              f"{'PRIOR-FALLBACK  ' if road.fell_back else ''}"
              f"water {surf.water_frac * 100:4.1f}%  mud {surf.mud_frac * 100:4.1f}%  "
              f"unassessable {surf.occluded_frac * 100:4.1f}%")
        saved += 1
    cap.release()
    print(f"\n{saved} previews -> {out_dir}")
    print("Green outline = road surface; hatched blue = water, brown = mud.")


def _cmd_train(args):
    from rdd.model.train import train
    from rdd.utils.logging import setup_logging
    from rdd.utils.manifest import set_seeds

    setup_logging()
    cfg = _load(args)
    set_seeds(int(cfg.get_path("run.seed", 0)))
    best = train(cfg, labels_root=args.labels, fps=args.fps)
    print(f"Best weights: {best}")


def _cmd_infer(args):
    from rdd.depth.severity import score_tracks
    from rdd.ingest.telemetry import extract_telemetry
    from rdd.inference.detect_track import run_inference
    from rdd.model.loader import load_model
    from rdd.preprocess.scale import build_area_scaler
    from rdd.quality.enhance import resolve_spec
    from rdd.quality.metrics import assess_video
    from rdd.report.writer import write_csv, write_json, write_report
    from rdd.utils.logging import setup_logging
    from rdd.viewpoint import resolve_view

    setup_logging()
    cfg = _load(args)
    out_dir = Path(args.output or "out") / cfg.get_path("run.name", "default")
    out_dir.mkdir(parents=True, exist_ok=True)

    src = Path(args.input)
    gps = extract_telemetry(src, src, cfg)
    quality = assess_video(src, cfg)
    spec = resolve_spec(cfg, quality)

    import cv2

    cap = cv2.VideoCapture(str(src))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    view = resolve_view(cfg, w, h)
    scaler = build_area_scaler(cfg, view, w, h)
    model = load_model(cfg, weights=args.weights or cfg.get_path("inference.weights"))

    res = run_inference(src, model, cfg, gps=gps, out_dir=out_dir, view=view,
                        profile=quality, spec=spec, scaler=scaler)
    sev = score_tracks(res.counter.confirmed_tracks(), cfg, counter=res.counter)

    write_csv(res.counter, sev, out_dir)
    write_json(res, out_dir, severity=sev)
    write_report(res, res.counter, sev, cfg, out_dir, rectified_video=src,
                 gps=gps, view=view, quality=quality)
    print(f"Unique defects     : {res.unique_counts}")
    print(f"Indeterminate      : {sev.n_indeterminate}")
    print(f"Road unassessable  : {res.surface.unassessable_frac * 100:.1f}%")
    print(f"Artifacts          : {out_dir}")


def _cmd_annotate(args):
    from rdd.annotate.frame_picker import pick_frames
    from rdd.utils.logging import setup_logging

    setup_logging()
    cfg = _load(args)
    frames = sorted(Path(args.frames).glob("*.jpg"))
    if not frames:
        raise SystemExit(f"No .jpg frames in {args.frames} — run preprocess first.")
    picked = pick_frames(frames, cfg)
    out = Path(args.output or "out") / "frames_to_label.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(str(p) for p in picked), encoding="utf-8")
    print(f"Picked {len(picked)} frames to label first -> {out}")


def _common(sp, *, view: bool = True) -> None:
    sp.add_argument("--config", default="config.yaml")
    sp.add_argument("--output", default="out")
    sp.add_argument("--device", default=None, help="override run.device")
    sp.add_argument("--set", action="append", metavar="KEY=VALUE", default=None,
                    help="override any config key, e.g. "
                         "--set view.drone.gsd_m_per_px=0.02 (repeatable)")
    if view:
        sp.add_argument("--view", default=None,
                        choices=["car_360", "car_flat", "drone_nadir"],
                        help="override view.profile")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Road Defect Detection Pipeline")
    p.add_argument("--input", help="input video (equirect .mp4 or Insta360 .insv/.insp)")
    _common(p)
    p.set_defaults(func=_cmd_run)

    sub = p.add_subparsers(dest="command")

    sp = sub.add_parser("preprocess", help="reproject + quality + sampling")
    sp.add_argument("--input", required=True)
    _common(sp)
    sp.set_defaults(func=_cmd_preprocess)

    sq = sub.add_parser("quality", help="report video quality; no processing")
    sq.add_argument("--input", required=True)
    sq.add_argument("--csv", default=None, help="write per-frame metrics here")
    _common(sq)
    sq.set_defaults(func=_cmd_quality)

    sr = sub.add_parser("roadseg", help="preview road + surface masks for tuning")
    sr.add_argument("--input", required=True)
    sr.add_argument("--n", type=int, default=8, help="how many preview frames")
    _common(sr)
    sr.set_defaults(func=_cmd_roadseg)

    st = sub.add_parser("train", help="fine-tune on labels (segment-split)")
    st.add_argument("--labels", default=None)
    st.add_argument("--fps", type=float, default=30.0)
    _common(st, view=False)
    st.set_defaults(func=_cmd_train)

    si = sub.add_parser("infer", help="detect+track+report on a rectified video")
    si.add_argument("--input", required=True)
    si.add_argument("--weights", default=None)
    _common(si)
    si.set_defaults(func=_cmd_infer)

    sa = sub.add_parser("annotate", help="active-learning frame picker")
    sa.add_argument("--frames", required=True)
    _common(sa, view=False)
    sa.set_defaults(func=_cmd_annotate)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None and not args.input:
        parser.error("--input is required for the end-to-end run (or use a subcommand)")
    args.func(args)


if __name__ == "__main__":
    main()
