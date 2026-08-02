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


# Speed presets. Each trades detection *density* for throughput; none changes how a
# detection is measured once found.
# Deliberately NOT included: quality.assess.sample_frames. Lowering it looks like a
# free speedup but it changes the learned sharpness threshold — a noisier median from
# fewer samples — and measurably started gating out real frames (30 of 60 on a test
# clip). A preset must not silently change which frames get assessed.
# The dominant cost on long footage is per-frame CPU work (road segmentation, surface
# analysis, optical flow) running while the GPU idles — so the levers that matter are
# "do it on fewer frames", not "make the GPU work harder".
_PRESETS = {
    "fast": {
        "inference.frame_stride": 3,      # ~28 cm of road between frames at 30 km/h
        "roadseg.stride": 3,
        "surface.stride": 3,
        "detect.conditions_stride": 15,
        "validity.traffic.stride": 9,
        "validity.egomotion.work_width": 320,
    },
    "turbo": {
        # For a first look at long footage. Coverage per metre drops noticeably.
        "inference.frame_stride": 8,
        "roadseg.stride": 8,
        "surface.stride": 8,
        "detect.conditions_stride": 40,
        "validity.traffic.enabled": False,
        "validity.egomotion.work_width": 256,
        "inference.imgsz": 768,
    },
    "accurate": {
        "inference.frame_stride": 1,
        "roadseg.stride": 1,
        "surface.stride": 1,
        "detect.conditions_stride": 3,
    },
}


def _overrides(args) -> dict:
    """CLI flags as dotted config overrides."""
    import yaml

    out = {}
    preset = getattr(args, "preset", None)
    if preset:
        if preset not in _PRESETS:
            raise SystemExit(f"Unknown preset {preset!r}; choose from {sorted(_PRESETS)}")
        out.update(_PRESETS[preset])
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
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    view = resolve_view(cfg, w, h)
    seg = build_segmenter(cfg, view)
    out_dir = Path(args.output or "out") / "roadseg_preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    from rdd.quality.enhance import enhance_frame
    from rdd.utils.video import iter_sampled_frames

    print(f"\nviewpoint {view.name}, prior {view.road_prior.kind}", flush=True)
    saved = 0
    # Seeks to the sample points rather than decoding the whole file: a preview of
    # four frames should not cost a full pass over an hour of 4K footage.
    for idx, raw in iter_sampled_frames(args.input, args.n):
        frame = enhance_frame(raw, spec) if spec.enabled else raw
        road = seg.segment(frame)
        surf = analyse_surface(frame, road, cfg)

        vis = draw_surface(draw_road(frame.copy(), road, cfg), surf, cfg)
        path = out_dir / f"preview_{idx:07d}.jpg"
        cv2.imwrite(str(path), vis)
        # Flushed so progress is visible while running under a notebook or a pipe.
        print(f"  frame {idx:6d}  road {road.coverage() * 100:5.1f}% of frame  "
              f"conf {road.confidence:.2f}  "
              f"{'PRIOR-FALLBACK  ' if road.fell_back else ''}"
              f"water {surf.water_frac * 100:4.1f}%  mud {surf.mud_frac * 100:4.1f}%  "
              f"unassessable {surf.occluded_frac * 100:4.1f}%", flush=True)
        saved += 1
    print(f"\n{saved} previews -> {out_dir}")
    print("Green outline = road surface; hatched blue = water, brown = mud.")


def _cmd_validity(args):
    """Print the per-frame assessability timeline — which frames are usable, and why not."""
    import cv2

    from rdd.geometry.autocal import calibrate_video
    from rdd.quality.enhance import enhance_frame, resolve_spec
    from rdd.quality.metrics import assess_video, judge, measure_frame
    from rdd.roadseg.base import build_segmenter
    from rdd.surface.condition import analyse_surface
    from rdd.utils.logging import setup_logging
    from rdd.validity.checker import ValidityChecker
    from rdd.viewpoint import resolve_view

    setup_logging()
    cfg = _load(args)
    if args.no_traffic:
        cfg.set_path("validity.traffic.enabled", False)
    quality = assess_video(args.input, cfg)
    spec = resolve_spec(cfg, quality)

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    view = resolve_view(cfg, w, h)
    calib = calibrate_video(args.input, cfg, view=view, spec=spec)
    seg = build_segmenter(cfg, view)
    checker = ValidityChecker(cfg, camera=calib.camera, zones=calib.zones)

    print(f"\n{'frame':>7} {'t(s)':>7}  {'verdict':<12} reasons")
    print("-" * 78)
    rows, idx = [], -1
    while True:
        ok, raw = cap.read()
        if not ok:
            break
        idx += 1
        if idx % max(1, args.stride):
            continue
        frame = enhance_frame(raw, spec) if spec.enabled else raw
        road = seg.segment(frame)
        surf = analyse_surface(frame, road, cfg)
        q = judge(measure_frame(raw, idx), quality)
        v = checker.check(idx, idx / fps, frame, road=road, surface=surf, quality=q)
        rows.append(v)
        if idx % max(1, args.every) == 0:
            state = "NOT ASSESSED" if v.blocked else ("degraded" if v.degraded else "ok")
            detail = "; ".join(v.block_reasons or v.degrade_reasons) or "-"
            print(f"{idx:>7} {idx / fps:>7.2f}  {state:<12} {detail[:52]}")
    cap.release()

    s = checker.stats.summary()
    print("-" * 78)
    print(f"assessable      : {s['frames_assessable']}/{s['frames']} "
          f"({100 * s['frame_coverage']:.1f}% of frames)")
    print(f"degraded        : {s['frames_degraded']}")
    if s["blocked_by_gate"]:
        print("excluded by     :")
        for gate, n in s["blocked_by_gate"].items():
            print(f"   {gate:<20} {n:>5} frames")
    if s["degraded_by_gate"]:
        print("degraded by     :")
        for gate, n in s["degraded_by_gate"].items():
            print(f"   {gate:<20} {n:>5} frames")
    print(f"longest gap     : {s['longest_unassessed_run_frames']} frames")

    if args.json:
        import json

        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"summary": s, "frames": [v.as_dict() for v in rows]}, indent=2),
            encoding="utf-8")
        print(f"per-frame verdicts -> {path}")


def _cmd_evaluate(args):
    """Measure precision per unique defect and calibrate per-class thresholds.

    Consumes a finished run's `defects.csv` plus a ground-truth file, so it can be
    re-run at different targets without re-running inference.
    """
    import json

    import pandas as pd

    from rdd.eval.precision import certify, load_ground_truth
    from rdd.utils.logging import setup_logging

    setup_logging()
    cfg = _load(args)
    truth = load_ground_truth(args.truth)

    df = pd.read_csv(args.defects)

    class _Track:
        """Just enough of a Track for the matcher."""

        def __init__(self, row):
            self.track_id = int(row["track_id"])
            self.cls_name = str(row["class"])
            self.first_frame = int(row["first_frame"])
            self.last_frame = int(row["last_frame"])
            self.peak_conf = float(row.get("peak_conf", 1.0) or 1.0)

    tracks = [_Track(r) for _, r in df.iterrows()]
    coverage = None
    if args.summary and Path(args.summary).exists():
        s = json.loads(Path(args.summary).read_text(encoding="utf-8"))
        coverage = (s.get("pipeline", {}).get("validity", {}) or {}).get("frame_coverage")

    report = certify(tracks, truth, cfg, route_coverage=coverage)
    print()
    print(report.table())
    print()
    print(f"Certified  : {', '.join(report.certified) or 'none'}")
    print(f"Indicative : {', '.join(report.indicative) or 'none'}")

    out = Path(args.out or "out/calibration.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    import yaml

    out.write_text(yaml.safe_dump({
        "target_precision": report.target_precision,
        "thresholds": report.thresholds(),
        "certified": report.certified,
        "indicative": report.indicative,
        "per_class": {c: v.as_dict() for c, v in report.per_class.items()},
    }, sort_keys=False), encoding="utf-8")
    print(f"\nCalibration -> {out}")
    print("Apply per-class thresholds by setting inference.conf to the lowest, then "
          "filtering the rest in the report.")


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
    sp.add_argument("--preset", default=None, choices=sorted(_PRESETS),
                    help="speed preset: fast (~3x), turbo (~8x, first look), accurate")
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

    sv = sub.add_parser("validity", help="per-frame assessability timeline")
    sv.add_argument("--input", required=True)
    sv.add_argument("--every", type=int, default=5, help="print every Nth frame")
    sv.add_argument("--stride", type=int, default=1,
                    help="check every Nth frame (traffic detection is slow on CPU)")
    sv.add_argument("--no-traffic", action="store_true",
                    help="skip COCO vehicle detection (much faster)")
    sv.add_argument("--json", default=None, help="write per-frame verdicts here")
    _common(sv)
    sv.set_defaults(func=_cmd_validity)

    se = sub.add_parser("evaluate", help="precision per unique defect + thresholds")
    se.add_argument("--defects", required=True, help="defects.csv from a run")
    se.add_argument("--truth", required=True,
                    help="ground truth CSV/JSON: class,first_frame,last_frame")
    se.add_argument("--summary", default=None, help="summary.json, for route coverage")
    se.add_argument("--out", default=None, help="where to write calibration.yaml")
    _common(se, view=False)
    se.set_defaults(func=_cmd_evaluate)

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
