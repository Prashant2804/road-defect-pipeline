#!/usr/bin/env python
"""Road Defect Detection Pipeline — single CLI entrypoint.

End-to-end (default):
    python run.py --input <video> --config config.yaml --output out/

Individual stages (each runnable on its own):
    python run.py preprocess --input <video> --config config.yaml
    python run.py train      --labels data/labels --config config.yaml
    python run.py infer      --input <rectified.mp4> --config config.yaml
    python run.py annotate   --frames data/rectified --config config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `src/` importable without installation.
sys.path.insert(0, str(Path(__file__).parent / "src"))


def _cmd_run(args):
    from rdd.pipeline import run_pipeline

    out = run_pipeline(args.input, args.config, args.output)
    print("\n=== RESULT ===")
    print(f"Unique defects: {out['unique_counts']}")
    print(f"Artifacts: {out['run_dir']}")


def _cmd_preprocess(args):
    from rdd.config import load_config
    from rdd.ingest.telemetry import extract_telemetry
    from rdd.ingest.video import ingest_video
    from rdd.preprocess.reproject import reproject_video
    from rdd.preprocess.sampling import sample_frames
    from rdd.utils.logging import setup_logging

    setup_logging()
    cfg = load_config(args.config)
    ing = ingest_video(args.input, cfg)
    gps = extract_telemetry(ing.video_path, ing.source_path, cfg)
    out_dir = Path(args.output or "out") / cfg.get_path("run.name", "default") / "preprocess"
    rect = reproject_video(ing.video_path, out_dir, cfg)
    sampled = sample_frames(rect.video_path, cfg.get_path("preprocess.sampling.frames_dir"), gps, cfg)
    print(f"Rectified: {rect.video_path}\nSampled {len(sampled.frames)} frames ({sampled.mode})")


def _cmd_train(args):
    from rdd.config import load_config
    from rdd.model.train import train
    from rdd.utils.logging import setup_logging
    from rdd.utils.manifest import set_seeds

    setup_logging()
    cfg = load_config(args.config)
    set_seeds(int(cfg.get_path("run.seed", 0)))
    best = train(cfg, labels_root=args.labels, fps=args.fps)
    print(f"Best weights: {best}")


def _cmd_infer(args):
    from rdd.config import load_config
    from rdd.inference.detect_track import run_inference
    from rdd.model.loader import load_model
    from rdd.report.writer import write_csv, write_json, write_report
    from rdd.utils.logging import setup_logging

    setup_logging()
    cfg = load_config(args.config)
    out_dir = Path(args.output or "out") / cfg.get_path("run.name", "default")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(cfg, weights=args.weights or cfg.get_path("inference.weights"))
    res = run_inference(args.input, model, cfg, out_dir=out_dir)
    from rdd.depth.severity import score_tracks

    sev = score_tracks(res.counter.confirmed_tracks(), cfg)
    write_csv(res.counter, sev, out_dir)
    write_json(res, out_dir)
    write_report(res, res.counter, sev, cfg, out_dir, rectified_video=Path(args.input))
    print(f"Unique defects: {res.unique_counts}\nArtifacts: {out_dir}")


def _cmd_annotate(args):
    from rdd.annotate.frame_picker import pick_frames
    from rdd.config import load_config
    from rdd.utils.logging import setup_logging

    setup_logging()
    cfg = load_config(args.config)
    frames = sorted(Path(args.frames).glob("*.jpg"))
    picked = pick_frames(frames, cfg)
    out = Path(args.output or "out") / "frames_to_label.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(str(p) for p in picked), encoding="utf-8")
    print(f"Picked {len(picked)} frames to label first -> {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Road Defect Detection Pipeline")
    p.add_argument("--input", help="input video (equirect .mp4 or Insta360 .insv/.insp)")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--output", default="out")
    p.set_defaults(func=_cmd_run)

    sub = p.add_subparsers(dest="command")

    sp = sub.add_parser("preprocess", help="360->flat + sampling only")
    sp.add_argument("--input", required=True)
    sp.add_argument("--config", default="config.yaml")
    sp.add_argument("--output", default="out")
    sp.set_defaults(func=_cmd_preprocess)

    st = sub.add_parser("train", help="fine-tune on labels (segment-split)")
    st.add_argument("--labels", default=None)
    st.add_argument("--config", default="config.yaml")
    st.add_argument("--fps", type=float, default=30.0)
    st.set_defaults(func=_cmd_train)

    si = sub.add_parser("infer", help="detect+track+report on a rectified video")
    si.add_argument("--input", required=True)
    si.add_argument("--weights", default=None)
    si.add_argument("--config", default="config.yaml")
    si.add_argument("--output", default="out")
    si.set_defaults(func=_cmd_infer)

    sa = sub.add_parser("annotate", help="active-learning frame picker")
    sa.add_argument("--frames", required=True)
    sa.add_argument("--config", default="config.yaml")
    sa.add_argument("--output", default="out")
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
