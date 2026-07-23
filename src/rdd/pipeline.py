"""End-to-end orchestration: ingest -> preprocess -> inference -> depth -> report.

Each step is a plain function call into a stage module, so stages remain
independently runnable/testable. Runs without GPS and without depth.
"""
from __future__ import annotations

from pathlib import Path

from .config import Cfg, load_config
from .utils.device import resolve_device
from .utils.logging import setup_logging
from .utils.manifest import Manifest, set_seeds


def run_pipeline(input_path: str, config_path: str, output_dir: str | None = None) -> dict:
    cfg: Cfg = load_config(config_path)
    if output_dir:
        cfg["run"]["output_dir"] = output_dir

    run_name = cfg.get_path("run.name", "default")
    run_dir = Path(cfg.get_path("run.output_dir", "out")) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logging(run_dir / "run.log")
    set_seeds(int(cfg.get_path("run.seed", 0)))
    device = resolve_device(cfg.get_path("run.device", "auto"))
    manifest = Manifest(run_dir, dict(cfg))
    log.info("=== Road Defect Pipeline: run '%s' (device=%s) ===", run_name, device)

    # 1. INGEST ------------------------------------------------------------
    from .ingest.video import ingest_video
    from .ingest.telemetry import extract_telemetry

    ing = ingest_video(input_path, cfg)
    gps = extract_telemetry(ing.video_path, ing.source_path, cfg)
    manifest.record("ingest", input=str(ing.source_path), equirect=str(ing.video_path),
                    converted=ing.was_converted, gps_fixes=len(gps))

    # 2. PREPROCESS --------------------------------------------------------
    from .preprocess.reproject import reproject_video
    from .preprocess.sampling import sample_frames

    rectified = reproject_video(ing.video_path, run_dir / "preprocess", cfg)
    frames_dir = Path(cfg.get_path("preprocess.sampling.frames_dir", "data/rectified"))
    sampled = sample_frames(rectified.video_path, frames_dir, gps, cfg)
    manifest.record("preprocess", rectified=str(rectified.video_path),
                    sampled_frames=len(sampled.frames), sampling_mode=sampled.mode)

    # 3. MODEL -------------------------------------------------------------
    from .model.loader import load_model

    weights = cfg.get_path("inference.weights")
    model = load_model(cfg, weights=weights)
    manifest.record("model", weights=weights or "arch/warm-start default")

    # 4. INFERENCE ---------------------------------------------------------
    from .inference.detect_track import run_inference

    infer = run_inference(rectified.video_path, model, cfg, gps=gps, out_dir=run_dir)
    manifest.record("inference", annotated_video=str(infer.annotated_video),
                    raw_detections=infer.raw_detections, unique=infer.unique_counts)

    # 5. DEPTH (optional) --------------------------------------------------
    from .depth.estimator import estimate_track_depths
    from .depth.severity import score_tracks

    depths = estimate_track_depths(rectified.video_path, infer.counter, cfg)
    severity = score_tracks(infer.counter.confirmed_tracks(), cfg, depths=depths)
    manifest.record("depth", enabled=bool(cfg.get_path("depth.enabled", False)),
                    depth_available=depths is not None)

    # 6. REPORT ------------------------------------------------------------
    from .report.writer import write_csv, write_json, write_report

    outputs = {}
    if cfg.get_path("report.csv", True):
        outputs["csv"] = str(write_csv(infer.counter, severity, run_dir))
    if cfg.get_path("report.json", True):
        outputs["json"] = str(write_json(infer, run_dir))
    outputs["report"] = str(
        write_report(infer, infer.counter, severity, cfg, run_dir,
                     rectified_video=rectified.video_path, gps=gps)
    )
    outputs["annotated_video"] = str(infer.annotated_video)
    manifest.record("report", **outputs)

    manifest.save()
    log.info("=== Done. Unique defects: %s (total %d) ===",
             infer.unique_counts, sum(infer.unique_counts.values()))
    log.info("Artifacts in: %s", run_dir)
    return {"run_dir": str(run_dir), "unique_counts": infer.unique_counts, **outputs}
