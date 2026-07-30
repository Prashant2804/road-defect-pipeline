"""End-to-end orchestration.

    ingest -> viewpoint -> preprocess -> quality -> scale
           -> inference (road seg -> surface -> detect/track/gate)
           -> severity -> report

Each step is a plain function call into a stage module, so stages remain
independently runnable and testable. Runs without GPS, without depth, and without
ground scale — degrading explicitly rather than silently in each case.

Note the ordering: quality is profiled on the *rectified* video, because that is
what the detector consumes, and the enhancement spec derived from it is then used
for both the labeling frames and inference. Road segmentation and surface
analysis live inside the inference loop, since they are per-frame and their
outputs gate detections directly.
"""
from __future__ import annotations

from pathlib import Path

from .config import Cfg, load_config
from .utils.device import resolve_device
from .utils.logging import setup_logging
from .utils.manifest import Manifest, set_seeds


def run_pipeline(input_path: str, config_path: str, output_dir: str | None = None,
                 overrides: dict | None = None) -> dict:
    """Run every stage. `overrides` maps dotted config keys to values (CLI flags)."""
    cfg: Cfg = load_config(config_path)
    for dotted, value in (overrides or {}).items():
        cfg.set_path(dotted, value)
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
    from .ingest.telemetry import extract_telemetry
    from .ingest.video import ingest_video

    ing = ingest_video(input_path, cfg)
    gps = extract_telemetry(ing.video_path, ing.source_path, cfg)
    manifest.record("ingest", input=str(ing.source_path), equirect=str(ing.video_path),
                    converted=ing.was_converted, gps_fixes=len(gps))

    # 2. VIEWPOINT ---------------------------------------------------------
    from .viewpoint import resolve_view

    view = resolve_view(cfg, ing.width, ing.height)
    manifest.record("viewpoint", profile=view.name,
                    input_projection=view.input_projection,
                    needs_reprojection=view.needs_reprojection,
                    road_prior=view.road_prior.kind, notes=list(view.notes))

    # 3. PREPROCESS --------------------------------------------------------
    from .preprocess.reproject import reproject_video

    rectified = reproject_video(ing.video_path, run_dir / "preprocess", cfg, view=view)
    manifest.record("reproject", rectified=str(rectified.video_path),
                    size=[rectified.width, rectified.height],
                    source_size=[rectified.source_width, rectified.source_height],
                    note=rectified.resolution_note)

    # Re-resolve the viewpoint against the frames the detector will actually see:
    # drone GSD depends on the working image width.
    view = resolve_view(cfg, rectified.width or ing.width, rectified.height or ing.height)

    # 4. QUALITY -----------------------------------------------------------
    from .quality.enhance import resolve_spec
    from .quality.metrics import assess_video

    quality = assess_video(rectified.video_path, cfg)
    spec = resolve_spec(cfg, quality)
    log.info("Enhancement: %s (fingerprint %s)", spec.describe(), spec.fingerprint())
    manifest.record("quality", **quality.summary(),
                    enhancement=spec.describe(),
                    enhance_fingerprint=spec.fingerprint())

    # 5. SAMPLING (labeling set; inference uses the full video) -------------
    from .preprocess.sampling import sample_frames

    frames_dir = Path(cfg.get_path("preprocess.sampling.frames_dir", "data/rectified"))
    sampled = sample_frames(rectified.video_path, frames_dir, gps, cfg,
                            profile=quality, spec=spec)
    manifest.record("sampling", **sampled.summary())

    # 6. SCALE -------------------------------------------------------------
    from .preprocess.scale import build_area_scaler

    scaler = build_area_scaler(cfg, view,
                               rectified.width or 0, rectified.height or 0)
    manifest.record("scale", kind=scaler.kind, has_scale=scaler.has_scale,
                    description=scaler.describe())

    # 7. MODEL -------------------------------------------------------------
    from .model.loader import load_model

    weights = cfg.get_path("inference.weights")
    model = load_model(cfg, weights=weights)
    manifest.record("model", weights=weights or "arch/warm-start default")

    # 8. INFERENCE (road seg + surface + detect/track) ----------------------
    from .inference.detect_track import run_inference

    infer = run_inference(rectified.video_path, model, cfg, gps=gps, out_dir=run_dir,
                          view=view, profile=quality, spec=spec, scaler=scaler)
    manifest.record("inference", annotated_video=str(infer.annotated_video),
                    raw_detections=infer.raw_detections, unique=infer.unique_counts,
                    **infer.summary())

    # 9. DEPTH (optional) --------------------------------------------------
    from .depth.estimator import estimate_track_depths

    depths = estimate_track_depths(rectified.video_path, infer.counter, cfg)
    manifest.record("depth", enabled=bool(cfg.get_path("depth.enabled", False)),
                    depth_available=depths is not None)

    # 10. SEVERITY ---------------------------------------------------------
    from .depth.severity import score_tracks

    severity = score_tracks(infer.counter.confirmed_tracks(), cfg, depths=depths,
                            counter=infer.counter)
    manifest.record("severity", basis=severity.basis,
                    levels=severity.level_counts(),
                    indeterminate=severity.n_indeterminate, note=severity.scale_note)

    # 11. REPORT -----------------------------------------------------------
    from .report.writer import write_csv, write_json, write_report

    outputs = {}
    if cfg.get_path("report.csv", True):
        outputs["csv"] = str(write_csv(infer.counter, severity, run_dir))
    if cfg.get_path("report.json", True):
        outputs["json"] = str(write_json(infer, run_dir, severity=severity))
    outputs["report"] = str(
        write_report(infer, infer.counter, severity, cfg, run_dir,
                     rectified_video=rectified.video_path, gps=gps,
                     view=view, quality=quality)
    )
    outputs["annotated_video"] = str(infer.annotated_video)
    manifest.record("report", **outputs)

    manifest.save()

    unique_total = sum(infer.unique_counts.values())
    log.info("=== Done. %d unique defects %s ===", unique_total, infer.unique_counts)
    log.info("    %d measurable, %d indeterminate (under water/mud)",
             len(infer.counter.assessable_tracks()), severity.n_indeterminate)
    log.info("    %.1f%% of road surface unassessable",
             100 * infer.surface.unassessable_frac)
    log.info("Artifacts in: %s", run_dir)

    return {
        "run_dir": str(run_dir),
        "unique_counts": infer.unique_counts,
        "indeterminate": severity.n_indeterminate,
        "unassessable_road_frac": round(infer.surface.unassessable_frac, 4),
        "severity_basis": severity.basis,
        **outputs,
    }
