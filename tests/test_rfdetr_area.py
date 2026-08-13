"""Defect area from RF-DETR boxes: camera Jacobian, SAM fallback, tape check."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from rdd.utils.geo import GpsFix, GpsTrack
from tools.rfdetr_infer.camera import (
    GOPRO_HFOV_16_9,
    area_map_m2,
    build_camera_model,
    camera_from_infer_cfg,
    check_gsd_with_speed,
    pothole_irc_band,
    speed_mps_at,
    undistort_maps,
)
from tools.rfdetr_infer.config import InferConfig
from tools.rfdetr_infer.export_out import tracks_to_rows
from tools.rfdetr_infer.sam_area import BoxFallbackSegmenter, apply_mask
from tools.rfdetr_infer.track_simple import SimpleTracker, Track
from tools.rfdetr_infer.validate_area import compare, load_pipeline, load_tape, write_tape_template


def test_pothole_irc_bands():
    assert pothole_irc_band(None) is None
    assert pothole_irc_band(0.02) == "low"
    assert pothole_irc_band(0.10) == "medium"
    assert pothole_irc_band(0.50) == "high"


def test_area_map_recovers_a_one_metre_square():
    """A 1×1 m patch on the road must sum to ~1 m², not pixel count."""
    cam = build_camera_model(
        320, 240, height_m=2.0, pitch_deg=45.0, h_fov_deg=70.0
    )
    amap = area_map_m2(cam)
    fl = cam.pixel_from_ground(-0.5, 2.5)
    fr = cam.pixel_from_ground(0.5, 2.5)
    nr = cam.pixel_from_ground(0.5, 1.5)
    nl = cam.pixel_from_ground(-0.5, 1.5)
    assert all(p is not None and np.isfinite(p).all() for p in (fl, fr, nr, nl))
    poly = np.array([fl, fr, nr, nl], dtype=np.int32)
    mask = np.zeros(amap.shape, dtype=np.uint8)
    import cv2

    cv2.fillPoly(mask, [poly], 1)
    area = float(amap[mask.astype(bool)].sum())
    assert 0.75 < area < 1.35, f"expected ~1 m², got {area}"


def test_box_overestimates_compact_blob():
    cam = build_camera_model(
        160, 120, height_m=1.3, pitch_deg=12.0, h_fov_deg=80.0
    )
    amap = area_map_m2(cam)
    h, w = amap.shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = int(0.75 * h), w // 2
    ry, rx = 12, 18
    ellipse = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
    x0, x1 = cx - rx, cx + rx
    y0, y1 = cy - ry, cy + ry
    box = np.zeros((h, w), dtype=bool)
    box[y0:y1, x0:x1] = True
    a_el = float(amap[ellipse].sum())
    a_box = float(amap[box].sum())
    assert a_el > 0 and a_box > a_el
    assert a_box / a_el > 1.15


def test_tracker_keeps_nearest_box():
    trk = SimpleTracker(iou_match=0.1, max_age=5)
    names = ["pothole"]
    # Farther (higher in frame, smaller y2) then nearer (lower, still overlapping)
    trk.update(
        np.array([[40, 20, 90, 70]], dtype=float),
        np.array([0]),
        np.array([0.9]),
        names, 0, 0.0,
    )
    trk.update(
        np.array([[45, 40, 95, 120]], dtype=float),
        np.array([0]),
        np.array([0.4]),
        names, 3, 0.12,
    )
    t = trk.flush()[0]
    assert t.frame_best == 3
    assert t.bbox_best[3] == 120
    assert t.conf_max == 0.9


def test_speed_from_gps():
    gps = GpsTrack(fixes=[
        GpsFix(0.0, 23.0, 77.0),
        GpsFix(1.0, 23.0001, 77.0),  # ~11 m north
        GpsFix(2.0, 23.0002, 77.0),
    ])
    v = speed_mps_at(gps, 1.0)
    assert v is not None and 8 < v < 15


def test_gsd_check_skips_without_gps():
    cam = build_camera_model(160, 120, height_m=1.3, pitch_deg=8.0, h_fov_deg=86.0)
    chk = check_gsd_with_speed(cam, [], GpsTrack(), fps=30.0, z_m=3.0)
    assert chk.ok
    assert "no GPS" in chk.note


def test_undistort_identity_when_k_zero():
    import cv2

    m1, m2 = undistort_maps(64, 48, h_fov_deg=80.0, k1=0.0, k2=0.0)
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    img[20:30, 20:40] = 200
    out = cv2.remap(img, m1, m2, interpolation=cv2.INTER_LINEAR)
    assert img.shape == out.shape
    assert abs(int(img.sum()) - int(out.sum())) / max(int(img.sum()), 1) < 0.05


def test_camera_json_fills_infer_config(tmp_path: Path):
    from tools.rfdetr_infer.camera import apply_camera_json, load_camera_json
    from tools.rfdetr_infer.camera_measure import main as measure_main

    out = tmp_path / "camera.json"
    measure_main([
        "--height-m", "1.32",
        "--pitch-deg", "8",
        "--mode", "linear",
        "--out", str(out),
    ])
    data = load_camera_json(out)
    assert data["height_m"] == 1.32
    assert data["h_fov_deg"] == GOPRO_HFOV_16_9["linear"]
    cfg = InferConfig()
    apply_camera_json(cfg, data)
    cam = camera_from_infer_cfg(cfg, 1920, 1080)
    assert cam is not None
    assert abs(cam.extr.height_m - 1.32) < 1e-9


def test_apply_mask_intersects_near_field():
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    near = np.zeros((40, 60), dtype=bool)
    near[20:40, :] = True
    amap = np.full((40, 60), 0.001, dtype=np.float32)
    tr = Track(
        track_id=7, class_id=3, class_name="pothole",
        bbox=(10, 10, 30, 35), conf=0.8,
        frame_start=1, frame_end=1, t_start_s=0.0, t_end_s=0.0,
    )
    geom, meas = apply_mask(
        frame, tr.bbox, near, amap, BoxFallbackSegmenter(), "box_fallback", tr
    )
    assert geom[15, 20] == False  # above near field
    assert geom[25, 20] == True
    assert meas.area_m2 is not None and meas.area_m2 > 0
    assert meas.irc_band in ("low", "medium", "high")
    assert "overestimate" in meas.note


def test_measure_tracks_writes_area_on_tiny_video(tmp_path: Path):
    import cv2
    from tools.rfdetr_infer.config import InferConfig
    from tools.rfdetr_infer.sam_area import measure_tracks_on_video

    video = tmp_path / "tiny.mp4"
    w, h, fps = 160, 120, 10.0
    wr = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert wr.isOpened()
    for _ in range(5):
        fr = np.zeros((h, w, 3), dtype=np.uint8)
        fr[70:100, 40:90] = (40, 40, 40)
        wr.write(fr)
    wr.release()

    cam = build_camera_model(w, h, height_m=1.3, pitch_deg=12.0, h_fov_deg=80.0)
    amap = area_map_m2(cam)
    tr = Track(
        track_id=1, class_id=3, class_name="pothole",
        bbox=(40, 70, 90, 100), conf=0.8,
        frame_start=2, frame_end=2, t_start_s=0.2, t_end_s=0.2,
        frame_best=2,
    )
    cfg = InferConfig(z_near_m=0.1, z_far_m=20.0, road_top_y=0.2)
    qa = tmp_path / "qa"
    ms = measure_tracks_on_video(
        video, [tr], cfg, amap, BoxFallbackSegmenter(), "box_fallback", qa_dir=qa
    )
    assert len(ms) == 1
    assert ms[0].area_px > 0
    assert ms[0].area_m2 is not None and ms[0].area_m2 > 0
    assert (qa / "defect_0001.jpg").is_file()


def test_tracks_to_rows_include_area():
    tr = Track(
        track_id=1, class_id=3, class_name="pothole",
        bbox=(1, 2, 10, 20), conf=0.5,
        frame_start=0, frame_end=4, t_start_s=0.0, t_end_s=0.4,
    )
    rows = tracks_to_rows(
        [tr], GpsTrack(),
        {1: {"area_px": 100, "area_m2": 0.22, "area_source": "sam",
             "irc_band": "medium", "note": ""}},
    )
    assert rows[0]["area_m2"] == 0.22
    assert rows[0]["irc_band"] == "medium"
    assert rows[0]["frame_best"] == 0


def test_tape_validation_recommends_skip_labeling(tmp_path: Path):
    tape_p = tmp_path / "tape.csv"
    write_tape_template(tape_p)
    # Real tape file
    tape_p.write_text(
        "defect_id,length_m,width_m\n"
        "1,0.40,0.30\n"
        "2,0.80,0.70\n",
        encoding="utf-8",
    )
    def_p = tmp_path / "defects.csv"
    def_p.write_text(
        "defect_id,class,area_m2,irc_band\n"
        "1,pothole,0.096,low\n"
        "2,pothole,0.448,medium\n",
        encoding="utf-8",
    )
    tape = load_tape(tape_p, fill=0.8)
    pipe = load_pipeline(def_p)
    report = compare(tape, pipe)
    assert report["n_compared"] == 2
    assert report["mape"] is not None and report["mape"] < 0.05
    assert "Skip polygon" in report["recommendation"]


def test_tape_validation_flags_large_error(tmp_path: Path):
    tape = {1: {"tape_area_m2": 0.10, "tape_band": "medium"}}
    pipe = {1: {"area_m2": 0.40, "irc_band": "medium"}}
    report = compare(tape, pipe)
    assert report["mape"] is not None and report["mape"] > 0.5
    assert "Large area error" in report["recommendation"]
