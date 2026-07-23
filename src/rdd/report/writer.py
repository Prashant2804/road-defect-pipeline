"""Report generation: CSV (per unique track), JSON (counts), HTML/PDF summary.

Consumes the UniqueCounter from inference + optional severity scores. Emits:
  * defects.csv  — one row per CONFIRMED unique track
  * summary.json — counts per class, totals, raw-vs-unique
  * report.html  — counts, annotated crops, GPS map/timeline if available
  (report.pdf if report.format == pdf and reportlab is installed)
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

from ..utils.logging import get_logger

log = get_logger("rdd.report")

_CSV_COLUMNS = [
    "track_id", "class", "first_frame", "last_frame", "first_t_s", "last_t_s",
    "n_frames", "mask_area_px", "severity_level", "severity_score",
    "peak_conf", "lat", "lon",
]


def _rows(counter, severity: dict[int, dict] | None):
    for t in counter.confirmed_tracks():
        rep = t.representative()
        sev = (severity or {}).get(t.track_id, {})
        yield {
            "track_id": t.track_id,
            "class": t.cls_name,
            "first_frame": t.first_frame,
            "last_frame": t.last_frame,
            "first_t_s": t.observations[0].t,
            "last_t_s": t.observations[-1].t,
            "n_frames": t.n_frames,
            "mask_area_px": int(t.max_mask_area),
            "severity_level": sev.get("level", ""),
            "severity_score": sev.get("score", ""),
            "peak_conf": round(t.peak_conf, 3),
            "lat": rep.lat if rep.lat is not None else "",
            "lon": rep.lon if rep.lon is not None else "",
        }


def write_csv(counter, severity, out_dir: Path) -> Path:
    import pandas as pd

    rows = list(_rows(counter, severity))
    df = pd.DataFrame(rows, columns=_CSV_COLUMNS)
    path = out_dir / "defects.csv"
    df.to_csv(path, index=False)
    log.info("CSV: %s (%d unique defects)", path, len(rows))
    return path


def write_json(result, out_dir: Path) -> Path:
    unique = result.unique_counts
    summary = {
        "unique_counts_per_class": unique,
        "unique_total": sum(unique.values()),
        "raw_per_frame_detections": result.raw_detections,
        "note": "unique_total counts one physical defect per confirmed track ID; "
                "raw_per_frame_detections is the sum of detections across all frames "
                "and is NOT a defect count.",
    }
    path = out_dir / "summary.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("JSON: %s", path)
    return path


def _extract_crops(rectified_video: Path, counter, severity, cfg, out_dir: Path):
    """Save a few annotated crops per class from each track's representative frame."""
    import cv2

    rc = cfg.get_path("report.crops", {}) or {}
    if not rc.get("enabled", True) or rectified_video is None or not Path(rectified_video).exists():
        return []
    max_per_class = int(rc.get("max_per_class", 6))
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    by_class: dict[str, list] = {}
    for t in counter.confirmed_tracks():
        by_class.setdefault(t.cls_name, []).append(t)
    chosen = []
    for cls, tracks in by_class.items():
        tracks.sort(key=lambda t: t.max_mask_area, reverse=True)
        chosen.extend(tracks[:max_per_class])

    cap = cv2.VideoCapture(str(rectified_video))
    saved = []
    for t in chosen:
        rep = t.representative()
        cap.set(cv2.CAP_PROP_POS_FRAMES, rep.frame)
        ok, frame = cap.read()
        if not ok:
            continue
        x1, y1, x2, y2 = map(int, rep.bbox)
        pad = 15
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = x2 + pad, y2 + pad
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        sev = (severity or {}).get(t.track_id, {})
        fpath = crops_dir / f"{t.cls_name}_{t.track_id}.jpg"
        cv2.imwrite(str(fpath), crop)
        saved.append({"path": fpath, "cls": t.cls_name, "id": t.track_id,
                      "level": sev.get("level", "")})
    cap.release()
    return saved


def _img_b64(path: Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def write_report(result, counter, severity, cfg, out_dir: Path,
                 rectified_video: Path | None = None, gps=None) -> Path:
    out_dir = Path(out_dir)
    fmt = cfg.get_path("report.format", "html")
    crops = _extract_crops(rectified_video, counter, severity, cfg, out_dir)

    unique = result.unique_counts
    ctx = {
        "run_name": cfg.get_path("run.name", "default"),
        "unique_total": sum(unique.values()),
        "counts": unique,
        "raw": result.raw_detections,
        "min_track_len": counter.min_track_len,
        "has_gps": bool(gps and gps.has_data),
        "n_gps": len(gps) if gps else 0,
        "crops": crops,
        "annotated_video": result.annotated_video.name,
    }

    if fmt == "pdf":
        return _write_pdf(ctx, out_dir)
    return _write_html(ctx, out_dir)


def _write_html(ctx, out_dir: Path) -> Path:
    rows = "".join(
        f"<tr><td>{k}</td><td class='n'>{v}</td></tr>" for k, v in ctx["counts"].items()
    )
    crop_html = "".join(
        f"<figure><img src='data:image/jpeg;base64,{_img_b64(c['path'])}'/>"
        f"<figcaption>{c['cls']} #{c['id']} "
        f"{'· ' + c['level'] if c['level'] else ''}</figcaption></figure>"
        for c in ctx["crops"]
    )
    gps_note = (
        f"<p>GPS: {ctx['n_gps']} fixes — per-defect lat/lon in defects.csv.</p>"
        if ctx["has_gps"] else
        "<p>GPS: none — timestamps used as location proxy.</p>"
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Road Defect Report — {ctx['run_name']}</title>
<style>
 body{{font-family:system-ui,Segoe UI,Arial;margin:2rem;color:#1a1a1a}}
 h1{{margin-bottom:0}} .sub{{color:#666}}
 .big{{font-size:2.4rem;font-weight:700;color:#c0392b}}
 table{{border-collapse:collapse;margin:1rem 0}} td,th{{border:1px solid #ddd;padding:6px 14px}}
 td.n{{text-align:right;font-variant-numeric:tabular-nums}}
 .grid{{display:flex;flex-wrap:wrap;gap:12px}}
 figure{{margin:0;width:220px}} figure img{{width:100%;border-radius:6px}}
 figcaption{{font-size:.8rem;color:#444}}
 .warn{{background:#fff8e1;border-left:4px solid #f39c12;padding:8px 12px;font-size:.9rem}}
</style></head><body>
<h1>Road Defect Report</h1>
<p class="sub">Run: {ctx['run_name']}</p>
<p class="big">{ctx['unique_total']} unique defects</p>
<table><tr><th>Class</th><th>Unique count</th></tr>{rows}</table>
<div class="warn">Unique counts = one physical defect per confirmed track
(≥ {ctx['min_track_len']} frames). Raw per-frame detections: {ctx['raw']}
(NOT a defect count).</div>
{gps_note}
<p>Annotated video: <code>{ctx['annotated_video']}</code></p>
<h2>Sample defects</h2>
<div class="grid">{crop_html or '<p>No crops available.</p>'}</div>
</body></html>"""
    path = out_dir / "report.html"
    path.write_text(html, encoding="utf-8")
    log.info("HTML report: %s", path)
    return path


def _write_pdf(ctx, out_dir: Path) -> Path:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.pdfgen import canvas
    except ImportError:
        log.warning("reportlab not installed — writing HTML instead of PDF")
        return _write_html(ctx, out_dir)

    path = out_dir / "report.pdf"
    c = canvas.Canvas(str(path), pagesize=A4)
    _, height = A4
    y = height - 2 * cm
    c.setFont("Helvetica-Bold", 18)
    c.drawString(2 * cm, y, "Road Defect Report")
    y -= 0.8 * cm
    c.setFont("Helvetica", 11)
    c.drawString(2 * cm, y, f"Run: {ctx['run_name']}")
    y -= 1.0 * cm
    c.setFont("Helvetica-Bold", 22)
    c.drawString(2 * cm, y, f"{ctx['unique_total']} unique defects")
    y -= 1.0 * cm
    c.setFont("Helvetica", 12)
    for k, v in ctx["counts"].items():
        c.drawString(2.5 * cm, y, f"{k}: {v}")
        y -= 0.6 * cm
    y -= 0.4 * cm
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(2 * cm, y, f"Raw per-frame detections: {ctx['raw']} (not a defect count)")
    c.save()
    log.info("PDF report: %s", path)
    return path
