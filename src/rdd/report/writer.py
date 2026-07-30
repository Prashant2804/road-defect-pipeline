"""Report generation: CSV (per unique track), JSON (counts), HTML/PDF summary.

The report's job is to be *defensible*, which means three things beyond counting:

  * separate "no defect here" from "could not see here" — the unassessable
    fraction of road surface is a headline number, not a footnote;
  * never present a severity as physical when it is only relative to the clip;
  * show what was skipped (unusable frames, off-road rejections) so coverage can
    be judged rather than assumed.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

from ..utils.logging import get_logger

log = get_logger("rdd.report")

_CSV_COLUMNS = [
    "track_id", "class", "first_frame", "last_frame", "first_t_s", "last_t_s",
    "n_frames", "mask_area_px", "area_m2", "severity_level", "severity_score",
    "severity_basis", "assessable", "occluded_frac", "road_overlap",
    "peak_conf", "lat", "lon", "note",
]


def _rows(counter, severity):
    for t in counter.confirmed_tracks():
        rep = t.representative()
        sev = severity.get(t.track_id) if severity is not None else None
        level = sev.level if sev else ""
        indeterminate = bool(sev and sev.is_indeterminate)
        yield {
            "track_id": t.track_id,
            "class": t.cls_name,
            "first_frame": t.first_frame,
            "last_frame": t.last_frame,
            "first_t_s": t.observations[0].t,
            "last_t_s": t.observations[-1].t,
            "n_frames": t.n_frames,
            "mask_area_px": int(t.max_mask_area),
            "area_m2": round(t.max_area_m2, 4) if t.max_area_m2 is not None else "",
            "severity_level": level,
            "severity_score": "" if (sev is None or sev.score is None) else sev.score,
            "severity_basis": sev.basis if sev else "",
            "assessable": "no" if indeterminate else "yes",
            "occluded_frac": round(t.median_occluded_frac, 3),
            "road_overlap": round(t.mean_road_overlap, 3),
            "peak_conf": round(t.peak_conf, 3),
            "lat": rep.lat if rep.lat is not None else "",
            "lon": rep.lon if rep.lon is not None else "",
            "note": sev.reason if sev else "",
        }


def write_csv(counter, severity, out_dir: Path) -> Path:
    import pandas as pd

    rows = list(_rows(counter, severity))
    df = pd.DataFrame(rows, columns=_CSV_COLUMNS)
    path = Path(out_dir) / "defects.csv"
    df.to_csv(path, index=False)
    n_indet = sum(1 for r in rows if r["assessable"] == "no")
    log.info("CSV: %s (%d unique defects, %d indeterminate)", path, len(rows), n_indet)
    return path


def write_json(result, out_dir: Path, severity=None) -> Path:
    unique = result.unique_counts
    counter = result.counter
    occluded = counter.occluded_counts()

    summary = {
        "unique_counts_per_class": unique,
        "unique_total": sum(unique.values()),
        "assessable_total": len(counter.assessable_tracks()),
        "indeterminate_total": sum(occluded.values()),
        "indeterminate_per_class": occluded,
        "raw_per_frame_detections": result.raw_detections,
        "notes": {
            "unique_total": "one physical defect per confirmed track ID",
            "raw_per_frame_detections": "sum of detections across all frames; NOT a defect count",
            "indeterminate": "confirmed defects lying under water/mud — detected but "
                             "not measurable, so deliberately not severity-scored",
        },
    }
    if severity is not None:
        summary["severity"] = {
            "levels": severity.level_counts(),
            "basis": severity.basis,
            "basis_note": severity.scale_note,
        }
    summary["pipeline"] = result.summary()

    path = Path(out_dir) / "summary.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("JSON: %s", path)
    return path


def _extract_crops(annotated_video: Path, counter, severity, cfg, out_dir: Path):
    """Save sample crops from each track's best frame.

    Read sequentially rather than seeking. `CAP_PROP_POS_FRAMES` on long-GOP
    H.264 lands on the nearest keyframe, so the previous seek-per-track approach
    could crop a different frame than the one the detection came from — a crop
    that does not contain the defect it is captioned with.

    Crops come from the *annotated* video so the mask overlay is visible, which
    is what makes them useful for verification.
    """
    import cv2

    rc = cfg.get_path("report.crops", {}) or {}
    if not rc.get("enabled", True):
        return []
    if annotated_video is None or not Path(annotated_video).exists():
        log.warning("Annotated video missing — no report crops")
        return []

    max_per_class = int(rc.get("max_per_class", 6))
    crops_dir = Path(out_dir) / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    by_class: dict[str, list] = {}
    for t in counter.confirmed_tracks():
        by_class.setdefault(t.cls_name, []).append(t)

    wanted: dict[int, list] = {}
    for tracks in by_class.values():
        tracks.sort(key=lambda t: t.max_mask_area, reverse=True)
        for t in tracks[:max_per_class]:
            wanted.setdefault(t.representative().frame, []).append(t)
    if not wanted:
        return []

    pad = int(rc.get("pad_px", 15))
    cap = cv2.VideoCapture(str(annotated_video))
    saved, idx = [], -1
    last_frame = max(wanted)
    while idx < last_frame:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        for t in wanted.get(idx, []):
            rep = t.representative()
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = (int(v) for v in rep.bbox)
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            sev = severity.get(t.track_id) if severity is not None else None
            fpath = crops_dir / f"{t.cls_name}_{t.track_id}.jpg"
            cv2.imwrite(str(fpath), crop)
            saved.append({
                "path": fpath, "cls": t.cls_name, "id": t.track_id,
                "level": sev.level if sev else "",
                "area_m2": t.max_area_m2,
                "indeterminate": bool(sev and sev.is_indeterminate),
            })
    cap.release()
    if len(saved) < sum(len(v) for v in wanted.values()):
        log.warning("Only %d of %d requested crops were readable from %s",
                    len(saved), sum(len(v) for v in wanted.values()),
                    Path(annotated_video).name)
    return saved


def _img_b64(path: Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _crop_figure(c: dict) -> str:
    """One <figure> for a sample crop, captioned with class, severity and size."""
    caption = f"{c['cls']} #{c['id']}"
    if c.get("level"):
        caption += f" · {c['level']}"
    if c.get("area_m2") is not None:
        caption += f" · {c['area_m2']:.2f} m²"
    cls_attr = "indet" if c.get("indeterminate") else ""
    return (
        f"<figure class='{cls_attr}'>"
        f"<img src='data:image/jpeg;base64,{_img_b64(c['path'])}'/>"
        f"<figcaption>{caption}</figcaption></figure>"
    )


def write_report(result, counter, severity, cfg, out_dir: Path,
                 rectified_video: Path | None = None, gps=None,
                 view=None, quality=None) -> Path:
    out_dir = Path(out_dir)
    fmt = cfg.get_path("report.format", "html")
    crops = _extract_crops(result.annotated_video, counter, severity, cfg, out_dir)

    unique = result.unique_counts
    occluded = counter.occluded_counts()
    surface = result.surface

    ctx = {
        "run_name": cfg.get_path("run.name", "default"),
        "unique_total": sum(unique.values()),
        "counts": unique,
        "indeterminate_total": sum(occluded.values()),
        "indeterminate_per_class": occluded,
        "assessable_total": len(counter.assessable_tracks()),
        "raw": result.raw_detections,
        "min_track_len": counter.min_track_len,
        "has_gps": bool(gps and gps.has_data),
        "n_gps": len(gps) if gps else 0,
        "crops": crops,
        "annotated_video": result.annotated_video.name,
        "view": getattr(view, "name", "unknown"),
        "scale_note": result.scale_note,
        "severity_levels": severity.level_counts() if severity else {},
        "severity_basis": severity.basis if severity else "",
        "severity_note": severity.scale_note if severity else "",
        "unassessable_frac": surface.unassessable_frac,
        "water_frac": surface.water_frac,
        "mud_frac": surface.mud_frac,
        "worst_frame_frac": surface.worst_frame_occluded_frac,
        "frames_total": result.frames_total,
        "frames_skipped": result.frames_skipped_quality,
        "skip_reasons": result.quality_skip_reasons,
        "off_road_rejected": counter.rejected_off_road,
        "roadseg": result.roadseg.summary(),
        "quality": quality.summary() if quality else {},
        "enhance": result.enhance_fingerprint,
    }

    if fmt == "pdf":
        return _write_pdf(ctx, out_dir)
    return _write_html(ctx, out_dir)


def _write_html(ctx, out_dir: Path) -> Path:
    rows = "".join(
        f"<tr><td>{k}</td><td class='n'>{v}</td>"
        f"<td class='n'>{ctx['indeterminate_per_class'].get(k, 0)}</td></tr>"
        for k, v in ctx["counts"].items()
    )
    sev_rows = "".join(
        f"<tr><td>{k}</td><td class='n'>{v}</td></tr>"
        for k, v in ctx["severity_levels"].items() if v
    )
    crop_html = "".join(_crop_figure(c) for c in ctx["crops"])
    gps_note = (
        f"<p>GPS: {ctx['n_gps']} fixes — per-defect lat/lon in <code>defects.csv</code>.</p>"
        if ctx["has_gps"] else
        "<p>GPS: none — frame timestamps used as the location proxy.</p>"
    )
    skip_note = (
        f"<li>{ctx['frames_skipped']} of {ctx['frames_total']} frames were too poor "
        f"to analyse ({', '.join(f'{k}: {v}' for k, v in ctx['skip_reasons'].items())}) "
        f"— they are banner-marked in the annotated video.</li>"
        if ctx["frames_skipped"] else
        f"<li>All {ctx['frames_total']} frames passed the quality check.</li>"
    )
    unassessable_pct = ctx["unassessable_frac"] * 100
    banner_class = "bad" if unassessable_pct >= 20 else "warn" if unassessable_pct >= 5 else "ok"

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Road Defect Report — {ctx['run_name']}</title>
<style>
 body{{font-family:system-ui,Segoe UI,Arial;margin:2rem;color:#1a1a1a;max-width:1000px}}
 h1{{margin-bottom:0}} .sub{{color:#666}}
 .big{{font-size:2.4rem;font-weight:700;color:#c0392b;margin:.4rem 0}}
 .row{{display:flex;gap:2rem;flex-wrap:wrap;align-items:flex-start}}
 table{{border-collapse:collapse;margin:1rem 0}} td,th{{border:1px solid #ddd;padding:6px 14px}}
 td.n{{text-align:right;font-variant-numeric:tabular-nums}}
 .grid{{display:flex;flex-wrap:wrap;gap:12px}}
 figure{{margin:0;width:220px}} figure img{{width:100%;border-radius:6px}}
 figure.indet img{{outline:3px solid #f39c12}}
 figcaption{{font-size:.8rem;color:#444}}
 .warn{{background:#fff8e1;border-left:4px solid #f39c12;padding:8px 12px;font-size:.9rem}}
 .bad{{background:#fdecea;border-left:4px solid #c0392b;padding:8px 12px;font-size:.9rem}}
 .ok{{background:#eafaf1;border-left:4px solid #27ae60;padding:8px 12px;font-size:.9rem}}
 .note{{background:#f4f6f8;border-left:4px solid #7f8c8d;padding:8px 12px;font-size:.85rem}}
 ul{{font-size:.9rem}} code{{background:#f4f6f8;padding:1px 4px;border-radius:3px}}
</style></head><body>
<h1>Road Defect Report</h1>
<p class="sub">Run: {ctx['run_name']} · viewpoint: <code>{ctx['view']}</code></p>

<p class="big">{ctx['unique_total']} unique defects</p>
<p>{ctx['assessable_total']} measured · <strong>{ctx['indeterminate_total']} detected but
not measurable</strong> (hidden under water/mud).</p>

<div class="{banner_class}">
<strong>{unassessable_pct:.1f}% of the road surface could not be assessed.</strong>
Water covered {ctx['water_frac'] * 100:.1f}% and mud {ctx['mud_frac'] * 100:.1f}% of the
observed road; the worst single frame was {ctx['worst_frame_frac'] * 100:.0f}% obscured.
Defects underneath standing water or mud are invisible, so an absence of detections
there is <em>not</em> evidence of an intact road.
</div>

<h2>Counts by class</h2>
<table><tr><th>Class</th><th>Unique</th><th>Of which indeterminate</th></tr>{rows}</table>

<h2>Severity</h2>
<table><tr><th>Level</th><th>Defects</th></tr>{sev_rows or '<tr><td colspan=2>none</td></tr>'}</table>
<div class="note"><strong>Basis: {ctx['severity_basis']}.</strong> {ctx['severity_note']}<br>
Ground scale: {ctx['scale_note']}</div>

<h2>Coverage &amp; method</h2>
<ul>
{skip_note}
<li>{ctx['off_road_rejected']} detections were rejected for falling outside the
segmented road surface.</li>
<li>Road mask: {ctx['roadseg'].get('mean_road_coverage', 0) * 100:.1f}% of frame on
average, mean confidence {ctx['roadseg'].get('mean_confidence', 0):.2f},
fell back to the geometric prior on {ctx['roadseg'].get('fallback_frames', 0)} frames.</li>
<li>Unique counts = one physical defect per confirmed track (≥ {ctx['min_track_len']}
frames). Raw per-frame detections: {ctx['raw']} — <em>not</em> a defect count.</li>
<li>Enhancement fingerprint <code>{ctx['enhance']}</code> — the same settings must be
used for labeling and inference.</li>
</ul>
{gps_note}
<p>Annotated video: <code>{ctx['annotated_video']}</code> — road outline, hatched
unassessable areas and per-defect labels are drawn in.</p>

<h2>Sample defects</h2>
<p class="sub">Amber outline = detected but not measurable.</p>
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
    width, height = A4
    y = height - 2 * cm

    def line(text: str, size: int = 11, font: str = "Helvetica", dy: float = 0.55):
        nonlocal y
        if y < 2 * cm:
            c.showPage()
            y = height - 2 * cm
        c.setFont(font, size)
        c.drawString(2 * cm, y, text[:110])
        y -= dy * cm

    line("Road Defect Report", 18, "Helvetica-Bold", 0.8)
    line(f"Run: {ctx['run_name']}   viewpoint: {ctx['view']}", 11)
    y -= 0.3 * cm
    line(f"{ctx['unique_total']} unique defects", 22, "Helvetica-Bold", 0.9)
    line(f"{ctx['assessable_total']} measured, "
         f"{ctx['indeterminate_total']} detected but not measurable", 11)
    y -= 0.2 * cm
    line(f"{ctx['unassessable_frac'] * 100:.1f}% of road surface could not be assessed",
         13, "Helvetica-Bold", 0.7)
    line(f"(water {ctx['water_frac'] * 100:.1f}%, mud {ctx['mud_frac'] * 100:.1f}%; "
         f"worst frame {ctx['worst_frame_frac'] * 100:.0f}%)", 10)
    line("Defects under water/mud are invisible; no detection there is not", 10)
    line("evidence of an intact road.", 10)

    y -= 0.3 * cm
    line("Counts by class", 13, "Helvetica-Bold", 0.6)
    for k, v in ctx["counts"].items():
        indet = ctx["indeterminate_per_class"].get(k, 0)
        line(f"  {k}: {v}" + (f"  ({indet} indeterminate)" if indet else ""), 11)

    y -= 0.3 * cm
    line("Severity", 13, "Helvetica-Bold", 0.6)
    for k, v in ctx["severity_levels"].items():
        if v:
            line(f"  {k}: {v}", 11)
    line(f"  basis: {ctx['severity_basis']}", 10)

    y -= 0.3 * cm
    line("Coverage", 13, "Helvetica-Bold", 0.6)
    line(f"  frames analysed: {ctx['frames_total'] - ctx['frames_skipped']} "
         f"of {ctx['frames_total']}", 10)
    line(f"  detections rejected off-road: {ctx['off_road_rejected']}", 10)
    line(f"  raw per-frame detections: {ctx['raw']} (not a defect count)", 10)
    line(f"  ground scale: {ctx['scale_note']}", 9)
    c.save()
    log.info("PDF report: %s", path)
    return path
