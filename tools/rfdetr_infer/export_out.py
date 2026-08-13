"""Write defects.csv / defects.json / summary.json."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .track_simple import Track


CSV_FIELDS = [
    "defect_id",
    "class",
    "conf",
    "t_start_s",
    "t_end_s",
    "frame_start",
    "frame_end",
    "frame_best",
    "hits",
    "lat",
    "lon",
    "chainage_m",
    "bbox_xyxy",
    "area_px",
    "area_m2",
    "area_source",
    "irc_band",
    "area_note",
    "near_frac",
]


def tracks_to_rows(tracks: list[Track], gps, measurements: dict | None = None) -> list[dict]:
    rows = []
    measurements = measurements or {}
    for tr in tracks:
        t_mid = 0.5 * (tr.t_start_s + tr.t_end_s)
        fix = gps.at_time(t_mid) if gps is not None and len(gps) else None
        chainage = (
            gps.distance_at_time(t_mid)
            if gps is not None and getattr(gps, "has_data", False)
            else None
        )
        x1, y1, x2, y2 = tr.bbox_best or tr.bbox
        meas = measurements.get(tr.track_id) or {}
        rows.append(
            {
                "defect_id": tr.track_id,
                "class": tr.class_name,
                "conf": round(tr.conf_max, 4),
                "conf_max": round(tr.conf_max, 4),
                "t_start_s": round(tr.t_start_s, 3),
                "t_end_s": round(tr.t_end_s, 3),
                "frame_start": tr.frame_start,
                "frame_end": tr.frame_end,
                "frame_best": tr.frame_best,
                "hits": tr.hits,
                "lat": None if fix is None else fix.lat,
                "lon": None if fix is None else fix.lon,
                "chainage_m": None if chainage is None else round(float(chainage), 2),
                "bbox_xyxy": f"{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}",
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "area_px": meas.get("area_px", ""),
                "area_m2": meas.get("area_m2", ""),
                "area_source": meas.get("area_source", ""),
                "irc_band": meas.get("irc_band") or "",
                "area_note": meas.get("note", ""),
                "near_frac": meas.get("near_frac", ""),
            }
        )
    return rows


def write_defects_csv(path: Path, rows: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def write_defects_json(path: Path, rows: list[dict]) -> Path:
    path = Path(path)
    clean = []
    for r in rows:
        clean.append({k: v for k, v in r.items() if k != "bbox_xyxy"})
    path.write_text(json.dumps(clean, indent=2), encoding="utf-8")
    return path


def write_summary(path: Path, summary: dict) -> Path:
    path = Path(path)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path
