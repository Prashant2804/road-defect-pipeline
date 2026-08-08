"""Rebuild map_trail.html dashboard for an existing infer run (no re-detect)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .export_out import write_defects_json
from .gps_io import load_gps, route_from_track
from .map_trail import write_map_trail


def _load_defects(run_dir: Path) -> list[dict]:
    js = run_dir / "defects.json"
    if js.exists():
        return json.loads(js.read_text(encoding="utf-8"))
    csv_path = run_dir / "defects.csv"
    if not csv_path.exists():
        raise SystemExit(f"No defects.json/csv under {run_dir}")
    import csv

    rows = []
    with csv_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in ("lat", "lon", "chainage_m", "conf", "t_start_s", "t_end_s"):
                if r.get(k) in ("", None):
                    r[k] = None
                else:
                    try:
                        r[k] = float(r[k])
                    except ValueError:
                        pass
            if r.get("defect_id") is not None:
                try:
                    r["defect_id"] = int(float(r["defect_id"]))
                except ValueError:
                    pass
            rows.append(r)
    return rows


def attach_gps_to_defects(rows: list[dict], gps) -> list[dict]:
    if gps is None or not len(gps):
        return rows
    out = []
    for r in rows:
        rr = dict(r)
        t_mid = 0.5 * (float(rr.get("t_start_s") or 0) + float(rr.get("t_end_s") or 0))
        fix = gps.at_time(t_mid)
        if fix is not None:
            rr["lat"] = fix.lat
            rr["lon"] = fix.lon
        ch = gps.distance_at_time(t_mid) if getattr(gps, "has_data", False) else None
        if ch is not None:
            rr["chainage_m"] = round(float(ch), 2)
        out.append(rr)
    return out


def rebuild(
    run_dir: Path,
    *,
    srt: Path | None = None,
    video: Path | None = None,
    maps_api_key: str | None = None,
    z_far_m: float = 5.0,
    title: str = "RF-DETR near-field defects",
) -> Path:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"Not a directory: {run_dir}")

    rows = _load_defects(run_dir)
    route: list[dict] = []
    route_path = run_dir / "route.json"
    if route_path.exists():
        route = json.loads(route_path.read_text(encoding="utf-8"))

    # Prefer re-parsing SRT so older GPS=no runs can be fixed
    vid = video
    if vid is None:
        cand = run_dir / "annotated.mp4"
        vid = cand if cand.exists() else None
    gps = load_gps(vid or run_dir / "annotated.mp4", srt)
    if gps.has_data or len(gps) >= 1:
        route = route_from_track(gps, dt=0.5)
        rows = attach_gps_to_defects(rows, gps)
        route_path.write_text(json.dumps(route, indent=2), encoding="utf-8")
        write_defects_json(run_dir / "defects.json", rows)

    key = maps_api_key or os.environ.get("GOOGLE_MAPS_API_KEY")
    out = write_map_trail(
        run_dir / "map_trail.html",
        route=route,
        defects=rows,
        title=title,
        video_src="annotated.mp4",
        z_far_m=z_far_m,
        maps_api_key=key,
    )
    n_gps = sum(1 for d in rows if d.get("lat") is not None)
    print(f"Wrote {out}")
    print(f"  route points: {len(route)}  defects with GPS: {n_gps}/{len(rows)}")
    print(f"  map backend: {'Google Maps' if key else 'Leaflet (set GOOGLE_MAPS_API_KEY for Google)'}")
    print(f"  open with a local server so video loads, e.g.:")
    print(f"    cd {run_dir} && python3 -m http.server 8765")
    print(f"    then visit http://localhost:8765/map_trail.html")
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rebuild synced video+map dashboard HTML for an existing infer run."
    )
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--srt", type=Path, default=None, help="GoPro/dashcam SRT with GPS")
    p.add_argument("--video", type=Path, default=None, help="Optional source video path")
    p.add_argument(
        "--maps-api-key",
        type=str,
        default=None,
        help="Google Maps JS API key (or set GOOGLE_MAPS_API_KEY)",
    )
    p.add_argument("--z-far", type=float, default=5.0, dest="z_far_m")
    p.add_argument("--title", type=str, default="RF-DETR near-field defects")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rebuild(
        args.run_dir,
        srt=args.srt,
        video=args.video,
        maps_api_key=args.maps_api_key,
        z_far_m=args.z_far_m,
        title=args.title,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
