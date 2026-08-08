"""Build synced video+map dashboard into a NEW folder (never mutates the source run)."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

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


def _link_or_copy(src: Path, dst: Path, *, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        print(f"  copying video {src.name} → {dst} (large)")
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(src.resolve(), dst)
        print(f"  symlinked video → {dst.name}")
    except OSError:
        print(f"  symlink failed; copying video {src.name}")
        shutil.copy2(src, dst)


def default_out_dir(run_dir: Path) -> Path:
    """Sibling folder: runs/.../ROAD-1-Gopro-v3_dashboard"""
    return run_dir.parent / f"{run_dir.name}_dashboard"


def rebuild(
    run_dir: Path,
    *,
    out_dir: Path | None = None,
    srt: Path | None = None,
    video: Path | None = None,
    maps_api_key: str | None = None,
    z_far_m: float = 5.0,
    title: str = "RF-DETR near-field defects",
    copy_video: bool = False,
) -> Path:
    """Read-only on ``run_dir``. All writes go to ``out_dir`` (new folder)."""
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Not a directory: {run_dir}")

    out_dir = Path(out_dir).resolve() if out_dir else default_out_dir(run_dir)
    if out_dir.resolve() == run_dir.resolve():
        raise SystemExit(
            "Refusing to write into --run-dir. Pass a different --out-dir "
            "(default is <run-dir>_dashboard)."
        )
    # Never nest inside the POC folder either unless user explicitly set out_dir
    # outside; default is sibling which is fine.

    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_defects(run_dir)
    route: list[dict] = []
    src_route = run_dir / "route.json"
    if src_route.exists():
        route = json.loads(src_route.read_text(encoding="utf-8"))

    vid = Path(video) if video else None
    if vid is None:
        cand = run_dir / "annotated.mp4"
        vid = cand if cand.exists() else None

    gps = load_gps(vid or (run_dir / "annotated.mp4"), srt)
    if len(gps) >= 1:
        route = route_from_track(gps, dt=0.5)
        rows = attach_gps_to_defects(rows, gps)

    # Write ONLY into out_dir
    (out_dir / "defects.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    (out_dir / "route.json").write_text(
        json.dumps(route, indent=2), encoding="utf-8"
    )
    (out_dir / "source_run.txt").write_text(
        f"Built from read-only source:\n{run_dir}\n", encoding="utf-8"
    )

    if vid is not None and vid.exists():
        _link_or_copy(vid, out_dir / "annotated.mp4", copy=copy_video)
    else:
        print("WARNING: no annotated.mp4 found — HTML video panel will be empty")

    key = maps_api_key or os.environ.get("GOOGLE_MAPS_API_KEY")
    html_path = write_map_trail(
        out_dir / "index.html",
        route=route,
        defects=rows,
        title=title,
        video_src="annotated.mp4",
        z_far_m=z_far_m,
        maps_api_key=key,
    )
    # Friendly alias name for Drive browsers
    shutil.copy2(html_path, out_dir / "map_dashboard.html")

    n_gps = sum(1 for d in rows if d.get("lat") is not None)
    print(f"Source run (unchanged): {run_dir}")
    print(f"Dashboard folder:       {out_dir}")
    print(f"  route points: {len(route)}  defects with GPS: {n_gps}/{len(rows)}")
    print(
        f"  map backend: "
        f"{'Google Maps' if key else 'Leaflet (set GOOGLE_MAPS_API_KEY for Google)'}"
    )
    print("  serve:")
    print(f"    cd {out_dir} && python3 -m http.server 8765")
    print("    http://localhost:8765/index.html")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build synced video+map dashboard into a NEW folder. "
            "Never modifies the source --run-dir (POC-safe)."
        )
    )
    p.add_argument("--run-dir", type=Path, required=True, help="Existing infer run (read-only)")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="New folder for dashboard (default: <run-dir>_dashboard sibling)",
    )
    p.add_argument("--srt", type=Path, default=None, help="GoPro/dashcam SRT with GPS")
    p.add_argument("--video", type=Path, default=None, help="Optional annotated.mp4 path")
    p.add_argument(
        "--maps-api-key",
        type=str,
        default=None,
        help="Google Maps JS API key (or set GOOGLE_MAPS_API_KEY)",
    )
    p.add_argument("--z-far", type=float, default=5.0, dest="z_far_m")
    p.add_argument("--title", type=str, default="RF-DETR near-field defects")
    p.add_argument(
        "--copy-video",
        action="store_true",
        help="Copy annotated.mp4 into out-dir (default: symlink; use for Drive upload packs)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rebuild(
        args.run_dir,
        out_dir=args.out_dir,
        srt=args.srt,
        video=args.video,
        maps_api_key=args.maps_api_key,
        z_far_m=args.z_far_m,
        title=args.title,
        copy_video=args.copy_video,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
