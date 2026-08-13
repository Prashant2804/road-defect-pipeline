"""Compare tape measurements of potholes to pipeline area_m2.

Tape CSV columns (header required):
  defect_id,length_m,width_m
or:
  defect_id,area_m2

Ellipse fill-factor (default 0.8) converts length×width to plan area when
area_m2 is omitted: area ≈ fill * length * width.

Example::

    .venv/bin/python -m tools.rfdetr_infer.validate_area \\
      --tape data/tape_potholes.csv \\
      --defects runs/rfdetr_infer/clip/defects.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .camera import POTHOLE_HIGH_M2, POTHOLE_MEDIUM_M2, pothole_irc_band


def _f(row: dict, *keys: str) -> float | None:
    for k in keys:
        if k in row and str(row[k]).strip() not in {"", "None", "none"}:
            try:
                return float(row[k])
            except ValueError:
                continue
    return None


def load_tape(path: Path, fill: float) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with Path(path).open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            did = row.get("defect_id") or row.get("id")
            if did is None or str(did).strip() == "":
                continue
            tid = int(float(did))
            area = _f(row, "area_m2", "tape_area_m2")
            length = _f(row, "length_m", "length")
            width = _f(row, "width_m", "width")
            if area is None:
                if length is None or width is None:
                    continue
                area = fill * length * width
            out[tid] = {
                "defect_id": tid,
                "tape_area_m2": area,
                "length_m": length,
                "width_m": width,
                "tape_band": pothole_irc_band(area),
            }
    return out


def load_pipeline(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with Path(path).open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            did = row.get("defect_id")
            if did is None or str(did).strip() == "":
                continue
            tid = int(float(did))
            area = _f(row, "area_m2")
            out[tid] = {
                "defect_id": tid,
                "class": row.get("class", ""),
                "area_m2": area,
                "area_source": row.get("area_source", ""),
                "irc_band": row.get("irc_band") or pothole_irc_band(area),
            }
    return out


def compare(tape: dict[int, dict], pipe: dict[int, dict]) -> dict:
    rows = []
    abs_pct = []
    band_ok = 0
    n_band = 0
    for tid, t in sorted(tape.items()):
        p = pipe.get(tid)
        pred = None if p is None else p.get("area_m2")
        truth = t["tape_area_m2"]
        err = None
        ape = None
        if pred is not None and truth > 0:
            err = pred - truth
            ape = abs(err) / truth
            abs_pct.append(ape)
        tb = t["tape_band"]
        pb = None if p is None else p.get("irc_band")
        if tb and pb:
            n_band += 1
            if tb == pb:
                band_ok += 1
        rows.append({
            "defect_id": tid,
            "tape_area_m2": round(truth, 4),
            "pipeline_area_m2": None if pred is None else round(pred, 4),
            "error_m2": None if err is None else round(err, 4),
            "ape": None if ape is None else round(ape, 3),
            "tape_band": tb,
            "pipeline_band": pb,
            "band_match": bool(tb and pb and tb == pb),
            "missing_in_pipeline": p is None,
        })
    n = len(abs_pct)
    mape = sum(abs_pct) / n if n else None
    median_ape = sorted(abs_pct)[n // 2] if n else None
    band_agree = (band_ok / n_band) if n_band else None

    if n == 0:
        rec = "no overlapping tape/pipeline areas — check defect_id alignment"
    elif mape is not None and mape <= 0.30 and (band_agree or 0) >= 0.80:
        rec = (
            "SAM+geometry looks good enough for IRC 3-band pothole area "
            "(MAPE ≤ 30%, band agreement ≥ 80%). Skip polygon labeling for now."
        )
    elif mape is not None and mape > 0.50:
        rec = (
            "Large area error. If pipeline >> tape, SAM is leaking or boxes are "
            "used as masks — inspect area_qa/. If errors are a stable scale "
            "factor, re-measure camera height/pitch. Only then collect polygons."
        )
    else:
        rec = (
            "Mixed. Inspect area_qa overlays. Re-measure height_m if GSD check "
            "failed. Train a small segmenter only if SAM systematically leaks "
            "on this surface."
        )

    return {
        "n_tape": len(tape),
        "n_compared": n,
        "mape": None if mape is None else round(mape, 3),
        "median_ape": None if median_ape is None else round(median_ape, 3),
        "band_agreement": None if band_agree is None else round(band_agree, 3),
        "n_band": n_band,
        "irc_thresholds_m2": {"medium": POTHOLE_MEDIUM_M2, "high": POTHOLE_HIGH_M2},
        "recommendation": rec,
        "rows": rows,
    }


def write_comparison_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "defect_id", "tape_area_m2", "pipeline_area_m2", "error_m2", "ape",
        "tape_band", "pipeline_band", "band_match", "missing_in_pipeline",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_tape_template(path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["defect_id", "length_m", "width_m", "notes"])
        w.writeheader()
        w.writerow({
            "defect_id": "1",
            "length_m": "0.40",
            "width_m": "0.30",
            "notes": "along-road x across-road; fill-factor 0.8 applied",
        })


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tape vs pipeline pothole area.")
    p.add_argument("--tape", type=Path, help="CSV with defect_id + length/width or area_m2")
    p.add_argument("--defects", type=Path, help="Pipeline defects.csv")
    p.add_argument("--fill", type=float, default=0.8,
                   help="length×width fill-factor when tape area_m2 is omitted")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--write-template",
        type=Path,
        default=None,
        help="Write an empty tape CSV and exit",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.write_template is not None:
        write_tape_template(args.write_template)
        print(f"Wrote template {args.write_template}")
        print("Fill 10–20 pothole rows (match defect_id from defects.csv), then re-run.")
        return 0
    if args.tape is None or args.defects is None:
        raise SystemExit("Need --tape and --defects (or --write-template)")
    tape = load_tape(args.tape, args.fill)
    pipe = load_pipeline(args.defects)
    report = compare(tape, pipe)
    out = args.out
    if out is None:
        out = Path(args.defects).parent / "area_validation.json"
    out = Path(out)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_comparison_csv(out.with_suffix(".csv"), report["rows"])
    print(f"compared {report['n_compared']} / {report['n_tape']} tape defects")
    print(f"MAPE={report['mape']}  median APE={report['median_ape']}  "
          f"band agreement={report['band_agreement']}")
    print(report["recommendation"])
    print(f"Wrote {out} and {out.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
