#!/usr/bin/env python3
"""Summarize Ultralytics RT-DETR training: precision, recall, mAP, losses.

Reads ``runs/rtdetr_stage2/results.csv`` (and related artifacts). Optionally
re-validates ``weights/best.pt`` for a fresh metrics dump.

Usage on the VM::

    ./scripts/check_rtdetr_run.sh
    ./scripts/check_rtdetr_run.sh --run-dir runs/rtdetr_stage2
    ./scripts/check_rtdetr_run.sh --val          # re-run val on best.pt
    ./scripts/check_rtdetr_run.sh --json-out runs/rtdetr_stage2/metrics_report.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import repo_root
from .taxonomy import CLASS_NAMES


def _fmt_bytes(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if x < 1024:
            return f"{x:.1f} {unit}"
        x /= 1024.0
    return f"{x:.1f} TB"


def _mtime(p: Path) -> str:
    return datetime.fromtimestamp(p.stat().st_mtime).isoformat(sep=" ", timespec="seconds")


def _f(v: Any) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        x = float(s)
    except ValueError:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _pick_col(headers: list[str], *needles: str) -> str | None:
    lower = {h: h.lower().replace(" ", "") for h in headers}
    for needle in needles:
        n = needle.lower().replace(" ", "")
        for h, hl in lower.items():
            if n == hl or n in hl:
                return h
    return None


def load_args_yaml(run_dir: Path) -> dict:
    path = run_dir / "args.yaml"
    if not path.is_file():
        return {}
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        # Minimal fallback: key: value lines
        out: dict = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" not in line or line.strip().startswith("#"):
                continue
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
        return out


def load_results_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [{k.strip(): (v.strip() if v is not None else "") for k, v in row.items() if k}
                for row in reader]
        headers = [h.strip() for h in (reader.fieldnames or [])]
    return headers, rows


def summarize_curve(headers: list[str], rows: list[dict[str, str]]) -> dict:
    if not rows:
        return {"epochs_logged": 0}

    col_epoch = _pick_col(headers, "epoch") or "epoch"
    col_prec = _pick_col(headers, "metrics/precision(B)", "metrics/precision", "precision")
    col_rec = _pick_col(headers, "metrics/recall(B)", "metrics/recall", "recall")
    col_map50 = _pick_col(headers, "metrics/mAP50(B)", "metrics/mAP50", "mAP50")
    col_map = _pick_col(
        headers, "metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95", "mAP"
    )
    col_tbox = _pick_col(headers, "train/box_loss", "train/giou_loss", "train/loss")
    col_vbox = _pick_col(headers, "val/box_loss", "val/giou_loss", "val/loss")
    col_time = _pick_col(headers, "time")

    def row_metrics(r: dict[str, str]) -> dict:
        return {
            "epoch": int(_f(r.get(col_epoch)) or 0),
            "precision": _f(r.get(col_prec)) if col_prec else None,
            "recall": _f(r.get(col_rec)) if col_rec else None,
            "mAP50": _f(r.get(col_map50)) if col_map50 else None,
            "mAP50_95": _f(r.get(col_map)) if col_map else None,
            "train_box_loss": _f(r.get(col_tbox)) if col_tbox else None,
            "val_box_loss": _f(r.get(col_vbox)) if col_vbox else None,
            "time_s": _f(r.get(col_time)) if col_time else None,
        }

    series = [row_metrics(r) for r in rows]
    last = series[-1]

    def best_by(key: str) -> dict | None:
        cand = [s for s in series if s.get(key) is not None]
        if not cand:
            return None
        return max(cand, key=lambda s: float(s[key]))  # type: ignore[arg-type]

    best_map = best_by("mAP50_95") or best_by("mAP50")
    best_prec = best_by("precision")
    best_rec = best_by("recall")

    return {
        "epochs_logged": len(series),
        "columns_used": {
            "precision": col_prec,
            "recall": col_rec,
            "mAP50": col_map50,
            "mAP50_95": col_map,
            "train_box_loss": col_tbox,
            "val_box_loss": col_vbox,
        },
        "last": last,
        "best_by_mAP": best_map,
        "best_by_precision": best_prec,
        "best_by_recall": best_rec,
        "series_tail": series[-5:],
    }


def list_weight_files(run_dir: Path) -> list[dict]:
    wdir = run_dir / "weights"
    out = []
    for name in ("best.pt", "last.pt"):
        p = wdir / name
        if p.is_file():
            out.append(
                {
                    "name": name,
                    "path": str(p.resolve()),
                    "size": _fmt_bytes(p.stat().st_size),
                    "mtime": _mtime(p),
                }
            )
    return out


def list_plots(run_dir: Path) -> list[str]:
    names = (
        "results.png",
        "results.csv",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "BoxPR_curve.png",
        "BoxF1_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
        "labels.jpg",
        "train_batch0.jpg",
        "val_batch0_pred.jpg",
        "val_batch0_labels.jpg",
    )
    return [n for n in names if (run_dir / n).exists()]


def run_ultralytics_val(
    weights: Path,
    data_yaml: Path | None,
    imgsz: int,
    device: str,
    batch: int,
) -> dict:
    from ultralytics import RTDETR

    model = RTDETR(str(weights))
    kwargs: dict[str, Any] = {
        "imgsz": imgsz,
        "device": device,
        "batch": batch,
        "plots": True,
        "verbose": True,
    }
    if data_yaml is not None and data_yaml.is_file():
        kwargs["data"] = str(data_yaml)

    print(f"\n==> Re-validating {weights} …")
    metrics = model.val(**kwargs)
    box = getattr(metrics, "box", None)
    out: dict[str, Any] = {"weights": str(weights.resolve())}
    if box is not None:
        out["precision"] = float(getattr(box, "mp", float("nan")))
        out["recall"] = float(getattr(box, "mr", float("nan")))
        out["mAP50"] = float(getattr(box, "map50", float("nan")))
        out["mAP50_95"] = float(getattr(box, "map", float("nan")))
        # Per-class if available
        names = getattr(metrics, "names", None) or {
            i: n for i, n in enumerate(CLASS_NAMES)
        }
        ap50 = getattr(box, "ap50", None)
        ap = getattr(box, "ap", None)
        p = getattr(box, "p", None)
        r = getattr(box, "r", None)
        per_class = []
        n_cls = 0
        if ap50 is not None:
            try:
                n_cls = len(ap50)
            except TypeError:
                n_cls = 0
        for i in range(n_cls):
            cname = names.get(i, CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i))
            entry: dict[str, Any] = {"id": i, "name": cname}
            try:
                entry["precision"] = float(p[i]) if p is not None else None
                entry["recall"] = float(r[i]) if r is not None else None
                entry["mAP50"] = float(ap50[i]) if ap50 is not None else None
                entry["mAP50_95"] = float(ap[i]) if ap is not None else None
            except Exception:
                pass
            per_class.append(entry)
        if per_class:
            out["per_class"] = per_class
    # Speed
    speed = getattr(metrics, "speed", None)
    if isinstance(speed, dict):
        out["speed_ms"] = {k: float(v) for k, v in speed.items()}
    return out


def build_report(
    run_dir: Path,
    *,
    do_val: bool = False,
    data_yaml: Path | None = None,
    device: str = "0",
    batch: int = 8,
) -> dict:
    run_dir = Path(run_dir).resolve()
    report: dict[str, Any] = {
        "run_dir": str(run_dir),
        "generated_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "exists": run_dir.is_dir(),
        "class_names": list(CLASS_NAMES),
    }
    if not run_dir.is_dir():
        report["error"] = f"Run directory not found: {run_dir}"
        return report

    args = load_args_yaml(run_dir)
    report["train_args"] = {
        k: args.get(k)
        for k in (
            "model",
            "data",
            "epochs",
            "batch",
            "imgsz",
            "device",
            "workers",
            "lr0",
            "lrf",
            "optimizer",
            "patience",
            "cache",
            "resume",
            "project",
            "name",
        )
        if k in args
    }

    results_csv = run_dir / "results.csv"
    report["results_csv"] = str(results_csv) if results_csv.is_file() else None
    if results_csv.is_file():
        headers, rows = load_results_csv(results_csv)
        report["metrics_from_csv"] = summarize_curve(headers, rows)
    else:
        report["metrics_from_csv"] = None
        report["warning"] = "results.csv missing — training may still be running"

    report["weights"] = list_weight_files(run_dir)
    report["artifacts"] = list_plots(run_dir)

    if do_val:
        weights = run_dir / "weights" / "best.pt"
        if not weights.is_file():
            weights = run_dir / "weights" / "last.pt"
        if not weights.is_file():
            report["val"] = {"error": "No best.pt / last.pt to validate"}
        else:
            data = data_yaml
            if data is None:
                raw = args.get("data")
                data = Path(raw) if raw else None
            imgsz = int(args.get("imgsz") or 640)
            report["val"] = run_ultralytics_val(
                weights, data, imgsz=imgsz, device=device, batch=batch
            )

    return report


def _pct(x: float | None) -> str:
    if x is None:
        return "—"
    # Ultralytics stores 0–1; also tolerate 0–100
    v = float(x)
    if v > 1.5:
        return f"{v:.2f}%"
    return f"{100.0 * v:.2f}%"


def _num(x: float | None, digits: int = 4) -> str:
    if x is None:
        return "—"
    return f"{float(x):.{digits}f}"


def print_report(report: dict) -> None:
    print("=" * 72)
    print("RT-DETR training metrics report")
    print("=" * 72)
    print(f"run_dir:      {report.get('run_dir')}")
    print(f"generated_at: {report.get('generated_at')}")
    if report.get("error"):
        print(f"ERROR: {report['error']}")
        return

    args = report.get("train_args") or {}
    if args:
        print("\n--- Train config ---")
        for k, v in args.items():
            print(f"  {k}: {v}")

    weights = report.get("weights") or []
    print("\n--- Weights ---")
    if not weights:
        print("  (none yet)")
    for w in weights:
        print(f"  {w['name']}: {w['size']}  mtime={w['mtime']}")
        print(f"    {w['path']}")

    arts = report.get("artifacts") or []
    if arts:
        print("\n--- Artifacts present ---")
        print("  " + ", ".join(arts))

    m = report.get("metrics_from_csv")
    print("\n--- Metrics from results.csv (validation each epoch) ---")
    if not m:
        print("  results.csv not found.")
        if report.get("warning"):
            print(f"  {report['warning']}")
    else:
        print(f"  epochs logged: {m.get('epochs_logged')}")
        last = m.get("last") or {}
        best = m.get("best_by_mAP") or {}
        print("\n  Last epoch:")
        print(
            f"    epoch={last.get('epoch')}  "
            f"precision={_pct(last.get('precision'))}  "
            f"recall={_pct(last.get('recall'))}  "
            f"mAP50={_pct(last.get('mAP50'))}  "
            f"mAP50-95={_pct(last.get('mAP50_95'))}"
        )
        print(
            f"    train_box_loss={_num(last.get('train_box_loss'))}  "
            f"val_box_loss={_num(last.get('val_box_loss'))}"
        )
        if best:
            print("\n  Best by mAP50-95 (fallback mAP50):")
            print(
                f"    epoch={best.get('epoch')}  "
                f"precision={_pct(best.get('precision'))}  "
                f"recall={_pct(best.get('recall'))}  "
                f"mAP50={_pct(best.get('mAP50'))}  "
                f"mAP50-95={_pct(best.get('mAP50_95'))}"
            )
        bp = m.get("best_by_precision")
        br = m.get("best_by_recall")
        if bp:
            print(
                f"  Best precision: epoch={bp.get('epoch')}  "
                f"precision={_pct(bp.get('precision'))}"
            )
        if br:
            print(
                f"  Best recall:    epoch={br.get('epoch')}  "
                f"recall={_pct(br.get('recall'))}"
            )
        tail = m.get("series_tail") or []
        if tail:
            print("\n  Last 5 epochs:")
            print(
                f"    {'ep':>4}  {'P':>8}  {'R':>8}  {'mAP50':>8}  {'mAP50-95':>8}"
            )
            for s in tail:
                print(
                    f"    {s.get('epoch') or 0:4d}  "
                    f"{_pct(s.get('precision')):>8}  "
                    f"{_pct(s.get('recall')):>8}  "
                    f"{_pct(s.get('mAP50')):>8}  "
                    f"{_pct(s.get('mAP50_95')):>8}"
                )

    val = report.get("val")
    if val:
        print("\n--- Fresh validation (model.val on best/last) ---")
        if val.get("error"):
            print(f"  ERROR: {val['error']}")
        else:
            print(f"  weights: {val.get('weights')}")
            print(
                f"  precision={_pct(val.get('precision'))}  "
                f"recall={_pct(val.get('recall'))}  "
                f"mAP50={_pct(val.get('mAP50'))}  "
                f"mAP50-95={_pct(val.get('mAP50_95'))}"
            )
            if val.get("speed_ms"):
                print(f"  speed_ms: {val['speed_ms']}")
            per = val.get("per_class") or []
            if per:
                print("\n  Per-class:")
                print(
                    f"    {'class':<22} {'P':>8} {'R':>8} {'mAP50':>8} {'mAP50-95':>8}"
                )
                for c in per:
                    print(
                        f"    {str(c.get('name')):<22} "
                        f"{_pct(c.get('precision')):>8} "
                        f"{_pct(c.get('recall')):>8} "
                        f"{_pct(c.get('mAP50')):>8} "
                        f"{_pct(c.get('mAP50_95')):>8}"
                    )

    print("\n--- How to read these ---")
    print("  precision  = of predicted boxes, fraction that match a GT box")
    print("  recall     = of GT boxes, fraction the model found")
    print("  mAP50      = mean Average Precision at IoU=0.50")
    print("  mAP50-95   = COCO-style mean AP over IoU 0.50…0.95 (stricter)")
    print("  plots: open runs/.../results.png and confusion_matrix.png on the VM")
    print("=" * 72)


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    p = argparse.ArgumentParser(
        description="Report RT-DETR Large training metrics (precision, recall, mAP)."
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=root / "runs" / "rtdetr_stage2",
        help="Ultralytics run directory (contains results.csv + weights/)",
    )
    p.add_argument(
        "--val",
        action="store_true",
        help="Re-run Ultralytics val on weights/best.pt for fresh metrics",
    )
    p.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Override data.yaml for --val (default: from args.yaml)",
    )
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--batch", type=int, default=8, help="Val batch size for --val")
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Also write the full report as JSON",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(
        args.run_dir,
        do_val=args.val,
        data_yaml=args.data,
        device=args.device,
        batch=args.batch,
    )
    print_report(report)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote JSON: {args.json_out.resolve()}")
    return 0 if report.get("exists") and not report.get("error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
