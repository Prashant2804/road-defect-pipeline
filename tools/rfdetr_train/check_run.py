#!/usr/bin/env python3
"""Inspect RF-DETR Stage-1 run directory: checkpoints, logs, and metrics.

Usage on the VM (after training finishes)::

    ./scripts/check_stage1_run.sh
    # or:
    .venv/bin/python -m tools.rfdetr_train.check_run
    .venv/bin/python -m tools.rfdetr_train.check_run --run-dir runs/rfdetr_stage1

Paste the full stdout here so we can decide next steps (resume, infer, more data, etc.).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from .config import repo_root
from .train_stage1 import find_best_checkpoint


PREFERRED_CKPTS = (
    "checkpoint_best_total.pth",
    "checkpoint_best_ema.pth",
    "checkpoint_best_regular.pth",
    "checkpoint.pth",
)

METRIC_PATTERNS = [
    ("mAP", re.compile(r"\bmAP(?:50)?(?:-95)?\b[^0-9]*([0-9]*\.?[0-9]+)", re.I)),
    ("AP50", re.compile(r"\bAP(?:50|_50)\b[^0-9]*([0-9]*\.?[0-9]+)", re.I)),
    ("AP75", re.compile(r"\bAP(?:75|_75)\b[^0-9]*([0-9]*\.?[0-9]+)", re.I)),
    ("loss", re.compile(r"\bloss\b[^0-9]*([0-9]*\.?[0-9]+)", re.I)),
    ("epoch", re.compile(r"\bepoch\b[^0-9]*(\d+)", re.I)),
]


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def _mtime(p: Path) -> str:
    return datetime.fromtimestamp(p.stat().st_mtime).isoformat(sep=" ", timespec="seconds")


def list_checkpoints(run_dir: Path) -> list[Path]:
    return sorted(run_dir.rglob("*.pth"), key=lambda p: p.stat().st_mtime)


def probe_checkpoint(path: Path) -> dict:
    info: dict = {
        "path": str(path),
        "size": _fmt_bytes(path.stat().st_size),
        "mtime": _mtime(path),
        "loadable": False,
    }
    try:
        import torch

        obj = torch.load(path, map_location="cpu", weights_only=False)
        info["loadable"] = True
        info["type"] = type(obj).__name__
        if isinstance(obj, dict):
            info["keys"] = sorted(str(k) for k in obj.keys())[:40]
            for k in ("epoch", "epochs", "best_epoch", "step", "iteration"):
                if k in obj:
                    info[k] = obj[k]
            for k in ("best_stats", "stats", "metrics", "coco_eval", "test_stats", "log"):
                if k in obj and obj[k] is not None:
                    try:
                        info[k] = (
                            obj[k]
                            if isinstance(obj[k], (dict, list, str, int, float))
                            else str(obj[k])[:500]
                        )
                    except Exception:
                        info[k] = "<unprintable>"
            # common weight containers
            for wk in ("model", "model_ema", "ema", "state_dict", "model_state_dict"):
                if wk in obj and hasattr(obj[wk], "keys"):
                    info[f"{wk}_tensors"] = len(list(obj[wk].keys()))
                    break
        elif hasattr(obj, "state_dict"):
            info["tensors"] = len(obj.state_dict())
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def find_log_files(run_dir: Path) -> list[Path]:
    patterns = (
        "*.log",
        "*.txt",
        "*.csv",
        "*.json",
        "log.txt",
        "results*.csv",
        "metrics*.json",
        "events.out.tfevents*",
    )
    found: set[Path] = set()
    for pat in patterns:
        found.update(run_dir.rglob(pat))
    # skip huge binary-ish except note tfevents presence
    return sorted(found, key=lambda p: (p.suffix, p.name))


def tail_text(path: Path, max_lines: int = 80, max_chars: int = 12000) -> str:
    try:
        if path.suffix in {".pth", ".pt", ".onnx"} or "tfevents" in path.name:
            return f"<binary/event file skipped: {path.name}>"
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"<read error: {e}>"
    lines = text.splitlines()
    chunk = "\n".join(lines[-max_lines:])
    if len(chunk) > max_chars:
        chunk = chunk[-max_chars:]
    return chunk


def scrape_metrics_from_text(text: str) -> dict[str, list]:
    hits: dict[str, list] = defaultdict(list)
    for name, rx in METRIC_PATTERNS:
        for m in rx.finditer(text):
            try:
                hits[name].append(float(m.group(1)))
            except ValueError:
                continue
    return dict(hits)


def summarize_dataset(repo: Path) -> dict:
    stage1 = repo / "data" / "rfdetr" / "stage1"
    out: dict = {"path": str(stage1), "exists": stage1.is_dir()}
    if not stage1.is_dir():
        return out
    for split in ("train", "valid", "test"):
        ann = stage1 / split / "_annotations.coco.json"
        if not ann.exists():
            out[split] = None
            continue
        try:
            doc = json.loads(ann.read_text(encoding="utf-8"))
            out[split] = {
                "images": len(doc.get("images", [])),
                "annotations": len(doc.get("annotations", [])),
                "categories": [c.get("name") for c in doc.get("categories", [])],
            }
        except Exception as e:
            out[split] = {"error": str(e)}
    return out


def build_report(run_dir: Path, repo: Path) -> dict:
    run_dir = run_dir.resolve()
    report: dict = {
        "run_dir": str(run_dir),
        "exists": run_dir.is_dir(),
        "dataset": summarize_dataset(repo),
        "checkpoints": [],
        "best_checkpoint": None,
        "log_files": [],
        "metric_hints": {},
        "log_tails": {},
        "tree_top": [],
    }
    if not run_dir.is_dir():
        return report

    # top-level listing
    for p in sorted(run_dir.iterdir()):
        if p.is_file():
            report["tree_top"].append(
                f"F {_fmt_bytes(p.stat().st_size):>10}  {p.name}"
            )
        else:
            n = sum(1 for _ in p.rglob("*"))
            report["tree_top"].append(f"D {n:>10} entries  {p.name}/")

    ckpts = list_checkpoints(run_dir)
    for ck in ckpts:
        report["checkpoints"].append(probe_checkpoint(ck))

    best = find_best_checkpoint(run_dir)
    if best is not None:
        report["best_checkpoint"] = str(best.resolve())

    logs = find_log_files(run_dir)
    # Prefer human-readable first
    text_logs = [
        p
        for p in logs
        if p.suffix.lower() in {".log", ".txt", ".csv", ".json"}
        and "tfevents" not in p.name
    ]
    event_logs = [p for p in logs if "tfevents" in p.name]
    report["log_files"] = [str(p.relative_to(run_dir)) for p in text_logs + event_logs]

    combined_metrics: dict[str, list] = defaultdict(list)
    for p in text_logs[:12]:
        body = tail_text(p, max_lines=200, max_chars=50000)
        report["log_tails"][str(p.relative_to(run_dir))] = tail_text(
            p, max_lines=60, max_chars=8000
        )
        for k, vals in scrape_metrics_from_text(body).items():
            combined_metrics[k].extend(vals)
    report["metric_hints"] = {
        k: {
            "n": len(v),
            "last": v[-1] if v else None,
            "min": min(v) if v else None,
            "max": max(v) if v else None,
        }
        for k, v in combined_metrics.items()
    }
    return report


def print_report(report: dict) -> None:
    print("=" * 72)
    print("RF-DETR Stage-1 run check")
    print("=" * 72)
    print(f"run_dir: {report['run_dir']}")
    print(f"exists:  {report['exists']}")
    print()

    ds = report.get("dataset") or {}
    print("--- Dataset (data/rfdetr/stage1) ---")
    print(f"path: {ds.get('path')}  exists={ds.get('exists')}")
    for split in ("train", "valid", "test"):
        info = ds.get(split)
        if info is None:
            print(f"  {split}: missing")
        elif isinstance(info, dict) and "error" in info:
            print(f"  {split}: ERROR {info['error']}")
        elif isinstance(info, dict):
            print(
                f"  {split}: {info.get('images')} images, "
                f"{info.get('annotations')} anns"
            )
    print()

    print("--- Run directory top-level ---")
    for line in report.get("tree_top") or []:
        print(" ", line)
    print()

    print("--- Checkpoints ---")
    ckpts = report.get("checkpoints") or []
    if not ckpts:
        print("  NONE FOUND")
    for c in ckpts:
        flag = ""
        if report.get("best_checkpoint") and Path(c["path"]).resolve() == Path(
            report["best_checkpoint"]
        ).resolve():
            flag = "  << BEST"
        print(f"  {c['path']}{flag}")
        print(f"    size={c.get('size')}  mtime={c.get('mtime')}  loadable={c.get('loadable')}")
        if c.get("epoch") is not None:
            print(f"    epoch={c.get('epoch')}")
        if c.get("keys"):
            print(f"    keys={c['keys'][:15]}{'...' if len(c['keys']) > 15 else ''}")
        for k in ("best_stats", "stats", "metrics", "test_stats"):
            if k in c:
                print(f"    {k}={c[k]}")
        if c.get("error"):
            print(f"    ERROR: {c['error']}")
    print(f"best_checkpoint: {report.get('best_checkpoint')}")
    print()

    print("--- Log / metric files ---")
    for rel in report.get("log_files") or []:
        print(f"  {rel}")
    if not report.get("log_files"):
        print("  (none — if you trained in tmux, also paste `tmux capture-pane` output)")
    print()

    print("--- Metric hints scraped from text logs ---")
    hints = report.get("metric_hints") or {}
    if not hints:
        print("  (no mAP/loss/epoch patterns found in text logs)")
    for k, v in hints.items():
        print(f"  {k}: last={v['last']}  min={v['min']}  max={v['max']}  n={v['n']}")
    print()

    print("--- Log tails (last ~60 lines each) ---")
    tails = report.get("log_tails") or {}
    if not tails:
        print("  (no text logs)")
    for rel, body in tails.items():
        print(f"\n##### {rel} #####")
        print(body)
    print()
    print("=" * 72)
    print("Copy everything above and paste it into chat for analysis.")
    print("=" * 72)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inspect RF-DETR Stage-1 run logs and weights.")
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Default: <repo>/runs/rfdetr_stage1",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Also write the structured report to this JSON path",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = repo_root()
    run_dir = Path(args.run_dir) if args.run_dir else (repo / "runs" / "rfdetr_stage1")
    report = build_report(run_dir, repo)
    print_report(report)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        # tails can be large; still useful
        args.json_out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\nWrote JSON report: {args.json_out}")
    return 0 if report.get("exists") and report.get("checkpoints") else 1


if __name__ == "__main__":
    raise SystemExit(main())
