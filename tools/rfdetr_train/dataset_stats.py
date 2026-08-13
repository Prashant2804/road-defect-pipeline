"""Show class-wise instance / image counts for Stage-1 and Stage-2 training data.

Prints ASCII tables for each known COCO root under data/rfdetr/:

  stage1              — Stage-1 Medium (CRRI)
  stage2              — multi-source Stage-2 merge (if present)
  custom_stage2       — your manual 6-class Drive zips (clean)
  custom_stage2_aug   — same + offline train augs (what fine-tune used)

Usage::

    .venv/bin/python -m tools.rfdetr_train.dataset_stats
    ./scripts/show_dataset_stats.sh
    .venv/bin/python -m tools.rfdetr_train.dataset_stats --json-out /tmp/stats.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .config import repo_root
from .taxonomy import CLASS_NAMES

# (label, relative path under data/rfdetr)
DEFAULT_DATASETS: list[tuple[str, str]] = [
    ("Stage-1 (CRRI)", "stage1"),
    ("Stage-2 (multi-source)", "stage2"),
    ("Custom Stage-2 (clean)", "custom_stage2"),
    ("Custom Stage-2 (aug train)", "custom_stage2_aug"),
]

SPLITS = ("train", "valid", "test")


def _ann_path(dataset_dir: Path, split: str) -> Path | None:
    for name in ("_annotations.coco.json", "annotations.json", f"{split}.json"):
        p = dataset_dir / split / name
        if p.is_file():
            return p
    # some layouts put json directly under split/
    d = dataset_dir / split
    if d.is_dir():
        cands = sorted(d.glob("*coco*.json")) + sorted(d.glob("*.json"))
        for p in cands:
            if p.is_file():
                return p
    return None


def load_split_stats(dataset_dir: Path, split: str) -> dict | None:
    ann = _ann_path(dataset_dir, split)
    if ann is None:
        return None
    doc = json.loads(ann.read_text(encoding="utf-8"))
    cats = {int(c["id"]): str(c["name"]) for c in doc.get("categories", [])}
    hist: Counter[str] = Counter()
    unknown: Counter[str] = Counter()
    for a in doc.get("annotations", []):
        cid = int(a["category_id"])
        name = cats.get(cid)
        if name is None:
            unknown[f"id:{cid}"] += 1
        else:
            hist[name] += 1
    # images with ≥1 ann per class (optional richness)
    img_by_class: Counter[str] = Counter()
    anns_by_img: dict[int, set[str]] = {}
    for a in doc.get("annotations", []):
        cid = int(a["category_id"])
        name = cats.get(cid)
        if name is None:
            continue
        iid = int(a["image_id"])
        anns_by_img.setdefault(iid, set()).add(name)
    for names in anns_by_img.values():
        for n in names:
            img_by_class[n] += 1

    return {
        "ann_path": str(ann),
        "images": len(doc.get("images", [])),
        "annotations": len(doc.get("annotations", [])),
        "categories": cats,
        "instances": {n: hist.get(n, 0) for n in CLASS_NAMES},
        "images_with_class": {n: img_by_class.get(n, 0) for n in CLASS_NAMES},
        "other_instances": dict(unknown),
        "extra_named": {
            n: c for n, c in hist.items() if n not in CLASS_NAMES
        },
    }


def _fmt_row(cols: list[str], widths: list[int]) -> str:
    return "| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths)) + " |"


def _sep(widths: list[int]) -> str:
    return "|-" + "-|-".join("-" * w for w in widths) + "-|"


def print_dataset_table(label: str, dataset_dir: Path, splits: list[str]) -> dict:
    print(f"\n{'=' * 72}")
    print(f"{label}")
    print(f"path: {dataset_dir}")
    print("=" * 72)

    if not dataset_dir.is_dir():
        print("  (directory missing — skip)")
        return {"label": label, "path": str(dataset_dir), "present": False}

    split_stats: dict[str, dict] = {}
    for sp in splits:
        st = load_split_stats(dataset_dir, sp)
        if st is not None:
            split_stats[sp] = st

    if not split_stats:
        print("  (no COCO splits found)")
        return {"label": label, "path": str(dataset_dir), "present": True, "splits": {}}

    # Header: Class | train inst | train imgs | valid ... | test ...
    headers = ["class"]
    for sp in splits:
        if sp in split_stats:
            headers.append(f"{sp} inst")
            headers.append(f"{sp} imgs")
    headers.append("notes")

    widths = [max(len(h), 20) for h in headers]
    widths[0] = max(widths[0], max(len(n) for n in CLASS_NAMES))

    print(_fmt_row(headers, widths))
    print(_sep(widths))

    for name in CLASS_NAMES:
        row = [name]
        for sp in splits:
            if sp not in split_stats:
                continue
            st = split_stats[sp]
            row.append(str(st["instances"].get(name, 0)))
            row.append(str(st["images_with_class"].get(name, 0)))
        row.append("")
        print(_fmt_row(row, widths))

    # totals
    tot = ["TOTAL"]
    for sp in splits:
        if sp not in split_stats:
            continue
        st = split_stats[sp]
        tot.append(str(st["annotations"]))
        tot.append(str(st["images"]))
    tot.append("anns / images")
    print(_sep(widths))
    print(_fmt_row(tot, widths))

    # extras
    for sp, st in split_stats.items():
        extras = st.get("extra_named") or {}
        other = st.get("other_instances") or {}
        if extras or other:
            print(f"\n  [{sp}] non-taxonomy / unknown:")
            for n, c in sorted(extras.items(), key=lambda x: -x[1]):
                print(f"    {n:22s} {c}")
            for n, c in sorted(other.items(), key=lambda x: -x[1]):
                print(f"    {n:22s} {c}")

    return {
        "label": label,
        "path": str(dataset_dir),
        "present": True,
        "splits": split_stats,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Class-wise tables for Stage-1 / Stage-2 training COCO datasets."
    )
    p.add_argument(
        "--work-root",
        type=Path,
        default=None,
        help="data/rfdetr root (default: <repo>/data/rfdetr)",
    )
    p.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma list: stage1,stage2,custom_stage2,custom_stage2_aug",
    )
    p.add_argument(
        "--splits",
        type=str,
        default="train,valid,test",
        help="Comma list of splits to show (default: train,valid,test)",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write full stats JSON",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    work = Path(args.work_root) if args.work_root else (repo_root() / "data" / "rfdetr")
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    wanted: set[str] | None = None
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}

    report: list[dict] = []
    print(f"Dataset root: {work.resolve()}")
    print("Instance = bounding-box count; imgs = images containing that class.")

    for label, rel in DEFAULT_DATASETS:
        key = rel
        if wanted is not None and key not in wanted:
            continue
        report.append(print_dataset_table(label, work / rel, splits))

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        # drop huge category maps if needed — keep as-is for debugging
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {out}")

    missing = [r["label"] for r in report if not r.get("present") or not r.get("splits")]
    if missing:
        print(
            "\nNote: missing/empty datasets (prepare/download first): "
            + ", ".join(missing)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
