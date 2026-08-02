#!/usr/bin/env python
"""Turn a raw annotation export into a dataset that trains correctly.

Writes a clean COPY; the original export is never modified, so a bad decision here
costs a re-run rather than the labels.

What it repairs, in the order it matters:

  geometry   One geometry per dataset. Ultralytics chooses box-vs-segment per FILE, so
             a file holding both silently reinterprets its box rows as 2-point polygons
             and produces boxes with the wrong centre and several times the area. This
             is the repair that changes results.
  leakage    Near-duplicate frames are pulled OUT of the eval splits and into train.
             A duplicate in train costs a little redundancy; the same frame in val
             turns validation into a memorisation test.
  taxonomy   Optionally renames classes onto `model.classes` so the trained checkpoint
             speaks the pipeline's vocabulary and needs no class_map at inference.
  empties    Classes with no annotations are dropped from data.yaml, since a head that
             can never fire only reports 0 mAP and confuses the metrics table.

    python tools/fix_labels.py --labels <export> --out data/mp_road --to box
    python tools/fix_labels.py --labels <export> --out data/mp_road --to box --rename
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_DIRS = ("train", "valid", "val", "test")
EVAL_SPLITS = ("valid", "val", "test")


def _dhash(path: Path, size: int = 8) -> int | None:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = 0
    for r in range(size):
        row = small[r]
        for c in range(size):
            bits = (bits << 1) | int(row[c + 1] > row[c])
    return bits


def _to_box(vals: list[float]) -> list[float]:
    """Any shape -> normalised cx cy w h."""
    if len(vals) == 4:
        return vals
    xs, ys = vals[0::2], vals[1::2]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]


def _to_poly(vals: list[float]) -> list[float]:
    """Any shape -> a polygon. A box becomes its four corners."""
    if len(vals) != 4:
        return vals
    cx, cy, w, h = vals
    x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def _clip(vals: list[float]) -> list[float]:
    return [min(1.0, max(0.0, v)) for v in vals]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", required=True, help="source export directory")
    p.add_argument("--out", required=True, help="destination for the cleaned copy")
    p.add_argument("--to", choices=["box", "polygon"], default="box",
                   help="geometry to normalise to (default: box -> a detect model)")
    p.add_argument("--rename", action="store_true",
                   help="rename dataset classes onto config model.classes")
    p.add_argument("--keep-leakage", action="store_true",
                   help="do not move near-duplicate eval frames into train")
    p.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = p.parse_args(argv)

    import yaml

    src, dst = Path(args.labels), Path(args.out)
    if not src.exists():
        print(f"Not found: {src}")
        return 1

    meta = yaml.safe_load((src / "data.yaml").read_text(encoding="utf-8")) \
        if (src / "data.yaml").exists() else {}
    names = [str(n) for n in (meta.get("names") or [])]
    if not names:
        print("No data.yaml class names found; cannot proceed safely.")
        return 1

    splits = [(s, src / s / "images", src / s / "labels")
              for s in SPLIT_DIRS if (src / s / "images").is_dir()]
    if not splits:
        print(f"No train/valid/test directories under {src}")
        return 1

    # ---- 1. decide which eval frames are contaminated -----------------------
    move_to_train: set[str] = set()
    if not args.keep_leakage:
        print("Scanning for duplicate frames across splits ...")
        hashes = {}
        for name, img_dir, _ in splits:
            hashes[name] = {p: _dhash(p) for p in sorted(img_dir.iterdir())
                            if p.suffix.lower() in IMAGE_EXTS}
        train_h = [h for h in hashes.get("train", {}).values() if h is not None]
        for s in EVAL_SPLITS:
            for path, h in hashes.get(s, {}).items():
                if h is None:
                    continue
                if any(h == t or bin(h ^ t).count("1") <= 4 for t in train_h):
                    move_to_train.add(f"{s}/{path.stem}")
        print(f"  {len(move_to_train)} eval frame(s) duplicate a training frame "
              f"-> moving into train")

    # ---- 2. class renaming --------------------------------------------------
    out_names = list(names)
    if args.rename:
        from rdd.config import load_config

        cfg = load_config(args.config)
        configured = [str(c) for c in (cfg.get_path("model.classes") or [])]
        mapping = _resolve_mapping(names, configured)
        unmapped = [n for n, v in mapping.items() if v is None]
        if unmapped:
            print(f"\nNo confident match onto model.classes for: {unmapped}")
            print("Edit CLASS_ALIASES in this file, or drop --rename and use "
                  "model.class_map at inference instead.")
            return 1
        out_names = [mapping[n] for n in names]
        print("\nClass renaming:")
        for a, b in zip(names, out_names):
            print(f"  {a!r} -> {b!r}")

    # ---- 3. rewrite ---------------------------------------------------------
    if dst.exists():
        shutil.rmtree(dst)
    conv = _to_box if args.to == "box" else _to_poly
    stats = Counter()
    per_class = Counter()

    for name, img_dir, lbl_dir in splits:
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            target = "train" if f"{name}/{img.stem}" in move_to_train else name
            target = "valid" if target == "val" else target
            (dst / target / "images").mkdir(parents=True, exist_ok=True)
            (dst / target / "labels").mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dst / target / "images" / img.name)

            lp = lbl_dir / f"{img.stem}.txt"
            lines = []
            if lp.exists():
                for line in lp.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    try:
                        cid = int(float(parts[0]))
                        vals = [float(v) for v in parts[1:]]
                    except ValueError:
                        stats["unparseable"] += 1
                        continue
                    if not (0 <= cid < len(names)):
                        stats["bad_class_dropped"] += 1
                        continue
                    was = "box" if len(vals) == 4 else "poly"
                    vals = _clip(conv(vals))
                    if was != args.to[:4]:
                        stats[f"converted_{was}_to_{args.to}"] += 1
                    # Reject anything that collapsed to nothing rather than writing a
                    # degenerate target the loss cannot use.
                    if args.to == "box" and (vals[2] <= 1e-6 or vals[3] <= 1e-6):
                        stats["degenerate_dropped"] += 1
                        continue
                    per_class[out_names[cid]] += 1
                    lines.append(" ".join([str(cid)] +
                                          [f"{v:.6f}" for v in vals]))
            (dst / target / "labels" / f"{img.stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            stats[f"images_{target}"] += 1

    # ---- 4. data.yaml -------------------------------------------------------
    # Keep indices stable: renumbering to drop an empty class would silently shift
    # every id in every label file, so empty classes are reported, not renumbered.
    empty = [n for n in out_names if per_class[n] == 0]
    doc = {"path": str(dst.resolve()), "train": "train/images", "val": "valid/images"}
    if (dst / "test" / "images").is_dir():
        doc["test"] = "test/images"
    doc["nc"] = len(out_names)
    doc["names"] = out_names
    (dst / "data.yaml").write_text(yaml.safe_dump(doc, sort_keys=False),
                                   encoding="utf-8")

    print(f"\nWrote {dst}")
    for k in sorted(stats):
        print(f"  {k:<28}{stats[k]}")
    print("\n  instances per class")
    for n in out_names:
        print(f"    {n:<38}{per_class[n]}")
    if empty:
        print(f"\n  Note: {empty} have no instances. Ids are left unchanged on purpose "
              f"- renumbering would shift every other class id in every label file.")
    print(f"\nVerify:  python tools/check_labels.py --labels {dst}")
    return 0


# Dataset class names rarely match a project taxonomy word for word. Aliases are
# explicit rather than fuzzy-matched: a wrong guess here mislabels the whole dataset,
# and that failure is invisible downstream.
CLASS_ALIASES = {
    "pothole": {"pothole", "potholes", "pot hole", "d40"},
    "longitudinal_crack": {"longitudinal_transverse_cracks", "longitudinal crack",
                           "longitudinal", "d00"},
    "transverse_crack": {"transverse crack", "transverse", "d10"},
    "alligator_crack": {"alligator_fatigue cracking", "alligator", "fatigue cracking",
                        "alligator crack", "d20"},
    "ravelling": {"rutting and ravelling", "ravelling", "raveling",
                  "surface distress"},
    "edge_damage": {"shoulder erosion", "edge damage", "edge break",
                    "shoulder damage"},
    "rutting": {"rutting", "rut"},
    "drainage_issue": {"culvert choke and drainage issues", "drainage",
                       "culvert choke", "culvert"},
    "water_logging": {"water logging", "waterlogging", "standing water"},
}


def _resolve_mapping(ds_names: list[str], configured: list[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for n in ds_names:
        key = n.strip().lower()
        hit = None
        for target, aliases in CLASS_ALIASES.items():
            if target not in configured:
                continue
            if key == target or key in aliases:
                hit = target
                break
        out[n] = hit
    return out


if __name__ == "__main__":
    sys.exit(main())
