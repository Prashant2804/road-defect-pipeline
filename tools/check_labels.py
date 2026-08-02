#!/usr/bin/env python
"""Validate an annotation set before spending GPU hours training on it.

Bad labels are expensive in a way bad code is not: training succeeds, metrics look
plausible, and the model is quietly wrong. Most of what goes wrong is mechanical and
checkable in seconds.

Checks, roughly in order of how badly each one bites:

  structure    every image has a label and vice versa; files readable
  taxonomy     dataset class names vs `model.classes`; emits a ready-made class_map
  geometry     normalised coords, non-degenerate shapes, boxes vs polygons
  aspect       whether a preprocessing resize distorted the images
  balance      instances per class - a class with too few examples cannot be learned
  leakage      near-duplicate images straddling the train/val split

The leakage check is the one people skip and the one that most often invalidates a
result. Frames grabbed from a video are near-duplicates of their neighbours, so a
RANDOM split puts the same physical pothole in train and val. Validation mAP then
measures memorisation and reads far too high - the model looks ready and is not.
Detection is by perceptual hash (dhash), which survives the JPEG recompression and
resizing an export applies, so it catches duplicates that are not byte-identical.

    python tools/check_labels.py --labels <export_dir>
    python tools/check_labels.py --labels <export_dir> --fix-clip
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_DIRS = ("train", "valid", "val", "test")

# Enough to learn a class at all / enough to certify 90% precision on it later.
MIN_TRAIN = 20
MIN_CERTIFY = 35


class Report:
    def __init__(self):
        self.errors: list[str] = []      # must fix: training will be wrong
        self.warnings: list[str] = []    # will degrade or misrepresent results
        self.notes: list[str] = []

    def error(self, m): self.errors.append(m)
    def warn(self, m): self.warnings.append(m)
    def note(self, m): self.notes.append(m)


def _dhash(path: Path, size: int = 8) -> int | None:
    """Perceptual hash: one bit per horizontal gradient step on a tiny greyscale copy.

    Survives recompression and rescaling, which byte hashing does not - and an export
    pipeline always recompresses, so byte hashing would report zero duplicates on a
    dataset that is full of them.
    """
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


def _discover(root: Path, rep: Report):
    """Return ([(split, images_dir, labels_dir)], dataset_class_names)."""
    names: list[str] = []
    yml = root / "data.yaml"
    if yml.exists():
        try:
            import yaml

            d = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
            names = [str(n) for n in (d.get("names") or [])]
            rf = d.get("roboflow") or {}
            if rf:
                rep.note(f"source: Roboflow project {rf.get('project')} "
                         f"v{rf.get('version')}")
        except Exception as e:
            rep.warn(f"data.yaml present but unreadable ({e}); assuming ids follow "
                     f"model.classes order")

    splits = []
    for s in SPLIT_DIRS:
        i, l = root / s / "images", root / s / "labels"
        if i.is_dir() and l.is_dir():
            splits.append((s, i, l))
    if not splits:
        if (root / "images").is_dir() and (root / "labels").is_dir():
            splits = [("all", root / "images", root / "labels")]
        elif any(p.suffix.lower() in IMAGE_EXTS for p in root.glob("*")):
            splits = [("all", root, root)]
    return splits, names


def _detect_foreign_format(root: Path, rep: Report) -> bool:
    """Spot COCO/VOC exports, which need converting rather than validating."""
    import json

    voc = list(root.rglob("*.xml"))[:1]
    if voc:
        rep.error(f"Found Pascal VOC XML ({voc[0].name}). Re-export as YOLO.")
        return True
    for j in list(root.rglob("*.json"))[:5]:
        try:
            d = json.loads(j.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(d, dict) and {"images", "annotations"} <= set(d):
            rep.error(f"Found COCO JSON ({j.name}). Re-export as YOLO .txt.")
            return True
    return False


def _check_split(name, img_dir, lbl_dir, classes, rep, fix_clip, stats):
    images = sorted(p for p in img_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    label_files = {p.stem: p for p in lbl_dir.glob("*.txt")}
    stems = {p.stem for p in images}
    if not images:
        rep.warn(f"[{name}] no images")
        return

    missing = sorted(stems - set(label_files))
    orphan = sorted(set(label_files) - stems)
    if missing:
        rep.warn(f"[{name}] {len(missing)} image(s) have no label file, so training "
                 f"treats them as containing NO defects. Deliberate negatives are "
                 f"useful; merely-unlabelled images teach the model to miss things. "
                 f"e.g. {missing[:2]}")
    if orphan:
        rep.warn(f"[{name}] {len(orphan)} label file(s) have no image: {orphan[:2]}")

    n_classes = len(classes)
    empty = 0
    for stem in sorted(stems & set(label_files)):
        lpath = label_files[stem]
        kinds = set()
        try:
            lines = [l.strip() for l in lpath.read_text(encoding="utf-8").splitlines()
                     if l.strip()]
        except Exception as e:
            rep.error(f"[{name}] {lpath.name}: unreadable ({e})")
            continue
        if not lines:
            empty += 1
            continue

        seen, out_lines = set(), []
        for ln, line in enumerate(lines, 1):
            parts = line.split()
            try:
                cid = int(float(parts[0]))
                vals = [float(v) for v in parts[1:]]
            except (ValueError, IndexError):
                rep.error(f"[{name}] {lpath.name}:{ln} unparseable: {line[:50]!r}")
                continue

            if not (0 <= cid < n_classes):
                stats["bad_class"].append((name, lpath.name, cid))
            else:
                stats["per_class"][classes[cid]] += 1
                stats["per_class_split"][(name, classes[cid])] += 1

            if len(vals) == 4:
                stats["boxes"] += 1
                kinds.add("box")
                _, _, bw, bh = vals
                if bw <= 0 or bh <= 0:
                    stats["degenerate"].append((lpath.name, ln, "zero-area box"))
                elif bw < 0.005 or bh < 0.005:
                    stats["tiny"].append((lpath.name, ln, f"{bw:.4f}x{bh:.4f}"))
                stats["areas"].append(bw * bh)
                key = (cid,) + tuple(round(v, 5) for v in vals)
                if key in seen:
                    stats["dupes"].append((lpath.name, ln))
                seen.add(key)
            elif len(vals) >= 6 and len(vals) % 2 == 0:
                stats["polys"] += 1
                kinds.add("poly")
                xs, ys = vals[0::2], vals[1::2]
                n = len(xs)
                area = abs(sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i]
                               for i in range(n))) / 2.0
                if n < 3 or area < 1e-6:
                    stats["degenerate"].append((lpath.name, ln, "zero-area polygon"))
                stats["areas"].append(area)
            else:
                rep.error(f"[{name}] {lpath.name}:{ln} has {len(vals)} coords - need 4 "
                          f"(box) or an even count >=6 (polygon)")
                continue

            bad = [v for v in vals if v < -1e-6 or v > 1 + 1e-6]
            if bad:
                stats["oob"].append((lpath.name, ln, round(max(bad), 3)))
                if fix_clip:
                    vals = [min(1.0, max(0.0, v)) for v in vals]
                    stats["fixed"] += 1
            out_lines.append(" ".join([str(cid)] + [f"{v:.6f}" for v in vals]))

        if len(kinds) > 1:
            stats["mixed_files"].append(f"{name}/{lpath.name}")
        if fix_clip and out_lines:
            lpath.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    stats["counts"][name] = {"images": len(images), "empty": empty}


def _check_images(splits, stats, sample=60):
    """Image dimensions - a stretch resize is a silent train/serve skew."""
    import cv2

    for _, img_dir, _ in splits:
        imgs = sorted(p for p in img_dir.iterdir()
                      if p.suffix.lower() in IMAGE_EXTS)[:sample]
        for p in imgs:
            im = cv2.imread(str(p))
            if im is not None:
                stats["sizes"][(im.shape[1], im.shape[0])] += 1


def _check_leakage(splits, stats):
    """Near-duplicate images straddling splits, by perceptual hash."""
    by_split: dict[str, dict[int, str]] = {}
    for name, img_dir, _ in splits:
        h: dict[int, str] = {}
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            d = _dhash(p)
            if d is not None:
                h.setdefault(d, p.name)
        by_split[name] = h

    cross = []
    names = list(by_split)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for s in sorted(set(by_split[a]) & set(by_split[b]))[:200]:
                cross.append((a, b, by_split[a][s], by_split[b][s]))
    stats["cross_dupes"] = cross

    # Near-matches (Hamming <= 4) between train and each eval split.
    near = []
    tr = by_split.get("train", {})
    for other in ("valid", "val", "test"):
        ev = by_split.get(other, {})
        if not tr or not ev:
            continue
        for eh, ename in ev.items():
            for th, tname in tr.items():
                if eh != th and bin(eh ^ th).count("1") <= 4:
                    near.append((other, ename, tname))
                    break
    stats["near_dupes"] = near
    stats["split_sizes"] = {k: len(v) for k, v in by_split.items()}


def _guess(ds_name: str, configured: list[str]) -> str | None:
    """Token-overlap guess from a dataset class name to a configured one."""
    import re

    def toks(s):
        return {t for t in re.split(r"[^a-z]+", s.lower()) if len(t) > 2}

    d = toks(ds_name)
    best, score = None, 0
    for c in configured:
        s = len(d & toks(c))
        if s > score:
            best, score = c, s
    return best


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", required=True, help="annotation root")
    p.add_argument("--config", default=str(ROOT / "config.yaml"))
    p.add_argument("--fix-clip", action="store_true",
                   help="clamp out-of-range coordinates in place")
    p.add_argument("--no-leakage", action="store_true",
                   help="skip the perceptual-hash duplicate scan")
    args = p.parse_args(argv)

    from rdd.config import load_config

    cfg = load_config(args.config)
    configured = [str(c) for c in (cfg.get_path("model.classes") or [])]
    root = Path(args.labels)
    rep = Report()

    print(f"\n{'=' * 78}\nAnnotation check: {root}\n{'=' * 78}")
    if not root.exists():
        print(f"  Not found: {root}")
        return 1
    if _detect_foreign_format(root, rep):
        for e in rep.errors:
            print(f"  ERROR: {e}")
        return 1

    splits, ds_names = _discover(root, rep)
    if not splits:
        print(f"  No image/label directories found under {root}")
        return 1
    classes = ds_names or configured
    if not ds_names:
        rep.warn("No data.yaml class list - assuming ids match model.classes order. "
                 "Verify this; a wrong assumption mislabels everything.")

    stats = {
        "per_class": Counter(), "per_class_split": Counter(), "counts": {},
        "bad_class": [], "degenerate": [], "oob": [], "dupes": [], "tiny": [],
        "areas": [], "boxes": 0, "polys": 0, "fixed": 0, "mixed_files": [],
        "sizes": Counter(), "cross_dupes": [], "near_dupes": [], "split_sizes": {},
    }

    for name, i, l in splits:
        _check_split(name, i, l, classes, rep, args.fix_clip, stats)
    _check_images(splits, stats)
    if not args.no_leakage:
        print("  scanning for duplicate frames across splits ...")
        _check_leakage(splits, stats)

    # ---------------- structure ----------------
    print("\n-- Dataset ------------------------------------------------------------")
    for n in rep.notes:
        print(f"  {n}")
    for name, c in stats["counts"].items():
        print(f"  {name:<8}{c['images']:>5} images  ({c['empty']} with no annotations)")
    print(f"  {'total':<8}{sum(c['images'] for c in stats['counts'].values()):>5} "
          f"images, {stats['boxes']} boxes, {stats['polys']} polygons")

    # ---------------- geometry ----------------
    print("\n-- Geometry -----------------------------------------------------------")
    if stats["sizes"]:
        for (w, h), n in stats["sizes"].most_common(4):
            print(f"  {w}x{h}  ({n} sampled)  aspect {w / h:.3f}")
        (w, h), _ = stats["sizes"].most_common(1)[0]
        if abs(w / h - 1.0) < 0.01:
            # The original aspect ratio is unrecoverable from an already-resized
            # export, so this is stated as conditional rather than asserted.
            rep.warn(
                f"Images are square ({w}x{h}). If the source frames were 16:9 - as "
                f"dashcam video normally is - a 'Stretch' resize squashed every one of "
                f"them, and that matters twice over: a stretched pothole is an ellipse, "
                f"so at inference on real 16:9 frames the model meets a shape it never "
                f"trained on; and anisotropic scaling CHANGES ANGLES, while "
                f"longitudinal-vs-transverse is an angle measurement. Check the source "
                f"aspect. If it was not square, re-export with Resize = 'Fit "
                f"(letterbox)' or no resize.")
        if w <= 640:
            rep.warn(
                f"Images are only {w}px wide. A hairline crack is a few pixels across "
                f"at 1920px and is simply gone by {w}px. Expect poor crack recall "
                f"regardless of training; export at 1280px or native.")
    if stats["boxes"] and stats["polys"]:
        mixed = stats["mixed_files"]
        print(f"  MIXED: {stats['boxes']} boxes and {stats['polys']} polygons")
        if mixed:
            # This one silently corrupts data rather than erroring, so it is the most
            # dangerous finding here. Ultralytics decides box-vs-segment PER FILE:
            #   if any(len(row) > 6 for row in file): treat every row as a polygon
            # A 4-value "x y w h" row in such a file is then reshaped to two POINTS,
            # (x,y) and (w,h), and segments2boxes builds a box from those. Verified on
            # this dataset: a box at (0.43, 0.83) size 0.14x0.29 came back as
            # (0.28, 0.56) size 0.28x0.53 - wrong centre, 3.6x the area, no warning.
            rep.error(
                f"{len(mixed)} label file(s) mix a 4-value box row with a polygon row. "
                f"Ultralytics picks box-vs-segment per FILE, so in each of these every "
                f"box row is reinterpreted as a 2-point polygon: 'x y w h' is read as "
                f"the points (x,y) and (w,h). The resulting box has the wrong centre "
                f"and several times the area, and nothing errors. Normalise the "
                f"geometry first:  python tools/fix_labels.py --labels <dir> --to box. "
                f"Affected: {mixed[:3]}")
        else:
            rep.error(
                f"Mixed geometry across files ({stats['boxes']} boxes, "
                f"{stats['polys']} polygons). No single file mixes both, so nothing is "
                f"corrupted, but a model trains as detect OR segment - normalise with "
                f"tools/fix_labels.py.")
    elif stats["boxes"]:
        print("  boxes only -> a DETECTION model (task=detect)")
        rep.warn(
            "Annotations are bounding boxes, but config.model.arch is a '-seg' "
            "architecture and the pipeline derives defect AREA from the mask. With "
            "boxes, area becomes the box: roughly right for a pothole, badly wrong "
            "for a crack, since the box around a thin diagonal crack overstates its "
            "area by an order of magnitude. Either re-annotate cracks as polygons, or "
            "report crack COUNT only and drop crack area/severity.")
    if stats["areas"]:
        import statistics as st

        a = sorted(stats["areas"])
        print(f"  annotation size: median {st.median(a) * 100:.2f}% of frame, "
              f"p5 {a[len(a) // 20] * 100:.2f}%, "
              f"p95 {a[-max(1, len(a) // 20)] * 100:.2f}%")

    # ---------------- taxonomy ----------------
    print("\n-- Taxonomy -----------------------------------------------------------")
    print(f"  dataset ({len(classes)}): {classes}")
    lower_ds = [c.lower() for c in ds_names]
    lower_cfg = [c.lower() for c in configured]
    mismatch = bool(ds_names) and lower_ds != lower_cfg
    if mismatch:
        print(f"  config  ({len(configured)}): {configured}")
        if set(lower_ds) <= set(lower_cfg):
            # Names already line up; only the indices differ. Easy to wave through and
            # exactly as damaging as unrecognisable names, because resolution is
            # positional: id 0 means the dataset's first class, not the config's.
            rep.error(
                f"Class NAMES are all valid, but the ORDER differs from model.classes "
                f"(this dataset has {len(ds_names)} of the {len(configured)} classes). "
                f"Detections resolve by INDEX, so a checkpoint trained here emits id 0 "
                f"for {ds_names[0]!r} while the pipeline reads id 0 as "
                f"{configured[0]!r}. Every label would be shifted. Add the identity "
                f"model.class_map below - it looks redundant and is not.")
        else:
            rep.error(
                "Dataset class names do not match model.classes. Detections resolve BY "
                "INDEX, so without a map every label in the report is wrong. Set "
                "model.class_map - a suggested mapping is printed below.")

    # ---------------- balance ----------------
    print("\n-- Instances per class ------------------------------------------------")
    print(f"  {'class':<38}{'total':>7}{'train':>7}{'valid':>7}{'test':>7}")
    for c in classes:
        t = stats["per_class"][c]
        row = (f"  {c:<38}{t:>7}"
               f"{stats['per_class_split'][('train', c)]:>7}"
               f"{stats['per_class_split'][('valid', c)]:>7}"
               f"{stats['per_class_split'][('test', c)]:>7}")
        if t == 0:
            row += "   <- none"
        elif t < MIN_TRAIN:
            row += "   <- too few to learn"
        elif t < MIN_CERTIFY:
            row += "   <- cannot certify 90%"
        print(row)

    absent = [c for c in classes if stats["per_class"][c] == 0]
    thin = [c for c in classes if 0 < stats["per_class"][c] < MIN_TRAIN]
    marginal = [c for c in classes if MIN_TRAIN <= stats["per_class"][c] < MIN_CERTIFY]
    if absent:
        rep.error(f"{len(absent)} class(es) declared but never annotated: {absent}. "
                  f"Remove them from data.yaml, or the model carries a head that can "
                  f"never fire and reports 0 mAP for no reason.")
    if thin:
        rep.warn(f"Too few instances to learn: {thin}. Expect near-zero recall on "
                 f"these until more are labelled.")
    if marginal:
        rep.warn(f"Enough to train, not enough to CERTIFY 90% precision: {marginal}. "
                 f"A Wilson lower bound of 0.90 needs about {MIN_CERTIFY} held-out "
                 f"instances with no errors; below that the claim is unprovable "
                 f"however good the model is.")

    # ---------------- annotation defects ----------------
    if stats["bad_class"]:
        ids = sorted({c for _, _, c in stats["bad_class"]})
        rep.error(f"{len(stats['bad_class'])} annotation(s) use out-of-range class "
                  f"id(s) {ids} (valid 0-{len(classes) - 1})")
    if stats["degenerate"]:
        rep.error(f"{len(stats['degenerate'])} zero-area shape(s), usually a stray "
                  f"click: {stats['degenerate'][:3]}")
    if stats["oob"]:
        m = f"{len(stats['oob'])} coordinate(s) outside [0,1]"
        if args.fix_clip:
            rep.warn(f"{m}; clamped {stats['fixed']}")
        else:
            rep.error(f"{m} - re-run with --fix-clip to clamp")
    if stats["dupes"]:
        rep.warn(f"{len(stats['dupes'])} duplicate annotation(s) within one file "
                 f"(double-click): {stats['dupes'][:3]}")
    if stats["tiny"]:
        rep.warn(f"{len(stats['tiny'])} annotation(s) under 0.5% of the frame. At this "
                 f"image size that is a handful of pixels - verify they are real: "
                 f"{stats['tiny'][:3]}")

    # ---------------- leakage ----------------
    if not args.no_leakage:
        print("\n-- Split integrity ----------------------------------------------------")
        cd, nd = stats["cross_dupes"], stats["near_dupes"]
        n_eval = sum(v for k, v in stats["split_sizes"].items()
                     if k in ("valid", "val", "test")) or 1
        print(f"  identical frames across splits : {len(cd)}")
        print(f"  near-identical (train <-> eval): {len(nd)}  "
              f"({100.0 * len(nd) / n_eval:.0f}% of eval)")
        if cd:
            rep.error(
                f"{len(cd)} image(s) appear in two splits at once, e.g. {cd[0][2]} in "
                f"{cd[0][0]} and {cd[0][3]} in {cd[0][1]}. A validation score measured "
                f"on training data means nothing.")
        if nd:
            pct = 100.0 * len(nd) / n_eval
            msg = (f"{len(nd)} eval image(s) ({pct:.0f}%) are near-duplicates of a "
                   f"TRAINING image - frames of the same scene split at random, so the "
                   f"model is partly tested on defects it has already memorised. "
                   f"Validation mAP reads above real performance. Re-split by SOURCE "
                   f"VIDEO or scene rather than at random.")
            # Severity scales with contamination: a handful of duplicates nudges the
            # number, a third of the eval set invalidates it.
            (rep.error if pct >= 10 else rep.warn)(
                msg if pct >= 10 else msg + f" At {pct:.0f}% the distortion is small.")

    # ---------------- verdict ----------------
    print(f"\n{'=' * 78}")
    if rep.warnings:
        print("WARNINGS - will degrade or misrepresent results\n")
        for w in rep.warnings:
            print(f"  * {w}\n")
    if rep.errors:
        print("ERRORS - fix before trusting a trained model\n")
        for e in rep.errors:
            print(f"  * {e}\n")

    if mismatch:
        print("Suggested config.yaml addition (check every line):\n")
        print("model:\n  class_map:")
        for c in classes:
            g = _guess(c, configured)
            print(f"    {c!r}: {g!r}" if g else f"    {c!r}: null  # TODO")
        print()

    if rep.errors:
        print("VERDICT: not ready to train as-is. See errors above.\n")
        return 1
    if rep.warnings:
        print("VERDICT: trainable, with the caveats above.\n")
        return 0
    print("VERDICT: clean.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
