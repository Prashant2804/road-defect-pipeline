#!/usr/bin/env python3
"""Download custom Drive zips, analyze, merge to 6-class COCO, offline train augs.

Writes ONLY under data/rfdetr/custom_* — never touches stage1/stage2 or infer runs.

Usage::

    .venv/bin/python -m tools.rfdetr_train.prepare_custom_stage2
    .venv/bin/python -m tools.rfdetr_train.prepare_custom_stage2 --skip-download
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from .augmentations import CUSTOM_ROAD_AUG, STRESS_LIGHT_AUG
from .coco_io import (
    ingest_to_coco,
    merge_coco_datasets,
    print_class_histogram,
    unwrap_zip_root,
)
from .config import repo_root
from .taxonomy import CLASS_NAMES, resolve_class

DEFAULT_DRIVE_URLS = [
    "https://drive.google.com/file/d/1TFZn9vQUXqbxkRykXD3egjkwMWtDDW4N/view",
    "https://drive.google.com/file/d/1Z0apCTay8FIkfSlE-wWYp9PLxQx-sfcb/view",
]


def _ensure_albumentations():
    try:
        import albumentations as A  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "albumentations required for offline augs. Install:\n"
            "  .venv/bin/pip install 'rfdetr[augment]' albumentations\n"
            f"Original error: {e}"
        ) from e


def _build_offline_pipeline(cfg: dict, *, seed: int):
    import albumentations as A

    transforms = []
    for name, kwargs in cfg.items():
        cls = getattr(A, name, None)
        if cls is None:
            print(f"WARNING: skip unknown transform {name}")
            continue
        try:
            transforms.append(cls(**kwargs))
        except TypeError:
            # Older/newer albumentations arg differences
            kw = dict(kwargs)
            if name == "GaussNoise":
                kw.pop("std_range", None)
                kw.setdefault("var_limit", (10.0, 50.0))
            if name == "CoarseDropout":
                kw.pop("fill_value", None)
            try:
                transforms.append(cls(**kw))
            except Exception as e:
                print(f"WARNING: skip {name}: {e}")
    if not transforms:
        transforms = [A.HorizontalFlip(p=0.5)]
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["category_ids"],
            min_visibility=0.2,
        ),
        seed=seed,
    )


def analyze_raw_tree(root: Path) -> dict:
    """Best-effort raw analysis before remapping."""
    report: dict = {"path": str(root), "coco": [], "yolo": False}
    for ann in root.rglob("_annotations.coco.json"):
        doc = json.loads(ann.read_text(encoding="utf-8"))
        cats = {c["id"]: c.get("name", str(c["id"])) for c in doc.get("categories", [])}
        hist = Counter()
        mapped = Counter()
        dropped = Counter()
        for a in doc.get("annotations", []):
            raw = cats.get(a.get("category_id"), str(a.get("category_id")))
            hist[raw] += 1
            resolved = resolve_class(raw)
            if resolved is None:
                dropped[raw] += 1
            else:
                mapped[resolved] += 1
        report["coco"].append(
            {
                "ann": str(ann.relative_to(root)),
                "images": len(doc.get("images", [])),
                "annotations": len(doc.get("annotations", [])),
                "categories": list(cats.values()),
                "raw_instance_counts": dict(hist),
                "mapped_instance_counts": dict(mapped),
                "dropped_instance_counts": dict(dropped),
            }
        )
    if list(root.rglob("data.yaml")) or list(root.rglob("*.txt")):
        report["yolo"] = True
    n_img = sum(
        1
        for p in root.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    report["image_files_found"] = n_img
    return report


def print_analysis(report: dict, title: str) -> None:
    print(f"\n===== ANALYZE: {title} =====")
    print(f"path: {report['path']}")
    print(f"image_files_found: {report.get('image_files_found')}")
    print(f"yolo_hints: {report.get('yolo')}")
    for block in report.get("coco") or []:
        print(f"\n  COCO {block['ann']}")
        print(f"    images={block['images']} anns={block['annotations']}")
        print(f"    categories={block['categories']}")
        print(f"    raw counts: {block['raw_instance_counts']}")
        print(f"    mapped→6class: {block['mapped_instance_counts']}")
        print(f"    dropped (null/other): {block['dropped_instance_counts']}")
    if not report.get("coco"):
        print("  (no _annotations.coco.json yet — will try YOLO ingest)")


def download_zips(urls: list[str], raw_dir: Path) -> list[Path]:
    from tools.rfdetr_infer.media_fetch import download_drive_file

    raw_dir.mkdir(parents=True, exist_ok=True)
    zips: list[Path] = []
    for i, url in enumerate(urls, start=1):
        dest = raw_dir / f"custom_zip_{i}.zip"
        download_drive_file(url, dest)
        zips.append(dest)
    return zips


def unzip_all(zips: list[Path], raw_dir: Path) -> list[Path]:
    roots: list[Path] = []
    for i, zpath in enumerate(zips, start=1):
        out = raw_dir / f"zip{i}"
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        print(f"Unzipping {zpath.name} → {out}")
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(out)
        roots.append(unwrap_zip_root(out))
    return roots


def expand_split_offline(
    src_split: Path,
    dst_split: Path,
    *,
    aug_cfg: dict,
    copies_per_image: int = 1,
    seed: int = 42,
    prefix: str = "aug",
) -> None:
    """Copy originals + write N random-aug clones (bboxes kept in COCO xywh)."""
    _ensure_albumentations()
    ann_path = src_split / "_annotations.coco.json"
    if not ann_path.is_file():
        return
    doc = json.loads(ann_path.read_text(encoding="utf-8"))
    dst_split.mkdir(parents=True, exist_ok=True)

    images_out: list[dict] = []
    anns_out: list[dict] = []
    next_img_id = 1
    next_ann_id = 1
    by_image: dict[int, list[dict]] = defaultdict(list)
    for a in doc.get("annotations", []):
        by_image[int(a["image_id"])].append(a)

    rng = random.Random(seed)
    pipeline = _build_offline_pipeline(aug_cfg, seed=seed)

    for im in doc.get("images", []):
        src_img = src_split / im["file_name"]
        if not src_img.exists():
            continue
        bgr = cv2.imread(str(src_img))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        boxes = []
        cats = []
        for a in by_image.get(int(im["id"]), []):
            bb = a.get("bbox")
            if not bb or len(bb) != 4:
                continue
            x, y, bw, bh = [float(v) for v in bb]
            if bw < 1 or bh < 1:
                continue
            boxes.append([x, y, bw, bh])
            cats.append(int(a["category_id"]))

        # Original
        new_name = im["file_name"]
        dst_img = dst_split / new_name
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)
        images_out.append(
            {
                "id": next_img_id,
                "file_name": new_name,
                "width": im.get("width") or w,
                "height": im.get("height") or h,
            }
        )
        for a in by_image.get(int(im["id"]), []):
            na = dict(a)
            na["id"] = next_ann_id
            na["image_id"] = next_img_id
            anns_out.append(na)
            next_ann_id += 1
        next_img_id += 1

        # Augmented clones
        for k in range(copies_per_image):
            if not boxes:
                # still enrich empty/background frames lightly
                local_seed = seed + next_img_id * 17 + k
                pipe = _build_offline_pipeline(aug_cfg, seed=local_seed)
                try:
                    out = pipe(image=rgb, bboxes=[], category_ids=[])
                except Exception:
                    continue
            else:
                local_seed = seed + next_img_id * 17 + k
                pipe = _build_offline_pipeline(aug_cfg, seed=local_seed)
                try:
                    out = pipe(image=rgb, bboxes=boxes, category_ids=cats)
                except Exception as e:
                    print(f"  aug skip {im['file_name']}: {e}")
                    continue
            aug_rgb = out["image"]
            aug_boxes = out.get("bboxes") or []
            aug_cats = out.get("category_ids") or []
            stem = Path(im["file_name"]).stem
            ext = Path(im["file_name"]).suffix or ".jpg"
            aug_name = f"{prefix}{k+1}__{stem}{ext}"
            aug_bgr = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(dst_split / aug_name), aug_bgr)
            ah, aw = aug_rgb.shape[:2]
            images_out.append(
                {
                    "id": next_img_id,
                    "file_name": aug_name,
                    "width": aw,
                    "height": ah,
                }
            )
            for bb, cid in zip(aug_boxes, aug_cats):
                x, y, bw, bh = [float(v) for v in bb]
                if bw < 1 or bh < 1:
                    continue
                anns_out.append(
                    {
                        "id": next_ann_id,
                        "image_id": next_img_id,
                        "category_id": int(cid),
                        "bbox": [x, y, bw, bh],
                        "area": float(bw * bh),
                        "iscrowd": 0,
                    }
                )
                next_ann_id += 1
            next_img_id += 1
            _ = rng  # keep Random used for reproducibility hooks

    merged = {
        "info": {"description": f"offline-aug {dst_split.name}"},
        "licenses": [],
        "categories": [
            {"id": i + 1, "name": n, "supercategory": "road_defect"}
            for i, n in enumerate(CLASS_NAMES)
        ],
        "images": images_out,
        "annotations": anns_out,
    }
    (dst_split / "_annotations.coco.json").write_text(
        json.dumps(merged), encoding="utf-8"
    )
    print(
        f"  offline {dst_split.name}: {len(images_out)} images, "
        f"{len(anns_out)} anns (copies_per_image={copies_per_image})"
    )


def build_aug_dataset(clean_dir: Path, aug_dir: Path, stress_dir: Path) -> Path:
    if aug_dir.exists():
        shutil.rmtree(aug_dir)
    aug_dir.mkdir(parents=True)

    # Train: originals + 1 aug clone
    expand_split_offline(
        clean_dir / "train",
        aug_dir / "train",
        aug_cfg=CUSTOM_ROAD_AUG,
        copies_per_image=1,
        seed=42,
        prefix="aug",
    )
    # Valid/test: clean copies only (honest metrics)
    for split in ("valid", "test"):
        src = clean_dir / split
        if not (src / "_annotations.coco.json").exists():
            continue
        dst = aug_dir / split
        dst.mkdir(parents=True, exist_ok=True)
        for p in src.iterdir():
            if p.is_file():
                shutil.copy2(p, dst / p.name)
        print(f"  copied clean {split}/ → {dst}")

    # Stress views (qualitative only)
    if stress_dir.exists():
        shutil.rmtree(stress_dir)
    stress_dir.mkdir(parents=True)
    for split in ("valid", "test"):
        src = clean_dir / split
        if not (src / "_annotations.coco.json").exists():
            continue
        expand_split_offline(
            src,
            stress_dir / split,
            aug_cfg=STRESS_LIGHT_AUG,
            copies_per_image=1,
            seed=99,
            prefix="stress",
        )
    return aug_dir


def prepare(
    urls: list[str],
    *,
    skip_download: bool = False,
    work_root: Path | None = None,
) -> Path:
    root = work_root or (repo_root() / "data" / "rfdetr")
    raw_dir = root / "custom_raw"
    parts_dir = root / "custom_parts"
    clean_dir = root / "custom_stage2"
    aug_dir = root / "custom_stage2_aug"
    stress_dir = root / "custom_stage2_stress"

    if skip_download:
        zips = sorted(raw_dir.glob("custom_zip_*.zip"))
        if not zips:
            raise SystemExit(f"No zips under {raw_dir}; omit --skip-download")
    else:
        zips = download_zips(urls, raw_dir)

    unzip_roots = unzip_all(zips, raw_dir)

    analysis = []
    for i, zr in enumerate(unzip_roots, start=1):
        rep = analyze_raw_tree(zr)
        print_analysis(rep, f"zip{i}")
        analysis.append(rep)
    (raw_dir / "analysis_report.json").write_text(
        json.dumps(analysis, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {raw_dir / 'analysis_report.json'}")

    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True)
    parts: list[Path] = []
    for i, zr in enumerate(unzip_roots, start=1):
        out = parts_dir / f"part{i}"
        print(f"\nIngest → remap 6-class: {zr} → {out}")
        ingest_to_coco(zr, out)
        print_class_histogram(out, "train")
        parts.append(out)

    print(f"\nMerge → {clean_dir}")
    merge_coco_datasets(parts, clean_dir)
    for split in ("train", "valid", "test"):
        if (clean_dir / split / "_annotations.coco.json").exists():
            print_class_histogram(clean_dir, split)

    print(f"\nOffline train augs → {aug_dir}")
    build_aug_dataset(clean_dir, aug_dir, stress_dir)
    print_class_histogram(aug_dir, "train")
    if (aug_dir / "valid" / "_annotations.coco.json").exists():
        print_class_histogram(aug_dir, "valid")

    print("\nPrepared:")
    print(f"  clean (metrics): {clean_dir}")
    print(f"  train+aug:       {aug_dir}")
    print(f"  stress views:    {stress_dir}")
    return aug_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Prepare custom Stage-2 COCO from two Drive zips (isolated paths)."
    )
    p.add_argument(
        "--url",
        action="append",
        dest="urls",
        default=None,
        help="Drive file URL (repeatable). Defaults to the two plan links.",
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse existing data/rfdetr/custom_raw/custom_zip_*.zip",
    )
    p.add_argument(
        "--work-root",
        type=Path,
        default=None,
        help="Default: <repo>/data/rfdetr",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    urls = args.urls or list(DEFAULT_DRIVE_URLS)
    if len(urls) < 1:
        raise SystemExit("Need at least one --url")
    prepare(urls, skip_download=args.skip_download, work_root=args.work_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
