"""Ingest UAV/drone pavement-distress sources into the fixed 6-class COCO layout.

Each source ships in a different raw format (Pascal VOC XML, flat YOLO txt with
fixed class ids, Roboflow COCO). These converters normalize all of them into
the same {train,valid,test}/_annotations.coco.json layout used by coco_io.py,
so they can be merged with tools.rfdetr_train.coco_io.merge_coco_datasets.
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

from .taxonomy import CLASS_NAMES, CLASS_TO_ID, coco_categories, resolve_class

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _find_image_xml_pairs(raw_root: Path) -> list[tuple[Path, Path]]:
    """Pair every XML annotation anywhere under raw_root with a same-stem image.

    Deliberately layout-agnostic (JPEGImages/Annotations, images/labels,
    flat folders, nested zips all work) since UAV-PDD2023 / UAPD Zenodo and
    Google Drive archives don't follow one fixed convention.
    """
    xml_by_stem: dict[str, Path] = {}
    for xml_path in raw_root.rglob("*.xml"):
        xml_by_stem.setdefault(xml_path.stem, xml_path)

    img_by_stem: dict[str, Path] = {}
    for img_path in raw_root.rglob("*"):
        if img_path.suffix.lower() in IMG_EXTS:
            img_by_stem.setdefault(img_path.stem, img_path)

    pairs = []
    for stem, xml_path in xml_by_stem.items():
        img_path = img_by_stem.get(stem)
        if img_path is not None:
            pairs.append((img_path, xml_path))
    return pairs


def _parse_voc_objects(xml_path: Path) -> list[tuple[str, float, float, float, float]]:
    root = ET.parse(xml_path).getroot()
    out = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        bb = obj.find("bndbox")
        if not name or bb is None:
            continue
        try:
            xmin = float(bb.findtext("xmin") or 0)
            ymin = float(bb.findtext("ymin") or 0)
            xmax = float(bb.findtext("xmax") or 0)
            ymax = float(bb.findtext("ymax") or 0)
        except ValueError:
            continue
        out.append((name, xmin, ymin, xmax, ymax))
    return out


def ingest_voc_dataset(
    raw_root: Path,
    out_dir: Path,
    *,
    val_frac: float = 0.12,
    test_frac: float = 0.0,
    seed: int = 1337,
    dedupe_hashes: set[str] | None = None,
) -> tuple[Path, set[str]]:
    """Convert a generic Pascal-VOC UAV dataset (UAV-PDD2023, UAPD) to COCO.

    Splits are created here (these sources ship no official split, or an
    ImageSets/Main list we don't rely on) using a fixed seed for reproducibility.
    If dedupe_hashes is given, images whose content hash is already in the set
    are skipped (used to drop UAPD frames that reappear in UAV-PDD2023).
    Returns (out_dir, updated_hash_set).
    """
    raw_root = Path(raw_root)
    pairs = _find_image_xml_pairs(raw_root)
    if not pairs:
        kids = sorted(p.name for p in raw_root.iterdir()) if raw_root.exists() else []
        raise RuntimeError(
            f"No image/xml pairs found under {raw_root}. Contents: {', '.join(kids[:20])}"
        )

    seen_hashes: set[str] = set(dedupe_hashes) if dedupe_hashes is not None else set()
    kept: list[tuple[Path, Path, str]] = []
    skipped_dupes = 0
    for img_path, xml_path in pairs:
        h = hashlib.sha1(img_path.read_bytes()).hexdigest()
        if h in seen_hashes:
            skipped_dupes += 1
            continue
        seen_hashes.add(h)
        kept.append((img_path, xml_path, h))

    if skipped_dupes:
        print(f"  {raw_root.name}: skipped {skipped_dupes} images already seen in another source")

    random.seed(seed)
    random.shuffle(kept)
    n = len(kept)
    n_val = max(1, int(val_frac * n))
    n_test = int(test_frac * n)
    splits = {
        "valid": kept[:n_val],
        "test": kept[n_val : n_val + n_test] if n_test else [],
        "train": kept[n_val + n_test :],
    }

    if out_dir.exists():
        shutil.rmtree(out_dir)

    for split, subset in splits.items():
        if not subset:
            continue
        sdir = out_dir / split
        sdir.mkdir(parents=True)
        images, anns = [], []
        img_id, ann_id = 1, 1
        for img_path, xml_path, _h in subset:
            try:
                with Image.open(img_path) as im:
                    w, h_px = im.size
            except Exception:
                continue
            kept_objs = 0
            objs = []
            for name, xmin, ymin, xmax, ymax in _parse_voc_objects(xml_path):
                resolved = resolve_class(name)
                if resolved is None:
                    continue
                bw, bh = max(0.0, xmax - xmin), max(0.0, ymax - ymin)
                if bw < 1 or bh < 1:
                    continue
                objs.append((resolved, xmin, ymin, bw, bh))
                kept_objs += 1
            if kept_objs == 0:
                continue
            shutil.copy2(img_path, sdir / img_path.name)
            images.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h_px})
            for resolved, xmin, ymin, bw, bh in objs:
                anns.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": CLASS_TO_ID[resolved] + 1,
                    "bbox": [xmin, ymin, bw, bh],
                    "area": float(bw * bh),
                    "iscrowd": 0,
                })
                ann_id += 1
            img_id += 1
        doc = {
            "info": {"description": f"{raw_root.name} VOC remapped {split}"},
            "licenses": [],
            "categories": coco_categories(CLASS_NAMES),
            "images": images,
            "annotations": anns,
        }
        (sdir / "_annotations.coco.json").write_text(json.dumps(doc), encoding="utf-8")
        print(f"  {raw_root.name} {split}: {len(images)} images, {len(anns)} anns")

    if not (out_dir / "train" / "_annotations.coco.json").exists():
        raise RuntimeError(f"VOC conversion of {raw_root} produced no train split")
    return out_dir, seen_hashes


# HighRPD's YOLO labels use dataset-fixed integer ids with NO data.yaml / classes.txt
# shipped (confirmed against the Mendeley record description): 0=line, 1=block, 2=pit.
_HIGHRPD_ID_TO_NAME = {0: "line crack", 1: "block crack", 2: "pit"}


def ingest_highrpd(
    raw_root: Path,
    out_dir: Path,
    *,
    val_frac: float = 0.12,
    seed: int = 50,
) -> Path:
    """Convert HighRPD's flat images/ + labels/ YOLO layout to COCO."""
    raw_root = Path(raw_root)
    img_dir = next((p for p in raw_root.rglob("images") if p.is_dir()), None)
    lbl_dir = next((p for p in raw_root.rglob("labels") if p.is_dir()), None)
    if img_dir is None or lbl_dir is None:
        kids = sorted(p.name for p in raw_root.iterdir()) if raw_root.exists() else []
        raise RuntimeError(
            f"Expected images/ and labels/ under {raw_root}. Contents: {', '.join(kids[:20])}"
        )

    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise RuntimeError(f"No images under {img_dir}")

    random.seed(seed)
    files = list(files)
    random.shuffle(files)
    n_val = max(1, int(val_frac * len(files)))
    splits = {"valid": files[:n_val], "train": files[n_val:]}

    if out_dir.exists():
        shutil.rmtree(out_dir)

    for split, subset in splits.items():
        sdir = out_dir / split
        sdir.mkdir(parents=True)
        images, anns = [], []
        img_id, ann_id = 1, 1
        for img_path in subset:
            with Image.open(img_path) as im:
                w, h = im.size
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            objs = []
            if lbl_path.exists():
                for line in lbl_path.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    raw_cls = int(float(parts[0]))
                    name = _HIGHRPD_ID_TO_NAME.get(raw_cls)
                    if name is None:
                        continue
                    resolved = resolve_class(name)
                    if resolved is None:
                        continue
                    xc, yc, bw, bh = map(float, parts[1:5])
                    x, y = (xc - bw / 2) * w, (yc - bh / 2) * h
                    bw_px, bh_px = bw * w, bh * h
                    if bw_px < 1 or bh_px < 1:
                        continue
                    objs.append((resolved, x, y, bw_px, bh_px))
            if not objs:
                continue
            shutil.copy2(img_path, sdir / img_path.name)
            images.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})
            for resolved, x, y, bw_px, bh_px in objs:
                anns.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": CLASS_TO_ID[resolved] + 1,
                    "bbox": [x, y, bw_px, bh_px],
                    "area": float(bw_px * bh_px),
                    "iscrowd": 0,
                })
                ann_id += 1
            img_id += 1
        doc = {
            "info": {"description": f"HighRPD remapped {split}"},
            "licenses": [],
            "categories": coco_categories(CLASS_NAMES),
            "images": images,
            "annotations": anns,
        }
        (sdir / "_annotations.coco.json").write_text(json.dumps(doc), encoding="utf-8")
        print(f"  HighRPD {split}: {len(images)} images, {len(anns)} anns")

    if not (out_dir / "train" / "_annotations.coco.json").exists():
        raise RuntimeError("HighRPD conversion produced no train split")
    return out_dir
