"""COCO / YOLO ingest and merge into the fixed 6-class layout."""
from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from pathlib import Path

from .taxonomy import CLASS_NAMES, CLASS_TO_ID, coco_categories, resolve_class


def remap_coco_json(src: Path, dst: Path) -> dict[str, int]:
    doc = json.loads(src.read_text(encoding="utf-8"))
    old_cats = {c["id"]: c.get("name", str(c["id"])) for c in doc.get("categories", [])}
    id_map: dict[int, int] = {}
    dropped: Counter[str] = Counter()

    for old_id, old_name in old_cats.items():
        resolved = resolve_class(old_name)
        if resolved is None:
            continue
        id_map[old_id] = CLASS_TO_ID[resolved] + 1

    new_anns = []
    for ann in doc.get("annotations", []):
        old_cid = ann.get("category_id")
        if old_cid not in id_map:
            dropped[f"dropped_ann:{old_cats.get(old_cid, old_cid)}"] += 1
            continue
        ann = dict(ann)
        ann["category_id"] = id_map[old_cid]
        if "bbox" not in ann and "segmentation" in ann:
            dropped["no_bbox"] += 1
            continue
        new_anns.append(ann)

    new_doc = {
        "info": doc.get("info", {"description": "rfdetr 6-class remapped"}),
        "licenses": doc.get("licenses", []),
        "categories": coco_categories(CLASS_NAMES),
        "images": doc.get("images", []),
        "annotations": new_anns,
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(new_doc), encoding="utf-8")
    print(
        f"  {src.name}: {len(new_anns)} anns kept, "
        f"{sum(dropped.values())} dropped, {len(new_doc['images'])} images"
    )
    if dropped:
        print("   ", dict(dropped))
    return dict(dropped)


def ensure_roboflow_coco_layout(dataset_dir: Path, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    splits_found = []
    for split in ("train", "valid", "test"):
        src_split = dataset_dir / split
        ann = src_split / "_annotations.coco.json"
        if not ann.exists():
            continue
        dst_split = out_dir / split
        dst_split.mkdir(parents=True)
        for img in src_split.iterdir():
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                shutil.copy2(img, dst_split / img.name)
        remap_coco_json(ann, dst_split / "_annotations.coco.json")
        splits_found.append(split)

    if "train" not in splits_found:
        raise RuntimeError(
            f"No train/_annotations.coco.json under {dataset_dir}. "
            "Export from Roboflow as COCO and re-run."
        )
    if "valid" not in splits_found:
        print(
            "WARNING: no valid/ split — RF-DETR early-stopping needs it. "
            "Will slice from train at merge if needed."
        )
    return out_dir


def yolo_to_coco_split(
    img_dir: Path, lbl_dir: Path, names: list[str], out_json: Path, out_img_dir: Path
) -> int:
    from PIL import Image

    if not img_dir.is_dir():
        return 0
    out_img_dir.mkdir(parents=True, exist_ok=True)
    images, anns = [], []
    ann_id, img_id = 1, 1
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in exts:
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        shutil.copy2(img_path, out_img_dir / img_path.name)
        images.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})

        lbl = lbl_dir / (img_path.stem + ".txt")
        if lbl.exists():
            for line in lbl.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                raw_cls = int(float(parts[0]))
                if raw_cls < 0 or raw_cls >= len(names):
                    continue
                resolved = resolve_class(names[raw_cls])
                if resolved is None:
                    continue
                cid = CLASS_TO_ID[resolved] + 1
                if len(parts) == 5:
                    xc, yc, bw, bh = map(float, parts[1:5])
                    x = (xc - bw / 2) * w
                    y = (yc - bh / 2) * h
                    bw_px, bh_px = bw * w, bh * h
                else:
                    coords = list(map(float, parts[1:]))
                    xs = [coords[i] * w for i in range(0, len(coords), 2)]
                    ys = [coords[i] * h for i in range(1, len(coords), 2)]
                    if not xs or not ys:
                        continue
                    x, y = min(xs), min(ys)
                    bw_px, bh_px = max(xs) - x, max(ys) - y
                anns.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cid,
                    "bbox": [x, y, bw_px, bh_px],
                    "area": float(max(bw_px, 0) * max(bh_px, 0)),
                    "iscrowd": 0,
                })
                ann_id += 1
        img_id += 1

    doc = {
        "info": {"description": "yolo to coco remapped"},
        "licenses": [],
        "categories": coco_categories(CLASS_NAMES),
        "images": images,
        "annotations": anns,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc), encoding="utf-8")
    return len(images)


def _resolve_yolo_split_dirs(
    dataset_dir: Path, rel: str
) -> tuple[Path, Path] | tuple[None, None]:
    p = Path(rel)
    base = p if p.is_absolute() else (dataset_dir / rel)
    base = base.resolve()
    if base.name == "images" and base.is_dir():
        lbl = base.parent / "labels"
        return (base, lbl) if lbl.is_dir() else (base, base.parent / "labels")
    if (base / "images").is_dir():
        return base / "images", base / "labels"
    if base.is_dir():
        return base, base.parent / "labels"
    return None, None


def convert_yolo_dataset(dataset_dir: Path, out_dir: Path) -> Path:
    import yaml

    dataset_dir = Path(dataset_dir)
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        cands = list(dataset_dir.rglob("data.yaml"))
        if not cands:
            raise RuntimeError(f"No data.yaml under {dataset_dir}")
        data_yaml = cands[0]
        dataset_dir = data_yaml.parent

    doc = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = doc.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    if not names:
        raise RuntimeError("data.yaml has no names")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    mapping = [
        ("train", ["train"]),
        ("valid", ["val", "valid"]),
        ("test", ["test"]),
    ]
    for out_split, keys in mapping:
        rel = None
        for k in keys:
            if doc.get(k):
                rel = doc[k]
                break
        if not rel:
            continue
        img_dir, lbl_dir = _resolve_yolo_split_dirs(dataset_dir, str(rel))
        if img_dir is None:
            print(f"  YOLO {out_split}: path not found for {rel}")
            continue
        n = yolo_to_coco_split(
            img_dir,
            lbl_dir,
            names,
            out_dir / out_split / "_annotations.coco.json",
            out_dir / out_split,
        )
        print(f"  YOLO {out_split}: {n} images -> {out_dir / out_split}")

    if not (out_dir / "train" / "_annotations.coco.json").exists():
        raise RuntimeError("YOLO conversion produced no train split")
    return out_dir


def unwrap_zip_root(path: Path) -> Path:
    kids = [p for p in path.iterdir() if not p.name.startswith((".", "__"))]
    if len(kids) == 1 and kids[0].is_dir():
        return kids[0]
    return path


def force_annotations_to_pothole(dataset_dir: Path) -> None:
    pothole_id = CLASS_TO_ID["pothole"] + 1
    for ann_path in dataset_dir.rglob("_annotations.coco.json"):
        doc = json.loads(ann_path.read_text(encoding="utf-8"))
        doc["categories"] = coco_categories(CLASS_NAMES)
        for a in doc.get("annotations", []):
            a["category_id"] = pothole_id
        ann_path.write_text(json.dumps(doc), encoding="utf-8")
        print(f"  forced pothole ids in {ann_path.relative_to(dataset_dir)}")


def prepare_bharatpothole(raw_root: Path, out_dir: Path) -> Path:
    raw_root = Path(raw_root)
    if list(raw_root.rglob("_annotations.coco.json")):
        coco_root = raw_root
        if not (coco_root / "train" / "_annotations.coco.json").exists():
            hits = list(raw_root.rglob("train/_annotations.coco.json"))
            if hits:
                coco_root = hits[0].parent.parent
        ensure_roboflow_coco_layout(coco_root, out_dir)
        force_annotations_to_pothole(out_dir)
        return out_dir

    if list(raw_root.rglob("data.yaml")):
        convert_yolo_dataset(raw_root, out_dir)
        force_annotations_to_pothole(out_dir)
        return out_dir

    from PIL import Image as _Image

    img_dirs = [p for p in raw_root.rglob("images") if p.is_dir()]
    for img_dir in img_dirs:
        lbl_dir = img_dir.parent / "labels"
        if not lbl_dir.is_dir():
            continue
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in exts]
        if not files:
            continue
        random.seed(1337)
        files = list(files)
        random.shuffle(files)
        n_val = max(1, int(0.15 * len(files)))
        splits = {"valid": files[:n_val], "train": files[n_val:]}
        if out_dir.exists():
            shutil.rmtree(out_dir)
        for split, subset in splits.items():
            sdir = out_dir / split
            sdir.mkdir(parents=True)
            images, anns = [], []
            ann_id, img_id = 1, 1
            pothole_id = CLASS_TO_ID["pothole"] + 1
            for img_path in subset:
                with _Image.open(img_path) as im:
                    w, h = im.size
                shutil.copy2(img_path, sdir / img_path.name)
                images.append(
                    {"id": img_id, "file_name": img_path.name, "width": w, "height": h}
                )
                lbl = lbl_dir / (img_path.stem + ".txt")
                if lbl.exists():
                    for line in lbl.read_text(encoding="utf-8").splitlines():
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        if len(parts) == 5:
                            xc, yc, bw, bh = map(float, parts[1:5])
                            x = (xc - bw / 2) * w
                            y = (yc - bh / 2) * h
                            bw_px, bh_px = bw * w, bh * h
                        else:
                            coords = list(map(float, parts[1:]))
                            xs = [coords[i] * w for i in range(0, len(coords), 2)]
                            ys = [coords[i] * h for i in range(1, len(coords), 2)]
                            if not xs:
                                continue
                            x, y = min(xs), min(ys)
                            bw_px, bh_px = max(xs) - x, max(ys) - y
                        anns.append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": pothole_id,
                            "bbox": [x, y, bw_px, bh_px],
                            "area": float(max(bw_px, 0) * max(bh_px, 0)),
                            "iscrowd": 0,
                        })
                        ann_id += 1
                img_id += 1
            doc = {
                "info": {"description": "bharatpothole remapped"},
                "licenses": [],
                "categories": coco_categories(CLASS_NAMES),
                "images": images,
                "annotations": anns,
            }
            (sdir / "_annotations.coco.json").write_text(json.dumps(doc), encoding="utf-8")
            print(f"  BharatPotHole {split}: {len(images)} images, {len(anns)} anns")
        return out_dir

    raise RuntimeError(
        f"Could not interpret BharatPotHole layout under {raw_root}. "
        "Expected COCO, YOLO data.yaml, or images/+labels/."
    )


def find_coco_root(search_roots: list[Path]) -> Path | None:
    """Locate a Roboflow-style COCO export (train/_annotations.coco.json)."""
    seen: set[Path] = set()
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        for ann in root.rglob("train/_annotations.coco.json"):
            cand = ann.parent.parent
            if cand not in seen:
                seen.add(cand)
                return cand
        for ann in root.rglob("_annotations.coco.json"):
            # if file is directly under train/
            if ann.parent.name == "train":
                cand = ann.parent.parent
                if cand not in seen:
                    return cand
    return None


def ingest_to_coco(raw_root: Path, out_dir: Path, force_pothole: bool = False) -> Path:
    raw_root = Path(raw_root)
    # If the given path is empty / wrong, search nearby for a COCO export
    if not list(raw_root.rglob("_annotations.coco.json")) and not list(
        raw_root.rglob("data.yaml")
    ):
        found = find_coco_root([raw_root, Path.cwd(), raw_root.parent])
        if found is not None:
            print(f"  discovered COCO root at {found}")
            raw_root = found

    if force_pothole:
        return prepare_bharatpothole(raw_root, out_dir)
    if list(raw_root.rglob("_annotations.coco.json")):
        coco_root = raw_root
        if not (coco_root / "train" / "_annotations.coco.json").exists():
            hits = list(raw_root.rglob("train/_annotations.coco.json"))
            if hits:
                coco_root = hits[0].parent.parent
        ensure_roboflow_coco_layout(coco_root, out_dir)
        return out_dir
    if list(raw_root.rglob("data.yaml")):
        convert_yolo_dataset(raw_root, out_dir)
        return out_dir
    kids = sorted(p.name for p in raw_root.iterdir()) if raw_root.exists() else []
    raise RuntimeError(
        f"No COCO/YOLO layout under {raw_root}. Contents: {', '.join(kids[:20])}"
    )


def merge_coco_datasets(parts: list[Path], out_dir: Path) -> Path:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for split in ("train", "valid", "test"):
        images, anns = [], []
        next_img_id, next_ann_id = 1, 1
        sdir = out_dir / split
        sdir.mkdir(parents=True, exist_ok=True)
        for part in parts:
            ann_path = part / split / "_annotations.coco.json"
            if not ann_path.exists():
                continue
            doc = json.loads(ann_path.read_text(encoding="utf-8"))
            id_remap: dict[int, int] = {}
            tag = part.name.replace(" ", "_")[:40]
            for im in doc.get("images", []):
                old_id = im["id"]
                src = part / split / im["file_name"]
                if not src.exists():
                    cands = list((part / split).rglob(Path(im["file_name"]).name))
                    if not cands:
                        continue
                    src = cands[0]
                new_name = f"{tag}__{Path(im['file_name']).name}"
                dst = sdir / new_name
                if dst.exists():
                    new_name = f"{tag}__{next_img_id}_{Path(im['file_name']).name}"
                    dst = sdir / new_name
                shutil.copy2(src, dst)
                id_remap[old_id] = next_img_id
                images.append({
                    "id": next_img_id,
                    "file_name": new_name,
                    "width": im.get("width"),
                    "height": im.get("height"),
                })
                next_img_id += 1
            for a in doc.get("annotations", []):
                if a.get("image_id") not in id_remap:
                    continue
                na = dict(a)
                na["id"] = next_ann_id
                na["image_id"] = id_remap[a["image_id"]]
                anns.append(na)
                next_ann_id += 1
        if images:
            merged = {
                "info": {"description": f"merged {split}"},
                "licenses": [],
                "categories": coco_categories(CLASS_NAMES),
                "images": images,
                "annotations": anns,
            }
            (sdir / "_annotations.coco.json").write_text(
                json.dumps(merged), encoding="utf-8"
            )
            print(f"  MERGED {split}: {len(images)} images, {len(anns)} anns")

    if not (out_dir / "train" / "_annotations.coco.json").exists():
        raise RuntimeError("Merge produced no train split")

    if not (out_dir / "valid" / "_annotations.coco.json").exists():
        print("WARNING: no valid/ — slicing 10% of train")
        full = json.loads((out_dir / "train" / "_annotations.coco.json").read_text())
        n_val = max(1, len(full["images"]) // 10)
        val_ids = {im["id"] for im in full["images"][-n_val:]}
        val_doc = {
            "info": {"description": "valid from train slice"},
            "licenses": [],
            "categories": coco_categories(CLASS_NAMES),
            "images": [im for im in full["images"] if im["id"] in val_ids],
            "annotations": [a for a in full["annotations"] if a["image_id"] in val_ids],
        }
        train_doc = {
            "info": full.get("info", {}),
            "licenses": [],
            "categories": coco_categories(CLASS_NAMES),
            "images": [im for im in full["images"] if im["id"] not in val_ids],
            "annotations": [a for a in full["annotations"] if a["image_id"] not in val_ids],
        }
        vdir = out_dir / "valid"
        vdir.mkdir(exist_ok=True)
        for im in val_doc["images"]:
            src = out_dir / "train" / im["file_name"]
            if src.exists():
                shutil.copy2(src, vdir / im["file_name"])
        (vdir / "_annotations.coco.json").write_text(json.dumps(val_doc), encoding="utf-8")
        (out_dir / "train" / "_annotations.coco.json").write_text(
            json.dumps(train_doc), encoding="utf-8"
        )
        print(f"  created valid/ with {len(val_doc['images'])} images")
    return out_dir


def print_class_histogram(dataset_dir: Path, split: str = "train") -> None:
    ann = dataset_dir / split / "_annotations.coco.json"
    doc = json.loads(ann.read_text(encoding="utf-8"))
    id_to_name = {c["id"]: c["name"] for c in doc["categories"]}
    hist = Counter(id_to_name[a["category_id"]] for a in doc["annotations"])
    print(f"\n{split} instance counts:")
    for n in CLASS_NAMES:
        print(f"  {n:22s} {hist.get(n, 0)}")
    print(f"Images: {len(doc['images'])}")


def prepare_rdd_voc(raw_root: Path, out_dir: Path, *, val_frac: float = 0.12) -> Path:
    """Convert RDD2022-style Pascal VOC (images + XML) into remapped COCO splits."""
    import xml.etree.ElementTree as ET

    from PIL import Image as _Image

    raw_root = Path(raw_root)
    # Common layouts: .../train/images + train/annotations, or images/ + xmls/
    img_dirs: list[Path] = []
    for cand in (
        raw_root / "train" / "images",
        raw_root / "India" / "train" / "images",
        raw_root / "RDD2022_India" / "train" / "images",
    ):
        if cand.is_dir():
            img_dirs.append(cand)
    if not img_dirs:
        img_dirs = [p for p in raw_root.rglob("images") if p.is_dir() and "train" in str(p)]
    if not img_dirs:
        raise RuntimeError(f"No train/images under RDD root {raw_root}")

    exts = {".jpg", ".jpeg", ".png"}
    samples: list[tuple[Path, Path]] = []
    for img_dir in img_dirs:
        ann_dir_cands = [
            img_dir.parent / "annotations",
            img_dir.parent / "annotations" / "xmls",
            img_dir.parent / "xmls",
            img_dir.parent / "labels",
        ]
        ann_dir = next((d for d in ann_dir_cands if d.is_dir()), None)
        if ann_dir is None:
            # XMLs next to images
            ann_dir = img_dir
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in exts:
                continue
            xml_path = ann_dir / f"{img_path.stem}.xml"
            if not xml_path.exists():
                # sometimes nested
                hits = list(ann_dir.rglob(f"{img_path.stem}.xml"))
                if not hits:
                    continue
                xml_path = hits[0]
            samples.append((img_path, xml_path))

    if not samples:
        raise RuntimeError(f"No image/XML pairs under {raw_root}")

    random.seed(2022)
    samples = list(samples)
    random.shuffle(samples)
    n_val = max(1, int(val_frac * len(samples)))
    splits = {"valid": samples[:n_val], "train": samples[n_val:]}

    if out_dir.exists():
        shutil.rmtree(out_dir)

    for split, subset in splits.items():
        sdir = out_dir / split
        sdir.mkdir(parents=True)
        images, anns = [], []
        ann_id, img_id = 1, 1
        for img_path, xml_path in subset:
            try:
                with _Image.open(img_path) as im:
                    w, h = im.size
            except Exception:
                continue
            try:
                root = ET.parse(xml_path).getroot()
            except ET.ParseError:
                continue
            kept = 0
            for obj in root.findall("object"):
                name = (obj.findtext("name") or "").strip()
                resolved = resolve_class(name)
                if resolved is None:
                    continue
                bb = obj.find("bndbox")
                if bb is None:
                    continue
                xmin = float(bb.findtext("xmin") or 0)
                ymin = float(bb.findtext("ymin") or 0)
                xmax = float(bb.findtext("xmax") or 0)
                ymax = float(bb.findtext("ymax") or 0)
                bw, bh = max(0.0, xmax - xmin), max(0.0, ymax - ymin)
                if bw < 1 or bh < 1:
                    continue
                anns.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": CLASS_TO_ID[resolved] + 1,
                    "bbox": [xmin, ymin, bw, bh],
                    "area": float(bw * bh),
                    "iscrowd": 0,
                })
                ann_id += 1
                kept += 1
            # Keep images even with 0 kept anns? Skip empty — reduces noise from unlabeled frames
            if kept == 0:
                continue
            shutil.copy2(img_path, sdir / img_path.name)
            images.append(
                {"id": img_id, "file_name": img_path.name, "width": w, "height": h}
            )
            img_id += 1
        doc = {
            "info": {"description": f"RDD VOC remapped {split}"},
            "licenses": [],
            "categories": coco_categories(CLASS_NAMES),
            "images": images,
            "annotations": anns,
        }
        (sdir / "_annotations.coco.json").write_text(json.dumps(doc), encoding="utf-8")
        print(f"  RDD VOC {split}: {len(images)} images, {len(anns)} anns")

    if not (out_dir / "train" / "_annotations.coco.json").exists():
        raise RuntimeError("RDD VOC conversion produced no train split")
    return out_dir


def cap_majority_class_images(
    dataset_dir: Path,
    *,
    majority: str = "pothole",
    max_fraction: float = 0.45,
    split: str = "train",
    seed: int = 42,
) -> None:
    """Drop excess images that only contain the majority class (in-place).

    Keeps every image that has at least one non-majority annotation. Among
    majority-only images, keeps enough so majority instances stay under
    ``max_fraction`` of all instances (best-effort).
    """
    ann_path = dataset_dir / split / "_annotations.coco.json"
    if not ann_path.exists():
        return
    doc = json.loads(ann_path.read_text(encoding="utf-8"))
    maj_id = CLASS_TO_ID[majority] + 1
    by_img: dict[int, list[dict]] = {}
    for a in doc["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)

    keep_ids: set[int] = set()
    maj_only: list[int] = []
    for im in doc["images"]:
        iid = im["id"]
        anns = by_img.get(iid, [])
        if not anns:
            continue
        if any(a["category_id"] != maj_id for a in anns):
            keep_ids.add(iid)
        else:
            maj_only.append(iid)

    # Count non-majority instances already kept
    non_maj = sum(
        1
        for iid in keep_ids
        for a in by_img.get(iid, [])
        if a["category_id"] != maj_id
    )
    maj_kept = sum(
        1
        for iid in keep_ids
        for a in by_img.get(iid, [])
        if a["category_id"] == maj_id
    )
    # Target: maj / (maj + non_maj) <= max_fraction  =>  maj <= max_fraction/(1-max_fraction) * non_maj
    if max_fraction >= 0.99 or non_maj == 0:
        print(f"  cap skipped (non_maj={non_maj}, max_fraction={max_fraction})")
        return
    max_maj = int((max_fraction / (1.0 - max_fraction)) * non_maj)
    budget = max(0, max_maj - maj_kept)

    random.seed(seed)
    random.shuffle(maj_only)
    # Prefer images with more majority boxes when filling budget
    maj_only.sort(key=lambda iid: len(by_img.get(iid, [])), reverse=True)
    for iid in maj_only:
        n = len(by_img.get(iid, []))
        if budget <= 0:
            break
        if n <= budget:
            keep_ids.add(iid)
            budget -= n

    before_imgs, before_anns = len(doc["images"]), len(doc["annotations"])
    new_images = [im for im in doc["images"] if im["id"] in keep_ids]
    new_anns = [a for a in doc["annotations"] if a["image_id"] in keep_ids]
    # Drop orphaned image files for majority-only discards (optional; save disk)
    keep_names = {im["file_name"] for im in new_images}
    sdir = dataset_dir / split
    for p in sdir.iterdir():
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} and p.name not in keep_names:
            p.unlink(missing_ok=True)

    doc["images"] = new_images
    doc["annotations"] = new_anns
    ann_path.write_text(json.dumps(doc), encoding="utf-8")
    print(
        f"  capped {majority}-only images on {split}: "
        f"{before_imgs}→{len(new_images)} imgs, {before_anns}→{len(new_anns)} anns"
    )
