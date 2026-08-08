"""Export Stage-2 COCO layout to Ultralytics YOLO format + data.yaml."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .config import repo_root
from .taxonomy import CLASS_NAMES, CLASS_TO_ID


def _coco_split_to_yolo(
    coco_dir: Path,
    split_in: str,
    out_root: Path,
    split_out: str,
) -> int:
    ann_path = coco_dir / split_in / "_annotations.coco.json"
    if not ann_path.exists():
        print(f"  skip missing {ann_path}")
        return 0

    doc = json.loads(ann_path.read_text(encoding="utf-8"))
    id_to_name = {c["id"]: c["name"] for c in doc.get("categories", [])}
    # Map COCO category_id → 0-based YOLO class (our taxonomy order)
    coco_to_yolo: dict[int, int] = {}
    for cid, name in id_to_name.items():
        if name in CLASS_TO_ID:
            coco_to_yolo[cid] = CLASS_TO_ID[name]
        else:
            # Already remapped datasets use 1..6 matching CLASS_NAMES order
            idx = int(cid) - 1
            if 0 <= idx < len(CLASS_NAMES):
                coco_to_yolo[cid] = idx

    by_img: dict[int, list[dict]] = {}
    for a in doc.get("annotations", []):
        by_img.setdefault(a["image_id"], []).append(a)

    img_out = out_root / "images" / split_out
    lbl_out = out_root / "labels" / split_out
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    n = 0
    for im in doc.get("images", []):
        src = coco_dir / split_in / im["file_name"]
        if not src.exists():
            hits = list((coco_dir / split_in).rglob(Path(im["file_name"]).name))
            if not hits:
                continue
            src = hits[0]
        w = float(im.get("width") or 0)
        h = float(im.get("height") or 0)
        if w <= 0 or h <= 0:
            try:
                from PIL import Image

                with Image.open(src) as pil:
                    w, h = float(pil.size[0]), float(pil.size[1])
            except Exception:
                continue

        dst_name = Path(im["file_name"]).name
        shutil.copy2(src, img_out / dst_name)

        lines: list[str] = []
        for a in by_img.get(im["id"], []):
            yid = coco_to_yolo.get(a["category_id"])
            if yid is None:
                continue
            bbox = a.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            x, y, bw, bh = map(float, bbox[:4])
            if bw <= 0 or bh <= 0:
                continue
            xc = (x + bw / 2.0) / w
            yc = (y + bh / 2.0) / h
            nw = bw / w
            nh = bh / h
            # clip
            xc = min(max(xc, 0.0), 1.0)
            yc = min(max(yc, 0.0), 1.0)
            nw = min(max(nw, 0.0), 1.0)
            nh = min(max(nh, 0.0), 1.0)
            lines.append(f"{yid} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        (lbl_out / (Path(dst_name).stem + ".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
        n += 1

    print(f"  {split_in}→{split_out}: {n} images")
    return n


def export_coco_to_yolo(coco_dir: Path, out_dir: Path) -> Path:
    coco_dir = Path(coco_dir)
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    n_train = _coco_split_to_yolo(coco_dir, "train", out_dir, "train")
    n_val = _coco_split_to_yolo(coco_dir, "valid", out_dir, "val")
    if n_val == 0:
        # Ultralytics needs a val split — fall back to train
        print("  WARNING: no valid/ — pointing val at train")
        n_val = _coco_split_to_yolo(coco_dir, "train", out_dir, "val")

    if n_train == 0:
        raise SystemExit(f"No train images exported from {coco_dir}")

    abs_out = out_dir.resolve()
    yaml_text = (
        f"# Auto-exported from {coco_dir}\n"
        f"path: {abs_out}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n"
    )
    for i, name in enumerate(CLASS_NAMES):
        yaml_text += f"  {i}: {name}\n"
    (out_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")
    print(f"Wrote {out_dir / 'data.yaml'}")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export Stage-2 COCO dataset to YOLO format for Ultralytics RT-DETR."
    )
    p.add_argument(
        "--coco-dir",
        type=Path,
        default=None,
        help="Default: data/rfdetr/stage2",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: data/rfdetr/stage2_yolo",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = repo_root()
    coco = args.coco_dir or (root / "data" / "rfdetr" / "stage2")
    out = args.out_dir or (root / "data" / "rfdetr" / "stage2_yolo")
    export_coco_to_yolo(coco, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
