#!/usr/bin/env python3
"""Build notebooks/colab_rfdetr_train.ipynb — run once, then delete this helper if desired."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "colab_rfdetr_train.ipynb"


def md(text: str) -> dict:
    lines = text.strip("\n").split("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [ln + "\n" for ln in lines],
    }


def code(text: str) -> dict:
    lines = text.strip("\n").split("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [ln + "\n" for ln in lines],
    }


cells: list[dict] = []

cells.append(md("""
# RF-DETR Medium — rural road defects (Colab)

> **Prefer a GPU VM / SSH?** For headless Stage-1 training (e.g. RTX 5090), use the
> repo scripts instead of this notebook:
> `cp .env.example .env` → `./scripts/setup_rfdetr_vm.sh` → `tmux new -s rfdetr` →
> `./scripts/run_stage1.sh`. See the README section **RF-DETR Stage 1 (headless VM)**.

Standalone Colab notebook: train **RFDETRMedium** on **India/rural-oriented** public
data (Stage 1), fine-tune on your own ~2000 rural images (Stage 2), then test on a
Drive or YouTube video with **road-segment → detect → gate**.

**Classes (fixed order — do not reorder):**

| id | name |
|---:|------|
| 0 | `alligator_crack` |
| 1 | `drainage_issue` |
| 2 | `longitudinal_crack` |
| 3 | `pothole` |
| 4 | `ravelling` |
| 5 | `edge_damage` |

### How to use this notebook

1. **Runtime → Change runtime type → T4 GPU** (or better). Training on CPU is refused.
2. Colab Secrets: `KAGGLE_USERNAME` + `KAGGLE_KEY` (Stage 1 BharatPotHole). Optional: `ROBOFLOW_API_KEY`.
3. Run cells top to bottom. Stage 2 and video test can be skipped if you only want Stage 1.
4. Export weights to Drive before the runtime recycles.

### Design choices (locked)

- **Rural-first Stage 1** — merge CRRI + BharatPotHole + ravelling/edge Roboflow sets (not asphalt RDD2022 alone).
- **All 6 classes in one model** — not one-class-at-a-time.
- **Train on full frames** — do **not** blank non-road pixels during training (avoids train/serve mismatch with normal Roboflow labels).
- **Segment then detect at video time** — geometric/classical road mask, then keep detections that overlap the road.
- **Recall-first inference** — low confidence; road gate cuts off-road false positives.
- Stage 1 merges CRRI + BharatPotHole + ravelling/edge sets into one 6-class head; **Stage 2 on your labels** still matters for drainage and village domain.

> There is no large public 6-class gravel-village set matching this taxonomy. Skipping
> asphalt RDD is intentional. Expecting Stage 1 alone to learn all 6 rural classes is not.
"""))

cells.append(md("## 1 · Check the GPU"))

cells.append(code("""
import subprocess, sys

print(subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
     "--format=csv,noheader"],
    capture_output=True, text=True,
).stdout.strip() or "NO GPU DETECTED")
print()

# Probe torch in a subprocess so we never importlib.reload(torch) in this process
# (that re-registers TORCH_LIBRARY and crashes Colab).
probe = subprocess.run(
    [sys.executable, "-c",
     "import torch; print(torch.__version__); print(torch.cuda.is_available()); "
     "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"],
    capture_output=True, text=True,
)
print(probe.stdout.strip() or probe.stderr.strip())
if "True" not in (probe.stdout or ""):
    raise SystemExit(
        "No CUDA GPU. Runtime → Change runtime type → T4 GPU, "
        "then Runtime → Restart session, and re-run from here."
    )
"""))

cells.append(md("""
## 2 · Install dependencies

Torch stays as Colab's CUDA build. We only add RF-DETR training extras and helpers.
"""))

cells.append(code("""
import subprocess, sys

pkgs = [
    "rfdetr[train]",
    "supervision>=0.25.0",
    "roboflow>=1.1.0",
    "kaggle",
    "yt-dlp",
    "opencv-python-headless",
    "pycocotools",
    "pillow",
    "tqdm",
    "pyyaml",
]
print("Installing (a few minutes on a fresh runtime) ...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

probe = subprocess.run(
    [sys.executable, "-c", "from rfdetr import RFDETRMedium; print('RFDETRMedium OK')"],
    capture_output=True, text=True,
)
print(probe.stdout.strip() or probe.stderr)
if probe.returncode != 0:
    raise SystemExit("rfdetr failed to import — scroll up for the pip error.")
print("Ready.")
"""))

cells.append(md("""
## 3 · Auth, paths, and shared config

Colab Secrets (left sidebar) → add → enable notebook access:

- `KAGGLE_USERNAME` + `KAGGLE_KEY` — Stage 1 BharatPotHole download
- `ROBOFLOW_API_KEY` — optional (Stage 2 Roboflow project, or alternate Universe sources)
"""))

cells.append(code("""
#@title Shared settings { display-mode: "form" }
MOUNT_DRIVE = True  #@param {type:"boolean"}
DRIVE_ROOT  = "/content/drive/MyDrive/rfdetr_road_defects"  #@param {type:"string"}
WORK_ROOT   = "/content/rfdetr_work"  #@param {type:"string"}

#@markdown ### Stage 1 (rural / India public data — BharatPotHole by default)
RUN_STAGE1          = True  #@param {type:"boolean"}
STAGE1_EPOCHS       = 50  #@param {type:"integer"}
STAGE1_BATCH        = 4  #@param {type:"integer"}
STAGE1_GRAD_ACCUM   = 4  #@param {type:"integer"}
STAGE1_LR           = 0.0001  #@param {type:"number"}
STAGE1_EARLY_STOP   = True  #@param {type:"boolean"}

#@markdown ### Stage 2 (your ~2000 images)
RUN_STAGE2          = True  #@param {type:"boolean"}
STAGE2_EPOCHS       = 30  #@param {type:"integer"}
STAGE2_BATCH        = 4  #@param {type:"integer"}
STAGE2_GRAD_ACCUM   = 4  #@param {type:"integer"}
STAGE2_LR           = 0.00002  #@param {type:"number"}

#@markdown ### T4 tip
#@markdown Effective batch ≈ BATCH × GRAD_ACCUM. Target ~16 on T4.
#@markdown If nvidia-smi shows >6 GB free after a few Stage-1 steps, raise BATCH to 8
#@markdown and set GRAD_ACCUM to 2.

from pathlib import Path
import json, os

CLASS_NAMES = [
    "alligator_crack",
    "drainage_issue",
    "longitudinal_crack",
    "pothole",
    "ravelling",
    "edge_damage",
]
CLASS_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}

WORK = Path(WORK_ROOT)
for sub in ("datasets/stage1", "datasets/stage2", "runs/stage1", "runs/stage2",
            "videos", "exports", "weights"):
    (WORK / sub).mkdir(parents=True, exist_ok=True)

if MOUNT_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    Path(DRIVE_ROOT).mkdir(parents=True, exist_ok=True)
    print(f"Drive root: {DRIVE_ROOT}")

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "").strip()
KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "").strip()
KAGGLE_KEY = os.environ.get("KAGGLE_KEY", "").strip()
try:
    from google.colab import userdata
    ROBOFLOW_API_KEY = (userdata.get("ROBOFLOW_API_KEY") or ROBOFLOW_API_KEY).strip()
    KAGGLE_USERNAME = (userdata.get("KAGGLE_USERNAME") or KAGGLE_USERNAME).strip()
    KAGGLE_KEY = (userdata.get("KAGGLE_KEY") or KAGGLE_KEY).strip()
except Exception:
    pass

if KAGGLE_USERNAME and KAGGLE_KEY:
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    (kaggle_dir / "kaggle.json").write_text(
        json.dumps({"username": KAGGLE_USERNAME, "key": KAGGLE_KEY})
    )
    os.chmod(kaggle_dir / "kaggle.json", 0o600)
    print("Kaggle credentials loaded.")
else:
    print("WARNING: KAGGLE_USERNAME/KAGGLE_KEY missing — use STAGE1_SOURCE=drive_zip "
          "with a BharatPotHole zip, or set the secrets.")

if ROBOFLOW_API_KEY:
    print("ROBOFLOW_API_KEY loaded.")
else:
    print("ROBOFLOW_API_KEY not set (ok unless using Roboflow downloads).")

(WORK / "exports" / "class_names.json").write_text(json.dumps(CLASS_NAMES, indent=2))
print("Classes:", CLASS_NAMES)
print(f"Work dir: {WORK}")
print(f"Stage1 effective batch: {STAGE1_BATCH * STAGE1_GRAD_ACCUM}")
print(f"Stage2 effective batch: {STAGE2_BATCH * STAGE2_GRAD_ACCUM}")
"""))

cells.append(md("""
## 4 · VRAM headroom (maximize T4)

Print free memory so you can bump `STAGE1_BATCH` / `STAGE2_BATCH` before a long train.
"""))

cells.append(code("""
import subprocess, sys

out = subprocess.run(
    [sys.executable, "-c",
     "import torch; "
     "free, total = torch.cuda.mem_get_info(); "
     "print(f'GPU: {torch.cuda.get_device_name(0)}'); "
     "print(f'VRAM free/total: {free/1e9:.2f} / {total/1e9:.2f} GB'); "
     "print('Suggested: STAGE*_BATCH=4, STAGE*_GRAD_ACCUM=4 (eff 16). Try batch=8 if free>8GB.')"],
    capture_output=True, text=True, check=True,
).stdout
print(out)
"""))

cells.append(md("""
## 5 · Helpers — COCO remap into the fixed 6-class taxonomy

RDD2022 / Universe exports use their own class names (`D00`, `D40`, `Pothole`, …).
Everything is rewritten into our six names before training so Stage 1 and Stage 2 share
one head. Classes we cannot map are **dropped** (not silently remapped by index).

**Transverse cracks (`D10`):** folded into `longitudinal_crack` so Stage 1 still uses
linear crack signal without inventing a 7th class. Change `CLASS_ALIASES` if you prefer
to drop them instead.
"""))

cells.append(code("""
import json, re, shutil, zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Canonical aliases -> one of CLASS_NAMES. Keys are lowercased / stripped.
CLASS_ALIASES = {
    # RDD / CRDDC codes
    "d00": "longitudinal_crack",
    "d10": "longitudinal_crack",   # transverse folded in (see markdown above)
    "d20": "alligator_crack",
    "d40": "pothole",
    "longitudinal crack": "longitudinal_crack",
    "longitudinal_crack": "longitudinal_crack",
    "longitudinal cracking": "longitudinal_crack",
    "longitudinal-crack": "longitudinal_crack",
    "lateral-crack": "longitudinal_crack",
    "lateral crack": "longitudinal_crack",
    "transverse crack": "longitudinal_crack",
    "transverse_crack": "longitudinal_crack",
    "transverse cracking": "longitudinal_crack",
    "tc": "longitudinal_crack",
    "lc": "longitudinal_crack",
    "alligator crack": "alligator_crack",
    "alligator_crack": "alligator_crack",
    "alligator cracking": "alligator_crack",
    "alligator": "alligator_crack",
    "fatigue crack": "alligator_crack",
    "reticular crack": "alligator_crack",
    "reticular_crack": "alligator_crack",
    "rc": "alligator_crack",
    "pothole": "pothole",
    "potholes": "pothole",
    "pot hole": "pothole",
    "pot-hole": "pothole",
    "high pothole": "pothole",
    "medium pothole": "pothole",
    "low pothole": "pothole",
    "ravelling": "ravelling",
    "raveling": "ravelling",
    "high ravelling": "ravelling",
    "medium ravelling": "ravelling",
    "low ravelling": "ravelling",
    "high raveling": "ravelling",
    "medium raveling": "ravelling",
    "low raveling": "ravelling",
    "edge_damage": "edge_damage",
    "edge damage": "edge_damage",
    "edgecrack": "edge_damage",
    "edge crack": "edge_damage",
    "edge cracking": "edge_damage",
    "edge-cracking": "edge_damage",
    "high edge cracking": "edge_damage",
    "medium edge cracking": "edge_damage",
    "low edge cracking": "edge_damage",
    "edge drop": "edge_damage",
    "edge-drop": "edge_damage",
    "edge_drop": "edge_damage",
    "lane/shoulder drop-off": "edge_damage",
    "lane shoulder drop-off": "edge_damage",
    "shoulder drop-off": "edge_damage",
    "drainage_issue": "drainage_issue",
    "drainage issue": "drainage_issue",
    "drainage": "drainage_issue",
    "water-stagnation": "drainage_issue",
    "water stagnation": "drainage_issue",
    "water_stagnation": "drainage_issue",
    "water-buildup": "drainage_issue",
    "water buildup": "drainage_issue",
    "waterlogging": "drainage_issue",
    "water logging": "drainage_issue",
    # drop non-targets explicitly
    "repair": None,
    "repaired": None,
    "patching": None,
    "patch": None,
    "block crack": None,
    "other corruption": None,
    "other": None,
    "rutting": None,
    "medium rutting": None,
    "high rutting": None,
    "low rutting": None,
    "striping": None,
    "bump": None,
    "loose-gravel": None,
    "sand-buildup": None,
    "facecrack": None,
    "cleaning-required": None,
}


def _norm(name: str) -> str:
    s = str(name).strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\\s+", " ", s)
    # strip severity prefixes used by some Roboflow projects
    for pref in ("high ", "medium ", "med ", "low "):
        if s.startswith(pref):
            s = s[len(pref):]
            break
    return s.strip()


def resolve_class(name: str) -> str | None:
    key = _norm(name)
    if key in CLASS_ALIASES:
        return CLASS_ALIASES[key]
    if key in CLASS_TO_ID:
        return key
    snake = key.replace(" ", "_")
    if snake in CLASS_TO_ID:
        return snake
    if snake in CLASS_ALIASES:
        return CLASS_ALIASES[snake]
    # substring fallback for compound Roboflow names
    for needle, dest in (
        ("alligator", "alligator_crack"),
        ("ravelling", "ravelling"),
        ("raveling", "ravelling"),
        ("pothole", "pothole"),
        ("edge cracking", "edge_damage"),
        ("edge crack", "edge_damage"),
        ("edge drop", "edge_damage"),
        ("longitudinal", "longitudinal_crack"),
        ("water stagnation", "drainage_issue"),
        ("waterlogging", "drainage_issue"),
        ("drainage", "drainage_issue"),
    ):
        if needle in key:
            return dest
    return None


def _coco_categories(names: list[str]) -> list[dict]:
    # RF-DETR / Roboflow COCO often use 1-based category ids.
    return [{"id": i + 1, "name": n, "supercategory": "road_defect"} for i, n in enumerate(names)]


def remap_coco_json(src: Path, dst: Path, image_dir: Path | None = None) -> dict[str, int]:
    # Rewrite one COCO JSON onto CLASS_NAMES. Returns dropped-by-reason counts.
    doc = json.loads(src.read_text(encoding="utf-8"))
    old_cats = {c["id"]: c.get("name", str(c["id"])) for c in doc.get("categories", [])}
    id_map: dict[int, int] = {}
    dropped = Counter()

    for old_id, old_name in old_cats.items():
        resolved = resolve_class(old_name)
        if resolved is None:
            continue
        id_map[old_id] = CLASS_TO_ID[resolved] + 1  # 1-based

    new_anns = []
    for ann in doc.get("annotations", []):
        old_cid = ann.get("category_id")
        if old_cid not in id_map:
            dropped[f"dropped_ann:{old_cats.get(old_cid, old_cid)}"] += 1
            continue
        ann = dict(ann)
        ann["category_id"] = id_map[old_cid]
        # detection boxes only
        if "bbox" not in ann and "segmentation" in ann:
            dropped["no_bbox"] += 1
            continue
        new_anns.append(ann)

    images = doc.get("images", [])
    keep_ids = {a["image_id"] for a in new_anns}
    # keep images even with zero anns (negatives help), unless empty export
    new_doc = {
        "info": doc.get("info", {"description": "rfdetr 6-class remapped"}),
        "licenses": doc.get("licenses", []),
        "categories": _coco_categories(CLASS_NAMES),
        "images": images,
        "annotations": new_anns,
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(new_doc), encoding="utf-8")
    print(f"  {src.name}: {len(new_anns)} anns kept, "
          f"{sum(dropped.values())} dropped, {len(images)} images")
    if dropped:
        print("   ", dict(dropped))
    return dict(dropped)


def ensure_roboflow_coco_layout(dataset_dir: Path, out_dir: Path) -> Path:
    # Copy/remap a Roboflow COCO export into out_dir/{train,valid,test}/_annotations.coco.json.
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # Roboflow COCO: train/, valid/, test/ each with _annotations.coco.json + images
    splits_found = []
    for split in ("train", "valid", "test"):
        src_split = dataset_dir / split
        ann = src_split / "_annotations.coco.json"
        if not ann.exists():
            # sometimes annotations sit at root with split folders of images only
            alt = dataset_dir / f"{split}/_annotations.coco.json"
            ann = alt if alt.exists() else ann
        if not ann.exists():
            continue
        dst_split = out_dir / split
        dst_split.mkdir(parents=True)
        # copy images
        for img in src_split.iterdir():
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                shutil.copy2(img, dst_split / img.name)
        remap_coco_json(ann, dst_split / "_annotations.coco.json")
        splits_found.append(split)

    if "train" not in splits_found:
        raise SystemExit(
            f"No train/_annotations.coco.json under {dataset_dir}. "
            "Export from Roboflow as COCO and re-run."
        )
    if "valid" not in splits_found:
        print("WARNING: no valid/ split - RF-DETR early-stopping needs it. "
              "Re-export with a validation split.")
    return out_dir


def yolo_to_coco_split(img_dir: Path, lbl_dir: Path, names: list[str],
                       out_json: Path, out_img_dir: Path) -> int:
    # Convert images/ + labels/ YOLO dirs into one COCO JSON + flat image folder.
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
        "categories": _coco_categories(CLASS_NAMES),
        "images": images,
        "annotations": anns,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc), encoding="utf-8")
    return len(images)


def _resolve_yolo_split_dirs(dataset_dir: Path, rel: str) -> tuple[Path, Path] | tuple[None, None]:
    # data.yaml paths are usually '.../train/images' or 'train'.
    p = Path(rel)
    base = p if p.is_absolute() else (dataset_dir / rel)
    base = base.resolve()
    if base.name == "images" and base.is_dir():
        lbl = base.parent / "labels"
        return (base, lbl) if lbl.is_dir() else (base, base.parent / "labels")
    if (base / "images").is_dir():
        return base / "images", base / "labels"
    if base.is_dir():
        # flat folder of images; labels live next to data.yaml layout
        return base, base.parent / "labels"
    return None, None


def convert_yolo_dataset(dataset_dir: Path, out_dir: Path) -> Path:
    import yaml

    dataset_dir = Path(dataset_dir)
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        cands = list(dataset_dir.rglob("data.yaml"))
        if not cands:
            raise SystemExit(f"No data.yaml under {dataset_dir}")
        data_yaml = cands[0]
        dataset_dir = data_yaml.parent

    doc = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = doc.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    if not names:
        raise SystemExit("data.yaml has no names")

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
            img_dir, lbl_dir, names,
            out_dir / out_split / "_annotations.coco.json",
            out_dir / out_split,
        )
        print(f"  YOLO {out_split}: {n} images -> {out_dir / out_split}")

    if not (out_dir / "train" / "_annotations.coco.json").exists():
        raise SystemExit("YOLO conversion produced no train split")
    return out_dir


def unwrap_zip_root(path: Path) -> Path:
    kids = [p for p in path.iterdir() if not p.name.startswith((".", "__"))]
    if len(kids) == 1 and kids[0].is_dir():
        return kids[0]
    return path


def force_annotations_to_pothole(dataset_dir: Path) -> None:
    # BharatPotHole (and similar) are pothole-only: rewrite every kept ann to pothole.
    pothole_id = CLASS_TO_ID["pothole"] + 1
    for ann_path in dataset_dir.rglob("_annotations.coco.json"):
        doc = json.loads(ann_path.read_text(encoding="utf-8"))
        doc["categories"] = _coco_categories(CLASS_NAMES)
        for a in doc.get("annotations", []):
            a["category_id"] = pothole_id
        ann_path.write_text(json.dumps(doc), encoding="utf-8")
        print(f"  forced pothole ids in {ann_path.relative_to(dataset_dir)}")


def prepare_bharatpothole(raw_root: Path, out_dir: Path) -> Path:
    # Accept COCO, YOLO with data.yaml, or flat images/+labels/ trees.
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

    # Heuristic: find a folder with images/ and labels/
    img_dirs = [p for p in raw_root.rglob("images") if p.is_dir()]
    for img_dir in img_dirs:
        lbl_dir = img_dir.parent / "labels"
        if not lbl_dir.is_dir():
            continue
        # Write a minimal YOLO layout with a single train split (+ copy 15% to valid)
        import random
        from PIL import Image as _Image

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
            # Build COCO directly
            images, anns = [], []
            ann_id, img_id = 1, 1
            pothole_id = CLASS_TO_ID["pothole"] + 1
            for img_path in subset:
                with _Image.open(img_path) as im:
                    w, h = im.size
                shutil.copy2(img_path, sdir / img_path.name)
                images.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})
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
                            "id": ann_id, "image_id": img_id, "category_id": pothole_id,
                            "bbox": [x, y, bw_px, bh_px],
                            "area": float(max(bw_px, 0) * max(bh_px, 0)), "iscrowd": 0,
                        })
                        ann_id += 1
                img_id += 1
            doc = {
                "info": {"description": "bharatpothole remapped"},
                "licenses": [],
                "categories": _coco_categories(CLASS_NAMES),
                "images": images,
                "annotations": anns,
            }
            (sdir / "_annotations.coco.json").write_text(json.dumps(doc), encoding="utf-8")
            print(f"  BharatPotHole {split}: {len(images)} images, {len(anns)} anns")
        return out_dir

    raise SystemExit(
        f"Could not interpret BharatPotHole layout under {raw_root}. "
        "Expected COCO, YOLO data.yaml, or images/+labels/."
    )


def ingest_to_coco(raw_root: Path, out_dir: Path, force_pothole: bool = False) -> Path:
    raw_root = Path(raw_root)
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
    raise SystemExit(
        f"No COCO/YOLO layout under {raw_root}. "
        "Contents: " + ", ".join(sorted(p.name for p in raw_root.iterdir())[:20])
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
            id_remap = {}
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
                "categories": _coco_categories(CLASS_NAMES),
                "images": images,
                "annotations": anns,
            }
            (sdir / "_annotations.coco.json").write_text(json.dumps(merged), encoding="utf-8")
            print(f"  MERGED {split}: {len(images)} images, {len(anns)} anns")
    if not (out_dir / "train" / "_annotations.coco.json").exists():
        raise SystemExit("Merge produced no train split")
    if not (out_dir / "valid" / "_annotations.coco.json").exists():
        print("WARNING: no valid/ — slicing 10% of train")
        full = json.loads((out_dir / "train" / "_annotations.coco.json").read_text())
        n_val = max(1, len(full["images"]) // 10)
        val_ids = {im["id"] for im in full["images"][-n_val:]}
        val_doc = {
            "info": {"description": "valid from train slice"},
            "licenses": [],
            "categories": _coco_categories(CLASS_NAMES),
            "images": [im for im in full["images"] if im["id"] in val_ids],
            "annotations": [a for a in full["annotations"] if a["image_id"] in val_ids],
        }
        train_doc = {
            "info": full.get("info", {}),
            "licenses": [],
            "categories": _coco_categories(CLASS_NAMES),
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
        (out_dir / "train" / "_annotations.coco.json").write_text(json.dumps(train_doc), encoding="utf-8")
        print(f"  created valid/ with {len(val_doc['images'])} images")
    return out_dir


print("Helpers ready:", len(CLASS_ALIASES), "aliases,", len(CLASS_NAMES), "target classes")
"""))

cells.append(md("""
## 6 · Stage 1 dataset — rural / India multi-source merge

No single public set covers all 6 classes on village roads. Default Stage 1 **merges**:

| Source | Role | Maps to |
|--------|------|---------|
| [CRRI pavement distress](https://universe.roboflow.com/crri/crri-road-pavement-distress-project) | India multi-class backbone | alligator, longitudinal, pothole, edge |
| [BharatPotHole](https://www.kaggle.com/datasets/surbhisaswatimohanty/bharatpothole) | rural / unpaved pothole look | pothole |
| [Road Crack Detection](https://universe.roboflow.com/projects-jszvc/road-crack-detection-htnrb) | adds ravelling | pothole, alligator, edge, longitudinal, ravelling |
| [Pavement Distress](https://universe.roboflow.com/college-7qowe/pavement-distress-datasets) | edge + ravelling (severity collapsed) | pothole, edge, ravelling |
| PWD-style water/edge (optional) | drainage + edge drop | drainage_issue, edge_damage |

Requires `ROBOFLOW_API_KEY` for Universe projects and `KAGGLE_*` for BharatPotHole.
Toggle sources in the form. Remap is by **name**; patching/rutting/etc. are dropped.

`drainage_issue` stays scarce until Stage 2 / PWD water-stagnation. Export Roboflow as
**COCO**, Fit/letterbox ≥1280 when you control the export.
"""))

cells.append(code("""
#@title Stage 1 multi-source merge { display-mode: "form" }
STAGE1_MODE = "multi_merge"  #@param ["multi_merge", "kaggle_bharatpothole_only", "drive_zip", "upload_zip", "local_folder"]

#@markdown ### Sources to include (multi_merge)
USE_CRRI = True  #@param {type:"boolean"}
USE_BHARATPOTHOLE = True  #@param {type:"boolean"}
USE_ROAD_CRACK_DET = True  #@param {type:"boolean"}
USE_PAVEMENT_DISTRESS = True  #@param {type:"boolean"}
USE_PWD_DRAINAGE = False  #@param {type:"boolean"}
#@markdown PWD projects are noisy (many non-defect classes). Enable only after inspecting samples.

#@markdown ### Kaggle
KAGGLE_DATASET = "surbhisaswatimohanty/bharatpothole"  #@param {type:"string"}

#@markdown ### Roboflow project versions (COCO download)
CRRI_WORKSPACE = "crri"  #@param {type:"string"}
CRRI_PROJECT = "crri-road-pavement-distress-project"  #@param {type:"string"}
CRRI_VERSION = 3  #@param {type:"integer"}

RCD_WORKSPACE = "projects-jszvc"  #@param {type:"string"}
RCD_PROJECT = "road-crack-detection-htnrb"  #@param {type:"string"}
RCD_VERSION = 1  #@param {type:"integer"}

PD_WORKSPACE = "college-7qowe"  #@param {type:"string"}
PD_PROJECT = "pavement-distress-datasets"  #@param {type:"string"}
PD_VERSION = 1  #@param {type:"integer"}

PWD_WORKSPACE = "pwd3601"  #@param {type:"string"}
PWD_PROJECT = "s_1-bcm7o"  #@param {type:"string"}
PWD_VERSION = 1  #@param {type:"integer"}

RF_FORMAT = "coco"  #@param ["coco", "yolov8"]

#@markdown ### Single-source fallbacks
STAGE1_DRIVE_ZIP = "/content/drive/MyDrive/datasets/stage1_merged.zip"  #@param {type:"string"}
STAGE1_LOCAL_DIR = "/content/datasets/stage1_raw"  #@param {type:"string"}

from pathlib import Path
import shutil, zipfile, subprocess, sys, json
from collections import Counter

STAGE1_RAW = WORK / "datasets" / "stage1_raw"
STAGE1_PARTS = WORK / "datasets" / "stage1_parts"
STAGE1_DIR = WORK / "datasets" / "stage1"
for d in (STAGE1_RAW, STAGE1_PARTS):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)

def _extract_zip(zpath: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest)
    return unwrap_zip_root(dest)

def _download_roboflow(workspace, project, version, dest: Path) -> Path:
    if not ROBOFLOW_API_KEY:
        raise SystemExit(f"ROBOFLOW_API_KEY required for {workspace}/{project}")
    from roboflow import Roboflow
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    print(f"  Roboflow {workspace}/{project} v{version} ...")
    ds = (rf.workspace(workspace).project(project)
            .version(version).download(RF_FORMAT, location=str(dest)))
    return Path(ds.location)

def _download_kaggle(dataset: str, dest: Path) -> Path:
    if not (KAGGLE_USERNAME and KAGGLE_KEY):
        raise SystemExit("KAGGLE_USERNAME/KAGGLE_KEY required for BharatPotHole")
    print(f"  Kaggle {dataset} ...")
    subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download",
         "-d", dataset, "-p", str(dest), "--unzip"],
        check=True,
    )
    return unwrap_zip_root(dest)

parts = []

if STAGE1_MODE == "multi_merge":
    plan = []
    if USE_CRRI:
        plan.append(("crri", "roboflow",
                     dict(workspace=CRRI_WORKSPACE, project=CRRI_PROJECT, version=CRRI_VERSION),
                     False))
    if USE_BHARATPOTHOLE:
        plan.append(("bharatpothole", "kaggle",
                     dict(dataset=KAGGLE_DATASET), True))
    if USE_ROAD_CRACK_DET:
        plan.append(("road_crack_det", "roboflow",
                     dict(workspace=RCD_WORKSPACE, project=RCD_PROJECT, version=RCD_VERSION),
                     False))
    if USE_PAVEMENT_DISTRESS:
        plan.append(("pavement_distress", "roboflow",
                     dict(workspace=PD_WORKSPACE, project=PD_PROJECT, version=PD_VERSION),
                     False))
    if USE_PWD_DRAINAGE:
        plan.append(("pwd_drainage", "roboflow",
                     dict(workspace=PWD_WORKSPACE, project=PWD_PROJECT, version=PWD_VERSION),
                     False))
    if not plan:
        raise SystemExit("Enable at least one Stage 1 source")

    for name, kind, kwargs, force_pothole in plan:
        print(f"\\n=== {name} ===")
        raw_dest = STAGE1_RAW / name
        raw_dest.mkdir(parents=True, exist_ok=True)
        try:
            if kind == "roboflow":
                raw = _download_roboflow(kwargs["workspace"], kwargs["project"],
                                         kwargs["version"], raw_dest)
            else:
                raw = _download_kaggle(kwargs["dataset"], raw_dest)
            part_out = STAGE1_PARTS / name
            ingest_to_coco(raw, part_out, force_pothole=force_pothole)
            parts.append(part_out)
            print(f"  OK -> {part_out}")
        except Exception as e:
            print(f"  SKIPPED {name}: {e}")

    if not parts:
        raise SystemExit("No Stage 1 sources downloaded successfully")
    print("\\n=== Merging ===")
    merge_coco_datasets(parts, STAGE1_DIR)

elif STAGE1_MODE == "kaggle_bharatpothole_only":
    raw = _download_kaggle(KAGGLE_DATASET, STAGE1_RAW / "bharat")
    prepare_bharatpothole(raw, STAGE1_DIR)

elif STAGE1_MODE == "drive_zip":
    z = Path(STAGE1_DRIVE_ZIP)
    if not z.exists():
        raise SystemExit(f"Missing zip: {z}")
    raw = _extract_zip(z, STAGE1_RAW / "drive")
    force = "pothole" in z.name.lower() or "bharat" in z.name.lower()
    ingest_to_coco(raw, STAGE1_DIR, force_pothole=force)

elif STAGE1_MODE == "upload_zip":
    from google.colab import files
    print("Upload a COCO/YOLO zip ...")
    up = files.upload()
    name = next(iter(up))
    zpath = STAGE1_RAW / name
    zpath.write_bytes(up[name])
    raw = _extract_zip(zpath, STAGE1_RAW / "unz")
    force = "pothole" in name.lower() or "bharat" in name.lower()
    ingest_to_coco(raw, STAGE1_DIR, force_pothole=force)

elif STAGE1_MODE == "local_folder":
    raw = Path(STAGE1_LOCAL_DIR)
    if not raw.exists():
        raise SystemExit(f"Missing {raw}")
    ingest_to_coco(raw, STAGE1_DIR, force_pothole=False)

train_ann = STAGE1_DIR / "train" / "_annotations.coco.json"
if not train_ann.exists():
    raise SystemExit("Stage 1 train annotations missing")
doc = json.loads(train_ann.read_text())
id_to_name = {c["id"]: c["name"] for c in doc["categories"]}
hist = Counter(id_to_name[a["category_id"]] for a in doc["annotations"])
print("\\nStage 1 train instance counts:")
for n in CLASS_NAMES:
    print(f"  {n:22s} {hist.get(n, 0)}")
print(f"Images: {len(doc['images'])}")
print(f"STAGE1_DIR = {STAGE1_DIR}")
print("If drainage_issue / ravelling are still near zero, enable more sources or rely on Stage 2.")
"""))

cells.append(md("""
## 7 · Stage 1 — train RFDETRMedium

COCO-pretrained Medium → fine-tune on the **merged rural/India** Stage 1 set.
Native Medium resolution is **576**. Training uses **full frames** (no road masking).
"""))

cells.append(code("""
#@title Train Stage 1 { display-mode: "form" }
#@markdown Re-run after tweaking STAGE1_* in the Shared settings cell.

import time
from pathlib import Path

STAGE1_WEIGHTS = None

if not RUN_STAGE1:
    print("RUN_STAGE1=False — skipping. Set STAGE1_WEIGHTS manually if jumping to Stage 2.")
else:
    if not (STAGE1_DIR / "train" / "_annotations.coco.json").exists():
        raise SystemExit("Stage 1 dataset missing — run the previous cell.")

    from rfdetr import RFDETRMedium

    out_dir = WORK / "runs" / "stage1"
    print(f"Training RFDETRMedium → {out_dir}")
    print(f"epochs={STAGE1_EPOCHS} batch={STAGE1_BATCH} grad_accum={STAGE1_GRAD_ACCUM} "
          f"lr={STAGE1_LR} (effective batch {STAGE1_BATCH * STAGE1_GRAD_ACCUM})")

    model = RFDETRMedium()
    t0 = time.time()
    train_kwargs = dict(
        dataset_dir=str(STAGE1_DIR),
        epochs=STAGE1_EPOCHS,
        batch_size=STAGE1_BATCH,
        grad_accum_steps=STAGE1_GRAD_ACCUM,
        lr=STAGE1_LR,
        output_dir=str(out_dir),
    )
    # early stopping flag name can vary slightly across rfdetr versions
    if STAGE1_EARLY_STOP:
        train_kwargs["early_stopping"] = True

    try:
        model.train(**train_kwargs)
    except TypeError as e:
        # older/newer API without early_stopping
        print("Retrying without early_stopping:", e)
        train_kwargs.pop("early_stopping", None)
        model.train(**train_kwargs)

    mins = (time.time() - t0) / 60
    print(f"Stage 1 finished in {mins:.1f} min")

    for name in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth",
                 "checkpoint_best_regular.pth", "checkpoint.pth"):
        cand = out_dir / name
        if cand.exists():
            STAGE1_WEIGHTS = cand
            break
    if STAGE1_WEIGHTS is None:
        raise SystemExit(f"No checkpoint in {out_dir} — check the training log.")
    print("Stage 1 weights:", STAGE1_WEIGHTS)

    # mirror to Drive
    if MOUNT_DRIVE:
        dest = Path(DRIVE_ROOT) / "stage1" / STAGE1_WEIGHTS.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(STAGE1_WEIGHTS, dest)
        print("Copied to", dest)
"""))

cells.append(md("## 8 · Evaluate Stage 1"))

cells.append(code("""
from pathlib import Path

if STAGE1_WEIGHTS is None:
    # allow resume from Drive
    fallback = Path(DRIVE_ROOT) / "stage1" / "checkpoint_best_total.pth"
    if fallback.exists():
        STAGE1_WEIGHTS = fallback
        print("Loaded Stage 1 weights from Drive:", STAGE1_WEIGHTS)
    else:
        print("No Stage 1 weights — skip eval or set STAGE1_WEIGHTS.")

if STAGE1_WEIGHTS is not None and (STAGE1_DIR / "train" / "_annotations.coco.json").exists():
    from rfdetr import RFDETRMedium
    model = RFDETRMedium(pretrain_weights=str(STAGE1_WEIGHTS))
    split = "test" if (STAGE1_DIR / "test" / "_annotations.coco.json").exists() else "val"
    print(f"Evaluating on split={split} ...")
    try:
        metrics = model.evaluate(dataset_dir=str(STAGE1_DIR), split=split)
        print(metrics)
    except Exception as e:
        print("evaluate() failed (", e, ") — trying valid split / train-time logs instead.")
        try:
            metrics = model.evaluate(dataset_dir=str(STAGE1_DIR), split="val")
            print(metrics)
        except Exception as e2:
            print("Eval unavailable:", e2)
            print("Rely on the training val/mAP curves under", WORK / "runs" / "stage1")

    print("\\nNote: Stage 1 merge should populate pothole / cracks / edge / ravelling.")
    print("drainage_issue often stays thin until Stage 2 or USE_PWD_DRAINAGE=True.")
else:
    print("Skipped Stage 1 evaluation.")
"""))

cells.append(md("""
## 9 · Stage 2 dataset — your ~2000 rural images

Same 6 class **names and order**. This step carries alligator, cracks, ravelling,
edge_damage, and drainage on **your** camera and surface type. Prefer Roboflow COCO
export (Fit/letterbox ≥1280) or a Drive zip — never Stretch 512.
"""))

cells.append(code("""
#@title Stage 2 dataset source { display-mode: "form" }
STAGE2_SOURCE = "drive_zip"  #@param ["drive_zip", "upload_zip", "roboflow_project", "local_folder"]

#@markdown ### Your Roboflow project (optional)
S2_RF_WORKSPACE = "nidhis-workspace-zyeyu"  #@param {type:"string"}
S2_RF_PROJECT   = "mp_road_annotation_poc"  #@param {type:"string"}
S2_RF_VERSION   = 3  #@param {type:"integer"}
S2_RF_FORMAT    = "coco"  #@param ["coco", "yolov8"]

#@markdown ### Drive / local
STAGE2_DRIVE_ZIP = "/content/drive/MyDrive/datasets/my_road_defects_coco.zip"  #@param {type:"string"}
STAGE2_LOCAL_DIR = "/content/datasets/my_road"  #@param {type:"string"}

from pathlib import Path
import shutil, zipfile, json
from collections import Counter

STAGE2_DIR = WORK / "datasets" / "stage2"
STAGE2_RAW = WORK / "datasets" / "stage2_raw"
if STAGE2_RAW.exists():
    shutil.rmtree(STAGE2_RAW)
STAGE2_RAW.mkdir(parents=True)

if not RUN_STAGE2:
    print("RUN_STAGE2=False — skipping dataset prep.")
else:
    raw_root = None

    if STAGE2_SOURCE == "roboflow_project":
        if not ROBOFLOW_API_KEY:
            raise SystemExit("Need ROBOFLOW_API_KEY for roboflow_project")
        from roboflow import Roboflow
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        print(f"Downloading {S2_RF_WORKSPACE}/{S2_RF_PROJECT} v{S2_RF_VERSION} ...")
        ds = (rf.workspace(S2_RF_WORKSPACE)
                .project(S2_RF_PROJECT)
                .version(S2_RF_VERSION)
                .download(S2_RF_FORMAT, location=str(STAGE2_RAW)))
        raw_root = Path(ds.location)

    elif STAGE2_SOURCE == "drive_zip":
        z = Path(STAGE2_DRIVE_ZIP)
        if not z.exists():
            raise SystemExit(
                f"Missing {z}\\nUpload your COCO/YOLO zip to Drive or switch STAGE2_SOURCE."
            )
        with zipfile.ZipFile(z) as zf:
            zf.extractall(STAGE2_RAW)
        raw_root = unwrap_zip_root(STAGE2_RAW)

    elif STAGE2_SOURCE == "upload_zip":
        from google.colab import files
        print("Upload your dataset zip ...")
        up = files.upload()
        name = next(iter(up))
        zpath = STAGE2_RAW / name
        zpath.write_bytes(up[name])
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(STAGE2_RAW / "unz")
        raw_root = unwrap_zip_root(STAGE2_RAW / "unz")

    elif STAGE2_SOURCE == "local_folder":
        raw_root = Path(STAGE2_LOCAL_DIR)
        if not raw_root.exists():
            raise SystemExit(f"Missing {raw_root}")

    assert raw_root is not None

    if list(raw_root.rglob("_annotations.coco.json")):
        coco_root = raw_root
        if not (coco_root / "train" / "_annotations.coco.json").exists():
            hits = list(raw_root.rglob("train/_annotations.coco.json"))
            if hits:
                coco_root = hits[0].parent.parent
        ensure_roboflow_coco_layout(coco_root, STAGE2_DIR)
    elif list(raw_root.rglob("data.yaml")):
        convert_yolo_dataset(raw_root, STAGE2_DIR)
    else:
        raise SystemExit(f"No COCO/YOLO dataset in {raw_root}")

    train_ann = STAGE2_DIR / "train" / "_annotations.coco.json"
    doc = json.loads(train_ann.read_text())
    id_to_name = {c["id"]: c["name"] for c in doc["categories"]}
    hist = Counter(id_to_name[a["category_id"]] for a in doc["annotations"])
    print("\\nStage 2 train instance counts:")
    for n in CLASS_NAMES:
        print(f"  {n:22s} {hist.get(n, 0)}")
    print("STAGE2_DIR =", STAGE2_DIR)
"""))

cells.append(md("""
## 10 · Stage 2 — fine-tune from Stage 1

Lower LR, fewer epochs. Initializes from `checkpoint_best_total.pth` (or EMA fallback).
"""))

cells.append(code("""
#@title Train Stage 2 { display-mode: "form" }
import time, shutil
from pathlib import Path

FINAL_WEIGHTS = None

if not RUN_STAGE2:
    print("RUN_STAGE2=False — using Stage 1 weights as FINAL_WEIGHTS if present.")
    FINAL_WEIGHTS = STAGE1_WEIGHTS
else:
    if STAGE1_WEIGHTS is None:
        for cand in (
            WORK / "runs" / "stage1" / "checkpoint_best_total.pth",
            Path(DRIVE_ROOT) / "stage1" / "checkpoint_best_total.pth",
        ):
            if Path(cand).exists():
                STAGE1_WEIGHTS = Path(cand)
                break
    if STAGE1_WEIGHTS is None:
        raise SystemExit("Need Stage 1 weights before Stage 2.")
    if not (STAGE2_DIR / "train" / "_annotations.coco.json").exists():
        raise SystemExit("Stage 2 dataset missing — run the previous cell.")

    from rfdetr import RFDETRMedium

    out_dir = WORK / "runs" / "stage2"
    print("Fine-tuning from", STAGE1_WEIGHTS)
    print(f"epochs={STAGE2_EPOCHS} batch={STAGE2_BATCH} grad_accum={STAGE2_GRAD_ACCUM} lr={STAGE2_LR}")

    model = RFDETRMedium(pretrain_weights=str(STAGE1_WEIGHTS))
    t0 = time.time()
    kwargs = dict(
        dataset_dir=str(STAGE2_DIR),
        epochs=STAGE2_EPOCHS,
        batch_size=STAGE2_BATCH,
        grad_accum_steps=STAGE2_GRAD_ACCUM,
        lr=STAGE2_LR,
        output_dir=str(out_dir),
        early_stopping=True,
    )
    try:
        model.train(**kwargs)
    except TypeError:
        kwargs.pop("early_stopping", None)
        model.train(**kwargs)

    print(f"Stage 2 finished in {(time.time() - t0) / 60:.1f} min")
    for name in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth",
                 "checkpoint_best_regular.pth", "checkpoint.pth"):
        cand = out_dir / name
        if cand.exists():
            FINAL_WEIGHTS = cand
            break
    if FINAL_WEIGHTS is None:
        raise SystemExit("Stage 2 produced no checkpoint")
    print("FINAL_WEIGHTS:", FINAL_WEIGHTS)

    if MOUNT_DRIVE:
        dest = Path(DRIVE_ROOT) / "stage2" / FINAL_WEIGHTS.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(FINAL_WEIGHTS, dest)
        shutil.copy2(WORK / "exports" / "class_names.json",
                     Path(DRIVE_ROOT) / "stage2" / "class_names.json")
        print("Copied to", dest)
"""))

cells.append(md("## 11 · Evaluate Stage 2"))

cells.append(code("""
from pathlib import Path

if FINAL_WEIGHTS is None:
    for cand in (
        WORK / "runs" / "stage2" / "checkpoint_best_total.pth",
        Path(DRIVE_ROOT) / "stage2" / "checkpoint_best_total.pth",
        STAGE1_WEIGHTS,
    ):
        if cand is not None and Path(cand).exists():
            FINAL_WEIGHTS = Path(cand)
            break

if FINAL_WEIGHTS is None:
    raise SystemExit("No FINAL_WEIGHTS — train Stage 1/2 first.")

print("Using", FINAL_WEIGHTS)
eval_dir = STAGE2_DIR if (STAGE2_DIR / "train" / "_annotations.coco.json").exists() else STAGE1_DIR
from rfdetr import RFDETRMedium
model = RFDETRMedium(pretrain_weights=str(FINAL_WEIGHTS))
split = "test" if (eval_dir / "test" / "_annotations.coco.json").exists() else "val"
try:
    metrics = model.evaluate(dataset_dir=str(eval_dir), split=split)
    print(metrics)
except Exception as e:
    print("evaluate failed:", e)
"""))

cells.append(md("""
## 12 · Test on Drive or YouTube video (road → detect → gate)

Flow per frame: **road mask → RF-DETR → keep boxes that overlap the road**.

- Training stayed on full frames; gating is inference-only (same pattern as the main pipeline).
- Recall-first default confidence **0.2**. Road gate cuts bushes / sky / roadside FPs.
- Tune `ROAD_PRIOR_*` if the green outline misses the carriageway on your camera.
"""))

cells.append(code("""
#@title Video inference { display-mode: "form" }
VIDEO_SOURCE   = "drive_path"  #@param ["drive_path", "drive_link", "youtube", "upload"]
VIDEO_PATH     = "/content/drive/MyDrive/road_videos/sample.mp4"  #@param {type:"string"}
DRIVE_LINK     = ""  #@param {type:"string"}
YOUTUBE_URL    = ""  #@param {type:"string"}
FRAME_STRIDE   = 5  #@param {type:"integer"}
CONF_THRESHOLD = 0.2  #@param {type:"number"}
MAX_FRAMES     = 0  #@param {type:"integer"}

#@markdown ### Road gate (segment then detect)
USE_ROAD_GATE     = True  #@param {type:"boolean"}
MIN_ROAD_OVERLAP  = 0.3  #@param {type:"number"}
ROAD_USE_CLASSICAL = True  #@param {type:"boolean"}
#@markdown Trapezoid prior (fractions of frame) — dashcam defaults:
ROAD_BOTTOM_Y = 1.0  #@param {type:"number"}
ROAD_TOP_Y = 0.55  #@param {type:"number"}
ROAD_BOTTOM_HALF_W = 0.48  #@param {type:"number"}
ROAD_TOP_HALF_W = 0.12  #@param {type:"number"}
ROAD_CENTER_X = 0.5  #@param {type:"number"}
#@markdown MAX_FRAMES=0 means process the whole video (after stride).

import re, shutil, subprocess, sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

def road_trapezoid_mask(h, w, bottom_y, top_y, bottom_half_w, top_half_w, center_x):
    cx = center_x * w
    by, ty = bottom_y * h, top_y * h
    bhw, thw = bottom_half_w * w, top_half_w * w
    pts = np.array([
        [int(cx - bhw), int(by)], [int(cx + bhw), int(by)],
        [int(cx + thw), int(ty)], [int(cx - thw), int(ty)],
    ], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool), pts


def classical_grow_road(frame_bgr, prior_bool, work_width=480, distance_tau=2.5):
    # Lightweight colour+texture grow seeded by the geometric prior (rural-friendly).
    h, w = frame_bgr.shape[:2]
    scale = work_width / float(w)
    small = cv2.resize(frame_bgr, (work_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    prior_s = cv2.resize(prior_bool.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST).astype(bool)
    er = max(1, int(0.05 * sw))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (er * 2 + 1, er * 2 + 1))
    seed = cv2.erode(prior_s.astype(np.uint8), kernel, iterations=1).astype(bool)
    if not seed.any():
        seed = prior_s
    search = cv2.dilate(prior_s.astype(np.uint8), kernel, iterations=1).astype(bool)

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tex = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    tex = cv2.GaussianBlur(np.abs(tex), (7, 7), 0)
    feats = np.dstack([lab[..., 0], lab[..., 1], lab[..., 2], tex])
    med = np.median(feats[seed], axis=0)
    mad = np.median(np.abs(feats[seed] - med), axis=0) + 1e-3
    z = np.abs(feats - med) / mad
    # weighted channels: L,a,b,tex
    dist = (1.0 * z[..., 0] + 1.0 * z[..., 1] + 1.0 * z[..., 2] + 1.3 * z[..., 3]) / 4.3
    cand = (dist < distance_tau) & search
    # fill holes so potholes stay inside the road
    filled = cand.astype(np.uint8)
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hole = np.zeros_like(filled)
    cv2.drawContours(hole, contours, -1, 1, thickness=-1)
    road_s = hole.astype(bool)
    if road_s.mean() < 0.02 or road_s.mean() > 0.95:
        road_s = prior_s
    road = cv2.resize(road_s.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return road


def box_road_overlap(xyxy, road_mask):
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    h, w = road_mask.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    patch = road_mask[y1:y2, x1:x2]
    return float(patch.mean()) if patch.size else 0.0


def gate_detections(detections, road_mask, min_overlap, empty_detections):
    if detections.xyxy is None or len(detections) == 0:
        return detections, 0
    keep = []
    for i, box in enumerate(detections.xyxy):
        keep.append(box_road_overlap(box, road_mask) >= min_overlap)
    keep = np.asarray(keep, dtype=bool)
    n_drop = int((~keep).sum())
    if keep.any():
        return detections[keep], n_drop
    return empty_detections, n_drop


VDIR = WORK / "videos"
VDIR.mkdir(exist_ok=True)
video_file = None

if VIDEO_SOURCE == "drive_path":
    video_file = Path(VIDEO_PATH)
    if not video_file.exists():
        raise SystemExit(f"Missing video: {video_file}")

elif VIDEO_SOURCE == "drive_link":
    if not DRIVE_LINK.strip():
        raise SystemExit("Paste DRIVE_LINK")
    m = re.search(r"/d/([A-Za-z0-9_-]{20,})", DRIVE_LINK) or \
        re.search(r"[?&]id=([A-Za-z0-9_-]{20,})", DRIVE_LINK)
    file_id = m.group(1) if m else DRIVE_LINK.strip()
    video_file = VDIR / "input.mp4"
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
    subprocess.run(["gdown", "--id", file_id, "-O", str(video_file)], check=True)

elif VIDEO_SOURCE == "youtube":
    if not YOUTUBE_URL.strip():
        raise SystemExit("Paste YOUTUBE_URL")
    video_file = VDIR / "youtube_input.mp4"
    subprocess.run([
        sys.executable, "-m", "yt_dlp",
        "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", str(video_file),
        YOUTUBE_URL.strip(),
    ], check=True)

elif VIDEO_SOURCE == "upload":
    from google.colab import files
    up = files.upload()
    name = next(iter(up))
    video_file = VDIR / name
    video_file.write_bytes(up[name])

print("Video:", video_file)
print(f"Road gate: {USE_ROAD_GATE}  min_overlap={MIN_ROAD_OVERLAP}  classical={ROAD_USE_CLASSICAL}")

from rfdetr import RFDETRMedium
import supervision as sv

_EMPTY = sv.Detections.empty()

if FINAL_WEIGHTS is None:
    raise SystemExit("FINAL_WEIGHTS is not set")

model = RFDETRMedium(pretrain_weights=str(FINAL_WEIGHTS))
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.4)

cap = cv2.VideoCapture(str(video_file))
if not cap.isOpened():
    raise SystemExit(f"Could not open {video_file}")

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out_path = VDIR / "annotated_rfdetr.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(out_path), fourcc, max(fps / max(FRAME_STRIDE, 1), 1.0), (w, h))

counts = {n: 0 for n in CLASS_NAMES}
gated_away = 0
gallery = []
frame_i, written = 0, 0
road_mask_ema = None

while True:
    ok, frame = cap.read()
    if not ok:
        break
    if frame_i % max(FRAME_STRIDE, 1) != 0:
        frame_i += 1
        continue

    prior, prior_pts = road_trapezoid_mask(
        h, w, ROAD_BOTTOM_Y, ROAD_TOP_Y, ROAD_BOTTOM_HALF_W, ROAD_TOP_HALF_W, ROAD_CENTER_X,
    )
    if USE_ROAD_GATE:
        if ROAD_USE_CLASSICAL:
            road = classical_grow_road(frame, prior)
        else:
            road = prior
        # light temporal smooth
        if road_mask_ema is None:
            road_mask_ema = road.astype(np.float32)
        else:
            road_mask_ema = 0.5 * road.astype(np.float32) + 0.5 * road_mask_ema
        road = road_mask_ema >= 0.5
    else:
        road = np.ones((h, w), dtype=bool)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    detections = model.predict(pil, threshold=CONF_THRESHOLD)

    if not isinstance(detections, sv.Detections):
        try:
            detections = sv.Detections.from_inference(detections)
        except Exception:
            if hasattr(detections, "xyxy"):
                detections = sv.Detections(
                    xyxy=np.asarray(detections.xyxy),
                    confidence=np.asarray(getattr(detections, "confidence", None))
                    if getattr(detections, "confidence", None) is not None else None,
                    class_id=np.asarray(getattr(detections, "class_id", None))
                    if getattr(detections, "class_id", None) is not None else None,
                )
            else:
                detections = sv.Detections.empty()

    if USE_ROAD_GATE and len(detections):
        detections, n_drop = gate_detections(detections, road, MIN_ROAD_OVERLAP, _EMPTY)
        gated_away += n_drop

    labels = []
    if detections.class_id is not None:
        for cid, conf in zip(
            detections.class_id,
            detections.confidence if detections.confidence is not None else [None] * len(detections),
        ):
            name = CLASS_NAMES[int(cid)] if 0 <= int(cid) < len(CLASS_NAMES) else str(cid)
            counts[name] = counts.get(name, 0) + 1
            labels.append(f"{name} {conf:.2f}" if conf is not None else name)

    annotated = frame.copy()
    if USE_ROAD_GATE:
        overlay = annotated.copy()
        overlay[road] = (
            (0.82 * overlay[road] + 0.18 * np.array([0, 200, 0])).astype(np.uint8)
        )
        annotated = overlay
        cv2.polylines(annotated, [prior_pts], isClosed=True, color=(0, 255, 0), thickness=2)

    annotated = box_annotator.annotate(annotated, detections)
    annotated = label_annotator.annotate(annotated, detections, labels=labels)
    writer.write(annotated)
    written += 1

    if detections.class_id is not None and len(detections) and len(gallery) < 12:
        gallery.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

    if MAX_FRAMES and written >= MAX_FRAMES:
        break
    frame_i += 1
    if written % 50 == 0:
        print(f"  wrote {written} frames ... (gated away so far: {gated_away})")

cap.release()
writer.release()
print(f"\\nAnnotated video: {out_path} ({written} frames)")
print(f"Detections dropped by road gate: {gated_away}")
print("Per-class detection counts (frame-level, on-road only if gate on):")
for n in CLASS_NAMES:
    print(f"  {n:22s} {counts.get(n, 0)}")

h264 = VDIR / "annotated_rfdetr_h264.mp4"
ff = subprocess.run([
    "ffmpeg", "-y", "-i", str(out_path),
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(h264),
], capture_output=True, text=True)
PLAYBACK = h264 if ff.returncode == 0 and h264.exists() else out_path
print("Playback file:", PLAYBACK)

if gallery:
    import matplotlib.pyplot as plt
    cols = 3
    rows = (len(gallery) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, img in zip(axes, gallery):
        ax.imshow(img)
        ax.axis("off")
    for ax in axes[len(gallery):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

ANNOTATED_VIDEO = PLAYBACK
"""))

cells.append(md("""
## 13 · Export weights + pipeline handoff

Save the checkpoint and class list. The full road-defect pipeline still expects YOLO
today — keep these artifacts for a later RF-DETR integration; for now you can use the
annotated video from this notebook directly.
"""))

cells.append(code("""
#@title Export { display-mode: "form" }
COPY_TO_DRIVE = True  #@param {type:"boolean"}
DOWNLOAD_LOCAL = False  #@param {type:"boolean"}
COPY_VIDEO = True  #@param {type:"boolean"}

import json, shutil
from pathlib import Path

if FINAL_WEIGHTS is None or not Path(FINAL_WEIGHTS).exists():
    raise SystemExit("FINAL_WEIGHTS missing")

export_dir = WORK / "exports"
export_dir.mkdir(exist_ok=True)
weights_out = export_dir / "rfdetr_medium_6class.pth"
shutil.copy2(FINAL_WEIGHTS, weights_out)
meta = {
    "model": "RFDETRMedium",
    "classes": CLASS_NAMES,
    "stage1_epochs": STAGE1_EPOCHS if RUN_STAGE1 else None,
    "stage2_epochs": STAGE2_EPOCHS if RUN_STAGE2 else None,
    "weights_source": str(FINAL_WEIGHTS),
    "notes": (
        "Standalone RF-DETR detector, rural-first (BharatPotHole Stage 1). "
        "Video path uses road-mask gating; training is full-frame. "
        "Not yet plugged into src/rdd YOLO inference. "
        "Use class_names.json for name<->id. Prefer checkpoint_best_total.pth."
    ),
}
(export_dir / "handoff.json").write_text(json.dumps(meta, indent=2))
print("Export bundle:")
print(" ", weights_out)
print(" ", export_dir / "class_names.json")
print(" ", export_dir / "handoff.json")

if COPY_TO_DRIVE and MOUNT_DRIVE:
    dest = Path(DRIVE_ROOT) / "exports"
    dest.mkdir(parents=True, exist_ok=True)
    for p in export_dir.iterdir():
        shutil.copy2(p, dest / p.name)
    if COPY_VIDEO and "ANNOTATED_VIDEO" in dir() and Path(ANNOTATED_VIDEO).exists():
        shutil.copy2(ANNOTATED_VIDEO, dest / Path(ANNOTATED_VIDEO).name)
    print("Copied to", dest)

if DOWNLOAD_LOCAL:
    from google.colab import files
    files.download(str(weights_out))
    files.download(str(export_dir / "class_names.json"))
    files.download(str(export_dir / "handoff.json"))

print("Done.")
print()
print("Next steps (outside this notebook):")
print("1. Inspect the annotated video - if recall is low, drop CONF_THRESHOLD")
print("   and/or add Stage-2 labels.")
print("2. If asphalt Stage-1 -> gravel Stage-2 still misses, prioritise labelling")
print("   hard negatives / false misses.")
print("3. Later: adapt src/rdd/model/loader.py (or a parallel RF-DETR path) to load")
print("   rfdetr_medium_6class.pth with class_names.json identity map")
print("   (same lesson as YOLO class_map).")
"""))

cells.append(md("""
---

### Troubleshooting

| Symptom | Fix |
|--------|-----|
| CUDA OOM | Lower `STAGE*_BATCH` to 2 and raise `GRAD_ACCUM` so product stays ~16 |
| Kaggle 401 / download fails | Set `KAGGLE_USERNAME` + `KAGGLE_KEY`, accept dataset terms on Kaggle website, or use `drive_zip` / disable `USE_BHARATPOTHOLE` |
| Roboflow 404 / skip source | Bump project `*_VERSION`, or turn that `USE_*` toggle off; merge continues with other sources |
| Stage 1 only predicts pothole | Enable CRRI + Road Crack Detection; then run Stage 2 on your 6-class rural labels |
| drainage_issue near zero | Set `USE_PWD_DRAINAGE=True` after inspecting PWD samples, or label more in Stage 2 |
| Great val mAP, empty rural video | Finish Stage 2; lower `CONF_THRESHOLD`; check road gate is not too tight |
| Green road outline misses carriageway | Tune `ROAD_TOP_Y` / half-widths; or set `ROAD_USE_CLASSICAL=False` for prior-only |
| Gate drops real defects | Lower `MIN_ROAD_OVERLAP` (e.g. 0.15) or widen the prior |
| Stretch 512 export | Re-export Fit/letterbox ≥1280 before Stage 2 |
"""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4", "name": "colab_rfdetr_train.ipynb"},
    },
    "cells": cells,
}

OUT.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {OUT} with {len(cells)} cells")
