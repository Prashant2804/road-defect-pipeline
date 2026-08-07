"""Download and prepare Stage-2 multi-source COCO merge for RFDETRLarge."""
from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

from .coco_io import (
    cap_majority_class_images,
    ingest_to_coco,
    merge_coco_datasets,
    prepare_bharatpothole,
    prepare_rdd_voc,
    print_class_histogram,
    unwrap_zip_root,
)
from .config import Stage2Config
from .download import download_kaggle, download_roboflow, load_dotenv


def download_url_zip(url: str, dest: Path) -> Path:
    """Download a zip URL into dest/ and extract; return unwrapped root."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "download.zip"
    if not zip_path.exists() or zip_path.stat().st_size < 1024:
        print(f"  GET {url}")
        urllib.request.urlretrieve(url, zip_path)
    print(f"  extracting {zip_path.name} ({zip_path.stat().st_size / 1e6:.1f} MB)")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest / "extracted")
    return unwrap_zip_root(dest / "extracted")


def _try_reuse_stage1_crri(cfg: Stage2Config, raw_dest: Path) -> Path | None:
    """Reuse already-downloaded Stage-1 CRRI raw if present."""
    candidates = [
        cfg.work_root / "stage1_raw" / "crri",
        cfg.work_root / "stage1" ,
    ]
    for c in candidates:
        if list(Path(c).rglob("_annotations.coco.json")) if c.exists() else []:
            print(f"  reusing existing data at {c}")
            if c.resolve() != raw_dest.resolve():
                if raw_dest.exists():
                    shutil.rmtree(raw_dest)
                shutil.copytree(c, raw_dest)
            return raw_dest
    return None


def prepare_stage2(cfg: Stage2Config, *, clean: bool = True) -> Path:
    raw_dir = cfg.raw_dir
    parts_dir = cfg.parts_dir
    out_dir = cfg.dataset_dir

    if clean:
        for d in (raw_dir, parts_dir):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
    else:
        raw_dir.mkdir(parents=True, exist_ok=True)
        parts_dir.mkdir(parents=True, exist_ok=True)

    # (name, kind, kwargs, force_pothole)
    plan: list[tuple[str, str, dict, bool]] = []

    if cfg.use_crri:
        plan.append(
            (
                "crri",
                "roboflow",
                {
                    "workspace": cfg.roboflow_workspace,
                    "project": cfg.roboflow_project,
                    "version": cfg.roboflow_version,
                },
                False,
            )
        )
    if cfg.use_rdd_india:
        plan.append(("rdd_india", "url_zip", {"url": cfg.rdd_india_url}, False))
    if cfg.use_bharatpothole:
        plan.append(
            ("bharatpothole", "kaggle", {"dataset": cfg.kaggle_dataset}, True)
        )
    if cfg.use_crack2:
        plan.append(
            (
                "crack2",
                "roboflow",
                {
                    "workspace": "crack-hrrpe",
                    "project": "crack-2-d7une",
                    "version": 1,
                },
                False,
            )
        )
    if cfg.use_pavement_distress:
        plan.append(
            (
                "pavement_distress",
                "roboflow",
                {
                    "workspace": "college-7qowe",
                    "project": "pavement-distress-datasets",
                    "version": 1,
                },
                False,
            )
        )
    if cfg.use_road_crack:
        plan.append(
            (
                "road_crack_det",
                "roboflow",
                {
                    "workspace": "projects-jszvc",
                    "project": "road-crack-detection-htnrb",
                    "version": 1,
                },
                False,
            )
        )
    if cfg.use_road_damage2:
        plan.append(
            (
                "road_damage2",
                "roboflow",
                {
                    "workspace": "takeoff-kk0rk",
                    "project": "road_damage_detection_2-q3ysu",
                    "version": 1,
                },
                False,
            )
        )
    if cfg.use_water_logging:
        plan.append(
            (
                "water_logging",
                "roboflow",
                {
                    "workspace": "water-logging",
                    "project": "water-logging",
                    "version": 1,
                },
                False,
            )
        )
    if cfg.use_drain_overflow:
        plan.append(
            (
                "drain_overflow",
                "roboflow",
                {
                    "workspace": "chaitanya-kharche",
                    "project": "drain-overflow",
                    "version": 1,
                },
                False,
            )
        )
    if cfg.use_pwd:
        plan.append(
            (
                "pwd_drainage",
                "roboflow",
                {
                    "workspace": "pwd3601",
                    "project": "s_1-bcm7o",
                    "version": 1,
                },
                False,
            )
        )

    parts: list[Path] = []
    for name, kind, kwargs, force_pothole in plan:
        print(f"\n=== {name} ===")
        raw_dest = raw_dir / name
        try:
            if kind == "roboflow":
                reused = None
                if name == "crri":
                    reused = _try_reuse_stage1_crri(cfg, raw_dest)
                if reused is None:
                    raw = download_roboflow(
                        kwargs["workspace"],
                        kwargs["project"],
                        kwargs["version"],
                        raw_dest,
                        fmt=cfg.roboflow_format,
                    )
                else:
                    raw = reused
            elif kind == "kaggle":
                raw_dest.mkdir(parents=True, exist_ok=True)
                raw = download_kaggle(kwargs["dataset"], raw_dest)
            elif kind == "url_zip":
                raw = download_url_zip(kwargs["url"], raw_dest)
            else:
                raise ValueError(kind)

            part_out = parts_dir / name
            if name == "rdd_india":
                prepare_rdd_voc(raw, part_out)
            elif force_pothole:
                prepare_bharatpothole(raw, part_out)
            else:
                ingest_to_coco(raw, part_out, force_pothole=False)
            parts.append(part_out)
            print(f"  OK → {part_out}")
        except (Exception, SystemExit) as e:
            print(f"  SKIPPED {name}: {e}")
            if name == "crri" and cfg.use_crri and len(plan) <= 2:
                raise

    if not parts:
        raise SystemExit("No Stage 2 sources downloaded successfully")

    print("\n=== Merging ===")
    if len(parts) == 1:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        shutil.copytree(parts[0], out_dir)
    else:
        merge_coco_datasets(parts, out_dir)

    print("\n=== Balancing (cap pothole-only images) ===")
    print_class_histogram(out_dir, "train")
    cap_majority_class_images(
        out_dir,
        majority="pothole",
        max_fraction=cfg.pothole_max_fraction,
        split="train",
    )
    print_class_histogram(out_dir, "train")
    if (out_dir / "valid" / "_annotations.coco.json").exists():
        print_class_histogram(out_dir, "valid")

    train_ann = out_dir / "train" / "_annotations.coco.json"
    if not train_ann.exists():
        raise SystemExit(f"Stage 2 train annotations missing under {out_dir}")

    print(f"\nSTAGE2_DIR = {out_dir}")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download and prepare RF-DETR Stage-2 multi-source COCO dataset."
    )
    p.add_argument("--work-root", type=Path, default=None)
    p.add_argument("--no-clean", action="store_true")
    p.add_argument("--no-crri", action="store_true")
    p.add_argument("--no-rdd-india", action="store_true")
    p.add_argument("--no-bharatpothole", action="store_true")
    p.add_argument("--no-crack2", action="store_true")
    p.add_argument("--no-pavement-distress", action="store_true")
    p.add_argument("--no-road-crack", action="store_true")
    p.add_argument("--no-road-damage2", action="store_true")
    p.add_argument("--no-water-logging", action="store_true")
    p.add_argument("--no-drain-overflow", action="store_true")
    p.add_argument("--no-pwd", action="store_true")
    p.add_argument(
        "--pothole-max-fraction",
        type=float,
        default=0.45,
        help="After merge, cap pothole-only images (default 0.45 of instances)",
    )
    p.add_argument("--env", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(args.env)
    cfg = Stage2Config(
        use_crri=not args.no_crri,
        use_rdd_india=not args.no_rdd_india,
        use_bharatpothole=not args.no_bharatpothole,
        use_crack2=not args.no_crack2,
        use_pavement_distress=not args.no_pavement_distress,
        use_road_crack=not args.no_road_crack,
        use_road_damage2=not args.no_road_damage2,
        use_water_logging=not args.no_water_logging,
        use_drain_overflow=not args.no_drain_overflow,
        use_pwd=not args.no_pwd,
        pothole_max_fraction=args.pothole_max_fraction,
    )
    if args.work_root is not None:
        cfg.work_root = Path(args.work_root)
    prepare_stage2(cfg, clean=not args.no_clean)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
