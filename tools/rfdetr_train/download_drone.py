"""Download and prepare the drone/UAV Stage-1 COCO dataset (6-class taxonomy).

Sources (see tools/rfdetr_train/drone_sources.py for citations/licenses):
  - UAV-PDD2023 (Zenodo, CC BY 4.0)      — VOC XML, 6-way crack+pothole taxonomy
  - UAPD        (Google Drive, public)   — VOC XML, same taxonomy, dedup'd against UAV-PDD2023
  - HighRPD     (Mendeley, CC BY 4.0)    — YOLO txt, 3 classes (line/block/pit)
  - Roboflow "Pothole detection by Drone" (MIT) — pothole-only, nadir-confirmed

No public drone dataset covers drainage_issue or edge_damage — see
docs/DRONE_DATASETS.md "Coverage gaps" for the recommended bootstrap plan.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from .coco_io import ensure_roboflow_coco_layout, merge_coco_datasets, print_class_histogram
from .config import DroneStage1Config
from .download import download_roboflow, extract_zip, load_dotenv
from .drone_ingest import ingest_highrpd, ingest_voc_dataset
from .drone_sources import SOURCE_GSD_CM_PX, resize_split_to_target

ZENODO_UAV_PDD2023_URL = "https://zenodo.org/api/records/8429208/files/UAV-PDD2023.zip/content"
MENDELEY_HIGHRPD_API = "https://data.mendeley.com/public-api/datasets/sywswj7djj"
UAPD_GDRIVE_FILE_ID = "1yQ0GMXFwwM5qdYY_5HzJBQqqjNtWJxEc"


def _download_url(url: str, dest_zip: Path, *, chunk_mb: int = 32) -> Path:
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "road-defect-pipeline/1.0"})
    chunk = chunk_mb * 1024 * 1024
    with urllib.request.urlopen(req) as resp, open(dest_zip, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        written = 0
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            f.write(buf)
            written += len(buf)
            if total:
                print(f"\r  {dest_zip.name}: {written / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)
    print()
    return dest_zip


def download_uav_pdd2023(dest: Path) -> Path:
    zip_path = dest / "UAV-PDD2023.zip"
    if not zip_path.exists():
        print(f"  Zenodo UAV-PDD2023 -> {zip_path}")
        _download_url(ZENODO_UAV_PDD2023_URL, zip_path)
    return extract_zip(zip_path, dest / "extracted")


def download_highrpd(dest: Path) -> Path:
    import json

    zip_path = dest / "HighRPD.zip"
    if not zip_path.exists():
        req = urllib.request.Request(MENDELEY_HIGHRPD_API, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req) as resp:
            meta = json.loads(resp.read())
        file_url = meta["files"][0]["content_details"]["download_url"]
        print(f"  Mendeley HighRPD -> {zip_path}")
        _download_url(file_url, zip_path)
    return extract_zip(zip_path, dest / "extracted")


def download_uapd(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "UAPD.zip"
    if not zip_path.exists():
        print(f"  Google Drive UAPD -> {zip_path}")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "gdown",
                f"https://drive.google.com/uc?id={UAPD_GDRIVE_FILE_ID}",
                "-O",
                str(zip_path),
            ],
            check=True,
        )
    return extract_zip(zip_path, dest / "extracted")


def prepare_drone_stage1(
    cfg: DroneStage1Config,
    *,
    clean: bool = True,
    use_uav_pdd2023: bool = True,
    use_uapd: bool = True,
    use_highrpd: bool = True,
    use_rf_pothole_drone: bool = True,
    local_overrides: dict[str, Path] | None = None,
) -> Path:
    raw_dir, parts_dir, out_dir = cfg.raw_dir, cfg.parts_dir, cfg.dataset_dir
    local_overrides = local_overrides or {}

    if clean:
        for d in (raw_dir, parts_dir):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
    else:
        raw_dir.mkdir(parents=True, exist_ok=True)
        parts_dir.mkdir(parents=True, exist_ok=True)

    parts: list[Path] = []
    voc_hashes: set[str] = set()

    # UAV-PDD2023 first (larger, canonical) so UAPD dedup drops the overlap, not the reverse.
    if use_uav_pdd2023:
        print("\n=== uav_pdd2023 ===")
        try:
            raw = local_overrides.get("uav_pdd2023") or download_uav_pdd2023(raw_dir / "uav_pdd2023")
            part_out = parts_dir / "uav_pdd2023"
            part_out, voc_hashes = ingest_voc_dataset(Path(raw), part_out, dedupe_hashes=voc_hashes)
            _apply_gsd(part_out, "uav_pdd2023")
            parts.append(part_out)
            print(f"  OK -> {part_out}")
        except Exception as e:
            print(f"  SKIPPED uav_pdd2023: {e}")

    if use_uapd:
        print("\n=== uapd ===")
        try:
            raw = local_overrides.get("uapd") or download_uapd(raw_dir / "uapd")
            part_out = parts_dir / "uapd"
            part_out, voc_hashes = ingest_voc_dataset(Path(raw), part_out, dedupe_hashes=voc_hashes)
            _apply_gsd(part_out, "uapd")
            parts.append(part_out)
            print(f"  OK -> {part_out}")
        except Exception as e:
            print(f"  SKIPPED uapd: {e}")

    if use_highrpd:
        print("\n=== highrpd ===")
        try:
            raw = local_overrides.get("highrpd") or download_highrpd(raw_dir / "highrpd")
            part_out = parts_dir / "highrpd"
            part_out = ingest_highrpd(Path(raw), part_out)
            _apply_gsd(part_out, "highrpd")
            parts.append(part_out)
            print(f"  OK -> {part_out}")
        except Exception as e:
            print(f"  SKIPPED highrpd: {e}")

    if use_rf_pothole_drone:
        print("\n=== rf_pothole_drone ===")
        try:
            raw_dest = raw_dir / "rf_pothole_drone"
            raw = local_overrides.get("rf_pothole_drone") or download_roboflow(
                "drone-zh0ho", "pothole-detection-zdizt", 1, raw_dest, fmt="coco"
            )
            part_out = parts_dir / "rf_pothole_drone"
            part_out = ensure_roboflow_coco_layout(raw, part_out)
            _apply_gsd(part_out, "rf_pothole_drone")
            parts.append(part_out)
            print(f"  OK -> {part_out}")
        except Exception as e:
            print(f"  SKIPPED rf_pothole_drone: {e}")

    if not parts:
        raise SystemExit("No drone Stage-1 sources prepared successfully")

    if len(parts) == 1:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        shutil.copytree(parts[0], out_dir)
    else:
        print("\n=== Merging ===")
        merge_coco_datasets(parts, out_dir)

    train_ann = out_dir / "train" / "_annotations.coco.json"
    if not train_ann.exists():
        raise SystemExit(f"Drone Stage-1 train annotations missing under {out_dir}")

    print_class_histogram(out_dir)
    print(f"\nDRONE_STAGE1_DIR = {out_dir}")
    print(
        "\nNOTE: drainage_issue and edge_damage are not covered by any public drone "
        "source — see docs/DRONE_DATASETS.md 'Coverage gaps' before training."
    )
    return out_dir


def _apply_gsd(part_out: Path, source_key: str) -> None:
    src_gsd = SOURCE_GSD_CM_PX.get(source_key)
    dst_gsd = SOURCE_GSD_CM_PX.get("_target")
    if src_gsd is None or dst_gsd is None:
        return
    for split in ("train", "valid", "test"):
        resize_split_to_target(part_out, split, src_gsd, dst_gsd)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download and prepare drone/UAV Stage-1 COCO dataset (6-class taxonomy)."
    )
    p.add_argument("--work-root", type=Path, default=None)
    p.add_argument("--no-clean", action="store_true")
    p.add_argument("--no-uav-pdd2023", action="store_true")
    p.add_argument("--no-uapd", action="store_true")
    p.add_argument("--no-highrpd", action="store_true")
    p.add_argument("--no-rf-pothole-drone", action="store_true")
    p.add_argument(
        "--local",
        action="append",
        default=[],
        metavar="SOURCE=PATH",
        help="Use a manually-downloaded folder/zip instead of fetching SOURCE "
        "(uav_pdd2023|uapd|highrpd|rf_pothole_drone). Repeatable.",
    )
    p.add_argument("--env", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(args.env)
    cfg = DroneStage1Config()
    if args.work_root is not None:
        cfg.work_root = Path(args.work_root)

    overrides: dict[str, Path] = {}
    for item in args.local:
        if "=" not in item:
            raise SystemExit(f"--local expects SOURCE=PATH, got: {item}")
        key, _, path_s = item.partition("=")
        p = Path(path_s)
        overrides[key] = extract_zip(p, cfg.raw_dir / f"{key}_local") if p.is_file() else p

    prepare_drone_stage1(
        cfg,
        clean=not args.no_clean,
        use_uav_pdd2023=not args.no_uav_pdd2023,
        use_uapd=not args.no_uapd,
        use_highrpd=not args.no_highrpd,
        use_rf_pothole_drone=not args.no_rf_pothole_drone,
        local_overrides=overrides,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
