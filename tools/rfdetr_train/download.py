"""Download and prepare Stage-1 COCO data (CRRI-first, optional extras)."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from .coco_io import (
    find_coco_root,
    ingest_to_coco,
    merge_coco_datasets,
    prepare_bharatpothole,
    print_class_histogram,
    unwrap_zip_root,
)
from .config import Stage1Config, repo_root


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader (no extra dependency). Does not override existing env."""
    env_path = path or (repo_root() / ".env")
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def download_roboflow(
    workspace: str,
    project: str,
    version: int,
    dest: Path,
    *,
    api_key: str | None = None,
    fmt: str = "coco",
) -> Path:
    """Download a Roboflow Universe export and return the real COCO root.

    Roboflow sometimes writes under cwd (e.g. ``CRRI-...-3/``) instead of
    ``dest``. We search for ``train/_annotations.coco.json`` after download.
    """
    api_key = api_key or os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        raise RuntimeError(f"ROBOFLOW_API_KEY required for {workspace}/{project}")

    from roboflow import Roboflow

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    cwd_before = set(Path.cwd().iterdir()) if Path.cwd().exists() else set()

    print(f"  Roboflow {workspace}/{project} v{version} → {dest}")
    rf = Roboflow(api_key=api_key)
    ds = (
        rf.workspace(workspace)
        .project(project)
        .version(version)
        .download(fmt, location=str(dest))
    )
    reported = Path(getattr(ds, "location", dest) or dest)

    search_roots = [reported, dest, Path.cwd(), dest.parent]
    # New top-level dirs created under cwd during download
    if Path.cwd().exists():
        for p in Path.cwd().iterdir():
            if p not in cwd_before and p.is_dir():
                search_roots.append(p)

    found = find_coco_root(search_roots)
    if found is None:
        # last resort: any _annotations under dest / cwd
        for root in search_roots:
            if not Path(root).exists():
                continue
            hits = list(Path(root).rglob("_annotations.coco.json"))
            if hits:
                # prefer train/
                for h in hits:
                    if h.parent.name == "train":
                        found = h.parent.parent
                        break
                if found is None:
                    found = hits[0].parent.parent
                break

    if found is None:
        kids = sorted(p.name for p in dest.iterdir()) if dest.exists() else []
        raise RuntimeError(
            f"Roboflow download finished but no COCO annotations found. "
            f"dest={dest} reported={reported} contents={kids[:20]}"
        )

    # Normalize into dest if Roboflow wrote elsewhere
    if found.resolve() != dest.resolve():
        print(f"  discovered COCO at {found} (reported={reported})")
        if dest.exists() and any(dest.iterdir()):
            # dest may be empty placeholder; clear and copy
            if not list(dest.rglob("_annotations.coco.json")):
                for child in list(dest.iterdir()):
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
        if not list(dest.rglob("_annotations.coco.json")):
            for item in found.iterdir():
                target = dest / item.name
                if item.is_dir():
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.copytree(item, target)
                else:
                    shutil.copy2(item, target)
            found = dest

    return found


def download_kaggle(dataset: str, dest: Path) -> Path:
    user = os.environ.get("KAGGLE_USERNAME", "")
    key = os.environ.get("KAGGLE_KEY", "")
    if not (user and key):
        raise RuntimeError("KAGGLE_USERNAME/KAGGLE_KEY required for Kaggle download")

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  Kaggle {dataset} → {dest}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "kaggle",
            "datasets",
            "download",
            "-d",
            dataset,
            "-p",
            str(dest),
            "--unzip",
        ],
        check=True,
    )
    return unwrap_zip_root(dest)


def extract_zip(zpath: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest)
    return unwrap_zip_root(dest)


def prepare_stage1(
    cfg: Stage1Config,
    *,
    clean: bool = True,
    use_road_crack: bool = False,
    use_pavement_distress: bool = False,
    use_pwd: bool = False,
    local_dir: Path | None = None,
    zip_path: Path | None = None,
) -> Path:
    """Download sources, ingest/remap, merge into cfg.dataset_dir."""
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

    if zip_path is not None:
        z = Path(zip_path)
        if not z.exists():
            raise FileNotFoundError(z)
        raw = extract_zip(z, raw_dir / "zip")
        force = "pothole" in z.name.lower() or "bharat" in z.name.lower()
        ingest_to_coco(raw, out_dir, force_pothole=force)
        print_class_histogram(out_dir)
        return out_dir

    if local_dir is not None:
        raw = Path(local_dir)
        if not raw.exists():
            raise FileNotFoundError(raw)
        ingest_to_coco(raw, out_dir, force_pothole=False)
        print_class_histogram(out_dir)
        return out_dir

    plan: list[tuple[str, str, dict, bool]] = [
        (
            "crri",
            "roboflow",
            {
                "workspace": cfg.roboflow_workspace,
                "project": cfg.roboflow_project,
                "version": cfg.roboflow_version,
            },
            False,
        ),
    ]
    if cfg.use_bharatpothole:
        plan.append(
            ("bharatpothole", "kaggle", {"dataset": cfg.kaggle_dataset}, True)
        )
    if use_road_crack:
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
    if use_pavement_distress:
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
    if use_pwd:
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
        raw_dest.mkdir(parents=True, exist_ok=True)
        try:
            if kind == "roboflow":
                raw = download_roboflow(
                    kwargs["workspace"],
                    kwargs["project"],
                    kwargs["version"],
                    raw_dest,
                    fmt=cfg.roboflow_format,
                )
            else:
                raw = download_kaggle(kwargs["dataset"], raw_dest)
            part_out = parts_dir / name
            if force_pothole:
                prepare_bharatpothole(raw, part_out)
            else:
                ingest_to_coco(raw, part_out, force_pothole=False)
            parts.append(part_out)
            print(f"  OK → {part_out}")
        except (Exception, SystemExit) as e:
            # Soft-skip optional extras; CRRI failure still soft when multi-source
            print(f"  SKIPPED {name}: {e}")

    if not parts:
        raise SystemExit("No Stage 1 sources downloaded successfully")

    if len(parts) == 1:
        # Single source: write directly to dataset_dir
        if out_dir.exists():
            shutil.rmtree(out_dir)
        shutil.copytree(parts[0], out_dir)
    else:
        print("\n=== Merging ===")
        merge_coco_datasets(parts, out_dir)

    train_ann = out_dir / "train" / "_annotations.coco.json"
    if not train_ann.exists():
        raise SystemExit(f"Stage 1 train annotations missing under {out_dir}")

    print_class_histogram(out_dir)
    print(f"\nSTAGE1_DIR = {out_dir}")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download and prepare RF-DETR Stage-1 COCO dataset (CRRI-first)."
    )
    p.add_argument(
        "--work-root",
        type=Path,
        default=None,
        help="Override data/rfdetr root",
    )
    p.add_argument("--no-clean", action="store_true", help="Keep existing raw/parts")
    p.add_argument(
        "--bharatpothole",
        action="store_true",
        help="Also download BharatPotHole from Kaggle",
    )
    p.add_argument("--road-crack", action="store_true", help="Add Road Crack Detection")
    p.add_argument(
        "--pavement-distress",
        action="store_true",
        help="Add Pavement Distress Roboflow set",
    )
    p.add_argument("--pwd", action="store_true", help="Add optional PWD Roboflow set")
    p.add_argument("--local-dir", type=Path, default=None, help="Ingest a local folder")
    p.add_argument("--zip", type=Path, default=None, help="Ingest a local zip")
    p.add_argument(
        "--env",
        type=Path,
        default=None,
        help="Path to .env (default: repo .env)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(args.env)
    cfg = Stage1Config(use_bharatpothole=args.bharatpothole)
    if args.work_root is not None:
        cfg.work_root = Path(args.work_root)
    prepare_stage1(
        cfg,
        clean=not args.no_clean,
        use_road_crack=args.road_crack,
        use_pavement_distress=args.pavement_distress,
        use_pwd=args.pwd,
        local_dir=args.local_dir,
        zip_path=args.zip,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
