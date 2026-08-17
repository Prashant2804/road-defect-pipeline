"""Download and prepare the drone/UAV Stage-1 COCO dataset (6-class taxonomy).

Sources (see tools/rfdetr_train/drone_sources.py for citations/licenses):
  - UAV-PDD2023 (Zenodo, CC BY 4.0)      — VOC XML, 6-way crack+pothole taxonomy
  - UAPD        (Google Drive, public)   — VOC XML, same taxonomy, dedup'd against UAV-PDD2023
  - HighRPD     (Mendeley, CC BY 4.0)    — YOLO txt, 3 classes (line/block/pit)
  - Roboflow "Pothole detection by Drone" (MIT) — pothole-only, nadir-confirmed
  - CQU-BPDD ravelling subset (CC BY-NC 4.0, opt-in via --cqu-bpdd-ravelling) —
    the only public ravelling source at a near-nadir angle; not a drone, and
    whole-image weak labels only. See docs/DRONE_DATASETS.md before enabling.
  - Roboflow "Pavement Distress Datasets" by COLLEGE (Public Domain) and
    "RD01" by RCDRD01 (MIT) — ground-level/handheld close-range shots, NOT
    drone or nadir, but the nearest available match for ravelling/edge_damage:
    Indian-PWD-style severity taxonomy, real instance-level boxes (not whole-
    image labels like CQU-BPDD). On by default; see docs/DRONE_DATASETS.md
    "ravelling / edge_damage sourcing" for the viewpoint-mismatch tradeoff.

drainage_issue has no public source at any camera angle — merge in your own
hand-labeled bootstrap via --extra-local drainage_issue=/path/to/coco.
See docs/DRONE_DATASETS.md "Coverage gaps".
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import io
import struct
import zipfile
import zlib

from .coco_io import (
    ensure_roboflow_coco_layout,
    ingest_to_coco,
    merge_coco_datasets,
    print_class_histogram,
)
from .config import DroneStage1Config
from .download import download_roboflow, extract_zip, load_dotenv
from .drone_ingest import ingest_classification_folder, ingest_highrpd, ingest_voc_dataset
from .drone_sources import CQU_BPDD_TRAIN_COUNTS, SOURCE_GSD_CM_PX, resize_split_to_target

ZENODO_UAV_PDD2023_URL = "https://zenodo.org/api/records/8429208/files/UAV-PDD2023.zip/content"
MENDELEY_HIGHRPD_API = "https://data.mendeley.com/public-api/datasets/sywswj7djj"
UAPD_GDRIVE_FILE_ID = "1yQ0GMXFwwM5qdYY_5HzJBQqqjNtWJxEc"
CQU_BPDD_URL = "https://huggingface.co/datasets/Ggggcs/CQU-BPDD/resolve/main/CQU-BPDD.zip"


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


class _HTTPRangeFile(io.RawIOBase):
    """Minimal seekable file-like object over HTTP Range requests, for zipfile.

    Used to read a remote ZIP's central directory (a few KB) without
    downloading the whole archive — zipfile only seeks/reads what it needs.
    """

    def __init__(self, url: str):
        self.url = url
        with urllib.request.urlopen(
            urllib.request.Request(url, method="HEAD", headers={"User-Agent": "road-defect-pipeline/1.0"})
        ) as resp:
            self.size = int(resp.headers["Content-Length"])
        self.pos = 0

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def readable(self) -> bool:
        return True

    def readinto(self, b: bytearray) -> int:
        end = min(self.pos + len(b), self.size) - 1
        if self.pos > end:
            return 0
        req = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={self.pos}-{end}", "User-Agent": "road-defect-pipeline/1.0"},
        )
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def download_cqu_bpdd_ravelling(dest: Path) -> Path:
    """Pull just the ravelling images out of CQU-BPDD's train split.

    CQU-BPDD (non-commercial license, see drone_sources.DRONE_SOURCES) is not
    a drone dataset — an in-vehicle inspection camera, not drone altitude —
    and only ships whole-image classification labels, no boxes. It's the only
    public source with any ravelling coverage at a near-nadir angle, so it's
    opt-in (--cqu-bpdd-ravelling) rather than part of the default merge.

    The archive nests train/val/test as three separately-DEFLATEd inner zips
    inside one outer zip, so we can fetch just train.zip's byte range from the
    outer archive (skip the far larger test.zip) — but the inner stream still
    has to be decompressed sequentially start-to-end to reach its own central
    directory, so this downloads and decompresses the full train.zip (~3.9GB
    compressed). Once local, folders are matched against CQU-BPDD's own
    published per-class counts (drone_sources.CQU_BPDD_TRAIN_COUNTS) rather
    than by name — the archive's internal folder names (e.g.
    "cementation_fissures") don't match the paper's English class names.
    """
    dest.mkdir(parents=True, exist_ok=True)
    local_train_zip = dest / "train.zip"

    if not local_train_zip.exists():
        print("  Locating train.zip inside the CQU-BPDD archive...")
        outer = zipfile.ZipFile(_HTTPRangeFile(CQU_BPDD_URL))
        train_info = next(i for i in outer.infolist() if i.filename == "train.zip")
        if train_info.compress_type != zipfile.ZIP_DEFLATED:
            raise RuntimeError(
                f"Expected train.zip to be DEFLATE-compressed inside the outer archive, "
                f"got compress_type={train_info.compress_type}"
            )
        # Read just the local file header to find where the compressed data starts.
        header = _range_get(CQU_BPDD_URL, train_info.header_offset, train_info.header_offset + 29)
        fnlen, exlen = header[26:28], header[28:30]
        data_start = train_info.header_offset + 30 + struct.unpack("<H", fnlen)[0] + struct.unpack("<H", exlen)[0]
        data_end = data_start + train_info.compress_size - 1
        print(
            f"  Streaming train.zip ({train_info.compress_size / 1e9:.2f} GB compressed) "
            "from the outer archive — this is the big one, expect several minutes..."
        )
        _stream_range_inflate(CQU_BPDD_URL, data_start, data_end, local_train_zip)

    print("  Identifying the ravelling folder by published per-class image count...")
    with zipfile.ZipFile(local_train_zip) as zf:
        by_folder: dict[str, list[str]] = {}
        for name in zf.namelist():
            parts = name.split("/")
            if len(parts) >= 3 and parts[0] == "train" and not name.endswith("/"):
                by_folder.setdefault(parts[1], []).append(name)
        counts = {k: len(v) for k, v in by_folder.items()}
        print(f"  folder counts: {counts}")
        target = CQU_BPDD_TRAIN_COUNTS["ravelling"]
        matches = [k for k, n in counts.items() if n == target]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one folder with {target} images (ravelling per "
                f"CQU-BPDD's own README table), found {matches}. Folder counts: {counts}. "
                "Inspect data/rfdetr_drone/stage1_raw/cqu_bpdd_ravelling/train.zip by hand."
            )
        ravel_folder = matches[0]
        print(f"  matched folder: {ravel_folder!r} ({target} images)")

        out_img_dir = dest / "ravelling_images"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        for name in by_folder[ravel_folder]:
            with zf.open(name) as src, open(out_img_dir / Path(name).name, "wb") as dst_f:
                shutil.copyfileobj(src, dst_f)

    return out_img_dir


def _range_get(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(
        url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "road-defect-pipeline/1.0"}
    )
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def _stream_range_inflate(url: str, start: int, end: int, dest: Path, *, chunk_mb: int = 16) -> None:
    """Fetch bytes[start:end] and stream-inflate (raw DEFLATE) them to dest.

    Bounded memory regardless of the (multi-GB) range size: only one chunk
    and zlib's internal state are held at a time.
    """
    chunk = chunk_mb * 1024 * 1024
    total = end - start + 1
    decompressor = zlib.decompressobj(-15)
    written = 0
    req = urllib.request.Request(
        url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "road-defect-pipeline/1.0"}
    )
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            f.write(decompressor.decompress(buf))
            written += len(buf)
            print(f"\r  {dest.name}: {written / 1e6:.0f}/{total / 1e6:.0f} MB compressed", end="", flush=True)
        f.write(decompressor.flush())
    print()


def prepare_drone_stage1(
    cfg: DroneStage1Config,
    *,
    clean: bool = True,
    use_uav_pdd2023: bool = True,
    use_uapd: bool = True,
    use_highrpd: bool = True,
    use_rf_pothole_drone: bool = True,
    use_college_pavement_distress: bool = True,
    use_rd01_pwd: bool = True,
    use_cqu_bpdd_ravelling: bool = False,
    local_overrides: dict[str, Path] | None = None,
    extra_local: list[tuple[str, Path]] | None = None,
) -> Path:
    raw_dir, parts_dir, out_dir = cfg.raw_dir, cfg.parts_dir, cfg.dataset_dir
    local_overrides = local_overrides or {}
    extra_local = extra_local or []

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

    # Ground-level/handheld close-range shots, NOT drone or nadir — the nearest
    # available match for ravelling/edge_damage rather than an exact one. See
    # docs/DRONE_DATASETS.md "ravelling / edge_damage sourcing" for the tradeoff.
    if use_college_pavement_distress:
        print("\n=== college_pavement_distress ===")
        try:
            raw_dest = raw_dir / "college_pavement_distress"
            raw = local_overrides.get("college_pavement_distress") or download_roboflow(
                "college-7qowe", "pavement-distress-datasets", 1, raw_dest, fmt="coco"
            )
            part_out = parts_dir / "college_pavement_distress"
            part_out = ensure_roboflow_coco_layout(raw, part_out)
            parts.append(part_out)
            print(f"  OK -> {part_out}")
        except Exception as e:
            print(f"  SKIPPED college_pavement_distress: {e}")

    if use_rd01_pwd:
        print("\n=== rd01_pwd ===")
        try:
            raw_dest = raw_dir / "rd01_pwd"
            raw = local_overrides.get("rd01_pwd") or download_roboflow(
                "rcdrd01", "rd01", 1, raw_dest, fmt="coco"
            )
            part_out = parts_dir / "rd01_pwd"
            part_out = ensure_roboflow_coco_layout(raw, part_out)
            parts.append(part_out)
            print(f"  OK -> {part_out}")
        except Exception as e:
            print(f"  SKIPPED rd01_pwd: {e}")

    if use_cqu_bpdd_ravelling:
        print("\n=== cqu_bpdd_ravelling ===")
        print(
            "  LICENSE: CQU-BPDD is CC BY-NC 4.0 (non-commercial only). Do not enable "
            "this source if the trained model ships in a commercial product."
        )
        print(
            "  NOTE: not a drone source (in-vehicle inspection camera) and whole-image "
            "weak labels only (full-frame box, not a tight crop) — see docs/DRONE_DATASETS.md."
        )
        try:
            raw = local_overrides.get("cqu_bpdd_ravelling") or download_cqu_bpdd_ravelling(
                raw_dir / "cqu_bpdd_ravelling"
            )
            part_out = parts_dir / "cqu_bpdd_ravelling"
            part_out = ingest_classification_folder(Path(raw), part_out, "ravelling")
            _apply_gsd(part_out, "cqu_bpdd_ravelling")
            parts.append(part_out)
            print(f"  OK -> {part_out}")
        except Exception as e:
            print(f"  SKIPPED cqu_bpdd_ravelling: {e}")

    # Hand-labeled bootstrap batches (your own drone footage, labeled via the
    # SAM-assisted workflow — see README "Labeling workflow (SAM-assisted)").
    # This is the real path to drainage_issue/edge_damage coverage: no public
    # drone dataset has either (see docs/DRONE_DATASETS.md). Any standard
    # COCO or YOLO export using this repo's class names works here — the same
    # auto-detecting ingest_to_coco the dashcam pipeline uses for --local-dir.
    for name, path in extra_local:
        print(f"\n=== {name} (hand-labeled bootstrap) ===")
        try:
            part_out = parts_dir / name
            part_out = ingest_to_coco(Path(path), part_out)
            _apply_gsd(part_out, name)
            parts.append(part_out)
            print(f"  OK -> {part_out}")
        except Exception as e:
            print(f"  SKIPPED {name}: {e}")

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
    if use_college_pavement_distress or use_rd01_pwd:
        print(
            "\nNOTE: ravelling/edge_damage instances above include ground-level/"
            "handheld close-range sources (college_pavement_distress, rd01_pwd) — "
            "not drone/nadir. Nearest available match, not an exact one; see "
            "docs/DRONE_DATASETS.md 'ravelling / edge_damage sourcing'."
        )
    if not extra_local:
        print(
            "\nNOTE: drainage_issue has no public source at any camera angle — "
            "see docs/DRONE_DATASETS.md 'Coverage gaps'. Use --extra-local "
            "drainage_issue=/path/to/coco_or_yolo once you have a hand-labeled "
            "bootstrap batch."
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
        "--no-college-pavement-distress",
        action="store_true",
        help="Drop the ground-level ravelling/edge_damage/pothole source (Public Domain, "
        "not drone/nadir — see docs/DRONE_DATASETS.md). On by default.",
    )
    p.add_argument(
        "--no-rd01-pwd",
        action="store_true",
        help="Drop the ground-level Indian-PWD-style ravelling/edge_damage source (MIT, "
        "not drone/nadir — see docs/DRONE_DATASETS.md). On by default.",
    )
    p.add_argument(
        "--cqu-bpdd-ravelling",
        action="store_true",
        help="Opt in to CQU-BPDD's ravelling subset (CC BY-NC 4.0 non-commercial only; "
        "not drone altitude; whole-image weak labels). Off by default — see "
        "docs/DRONE_DATASETS.md before enabling. Downloads+decompresses ~3.9GB.",
    )
    p.add_argument(
        "--local",
        action="append",
        default=[],
        metavar="SOURCE=PATH",
        help="Use a manually-downloaded folder/zip instead of fetching SOURCE "
        "(uav_pdd2023|uapd|highrpd|rf_pothole_drone|college_pavement_distress|"
        "rd01_pwd|cqu_bpdd_ravelling). Repeatable.",
    )
    p.add_argument(
        "--extra-local",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Merge in an additional hand-labeled COCO or YOLO dataset (e.g. your own "
        "drone footage labeled for drainage_issue/edge_damage via the SAM-assisted "
        "workflow). NAME is just a tag for the merged filenames. Repeatable.",
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

    extra: list[tuple[str, Path]] = []
    for item in args.extra_local:
        if "=" not in item:
            raise SystemExit(f"--extra-local expects NAME=PATH, got: {item}")
        name, _, path_s = item.partition("=")
        p = Path(path_s)
        extra.append((name, extract_zip(p, cfg.raw_dir / f"{name}_local") if p.is_file() else p))

    prepare_drone_stage1(
        cfg,
        clean=not args.no_clean,
        use_uav_pdd2023=not args.no_uav_pdd2023,
        use_uapd=not args.no_uapd,
        use_highrpd=not args.no_highrpd,
        use_rf_pothole_drone=not args.no_rf_pothole_drone,
        use_college_pavement_distress=not args.no_college_pavement_distress,
        use_rd01_pwd=not args.no_rd01_pwd,
        use_cqu_bpdd_ravelling=args.cqu_bpdd_ravelling,
        local_overrides=overrides,
        extra_local=extra,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
