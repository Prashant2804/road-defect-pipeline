"""Metadata for public drone/UAV pavement-distress sources + GSD normalization.

Ground sampling distance (GSD, cm/px) is what actually needs to match across
merged sources — not altitude by itself, since sensor and focal length vary
per drone/camera and change the GSD at a given altitude. None of the public
papers below publish full camera intrinsics (focal length + sensor size), so
the altitudes recorded here are for reference only — do NOT compute GSD from
them without confirming the camera. Measure GSD empirically instead (see
docs/DRONE_DATASETS.md "Calibrating GSD" for the recipe) and fill it into
SOURCE_GSD_CM_PX below, then run ``normalize_gsd.resize_split`` before
merging so a pothole of a given real-world size occupies a similar pixel
footprint regardless of which source it came from.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class DroneSource:
    key: str
    name: str
    url: str
    license: str
    format: str  # "voc" | "yolo" | "roboflow_coco"
    documented_altitude_m: float | None
    notes: str


DRONE_SOURCES: dict[str, DroneSource] = {
    "uav_pdd2023": DroneSource(
        key="uav_pdd2023",
        name="UAV-PDD2023",
        url="https://zenodo.org/records/8429208",
        license="CC BY 4.0",
        format="voc",
        documented_altitude_m=30.0,
        notes=(
            "2,440 images / 11,158 instances. Nadir camera, hovering or 0.8 m/s. "
            "Classes: longitudinal/transverse/oblique/alligator crack, repair, pothole."
        ),
    ),
    "uapd": DroneSource(
        key="uapd",
        name="UAPD (UAV Asphalt Pavement Distress Dataset)",
        url="https://github.com/tantantetetao/UAPD-Pavement-Distress-Dataset",
        license="Released for public research use (cite Zhu et al., Automation in Construction 2022)",
        format="voc",
        documented_altitude_m=None,
        notes=(
            "3,151 images, 512x512 crops from full-scale UAV frames. Same 6-way crack "
            "taxonomy as UAV-PDD2023 (earlier release from an overlapping author group) — "
            "dedupe against UAV-PDD2023 by file hash before merging, see ingest warning."
        ),
    ),
    "highrpd": DroneSource(
        key="highrpd",
        name="HighRPD",
        url="https://data.mendeley.com/datasets/sywswj7djj/1",
        license="CC BY 4.0",
        format="yolo",
        documented_altitude_m=50.0,
        notes=(
            "11,696 images tiled to 640x640 from DJI M300 + Zenmuse P1 (45MP) frames. "
            "3 classes only: line crack, block crack, pit (pothole) — no repair/patch class."
        ),
    ),
    "rf_pothole_drone": DroneSource(
        key="rf_pothole_drone",
        name="Roboflow: Pothole detection (by Drone)",
        url="https://universe.roboflow.com/drone-zh0ho/pothole-detection-zdizt",
        license="MIT",
        format="roboflow_coco",
        documented_altitude_m=None,
        notes="465 images, pothole-only, visually confirmed nadir crops. Good for scale diversity.",
    ),
}

# Fill in once measured — see docs/DRONE_DATASETS.md "Calibrating GSD".
# None = not yet measured; download_drone.py skips GSD normalization for that
# source until it's filled in (better to leave scale un-normalized than to
# silently resize by a guessed number). "_target" = the GSD your own drone
# will fly at — set it once you know your altitude + camera, then fill in
# each source above and every part gets rescaled to match it before merge.
SOURCE_GSD_CM_PX: dict[str, float | None] = {
    "uav_pdd2023": None,
    "uapd": None,
    "highrpd": None,
    "rf_pothole_drone": None,
    "_target": None,
}


def resize_split_to_target(
    coco_dir: Path,
    split: str,
    src_cm_per_px: float,
    dst_cm_per_px: float,
) -> None:
    """Rescale images + bbox/area in-place so 1 px == dst_cm_per_px cm.

    Call once per source, before merge_coco_datasets, with src_cm_per_px taken
    from SOURCE_GSD_CM_PX (measured, not guessed) and dst_cm_per_px = the GSD
    your drone will actually fly at. Skips (no-op) if src == dst.
    """
    scale = src_cm_per_px / dst_cm_per_px
    if abs(scale - 1.0) < 1e-6:
        return

    ann_path = coco_dir / split / "_annotations.coco.json"
    if not ann_path.exists():
        return
    doc = json.loads(ann_path.read_text(encoding="utf-8"))
    sdir = coco_dir / split

    for im in doc["images"]:
        img_path = sdir / im["file_name"]
        if not img_path.exists():
            continue
        with Image.open(img_path) as pil_im:
            new_w = max(1, round(pil_im.width * scale))
            new_h = max(1, round(pil_im.height * scale))
            resized = pil_im.resize((new_w, new_h), Image.LANCZOS)
            resized.save(img_path)
        im["width"], im["height"] = new_w, new_h

    for ann in doc["annotations"]:
        x, y, w, h = ann["bbox"]
        ann["bbox"] = [x * scale, y * scale, w * scale, h * scale]
        ann["area"] = float(w * scale * h * scale)

    ann_path.write_text(json.dumps(doc), encoding="utf-8")
    print(f"  GSD-normalized {split}: scale={scale:.4f} ({src_cm_per_px}cm/px -> {dst_cm_per_px}cm/px)")


def print_source_catalog() -> None:
    for src in DRONE_SOURCES.values():
        gsd = SOURCE_GSD_CM_PX.get(src.key)
        gsd_s = f"{gsd:.2f} cm/px (measured)" if gsd else "NOT MEASURED — see docs/DRONE_DATASETS.md"
        print(f"\n{src.name} [{src.key}]")
        print(f"  url: {src.url}")
        print(f"  license: {src.license}")
        print(f"  documented altitude: {src.documented_altitude_m or 'unspecified'} m")
        print(f"  GSD: {gsd_s}")
        print(f"  {src.notes}")
