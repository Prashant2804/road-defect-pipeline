"""Defaults tuned for a single RTX 5090 (32 GB) Stage-1 / Stage-2 runs."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class Stage1Config:
    # Paths (repo-relative by default)
    work_root: Path = field(default_factory=lambda: repo_root() / "data" / "rfdetr")
    output_dir: Path = field(default_factory=lambda: repo_root() / "runs" / "rfdetr_stage1")
    # Optional absolute override for prepared COCO root (train/valid/_annotations...)
    dataset_dir_override: Path | None = None

    # CRRI defaults (validated on Colab)
    roboflow_workspace: str = "crri"
    roboflow_project: str = "crri-road-pavement-distress-project"
    roboflow_version: int = 3
    roboflow_format: str = "coco"

    # Optional extras
    use_bharatpothole: bool = False
    kaggle_dataset: str = "surbhisaswatimohanty/bharatpothole"

    # Train — maximize 5090 without OOM on Medium @ 576
    epochs: int = 50
    batch_size: int = 16
    grad_accum_steps: int = 1
    lr: float = 1e-4
    num_workers: int = 8
    early_stopping: bool = True
    early_stopping_patience: int = 10
    resume: str | None = None
    # Optional Albumentations preset name (see augmentations.AUG_PRESETS); None = rfdetr default
    aug_preset: str | None = None

    @property
    def raw_dir(self) -> Path:
        return self.work_root / "stage1_raw"

    @property
    def parts_dir(self) -> Path:
        return self.work_root / "stage1_parts"

    @property
    def dataset_dir(self) -> Path:
        if self.dataset_dir_override is not None:
            return Path(self.dataset_dir_override)
        return self.work_root / "stage1"

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum_steps


@dataclass
class Stage2Config:
    """Multi-source merge + RFDETRLarge overnight training."""

    work_root: Path = field(default_factory=lambda: repo_root() / "data" / "rfdetr")
    output_dir: Path = field(default_factory=lambda: repo_root() / "runs" / "rfdetr_stage2")
    dataset_dir_override: Path | None = None

    # Reuse CRRI as India backbone
    roboflow_workspace: str = "crri"
    roboflow_project: str = "crri-road-pavement-distress-project"
    roboflow_version: int = 3
    roboflow_format: str = "coco"

    use_crri: bool = True
    use_rdd_india: bool = True
    use_bharatpothole: bool = True
    use_crack2: bool = True
    use_pavement_distress: bool = True
    use_road_crack: bool = True
    use_road_damage2: bool = True
    use_water_logging: bool = True
    use_drain_overflow: bool = True
    use_pwd: bool = True

    kaggle_dataset: str = "surbhisaswatimohanty/bharatpothole"
    rdd_india_url: str = (
        "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/"
        "RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_India.zip"
    )

    # Cap pothole-only images so rare classes aren't drowned
    pothole_max_fraction: float = 0.45

    # RFDETRLarge on 5090 — start conservative; bump batch if VRAM allows
    epochs: int = 100
    batch_size: int = 4
    grad_accum_steps: int = 4  # effective batch 16
    lr: float = 1e-4
    num_workers: int = 8
    early_stopping: bool = False  # full 100-epoch overnight by default
    resume: str | None = None

    @property
    def raw_dir(self) -> Path:
        return self.work_root / "stage2_raw"

    @property
    def parts_dir(self) -> Path:
        return self.work_root / "stage2_parts"

    @property
    def dataset_dir(self) -> Path:
        if self.dataset_dir_override is not None:
            return Path(self.dataset_dir_override)
        return self.work_root / "stage2"

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum_steps
