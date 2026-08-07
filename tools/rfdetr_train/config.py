"""Defaults tuned for a single RTX 5090 (32 GB) Stage-1 run."""
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
    resume: str | None = None

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
