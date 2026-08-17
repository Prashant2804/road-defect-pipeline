"""Train RFDETRMedium on the merged drone/UAV Stage-1 dataset (headless / VM).

Reuses train_stage1.train_stage1() — it only touches fields DroneStage1Config
also defines (dataset_dir, epochs, batch_size, ...), so no logic is duplicated.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import DroneStage1Config
from .download import load_dotenv
from .train_stage1 import train_stage1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train RFDETRMedium on the prepared drone/UAV COCO dataset."
    )
    p.add_argument("--dataset-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None, dest="batch_size")
    p.add_argument("--grad-accum", type=int, default=None, dest="grad_accum_steps")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--workers", type=int, default=None, dest="num_workers")
    p.add_argument("--no-early-stop", action="store_true")
    p.add_argument("--early-stopping-patience", type=int, default=None)
    p.add_argument("--aug-preset", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--pretrain-weights", type=str, default=None)
    p.add_argument("--env", type=Path, default=None)
    return p


def config_from_args(args: argparse.Namespace) -> DroneStage1Config:
    base = DroneStage1Config()
    return DroneStage1Config(
        work_root=base.work_root,
        output_dir=Path(args.output_dir) if args.output_dir else base.output_dir,
        dataset_dir_override=Path(args.dataset_dir) if args.dataset_dir else None,
        epochs=args.epochs if args.epochs is not None else base.epochs,
        batch_size=args.batch_size if args.batch_size is not None else base.batch_size,
        grad_accum_steps=(
            args.grad_accum_steps if args.grad_accum_steps is not None else base.grad_accum_steps
        ),
        lr=args.lr if args.lr is not None else base.lr,
        num_workers=args.num_workers if args.num_workers is not None else base.num_workers,
        early_stopping=not args.no_early_stop,
        early_stopping_patience=(
            args.early_stopping_patience
            if args.early_stopping_patience is not None
            else base.early_stopping_patience
        ),
        resume=args.resume or os.environ.get("RFDETR_RESUME") or None,
        pretrain_weights=(
            args.pretrain_weights or os.environ.get("RFDETR_PRETRAIN_WEIGHTS") or None
        ),
        aug_preset=args.aug_preset,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(args.env)
    cfg = config_from_args(args)
    best = train_stage1(cfg)
    print("DONE", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
