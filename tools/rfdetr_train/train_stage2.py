"""Train RFDETRLarge Stage 2 (headless / VM, overnight)."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from .config import Stage2Config
from .download import load_dotenv
from .train_stage1 import _filter_kwargs, find_best_checkpoint, print_gpu_info


def train_stage2(cfg: Stage2Config) -> Path:
    dataset_dir = Path(cfg.dataset_dir)
    train_ann = dataset_dir / "train" / "_annotations.coco.json"
    if not train_ann.exists():
        raise SystemExit(
            f"Dataset missing: {train_ann}\n"
            "Run: python -m tools.rfdetr_train.download_stage2\n"
            "  or: ./scripts/run_stage2.sh"
        )

    print_gpu_info()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from rfdetr import RFDETRLarge

    print(f"\nTraining RFDETRLarge → {out_dir}")
    print(
        f"epochs={cfg.epochs} batch={cfg.batch_size} grad_accum={cfg.grad_accum_steps} "
        f"lr={cfg.lr} workers={cfg.num_workers} "
        f"(effective batch {cfg.effective_batch})"
    )
    if cfg.resume:
        print(f"resume={cfg.resume}")

    model_kwargs: dict = {}
    if cfg.resume:
        try:
            model = RFDETRLarge(pretrain_weights=str(cfg.resume))
        except TypeError:
            model = RFDETRLarge()
            model_kwargs["resume"] = str(cfg.resume)
    else:
        model = RFDETRLarge()

    train_kwargs = dict(
        dataset_dir=str(dataset_dir),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        grad_accum_steps=cfg.grad_accum_steps,
        lr=cfg.lr,
        output_dir=str(out_dir),
        num_workers=cfg.num_workers,
        **model_kwargs,
    )
    if cfg.early_stopping:
        train_kwargs["early_stopping"] = True
    if cfg.resume and "resume" not in train_kwargs:
        train_kwargs["resume"] = str(cfg.resume)

    train_kwargs = _filter_kwargs(model.train, train_kwargs)

    t0 = time.time()
    try:
        model.train(**train_kwargs)
    except TypeError as e:
        print("Retrying with reduced kwargs after TypeError:", e)
        for drop in ("early_stopping", "num_workers", "resume"):
            train_kwargs.pop(drop, None)
        train_kwargs = _filter_kwargs(model.train, train_kwargs)
        model.train(**train_kwargs)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            print(
                "OOM during RFDETRLarge train. Retry with smaller batch, e.g.:\n"
                "  EXTRA_TRAIN_ARGS='--batch 2 --grad-accum 8' ./scripts/run_stage2.sh"
            )
        raise

    mins = (time.time() - t0) / 60
    print(f"Stage 2 finished in {mins:.1f} min")

    best = find_best_checkpoint(out_dir)
    if best is None:
        raise SystemExit(f"No checkpoint in {out_dir} — check the training log.")
    print("Best checkpoint:", best.resolve())
    return best


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train RFDETRLarge Stage 2 on prepared multi-source COCO data."
    )
    p.add_argument("--dataset-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None, dest="batch_size")
    p.add_argument("--grad-accum", type=int, default=None, dest="grad_accum_steps")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--workers", type=int, default=None, dest="num_workers")
    p.add_argument(
        "--early-stop",
        action="store_true",
        help="Enable early stopping (off by default for full overnight 100ep)",
    )
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--env", type=Path, default=None)
    return p


def config_from_args(args: argparse.Namespace) -> Stage2Config:
    base = Stage2Config()
    return Stage2Config(
        work_root=base.work_root,
        output_dir=Path(args.output_dir) if args.output_dir else base.output_dir,
        dataset_dir_override=Path(args.dataset_dir) if args.dataset_dir else None,
        epochs=args.epochs if args.epochs is not None else base.epochs,
        batch_size=args.batch_size if args.batch_size is not None else base.batch_size,
        grad_accum_steps=(
            args.grad_accum_steps
            if args.grad_accum_steps is not None
            else base.grad_accum_steps
        ),
        lr=args.lr if args.lr is not None else base.lr,
        num_workers=(
            args.num_workers if args.num_workers is not None else base.num_workers
        ),
        early_stopping=bool(args.early_stop),
        resume=args.resume or os.environ.get("RFDETR_RESUME") or None,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(args.env)
    cfg = config_from_args(args)
    best = train_stage2(cfg)
    print("DONE", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
