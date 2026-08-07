"""Train RFDETRMedium Stage 1 (headless / VM)."""
from __future__ import annotations

import argparse
import inspect
import os
import subprocess
import time
from pathlib import Path

from .config import Stage1Config
from .download import load_dotenv


def print_gpu_info() -> None:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        print("GPU:", (out.stdout or out.stderr or "nvidia-smi failed").strip())
    except FileNotFoundError:
        print("GPU: nvidia-smi not found")

    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(
                f"torch.cuda: {torch.cuda.get_device_name(0)} | "
                f"free={free / 1024**3:.1f} GiB / total={total / 1024**3:.1f} GiB"
            )
            print(
                "Tip: if free > ~20 GiB after model load, try --batch 24 or 32; "
                "on OOM back off by 4."
            )
        else:
            print("torch.cuda.is_available() = False")
    except Exception as e:
        print(f"torch probe skipped: {e}")


def find_best_checkpoint(output_dir: Path) -> Path | None:
    for name in (
        "checkpoint_best_total.pth",
        "checkpoint_best_ema.pth",
        "checkpoint_best_regular.pth",
        "checkpoint.pth",
    ):
        cand = output_dir / name
        if cand.exists():
            return cand
    pths = sorted(
        output_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return pths[0] if pths else None


def _filter_kwargs(fn, kwargs: dict) -> dict:
    try:
        sig = inspect.signature(fn)
        params = sig.parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return kwargs
        return {k: v for k, v in kwargs.items() if k in params}
    except (TypeError, ValueError):
        return kwargs


def train_stage1(cfg: Stage1Config) -> Path:
    dataset_dir = Path(cfg.dataset_dir)
    train_ann = dataset_dir / "train" / "_annotations.coco.json"
    if not train_ann.exists():
        raise SystemExit(
            f"Dataset missing: {train_ann}\n"
            "Run: python -m tools.rfdetr_train.download   (or ./scripts/run_stage1.sh)"
        )

    print_gpu_info()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from rfdetr import RFDETRMedium

    print(f"\nTraining RFDETRMedium → {out_dir}")
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
            model = RFDETRMedium(pretrain_weights=str(cfg.resume))
        except TypeError:
            model = RFDETRMedium()
            model_kwargs["resume"] = str(cfg.resume)
    else:
        model = RFDETRMedium()

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

    mins = (time.time() - t0) / 60
    print(f"Stage 1 finished in {mins:.1f} min")

    best = find_best_checkpoint(out_dir)
    if best is None:
        raise SystemExit(f"No checkpoint in {out_dir} — check the training log.")
    print("Best checkpoint:", best.resolve())
    return best


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train RFDETRMedium Stage 1 on prepared COCO data."
    )
    p.add_argument("--dataset-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None, dest="batch_size")
    p.add_argument("--grad-accum", type=int, default=None, dest="grad_accum_steps")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--workers", type=int, default=None, dest="num_workers")
    p.add_argument("--no-early-stop", action="store_true")
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint.pth (or best) to continue / warm-start",
    )
    p.add_argument("--env", type=Path, default=None)
    return p


def config_from_args(args: argparse.Namespace) -> Stage1Config:
    base = Stage1Config()
    return Stage1Config(
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
        early_stopping=not args.no_early_stop,
        resume=args.resume or os.environ.get("RFDETR_RESUME") or None,
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
