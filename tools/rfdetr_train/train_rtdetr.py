"""Train Ultralytics RT-DETR-l on Stage-2 YOLO export (parallel-safe defaults)."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import repo_root
from .download import load_dotenv


def train_rtdetr(
    *,
    data_yaml: Path,
    output_dir: Path,
    model: str = "rtdetr-l.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 8,
    workers: int = 4,
    device: str = "0",
    memory_fraction: float = 0.45,
    resume: bool = False,
) -> Path:
    data_yaml = Path(data_yaml)
    if not data_yaml.is_file():
        raise SystemExit(
            f"Missing {data_yaml}\n"
            "Run: python -m tools.rfdetr_train.export_yolo "
            "--coco-dir data/rfdetr/stage2 --out-dir data/rfdetr/stage2_yolo"
        )

    # Soft VRAM cap so RF-DETR Large (already running) keeps headroom.
    frac = max(0.15, min(float(memory_fraction), 0.85))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(frac, device=int(device))
            print(
                f"CUDA memory fraction={frac:.2f} on device {device} "
                f"(leave room for RF-DETR Large)"
            )
    except Exception as e:
        print(f"WARNING: could not set memory fraction: {e}")

    from ultralytics import RTDETR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    project = str(output_dir.parent)
    name = output_dir.name

    print(
        f"Training Ultralytics {model} → {output_dir}\n"
        f"  data={data_yaml} epochs={epochs} imgsz={imgsz} "
        f"batch={batch} workers={workers} device={device}"
    )
    det = RTDETR(model)
    det.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        resume=resume,
        patience=0,  # match overnight full-run intent; set >0 to early-stop
        plots=True,
        save=True,
    )

    best = output_dir / "weights" / "best.pt"
    last = output_dir / "weights" / "last.pt"
    if best.exists():
        print("Best checkpoint:", best.resolve())
        return best
    if last.exists():
        print("Last checkpoint:", last.resolve())
        return last
    raise SystemExit(f"No weights under {output_dir / 'weights'}")


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    p = argparse.ArgumentParser(
        description="Train Ultralytics RT-DETR-l on Stage-2 YOLO data (parallel to RF-DETR)."
    )
    p.add_argument(
        "--data",
        type=Path,
        default=root / "data" / "rfdetr" / "stage2_yolo" / "data.yaml",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=root / "runs" / "rtdetr_stage2",
    )
    p.add_argument("--model", type=str, default="rtdetr-l.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="0")
    p.add_argument(
        "--memory-fraction",
        type=float,
        default=0.45,
        help="Fraction of GPU VRAM this process may use (default 0.45)",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--env", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(args.env)
    best = train_rtdetr(
        data_yaml=args.data,
        output_dir=args.output_dir,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        memory_fraction=args.memory_fraction,
        resume=args.resume,
    )
    print("DONE", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
