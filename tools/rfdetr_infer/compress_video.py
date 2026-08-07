"""Compress annotated OpenCV mp4v output to H.264 for Drive / preview."""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def compress_mp4(
    src: Path,
    dst: Path | None = None,
    *,
    crf: int = 23,
    preset: str = "medium",
    replace_src: bool = False,
) -> Path:
    """Re-encode with libx264. Default CRF 23 is a good size/quality tradeoff."""
    src = Path(src)
    if not src.is_file():
        raise FileNotFoundError(src)
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Install: sudo apt install -y ffmpeg")

    if dst is None:
        dst = src.with_name(src.stem + "_h264.mp4")
    dst = Path(dst)
    if dst.resolve() == src.resolve():
        tmp = src.with_suffix(".tmp.mp4")
    else:
        tmp = dst

    src_mb = src.stat().st_size / (1024 * 1024)
    print(f"Compressing {src.name} ({src_mb:.1f} MB) → {tmp.name} (crf={crf}, preset={preset})")
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-stats",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(int(crf)),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(tmp),
    ]
    subprocess.run(cmd, check=True)
    out_mb = tmp.stat().st_size / (1024 * 1024)
    print(f"Done: {out_mb:.1f} MB ({100.0 * out_mb / max(src_mb, 1e-6):.1f}% of original)")

    if replace_src:
        src.unlink(missing_ok=True)
        tmp.rename(src)
        return src
    if tmp != dst:
        tmp.rename(dst)
    return dst


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compress annotated.mp4 to H.264 for upload/preview.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--crf", type=int, default=23, help="Lower=better quality/larger (18–28 typical)")
    p.add_argument("--preset", type=str, default="medium")
    p.add_argument(
        "--replace",
        action="store_true",
        help="Replace the input file with the compressed H.264 version",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = compress_mp4(
        args.input,
        args.output,
        crf=args.crf,
        preset=args.preset,
        replace_src=args.replace,
    )
    print("Output:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
