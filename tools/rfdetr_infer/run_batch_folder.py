"""Batch: download all MP4s from a Drive folder → Stage-2 Medium infer → upload.

Uses custom Stage-2 6-class weights @ conf 0.30 and extended near-field ROI.
Does not change default single-video infer scripts.

Example::

    .venv/bin/python -m tools.rfdetr_infer.run_batch_folder \\
      --folder 'https://drive.google.com/drive/folders/FOLDER_ID' \\
      --upload

    ./scripts/run_rfdetr_batch_folder.sh \\
      --folder 'https://drive.google.com/drive/folders/FOLDER_ID' \\
      --upload
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from .config import InferConfig, repo_root
from .media_fetch import (
    download_drive_folder,
    drive_folder_id,
    find_srt_in_dir,
    find_videos_in_dir,
    is_drive_folder_url,
)
from .run import run_inference

DEFAULT_SOURCE_FOLDER = (
    "https://drive.google.com/drive/folders/1iAd2NiejJeYcJqSCUvnt5c3O6Ddi3KLB"
)
DEFAULT_UPLOAD_FOLDER = (
    "https://drive.google.com/drive/folders/1gFw80e4fMdL3ztDlUxVdQinNQlskpoz-"
)

DEFAULT_CONF = 0.30
DEFAULT_NMS_IOU = 0.5
DEFAULT_ROAD_TOP_Y = 0.32
DEFAULT_Z_FAR_M = 8.0


def _safe_name(stem: str, *, max_len: int = 48) -> str:
    s = re.sub(r"[^\w.\-]+", "-", stem.strip(), flags=re.UNICODE)
    s = re.sub(r"-{2,}", "-", s).strip("-._")
    return (s or "video")[:max_len]


def _default_weights(root: Path) -> Path:
    total = root / "runs" / "rfdetr_medium_custom_stage2" / "checkpoint_best_total.pth"
    if total.is_file():
        return total.resolve()
    ema = root / "runs" / "rfdetr_medium_custom_stage2" / "checkpoint_best_ema.pth"
    if ema.is_file():
        return ema.resolve()
    return total


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Download all videos from a Drive folder, run custom Stage-2 Medium "
            "(@ conf 0.30, extended ROI) on each, optionally upload to Drive."
        )
    )
    p.add_argument(
        "--folder",
        type=str,
        default=DEFAULT_SOURCE_FOLDER,
        help="Google Drive folder URL containing MP4s",
    )
    p.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Custom Stage-2 Medium checkpoint (default: custom_stage2 best)",
    )
    p.add_argument(
        "--media-dir",
        type=Path,
        default=None,
        help="Cache for Drive downloads (default: data/rfdetr/infer_media)",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Per-video dirs under this root (default: runs/rfdetr_infer/batch-<id>)",
    )
    p.add_argument("--conf", type=float, default=DEFAULT_CONF)
    p.add_argument("--stride", type=int, default=3, dest="frame_stride")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--z-near", type=float, default=0.5, dest="z_near_m")
    p.add_argument("--z-far", type=float, default=DEFAULT_Z_FAR_M, dest="z_far_m")
    p.add_argument("--road-top-y", type=float, default=DEFAULT_ROAD_TOP_Y)
    p.add_argument("--road-bottom-y", type=float, default=1.0)
    p.add_argument("--road-bottom-half-w", type=float, default=0.78)
    p.add_argument("--road-top-half-w", type=float, default=0.50)
    p.add_argument("--road-center-x", type=float, default=0.52)
    p.add_argument("--min-overlap", type=float, default=0.15)
    p.add_argument("--nms-iou", type=float, default=DEFAULT_NMS_IOU, dest="nms_iou")
    p.add_argument("--near-wash-alpha", type=float, default=0.28)
    p.add_argument("--far-wash-alpha", type=float, default=0.0)
    p.add_argument("--crf", type=int, default=23)
    p.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download Drive folder even if a local cache exists",
    )
    p.add_argument(
        "--upload",
        action="store_true",
        help="Upload each run to Drive after infer",
    )
    p.add_argument(
        "--upload-folder",
        type=str,
        default=DEFAULT_UPLOAD_FOLDER,
        help="Parent Drive folder for uploads",
    )
    p.add_argument(
        "--client-secret",
        type=Path,
        default=None,
        help="OAuth client JSON (default: ~/secrets/drive_oauth_client.json)",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a video if out-dir already has annotated.mp4",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = repo_root()

    folder_url = str(args.folder).strip()
    if not is_drive_folder_url(folder_url):
        raise SystemExit(f"--folder must be a Google Drive folder URL, got: {folder_url}")

    fid = drive_folder_id(folder_url) or "unknown"
    weights = args.weights
    if weights is None:
        weights = _default_weights(root)
    else:
        weights = Path(weights).expanduser()
    if not weights.is_file():
        raise SystemExit(
            f"--weights is not a file: {weights}\n"
            "Expected e.g. runs/rfdetr_medium_custom_stage2/checkpoint_best_total.pth"
        )
    weights = weights.resolve()

    media_dir = args.media_dir or (root / "data" / "rfdetr" / "infer_media")
    media_dir = Path(media_dir)
    media_dir.mkdir(parents=True, exist_ok=True)

    cache = media_dir / f"drive_folder_{fid}"
    if args.force_download and cache.exists():
        import shutil

        print(f"==> --force-download: removing cache {cache}")
        shutil.rmtree(cache)

    print(f"==> Download Drive folder {fid}")
    print("    (must be shared as Anyone with the link)")
    local_folder = download_drive_folder(folder_url, media_dir)
    videos = find_videos_in_dir(local_folder)
    if not videos:
        raise SystemExit(f"No MP4/video files under {local_folder}")

    print(f"==> Found {len(videos)} video(s):")
    for v in videos:
        print(f"    - {v.relative_to(local_folder)} ({v.stat().st_size / 1e6:.1f} MB)")

    out_root = args.out_root
    if out_root is None:
        out_root = root / "runs" / "rfdetr_infer" / f"batch-{fid[:8]}"
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(
        f"==> Batch infer: weights={weights.name} conf={args.conf} "
        f"road_top_y={args.road_top_y} z_far={args.z_far_m}"
    )
    print(f"    out_root={out_root}")

    client_secret = args.client_secret
    if client_secret is None:
        client_secret = Path.home() / "secrets" / "drive_oauth_client.json"
    else:
        client_secret = Path(client_secret).expanduser()

    results: list[tuple[Path, Path]] = []
    for i, video in enumerate(videos, 1):
        safe = _safe_name(video.stem)
        out_dir = out_root / safe
        ann = out_dir / "annotated.mp4"
        print(f"\n===== [{i}/{len(videos)}] {video.name} → {out_dir.name} =====")
        if args.skip_existing and ann.is_file() and ann.stat().st_size > 0:
            print(f"  skip (exists): {ann}")
            results.append((video, out_dir))
            continue

        srt = find_srt_in_dir(local_folder, prefer_stem=video.stem)
        if srt is None:
            # sibling next to the video
            sib = video.with_suffix(".srt")
            if sib.is_file():
                srt = sib
        print(f"  srt: {srt or '(none)'}")

        cfg = InferConfig(
            video=video,
            weights=weights,
            srt=srt,
            out_dir=out_dir,
            conf=args.conf,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            z_near_m=args.z_near_m,
            z_far_m=args.z_far_m,
            road_top_y=args.road_top_y,
            road_bottom_y=args.road_bottom_y,
            road_bottom_half_w=args.road_bottom_half_w,
            road_top_half_w=args.road_top_half_w,
            road_center_x=args.road_center_x,
            use_classical_road=False,
            min_overlap=args.min_overlap,
            require_center=False,
            clip_to_mask=False,
            nms_iou=args.nms_iou,
            near_wash_alpha=args.near_wash_alpha,
            far_wash_alpha=args.far_wash_alpha,
            crf=args.crf,
            backend="rfdetr",
        )
        run_inference(cfg)
        results.append((video, out_dir))

        if args.upload:
            from .upload_drive import upload_run

            sub = f"medium-custom-s2-c030-batch-{fid[:8]}-{safe}"
            print(f"==> Upload → subfolder {sub}")
            if not client_secret.is_file():
                raise SystemExit(
                    f"Missing OAuth client: {client_secret}\n"
                    "Pass --client-secret or place drive_oauth_client.json in ~/secrets/"
                )
            upload_run(
                out_dir,
                args.upload_folder,
                client_secret=client_secret,
                subfolder=sub,
            )

    print("\n===== BATCH DONE =====")
    for video, out_dir in results:
        print(f"  {video.name} → {out_dir}")
    if args.upload:
        print(f"Drive parent: {args.upload_folder}")
    else:
        print("Tip: re-run with --upload (or upload each run-dir) to push to Drive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
