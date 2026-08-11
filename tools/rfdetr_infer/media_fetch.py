"""Download dashcam media from GCS, HTTPS, or Google Drive (file/folder)."""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urlparse

_DRIVE_FILE_RE = re.compile(
    r"drive\.google\.com/(?:file/d/|open\?id=|uc\?.*id=)([A-Za-z0-9_-]{20,})",
    re.I,
)
_DRIVE_FOLDER_RE = re.compile(
    r"drive\.google\.com/(?:drive/)?folders/([A-Za-z0-9_-]{20,})",
    re.I,
)
_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
_SRT_EXTS = {".srt", ".SRT"}


def looks_like_url(value: str | Path) -> bool:
    s = str(value).strip()
    return s.startswith(("gs://", "https://", "http://"))


def is_drive_folder_url(url: str) -> bool:
    return _DRIVE_FOLDER_RE.search(url) is not None


def is_drive_file_url(url: str) -> bool:
    return _DRIVE_FILE_RE.search(url) is not None and not is_drive_folder_url(url)


def drive_folder_id(url: str) -> str | None:
    m = _DRIVE_FOLDER_RE.search(url)
    return m.group(1) if m else None


def drive_file_id(url: str) -> str | None:
    m = _DRIVE_FILE_RE.search(url)
    return m.group(1) if m else None


def _fmt_size(n: int) -> str:
    x = float(n)
    for u in ("B", "KB", "MB", "GB"):
        if x < 1024:
            return f"{x:.1f} {u}"
        x /= 1024.0
    return f"{x:.1f} TB"


def _filename_from_url(url: str, default: str) -> str:
    if url.startswith("gs://"):
        name = url.rstrip("/").split("/")[-1]
        return unquote(name) or default
    path = unquote(urlparse(url).path)
    name = Path(path).name
    if name and "." in name and "folders" not in path:
        return name
    m = re.search(r"[?&](?:name|file)=([^&]+)", url)
    if m:
        return unquote(m.group(1).split("/")[-1]) or default
    return default


def _ensure_gdown() -> str:
    gdown = shutil.which("gdown")
    if gdown:
        return gdown
    # fall back to python -m gdown
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("  Installing gdown for Google Drive downloads...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "gdown"],
            check=True,
        )
    return sys.executable


def _run_gdown(args: list[str]) -> None:
    """Run gdown (module first). Retry without optional flags on older CLIs."""
    launchers = [
        [sys.executable, "-m", "gdown"],
    ]
    gdown_bin = shutil.which("gdown")
    if gdown_bin:
        launchers.append([gdown_bin])

    optional = {"--remaining-ok", "--fuzzy"}
    arg_variants = [args]
    cleaned = [a for a in args if a not in optional]
    if cleaned != args:
        arg_variants.append(cleaned)

    last_err: Exception | None = None
    for launcher in launchers:
        for variant in arg_variants:
            cmd = [*launcher, *variant]
            print("  $", " ".join(cmd))
            try:
                subprocess.run(cmd, check=True)
                return
            except subprocess.CalledProcessError as e:
                last_err = e
                # exit 2 is often argparse / bad flag — try next variant
                continue
    assert last_err is not None
    raise last_err


def find_video_in_dir(root: Path) -> Path | None:
    cands = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in _VIDEO_EXTS
    ]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_size, reverse=True)
    return cands[0]


def find_srt_in_dir(root: Path, prefer_stem: str | None = None) -> Path | None:
    cands = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() == ".srt"
    ]
    if not cands:
        return None
    if prefer_stem:
        for p in cands:
            if p.stem == prefer_stem:
                return p
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def download_drive_folder(url: str, dest_dir: Path) -> Path:
    """Download a shared Drive folder into dest_dir/<folder_id>/."""
    fid = drive_folder_id(url)
    if not fid:
        raise ValueError(f"Not a Google Drive folder URL: {url}")
    out = Path(dest_dir) / f"drive_folder_{fid}"
    # Reuse if we already have a video inside
    if out.exists():
        existing = find_video_in_dir(out)
        if existing is not None:
            print(f"  Drive folder cache hit: {out} ({existing.name})")
            return out
    out.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading Google Drive folder {fid} → {out}")
    print("  (folder must be shared as 'Anyone with the link' or accessible to this VM)")
    # Do not pass --remaining-ok: many installed gdown versions reject it.
    _run_gdown(["--folder", url, "-O", str(out)])
    if find_video_in_dir(out) is None:
        raise RuntimeError(
            f"Drive folder downloaded but no video (*{', *'.join(sorted(_VIDEO_EXTS))}) "
            f"found under {out}. Check sharing permissions / folder contents."
        )
    return out


def download_drive_file(url: str, dest: Path) -> Path:
    fid = drive_file_id(url)
    if not fid:
        raise ValueError(f"Not a Google Drive file URL: {url}")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  Drive file cache hit: {dest}")
        return dest
    print(f"  Downloading Google Drive file {fid} → {dest}")
    # Prefer uc?id= (works on older gdown). Avoid --fuzzy — many builds reject it.
    direct = f"https://drive.google.com/uc?id={fid}"
    try:
        _run_gdown([direct, "-O", str(dest)])
    except Exception:
        _run_gdown([url, "-O", str(dest)])
    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(
            f"Drive file download failed/empty: {dest}. "
            "Share the file as 'Anyone with the link' or use a direct gs:// URL."
        )
    print(f"  OK {_fmt_size(dest.stat().st_size)} → {dest}")
    return dest


def _download_gs(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  GCS cache hit: {dest}")
        return dest

    gsutil = shutil.which("gsutil")
    gcloud = shutil.which("gcloud")
    print(f"  Downloading {url} → {dest}")
    if gsutil:
        subprocess.run([gsutil, "-m", "cp", url, str(dest)], check=True)
    elif gcloud:
        subprocess.run(["gcloud", "storage", "cp", url, str(dest)], check=True)
    else:
        raise RuntimeError(
            "Neither gsutil nor gcloud found. Install Google Cloud SDK "
            "or pass a signed https:// / Drive file URL instead of gs://"
        )
    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(f"GCS download produced empty file: {dest}")
    print(f"  OK {_fmt_size(dest.stat().st_size)} → {dest}")
    return dest


def _download_http(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  HTTP cache hit: {dest}")
        return dest

    print(f"  Downloading {url} → {dest}")
    curl = shutil.which("curl")
    if curl:
        subprocess.run(
            [curl, "-L", "--fail", "--retry", "3", "-o", str(dest), url],
            check=True,
        )
    else:
        urllib.request.urlretrieve(url, dest)  # noqa: S310
    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(f"HTTP download produced empty file: {dest}")
    # Detect HTML error pages (common when curling a Drive folder URL)
    head = dest.read_bytes()[:200].lower()
    if b"<html" in head or b"<!doctype" in head:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"Download looks like an HTML page, not media: {url}\n"
            "For Google Drive use a *file* link or a shared *folder* link "
            "(handled via gdown), not a raw curl of the folder page."
        )
    print(f"  OK {_fmt_size(dest.stat().st_size)} → {dest}")
    return dest


def fetch_media(url_or_path: str | Path, dest_dir: Path, default_name: str) -> Path:
    """Return a local path. Downloads if given gs://, Drive, or http(s)://."""
    value = str(url_or_path).strip()
    if not looks_like_url(value):
        p = Path(value).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if is_drive_folder_url(value):
        folder = download_drive_folder(value, dest_dir)
        # Prefer video; if default suggests srt, prefer srt
        if default_name.lower().endswith(".srt"):
            srt = find_srt_in_dir(folder)
            if srt is None:
                raise RuntimeError(f"No .srt found in Drive folder {folder}")
            print(f"  Using SRT: {srt}")
            return srt
        vid = find_video_in_dir(folder)
        if vid is None:
            raise RuntimeError(f"No video found in Drive folder {folder}")
        print(f"  Using video: {vid}")
        return vid

    if is_drive_file_url(value):
        name = _filename_from_url(value, default_name)
        return download_drive_file(value, dest_dir / name)

    name = _filename_from_url(value, default_name)
    dest = dest_dir / name

    if value.startswith("gs://"):
        return _download_gs(value, dest)
    if value.startswith(("https://", "http://")):
        return _download_http(value, dest)
    raise ValueError(f"Unsupported URL scheme: {value}")


def resolve_video_and_srt(
    video: str,
    srt: str | None,
    dest_dir: Path,
) -> tuple[Path, Path | None]:
    """Resolve video (+ optional SRT). Same Drive folder URL is downloaded once."""
    dest_dir = Path(dest_dir)
    v = str(video).strip()
    s = str(srt).strip() if srt else None

    if (
        s
        and is_drive_folder_url(v)
        and is_drive_folder_url(s)
        and drive_folder_id(v) == drive_folder_id(s)
    ):
        print("==> Google Drive folder (video + SRT from same folder)")
        folder = download_drive_folder(v, dest_dir)
        vid = find_video_in_dir(folder)
        if vid is None:
            raise RuntimeError(f"No video in {folder}")
        srt_path = find_srt_in_dir(folder, prefer_stem=vid.stem)
        print(f"  video: {vid}")
        print(f"  srt:   {srt_path or '(none found)'}")
        return vid, srt_path

    print("==> Resolving video")
    vid = fetch_media(v, dest_dir, default_name="input.mp4")
    srt_path = None
    if s:
        print("==> Resolving SRT")
        srt_path = fetch_media(s, dest_dir, default_name=f"{vid.stem}.srt")
    return vid, srt_path
