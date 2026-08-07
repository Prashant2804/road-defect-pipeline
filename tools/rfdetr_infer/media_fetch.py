"""Download dashcam media from GCS (gs://) or HTTPS URLs."""
from __future__ import annotations

import re
import shutil
import subprocess
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urlparse


def looks_like_url(value: str | Path) -> bool:
    s = str(value).strip()
    return s.startswith(("gs://", "https://", "http://"))


def _filename_from_url(url: str, default: str) -> str:
    if url.startswith("gs://"):
        name = url.rstrip("/").split("/")[-1]
        return unquote(name) or default
    path = unquote(urlparse(url).path)
    name = Path(path).name
    if name and "." in name:
        return name
    # GCS JSON API style sometimes embeds the object name in the query
    m = re.search(r"[?&](?:name|file)=([^&]+)", url)
    if m:
        return unquote(m.group(1).split("/")[-1]) or default
    return default


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
        subprocess.run(
            ["gcloud", "storage", "cp", url, str(dest)],
            check=True,
        )
    else:
        raise RuntimeError(
            "Neither gsutil nor gcloud found. Install Google Cloud SDK "
            "or pass a signed https:// URL instead of gs://"
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
    # Prefer curl for progress + redirects (signed GCS URLs)
    curl = shutil.which("curl")
    if curl:
        subprocess.run(
            [curl, "-L", "--fail", "--retry", "3", "-o", str(dest), url],
            check=True,
        )
    else:
        urllib.request.urlretrieve(url, dest)  # noqa: S310 — user-supplied URL
    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(f"HTTP download produced empty file: {dest}")
    print(f"  OK {_fmt_size(dest.stat().st_size)} → {dest}")
    return dest


def _fmt_size(n: int) -> str:
    x = float(n)
    for u in ("B", "KB", "MB", "GB"):
        if x < 1024:
            return f"{x:.1f} {u}"
        x /= 1024.0
    return f"{x:.1f} TB"


def fetch_media(url_or_path: str | Path, dest_dir: Path, default_name: str) -> Path:
    """Return a local path. Downloads if given gs:// or http(s)://."""
    value = str(url_or_path).strip()
    if not looks_like_url(value):
        p = Path(value).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = _filename_from_url(value, default_name)
    dest = dest_dir / name

    if value.startswith("gs://"):
        return _download_gs(value, dest)
    if value.startswith(("https://", "http://")):
        return _download_http(value, dest)
    raise ValueError(f"Unsupported URL scheme: {value}")
