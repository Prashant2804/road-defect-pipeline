"""Upload RF-DETR inference outputs to a Google Drive folder."""
from __future__ import annotations

import argparse
import mimetypes
import re
import sys
from pathlib import Path

_FOLDER_RE = re.compile(
    r"(?:drive\.google\.com/(?:drive/)?folders/|id=)([A-Za-z0-9_-]{20,})",
    re.I,
)

DEFAULT_FILES = (
    "annotated.mp4",
    "defects.csv",
    "defects.json",
    "map_trail.html",
    "summary.json",
)


def folder_id_from_url(url_or_id: str) -> str:
    s = url_or_id.strip()
    m = _FOLDER_RE.search(s)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", s):
        return s
    raise SystemExit(
        f"Could not parse Google Drive folder id from: {url_or_id}\n"
        "Pass a folders/ URL or the raw folder id."
    )


def collect_files(run_dir: Path, names: tuple[str, ...] | None = None) -> list[Path]:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"Run dir not found: {run_dir}")
    names = names or DEFAULT_FILES
    found = []
    missing = []
    for name in names:
        p = run_dir / name
        if p.is_file():
            found.append(p)
        else:
            missing.append(name)
    if missing:
        print("WARNING: missing (skipped):", ", ".join(missing))
    if not found:
        raise SystemExit(f"No uploadable files in {run_dir}")
    return found


def _drive_service(service_account: Path | None = None):
    """Build Drive API client.

    Prefer a service-account JSON (no browser OAuth — Google blocks full Drive
    scope on the default gcloud client). Fall back to ADC only if requested.
    """
    try:
        from google.oauth2 import service_account as sa_mod
        from googleapiclient.discovery import build
    except ImportError as e:
        raise SystemExit(
            "Missing Drive upload deps. Install with:\n"
            "  .venv/bin/pip install google-api-python-client google-auth google-auth-httplib2\n"
            f"Import error: {e}"
        ) from e

    scopes = ["https://www.googleapis.com/auth/drive"]

    if service_account is not None:
        sa_path = Path(service_account)
        if not sa_path.is_file():
            raise SystemExit(f"Service account JSON not found: {sa_path}")
        creds = sa_mod.Credentials.from_service_account_file(str(sa_path), scopes=scopes)
        print(f"Using service account: {creds.service_account_email}")
        return build("drive", "v3", credentials=creds, cache_discovery=False)

    # ADC / gcloud browser login — often blocked for drive scope ("This app is blocked")
    raise SystemExit(
        "Browser OAuth for Drive is blocked by Google on the default Cloud SDK client.\n\n"
        "Use ONE of these instead:\n\n"
        "A) Upload to GCS (recommended, already works on this VM):\n"
        "   ./scripts/upload_infer_to_gcs.sh \\\n"
        "     --run-dir 'runs/rfdetr_infer/ROAD-1 Gopro' \\\n"
        "     --gcs gs://YOUR_BUCKET/rfdetr_infer/ROAD-1-Gopro\n\n"
        "B) Drive via service account (no browser):\n"
        "   1. GCP Console → IAM → Service Accounts → Create key (JSON)\n"
        "   2. Share the Drive folder with the SA email as Editor\n"
        "   3. ./scripts/upload_infer_results.sh \\\n"
        "        --run-dir '...' --folder 'https://drive.google.com/...' \\\n"
        "        --service-account /path/to/sa.json\n"
    )


def _auth_help() -> str:
    return (
        "Cannot access that Drive folder with the current credentials.\n"
        "If using a service account: share the folder with the SA email as Editor.\n"
        "Or upload to GCS instead: ./scripts/upload_infer_to_gcs.sh --help"
    )


def _find_existing(service, folder_id: str, name: str) -> str | None:
    # Escape single quotes in name for Drive query
    safe = name.replace("\\", "\\\\").replace("'", "\\'")
    q = (
        f"name = '{safe}' and '{folder_id}' in parents "
        f"and trashed = false"
    )
    resp = (
        service.files()
        .list(
            q=q,
            spaces="drive",
            fields="files(id, name)",
            pageSize=5,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        )
        .execute()
    )
    files = resp.get("files") or []
    return files[0]["id"] if files else None


def upload_file(service, local: Path, folder_id: str, overwrite: bool = True) -> str:
    from googleapiclient.http import MediaFileUpload

    mime, _ = mimetypes.guess_type(str(local))
    mime = mime or "application/octet-stream"
    media = MediaFileUpload(str(local), mimetype=mime, resumable=True)
    existing = _find_existing(service, folder_id, local.name) if overwrite else None

    size_mb = local.stat().st_size / (1024 * 1024)
    if existing:
        print(f"  updating {local.name} ({size_mb:.1f} MB) ...")
        service.files().update(
            fileId=existing, media_body=media, supportsAllDrives=True
        ).execute()
        return existing

    print(f"  uploading {local.name} ({size_mb:.1f} MB) ...")
    meta = {"name": local.name, "parents": [folder_id]}
    created = (
        service.files()
        .create(
            body=meta,
            media_body=media,
            fields="id, name, webViewLink",
            supportsAllDrives=True,
        )
        .execute()
    )
    return created["id"]


def upload_run(
    run_dir: Path,
    folder_url: str,
    *,
    overwrite: bool = True,
    extra: list[str] | None = None,
    service_account: Path | None = None,
) -> None:
    folder_id = folder_id_from_url(folder_url)
    names = list(DEFAULT_FILES)
    if extra:
        names.extend(extra)
    files = collect_files(run_dir, tuple(dict.fromkeys(names)))

    print(f"Destination folder id: {folder_id}")
    print(f"Uploading {len(files)} file(s) from {run_dir}")
    service = _drive_service(service_account)

    try:
        meta = (
            service.files()
            .get(
                fileId=folder_id,
                fields="id, name, mimeType",
                supportsAllDrives=True,
            )
            .execute()
        )
        print(f"Folder: {meta.get('name')} ({meta.get('mimeType')})")
    except Exception as e:
        raise SystemExit(
            f"Cannot access Drive folder {folder_id}: {e}\n\n{_auth_help()}"
        ) from e

    for path in files:
        fid = upload_file(service, path, folder_id, overwrite=overwrite)
        print(f"    ok id={fid}")

    print("\nDone.")
    print(f"Open: https://drive.google.com/drive/folders/{folder_id}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Upload near-field inference results to a Google Drive folder."
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Local results dir, e.g. runs/rfdetr_infer/ROAD-1\\ Gopro",
    )
    p.add_argument(
        "--folder",
        required=True,
        help="Google Drive folder URL or folder id",
    )
    p.add_argument(
        "--service-account",
        type=Path,
        default=None,
        help="Path to GCP service-account JSON (required; browser OAuth is blocked)",
    )
    p.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Always create new files instead of updating same names",
    )
    p.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help="Extra filenames inside run-dir to upload",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    upload_run(
        args.run_dir,
        args.folder,
        overwrite=not args.no_overwrite,
        extra=list(args.extra or []),
        service_account=args.service_account,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
