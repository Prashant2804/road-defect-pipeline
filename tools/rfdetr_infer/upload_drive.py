"""Upload RF-DETR inference outputs to a Google Drive folder.

Uses YOUR Desktop OAuth client (Testing mode) — not the blocked default gcloud
Cloud SDK client. One-time browser login saves token.json for later runs.
"""
from __future__ import annotations

import argparse
import mimetypes
import re
from pathlib import Path

_FOLDER_RE = re.compile(
    r"(?:drive\.google\.com/(?:drive/)?folders/|id=)([A-Za-z0-9_-]{20,})",
    re.I,
)

SCOPES = ["https://www.googleapis.com/auth/drive"]

DEFAULT_FILES = (
    "annotated.mp4",
    "defects.csv",
    "defects.json",
    "map_trail.html",
    "summary.json",
)

SETUP_HELP = """
One-time Google Cloud setup (avoids "This app is blocked" from gcloud):

  1. Console → project 614564067545 (or yours)
  2. APIs & Services → enable Google Drive API
  3. OAuth consent screen → External → Publishing: Testing
     → add YOUR Gmail under Test users
  4. Credentials → Create OAuth client ID → Desktop app
     → Download JSON → save as ~/secrets/drive_oauth_client.json
  5. Share the destination Drive folder with that same Gmail (Editor)

Then:
  ./scripts/upload_infer_results.sh \\
    --run-dir 'runs/rfdetr_infer/ROAD-1 Gopro' \\
    --folder  'https://drive.google.com/drive/folders/FOLDER_ID' \\
    --client-secret ~/secrets/drive_oauth_client.json
""".strip()


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


def _default_token_path(client_secret: Path) -> Path:
    cfg = Path.home() / ".config" / "rfdetr_drive"
    cfg.mkdir(parents=True, exist_ok=True)
    # Bind token to client file so switching clients does not reuse a bad token
    return cfg / f"token_{client_secret.stem}.json"


def _creds_from_client_secret(
    client_secret: Path,
    token_path: Path | None,
    auth_port: int = 8090,
):
    """Installed-app OAuth (Desktop). First run needs a browser once."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as e:
        raise SystemExit(
            "Missing deps. Install with:\n"
            "  .venv/bin/pip install google-api-python-client google-auth "
            "google-auth-httplib2 google-auth-oauthlib\n"
            f"Import error: {e}"
        ) from e

    client_secret = Path(client_secret).expanduser()
    if not client_secret.is_file():
        raise SystemExit(
            f"OAuth client secret not found: {client_secret}\n\n{SETUP_HELP}"
        )

    token_path = (
        Path(token_path).expanduser()
        if token_path
        else _default_token_path(client_secret)
    )
    creds = None
    if token_path.is_file():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except Exception as e:
            print(f"Token refresh failed ({e}); re-authenticating...")
            creds = None

    if not creds or not creds.valid:
        print("Google sign-in required (use the Gmail listed as OAuth Test user).")
        print(f"Client: {client_secret}")
        print()
        print("If you are on SSH with no browser on the VM, from your LAPTOP run:")
        print(f"  ssh -L {auth_port}:localhost:{auth_port} ubuntu@YOUR_VM_IP")
        print("Keep that tunnel open, then open the URL printed below in your laptop browser.")
        print()
        flow = InstalledAppFlow.from_client_secrets_file(str(client_secret), SCOPES)
        creds = flow.run_local_server(
            port=auth_port,
            open_browser=False,
            authorization_prompt_message=(
                "Open this URL in your laptop browser (with the SSH -L tunnel up):\n{url}\n"
            ),
            success_message=(
                "Auth OK — you can close this tab and return to the SSH session."
            ),
        )
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json(), encoding="utf-8")
        print(f"Saved token: {token_path}")

    return creds


def _creds_from_service_account(service_account: Path):
    from google.oauth2 import service_account as sa_mod

    sa_path = Path(service_account).expanduser()
    if not sa_path.is_file():
        raise SystemExit(f"Service account JSON not found: {sa_path}")
    creds = sa_mod.Credentials.from_service_account_file(str(sa_path), scopes=SCOPES)
    print(f"Using service account: {creds.service_account_email}")
    return creds


def _drive_service(
    *,
    client_secret: Path | None = None,
    token_path: Path | None = None,
    service_account: Path | None = None,
    auth_port: int = 8090,
):
    try:
        from googleapiclient.discovery import build
    except ImportError as e:
        raise SystemExit(
            "Missing google-api-python-client. "
            "pip install google-api-python-client google-auth-oauthlib\n"
            f"{e}"
        ) from e

    if service_account is not None:
        creds = _creds_from_service_account(service_account)
    elif client_secret is not None:
        creds = _creds_from_client_secret(
            client_secret, token_path, auth_port=auth_port
        )
    else:
        raise SystemExit(
            "Pass --client-secret ~/secrets/drive_oauth_client.json\n\n" + SETUP_HELP
        )

    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _auth_help() -> str:
    return (
        "Cannot access that Drive folder.\n"
        "- Sign in with the Gmail that owns/edits the folder\n"
        "- That Gmail must be listed under OAuth consent → Test users\n"
        "- Folder must be shared with that account as Editor\n\n"
        + SETUP_HELP
    )


def _find_existing(service, folder_id: str, name: str) -> str | None:
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
    client_secret: Path | None = None,
    token_path: Path | None = None,
    service_account: Path | None = None,
    auth_port: int = 8090,
) -> None:
    folder_id = folder_id_from_url(folder_url)
    names = list(DEFAULT_FILES)
    if extra:
        names.extend(extra)
    files = collect_files(run_dir, tuple(dict.fromkeys(names)))

    print(f"Destination folder id: {folder_id}")
    print(f"Uploading {len(files)} file(s) from {run_dir}")
    service = _drive_service(
        client_secret=client_secret,
        token_path=token_path,
        service_account=service_account,
        auth_port=auth_port,
    )

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
        description="Upload near-field inference results to Google Drive "
        "(Desktop OAuth client — not gcloud ADC).",
        epilog=SETUP_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        "--client-secret",
        type=Path,
        default=None,
        help="Downloaded Desktop OAuth client JSON (required unless --service-account)",
    )
    p.add_argument(
        "--token",
        type=Path,
        default=None,
        help="Where to store/reuse OAuth token (default: ~/.config/rfdetr_drive/token_*.json)",
    )
    p.add_argument(
        "--auth-port",
        type=int,
        default=8090,
        help="Local port for OAuth redirect (SSH: ssh -L 8090:localhost:8090 ...)",
    )
    p.add_argument(
        "--service-account",
        type=Path,
        default=None,
        help="Optional: SA JSON instead of Desktop OAuth (share folder with SA email)",
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
    if args.client_secret is None and args.service_account is None:
        raise SystemExit(
            "Missing --client-secret (or --service-account).\n\n" + SETUP_HELP
        )
    upload_run(
        args.run_dir,
        args.folder,
        overwrite=not args.no_overwrite,
        extra=list(args.extra or []),
        client_secret=args.client_secret,
        token_path=args.token,
        service_account=args.service_account,
        auth_port=args.auth_port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
