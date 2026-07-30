"""Thin FFmpeg/ffprobe wrapper with clear errors when the binary is missing.

The whole preprocess/ stage depends on ffmpeg's `v360` filter. We detect its
absence up front and raise an actionable message instead of a cryptic
FileNotFoundError deep in a subprocess call.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .logging import get_logger

log = get_logger("rdd.ffmpeg")

_INSTALL_HINT = (
    "FFmpeg not found on PATH. Install it and ensure the `v360` filter is built in:\n"
    "  Windows : winget install Gyan.FFmpeg   (or download from https://www.gyan.dev/ffmpeg/builds/)\n"
    "  macOS   : brew install ffmpeg\n"
    "  Linux   : sudo apt install ffmpeg\n"
    "Verify with: ffmpeg -hide_banner -filters | findstr v360   (Windows)  /  grep v360  (unix)"
)


class FFmpegNotFound(RuntimeError):
    pass


def ensure_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise FFmpegNotFound(_INSTALL_HINT)
    return exe


def has_v360() -> bool:
    if not shutil.which("ffmpeg"):
        return False
    try:
        res = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            capture_output=True, text=True, timeout=15,
        )
        return "v360" in res.stdout
    except Exception:
        return False


def run(args: list[str], desc: str = "ffmpeg") -> None:
    """Run an ffmpeg command (args already exclude the leading 'ffmpeg')."""
    ensure_ffmpeg()
    cmd = ["ffmpeg", "-hide_banner", "-y", *args]
    log.info("%s: %s", desc, " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        log.error("ffmpeg failed (%d):\n%s", res.returncode, res.stderr[-2000:])
        raise RuntimeError(f"{desc} failed; see log above")


def encode_args(crf: int = 18, preset: str = "medium", lossless: bool = False) -> list[str]:
    """Video encode flags shared by every stage that writes an mp4.

    `lossless` switches to x264 lossless (crf 0). Files get large fast, but the
    intermediate rectified video is what the detector actually sees, so a lossy
    re-encode there costs real detection accuracy on faint cracks.
    """
    if lossless:
        return ["-c:v", "libx264", "-preset", preset, "-crf", "0",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    return ["-c:v", "libx264", "-preset", preset, "-crf", str(int(crf)),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart"]


class VideoWriter:
    """Write BGR frames to an H.264 mp4 by piping rawvideo into ffmpeg.

    OpenCV's `VideoWriter` with the `mp4v` fourcc encodes MPEG-4 Part 2 at a
    fixed, fairly low quality — visibly worse than the source and a poor way to
    ship an inspection deliverable. Piping to ffmpeg gets us x264 with a
    controllable CRF. Falls back to OpenCV if ffmpeg is unavailable.
    """

    def __init__(self, path: str | Path, width: int, height: int, fps: float,
                 crf: int = 18, preset: str = "medium"):
        self.path = Path(path)
        self.width, self.height = int(width), int(height)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self._proc = None
        self._cv = None
        self._closed = False
        self.frames_written = 0

        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"VideoWriter needs positive dimensions, got {width}x{height}")

        try:
            ensure_ffmpeg()
        except FFmpegNotFound:
            log.warning("ffmpeg missing — annotated video falls back to OpenCV mp4v "
                        "(lower quality)")
            self._open_cv_fallback()
            return

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", "bgr24", "-r", f"{self.fps}", "-i", "-",
            "-an", *encode_args(crf=crf, preset=preset), str(self.path),
        ]
        try:
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError as e:
            log.warning("Could not start ffmpeg writer (%s) — using OpenCV fallback", e)
            self._open_cv_fallback()

    def _open_cv_fallback(self) -> None:
        import cv2

        self._cv = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps, (self.width, self.height),
        )
        if not self._cv.isOpened():
            raise RuntimeError(f"Cannot open any video writer for {self.path}")

    def write(self, frame) -> None:
        if self._closed:
            raise RuntimeError("write() after close()")
        h, w = frame.shape[:2]
        if (w, h) != (self.width, self.height):
            import cv2

            frame = cv2.resize(frame, (self.width, self.height),
                               interpolation=cv2.INTER_AREA)
        if self._cv is not None:
            self._cv.write(frame)
        else:
            import numpy as np

            buf = frame if frame.flags["C_CONTIGUOUS"] else np.ascontiguousarray(frame)
            try:
                self._proc.stdin.write(buf.tobytes())      # type: ignore[union-attr]
            except (BrokenPipeError, OSError) as e:
                stderr = b""
                if self._proc is not None and self._proc.stderr is not None:
                    stderr = self._proc.stderr.read() or b""
                raise RuntimeError(
                    f"ffmpeg writer died after {self.frames_written} frames: {e}\n"
                    f"{stderr.decode(errors='replace')[-1000:]}"
                ) from e
        self.frames_written += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._cv is not None:
            self._cv.release()
            return
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except OSError:
            pass
        rc = self._proc.wait()
        if rc != 0:
            err = (self._proc.stderr.read() if self._proc.stderr else b"") or b""
            log.error("ffmpeg writer exited %d:\n%s", rc,
                      err.decode(errors="replace")[-2000:])
            raise RuntimeError(f"ffmpeg failed writing {self.path}")

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        # On an exception, still tear the process down but don't mask the original error.
        if exc[0] is not None:
            self._closed = True
            if self._cv is not None:
                self._cv.release()
            elif self._proc is not None:
                try:
                    if self._proc.stdin:
                        self._proc.stdin.close()
                except OSError:
                    pass
                self._proc.terminate()
            return
        self.close()


def probe(path: str | Path) -> dict:
    """ffprobe -> parsed json (streams + format). Empty dict if ffprobe missing."""
    if not shutil.which("ffprobe"):
        log.warning("ffprobe not found; skipping metadata probe of %s", path)
        return {}
    res = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        return {}
    try:
        return json.loads(res.stdout)
    except json.JSONDecodeError:
        return {}
