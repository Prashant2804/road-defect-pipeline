"""H.264 video writer via ffmpeg (avoids huge/poor OpenCV mp4v)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _even(n: int) -> int:
    """yuv420p requires even dimensions."""
    n = int(n)
    return n if n % 2 == 0 else n - 1


class FfmpegH264Writer:
    """Pipe BGR frames to ffmpeg libx264. Falls back to OpenCV mp4v if needed."""

    def __init__(
        self,
        path: str | Path,
        width: int,
        height: int,
        fps: float,
        *,
        crf: int = 23,
        preset: str = "medium",
    ):
        self.path = Path(path)
        self.width = _even(width)
        self.height = _even(height)
        if self.width < 2 or self.height < 2:
            raise ValueError(f"Invalid video size {width}x{height}")
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.crf = int(crf)
        self.preset = preset
        self._proc: subprocess.Popen | None = None
        self._cv = None
        self._closed = False
        self.frames_written = 0
        self.backend = "ffmpeg"

        if shutil.which("ffmpeg") is None:
            self._open_cv()
            return

        # -framerate before -i sets input rate; do NOT use stderr=PIPE (deadlocks
        # after enough log bytes and truncates long encodes).
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-framerate",
            f"{self.fps:.6f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            self.preset,
            "-crf",
            str(self.crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.path),
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                bufsize=10**7,
            )
            print(
                f"Writing H.264 annotated video "
                f"({self.width}x{self.height} @ {self.fps:.2f}fps, "
                f"crf={self.crf}, preset={self.preset}) → {self.path}"
            )
        except OSError as e:
            print(f"WARNING: ffmpeg writer failed ({e}); falling back to OpenCV mp4v")
            self._open_cv()

    def _open_cv(self) -> None:
        import cv2

        self.backend = "opencv_mp4v"
        self._cv = cv2.VideoWriter(
            str(self.path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.width, self.height),
        )
        if not self._cv.isOpened():
            raise RuntimeError(f"Cannot open video writer for {self.path}")
        print(f"WARNING: using OpenCV mp4v (lower quality) → {self.path}")

    def write(self, frame) -> None:
        if self._closed:
            raise RuntimeError("write() after close()")
        import cv2
        import numpy as np

        h, w = frame.shape[:2]
        if (w, h) != (self.width, self.height):
            frame = cv2.resize(
                frame, (self.width, self.height), interpolation=cv2.INTER_AREA
            )
        if self._cv is not None:
            self._cv.write(frame)
        else:
            buf = frame if frame.flags["C_CONTIGUOUS"] else np.ascontiguousarray(frame)
            assert self._proc is not None and self._proc.stdin is not None
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"ffmpeg exited early (code={self._proc.returncode}) after "
                    f"{self.frames_written} frames — annotated video would be truncated"
                )
            try:
                self._proc.stdin.write(buf.tobytes())
            except (BrokenPipeError, OSError) as e:
                raise RuntimeError(
                    f"ffmpeg died after {self.frames_written} frames: {e}"
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
            raise RuntimeError(
                f"ffmpeg exit {rc} after {self.frames_written} frames "
                f"(expected a full-length annotated.mp4)"
            )
        print(f"ffmpeg closed OK — wrote {self.frames_written} frames → {self.path}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False
