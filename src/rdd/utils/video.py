"""Frame sampling helpers.

The distinction that matters here is between *sampling* and *addressing*.

**Sampling** — "give me 60 frames spread across this clip" for quality statistics or a
mask preview. Any frame near the requested position will do.

**Addressing** — "give me exactly frame 4,182, the one that produced this detection", as
the report crops need. Those call sites read sequentially and must keep doing so,
because `CAP_PROP_POS_FRAMES` on long-GOP H.264 lands on the nearest keyframe and can
silently hand back a different frame than the one asked for.

The sampling strategy below was chosen by measurement, not intuition. On a 200 s 1080p
clip, extracting 60 frames:

    ffmpeg, keyframes only      8.2 s     <- chosen
    OpenCV grab()+retrieve()   19.9 s     <- fallback
    OpenCV read() every frame  27.9 s
    OpenCV seek per sample     62.0 s
    ffmpeg -ss per sample     143.5 s

Both "obvious" optimisations are traps. Seeking per sample is *slower* than decoding
everything, because each seek rewinds to a keyframe and re-decodes forward. Spawning an
ffmpeg process per sample is slower still. What actually wins is asking the decoder to
skip non-keyframes entirely, so the expensive inter-frame reconstruction never happens.

Keyframes are encoded slightly differently from P/B frames, so there was a fair worry
that quality statistics measured on them would be biased. Measured on the same clip:
median sharpness 1279 via keyframes against 1293 via full decode, a 1.1% difference.
Not enough to matter for threshold-setting.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .logging import get_logger

log = get_logger("rdd.utils.video")


def probe_video(path: str | Path) -> tuple[int, int, int, float]:
    """(width, height, frame_count, fps). Zeros where the container will not say."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()
    return w, h, max(0, n), fps


def keyframe_times(path) -> list[float]:
    """Presentation timestamps of every keyframe, via ffprobe. Metadata only, no decode."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey",
             "-show_entries", "frame=pts_time", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=300)
    except Exception as e:
        log.debug("ffprobe keyframe listing failed (%s)", e)
        return []
    out = []
    for tok in (r.stdout or "").replace(",", " ").split():
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return sorted(out)


def _keyframe_samples(path, n_target: int, w: int, h: int, times: list[float],
                      fps: float):
    """Decode only keyframes, evenly subsampled. Yields (index, frame).

    `-skip_frame nokey` makes the decoder discard non-keyframes before reconstruction,
    which is where nearly all the time goes. Under that flag the filter's frame counter
    `n` counts keyframes, so selecting every STEP-th of them spreads the samples across
    the whole clip. The exact timestamps come from ffprobe, so the reported frame
    indices are real rather than inferred from an assumed spacing.
    """
    import numpy as np

    if not times or w <= 0 or h <= 0:
        return
    # Round rather than floor, and do NOT truncate to n_target afterwards. Flooring
    # then slicing [:n_target] silently keeps only the FIRST n_target keyframes
    # whenever the clip has between 1x and 2x that many — sampling the opening half of
    # a route and calling it representative. Rounding lands the count near the target
    # while still spanning the whole clip.
    step = max(1, round(len(times) / n_target))
    chosen = times[::step]
    if len(chosen) > 2 * n_target:          # pathological metadata guard
        chosen = chosen[:2 * n_target]
    if not chosen:
        return

    # The comma inside mod() must be escaped: unescaped, ffmpeg reads it as a filter
    # separator in the filtergraph. Raw string so Python leaves the backslash alone.
    select = (rf"select='not(mod(n\,{step}))'" if step > 1 else "select=1")

    def build(sync_flag: str) -> list[str]:
        # Built whole rather than spliced: inserting the sync flag by index once landed
        # it between -vf and its value, so ffmpeg took "-fps_mode" as the filtergraph
        # and silently produced nothing.
        return ["ffmpeg", "-v", "error", "-skip_frame", "nokey", "-i", str(path),
                "-vf", select, sync_flag, "passthrough",
                "-frames:v", str(len(chosen)),
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]

    proc = None
    for sync in ("-fps_mode", "-vsync"):     # -fps_mode is absent from older ffmpeg
        try:
            proc = subprocess.run(build(sync), capture_output=True, timeout=900)
        except Exception as e:
            log.debug("keyframe extraction failed (%s)", e)
            return
        if proc.returncode == 0 and proc.stdout:
            break
        log.debug("ffmpeg %s: %s", sync,
                  (proc.stderr or b"").decode(errors="replace")[-300:])
    if proc is None or proc.returncode != 0 or not proc.stdout:
        return

    size = w * h * 3
    count = min(len(proc.stdout) // size, len(chosen))
    for i in range(count):
        frame = np.frombuffer(proc.stdout[i * size:(i + 1) * size],
                              dtype=np.uint8).reshape(h, w, 3)
        yield int(round(chosen[i] * fps)) if fps > 0 else i, frame


def _opencv_samples(path, n_target: int, total: int):
    """Fallback: grab() demuxes without decoding; retrieve() decodes only what we keep."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    try:
        stride = max(1, total // n_target) if total > 0 else 1
        idx, produced = -1, 0
        while produced < n_target:
            if not cap.grab():
                break
            idx += 1
            if idx % stride:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                continue
            produced += 1
            yield idx, frame
    finally:
        cap.release()


def iter_sampled_frames(video_path: str | Path, n_target: int,
                        prefer_keyframes: bool = True):
    """Yield up to `n_target` (index, frame) pairs spread across the clip.

    Chooses between the two strategies rather than always preferring one, because
    keyframe density varies enormously: GoPro footage has a keyframe every second or
    two (hundreds available, so the fast path wins outright), while a long-GOP encode
    may hold only a dozen in several minutes — too few to characterise a clip.
    """
    path = Path(video_path)
    n_target = max(1, int(n_target))
    w, h, total, fps = probe_video(path)

    # Short clip: decoding it is already cheap and ffprobe would cost more than it saves.
    long_enough = total > n_target * 5
    if prefer_keyframes and long_enough and shutil.which("ffmpeg"):
        times = keyframe_times(path)
        if len(times) >= max(8, n_target // 2):
            produced = 0
            for idx, frame in _keyframe_samples(path, n_target, w, h, times, fps):
                produced += 1
                yield idx, frame
            if produced:
                return
        elif times:
            log.debug("only %d keyframes for %d requested samples - decoding instead",
                      len(times), n_target)

    yield from _opencv_samples(path, n_target, total)
