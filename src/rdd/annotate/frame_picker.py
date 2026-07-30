"""Active-learning frame picker: surface the most useful frames to label first.

Strategies:
  * diversity   : pick frames that are visually most different (greedy farthest-
                  point on a cheap color/gradient feature) — maximises coverage.
  * uncertainty : run the current model and pick frames where it's least
                  confident / most ambiguous — needs a loaded model.
  * random      : baseline.

Returns an ordered list of frame paths (best first).
"""
from __future__ import annotations

from pathlib import Path

from ..utils.logging import get_logger

log = get_logger("rdd.annotate.picker")


def _feature(img) -> "list[float]":
    import cv2
    import numpy as np

    small = cv2.resize(img, (64, 64))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def pick_frames(frame_paths: list[Path], cfg, model=None) -> list[Path]:
    import cv2
    import numpy as np

    fc = cfg.get_path("annotate.frame_picker", {}) or {}
    strategy = fc.get("strategy", "diversity")
    n = int(fc.get("n_frames", 200))
    frame_paths = [Path(p) for p in frame_paths]

    if strategy == "random":
        import random

        random.shuffle(frame_paths)
        return frame_paths[:n]

    if strategy == "uncertainty":
        if model is None:
            log.warning("uncertainty strategy needs a model; falling back to diversity")
        else:
            scored = []
            for p in frame_paths:
                res = model.predict(str(p), verbose=False)[0]
                confs = res.boxes.conf.cpu().tolist() if res.boxes is not None else []
                # ambiguity: detections near 0.5 conf, or no detections at all
                amb = sum(1 - abs(c - 0.5) * 2 for c in confs) if confs else 1.0
                scored.append((amb, p))
            scored.sort(reverse=True)
            return [p for _, p in scored[:n]]

    # diversity: greedy farthest-point sampling on color-histogram features
    readable, features = [], []
    for p in frame_paths:
        img = cv2.imread(str(p))
        if img is None:
            log.warning("Skipping unreadable frame %s", p)
            continue
        readable.append(p)
        features.append(_feature(img))
    if not features:
        log.warning("No readable frames among %d paths", len(frame_paths))
        return []
    frame_paths = readable
    feats = np.array(features)
    chosen_idx = [0]
    dists = np.linalg.norm(feats - feats[0], axis=1)
    while len(chosen_idx) < min(n, len(feats)):
        nxt = int(np.argmax(dists))
        chosen_idx.append(nxt)
        dists = np.minimum(dists, np.linalg.norm(feats - feats[nxt], axis=1))
    return [frame_paths[i] for i in chosen_idx]
