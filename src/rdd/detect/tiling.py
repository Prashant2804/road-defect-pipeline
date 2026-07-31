"""Sliced inference: run the detector at native resolution on overlapping tiles.

Necessary for cracks, and the reason is arithmetic rather than modelling. A detector
at `imgsz: 960` letterboxes a 1920-wide frame by half, so a crack three pixels wide
becomes one and a half — below the network's stride, so its features never survive the
first downsampling layer. No amount of training fixes a feature that was destroyed
before the first convolution.

Tiling the *road region only* at native resolution restores those pixels at a cost
proportional to road area rather than frame area, which on a dashcam is a small
fraction of the image. Detections are merged back into frame coordinates with NMS
across tile seams, where the same crack is otherwise found twice.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..utils.logging import get_logger

log = get_logger("rdd.detect.tiling")


@dataclass
class Tile:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0


@dataclass
class TilingConfig:
    enabled: bool = False
    tile_px: int = 640
    overlap: float = 0.25
    max_tiles: int = 12
    iou_merge: float = 0.45
    min_road_frac: float = 0.10      # skip tiles that are mostly not road

    @classmethod
    def from_cfg(cls, cfg) -> "TilingConfig":
        tc = cfg.get_path("detect.tiling", {}) or {}
        return cls(
            enabled=bool(tc.get("enabled", False)),
            tile_px=int(tc.get("tile_px", 640)),
            overlap=float(tc.get("overlap", 0.25)),
            max_tiles=int(tc.get("max_tiles", 12)),
            iou_merge=float(tc.get("iou_merge", 0.45)),
            min_road_frac=float(tc.get("min_road_frac", 0.10)),
        )


def plan_tiles(region_mask, tc: TilingConfig) -> list[Tile]:
    """Overlapping tiles covering the bounding box of `region_mask`.

    Only the region of interest is tiled — sky and verge contribute no defects and
    tiling them would multiply cost for nothing.
    """
    import numpy as np

    if region_mask is None or not region_mask.any():
        return []

    ys, xs = np.nonzero(region_mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    H, W = region_mask.shape[:2]

    step = max(16, int(tc.tile_px * (1.0 - min(0.9, max(0.0, tc.overlap)))))
    tiles: list[Tile] = []
    for ty in range(y0, y1, step):
        for tx in range(x0, x1, step):
            bx1 = min(W, tx + tc.tile_px)
            by1 = min(H, ty + tc.tile_px)
            bx0 = max(0, bx1 - tc.tile_px)
            by0 = max(0, by1 - tc.tile_px)
            sub = region_mask[by0:by1, bx0:bx1]
            if sub.size == 0:
                continue
            if float(sub.sum()) / float(sub.size) < tc.min_road_frac:
                continue
            t = Tile(bx0, by0, bx1, by1)
            if t not in tiles:
                tiles.append(t)

    if len(tiles) > tc.max_tiles:
        # Keep the tiles with the most road in them: near-field tiles carry the
        # resolution that made tiling worthwhile in the first place.
        tiles.sort(key=lambda t: -float(region_mask[t.y0:t.y1, t.x0:t.x1].sum()))
        tiles = tiles[:tc.max_tiles]
    return tiles


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def merge_detections(dets: list[dict], iou_thresh: float = 0.45) -> list[dict]:
    """Greedy NMS over tile detections, per class.

    Needed because overlap is deliberate: a crack crossing a seam is detected in both
    tiles, and without merging it would be counted twice and then tracked as two
    separate defects.
    """
    kept: list[dict] = []
    for det in sorted(dets, key=lambda d: -float(d.get("conf", 0.0))):
        bbox = det.get("bbox")
        if bbox is None:
            kept.append(det)
            continue
        clash = False
        for k in kept:
            if k.get("cls_id") != det.get("cls_id"):
                continue
            if k.get("bbox") is None:
                continue
            if _iou(bbox, k["bbox"]) >= iou_thresh:
                clash = True
                break
        if not clash:
            kept.append(det)
    return kept


def run_tiled(model, frame, region_mask, cfg, tc: TilingConfig | None = None,
              **predict_kwargs) -> list[dict]:
    """Detect on tiles and return detections in frame coordinates.

    Returns raw dicts (bbox, mask, conf, cls_id) rather than tracked observations —
    tiling is a detection-time concern and tracking still happens once, on the merged
    result, so track identity stays global.
    """
    import numpy as np

    tc = tc or TilingConfig.from_cfg(cfg)
    tiles = plan_tiles(region_mask, tc)
    if not tiles:
        return []

    H, W = frame.shape[:2]
    out: list[dict] = []
    for t in tiles:
        crop = frame[t.y0:t.y1, t.x0:t.x1]
        try:
            res = model.predict(crop, verbose=False, **predict_kwargs)
        except Exception as e:
            log.warning("Tiled predict failed on tile %s (%s) — skipping",
                        (t.x0, t.y0), e)
            continue
        r = res[0] if res else None
        boxes = getattr(r, "boxes", None) if r is not None else None
        if boxes is None or not len(boxes):
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().tolist()
        clss = boxes.cls.int().cpu().tolist()
        masks = getattr(r, "masks", None)
        mdata = masks.data.cpu().numpy() if masks is not None and masks.data is not None else None

        for i in range(len(confs)):
            bx = xyxy[i]
            full_mask = None
            if mdata is not None and i < len(mdata):
                import cv2

                m = mdata[i]
                if m.shape != (t.h, t.w):
                    m = cv2.resize(m.astype("float32"), (t.w, t.h),
                                   interpolation=cv2.INTER_NEAREST)
                full_mask = np.zeros((H, W), dtype=bool)
                full_mask[t.y0:t.y1, t.x0:t.x1] = m.astype(bool)
            out.append({
                "bbox": (float(bx[0]) + t.x0, float(bx[1]) + t.y0,
                         float(bx[2]) + t.x0, float(bx[3]) + t.y0),
                "conf": float(confs[i]),
                "cls_id": int(clss[i]),
                "mask": full_mask,
            })

    merged = merge_detections(out, tc.iou_merge)
    if out:
        log.debug("tiling: %d tiles, %d raw -> %d merged", len(tiles), len(out),
                  len(merged))
    return merged
