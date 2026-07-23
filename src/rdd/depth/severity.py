"""Severity scoring from mask area (+ optional depth).

severity_score in [0,1] = w_area * norm(area) + w_depth * norm(depth).
Without depth (depth disabled) it degrades to area-only (w_depth folded out).
Binned into low/medium/high for the report.
"""
from __future__ import annotations

from ..utils.logging import get_logger

log = get_logger("rdd.depth.severity")


def _normalize(values: list[float]) -> dict[int, float]:
    if not values:
        return {}
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    return {i: (v - lo) / span for i, v in enumerate(values)}


def score_tracks(tracks, cfg, depths: dict[int, float] | None = None) -> dict[int, dict]:
    """Return {track_id: {"score": float, "level": str, "area": float, "depth": float|None}}."""
    sc = cfg.get_path("depth.severity", {}) or {}
    w_area = float(sc.get("w_area", 0.5))
    w_depth = float(sc.get("w_depth", 0.5))
    bins = sc.get("bins", {"low": 0.33, "medium": 0.66})
    depth_enabled = bool(cfg.get_path("depth.enabled", False)) and depths is not None

    areas = [t.max_mask_area for t in tracks]
    area_norm = _normalize(areas)
    depth_norm = {}
    if depth_enabled:
        depth_norm = _normalize([depths.get(t.track_id, 0.0) for t in tracks])

    out: dict[int, dict] = {}
    for i, t in enumerate(tracks):
        if depth_enabled:
            score = w_area * area_norm.get(i, 0.0) + w_depth * depth_norm.get(i, 0.0)
            score /= (w_area + w_depth) or 1.0
        else:
            score = area_norm.get(i, 0.0)  # area-only fallback
        level = "high" if score >= bins["medium"] else "medium" if score >= bins["low"] else "low"
        out[t.track_id] = {
            "score": round(score, 4),
            "level": level,
            "area": t.max_mask_area,
            "depth": depths.get(t.track_id) if depth_enabled else None,
        }
    return out
