"""Per-class assessment zones: where this camera can actually see each defect type.

The problem being solved is a specific and expensive kind of dishonesty. A hairline
crack is a few millimetres wide. At 25 m ahead, one pixel of a 1080p dashcam covers
several centimetres of road along the direction of travel, so that crack is not
faint in the image — it is *absent*, below the sampling limit. A detector that finds
nothing there is not observing intact pavement; it is observing nothing at all.
Recorded as "no defect", that becomes a false negative dressed up as a clean road,
and it silently inflates any condition score computed over the whole frame.

So each class gets a distance band derived from a required ground resolution, and
everything outside it is reported as **not assessed**. The same band suppresses the
opposite error: far-field pixels are where noise, compression artifacts and
aliasing generate spurious thin-line detections, so excluding them raises precision
directly.

Bands are derived, not typed in. Change the camera, the mounting height or the
resolution and the zones move on their own — which is the point, because a hard
"assess 3–10 m" would be wrong the moment someone fits a different dashcam.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..utils.logging import get_logger
from .calibration import CameraModel

log = get_logger("rdd.geometry.zones")

# Required ground resolution per class, in metres per pixel. Each is roughly
# "smallest feature we must resolve, divided by the pixels needed to see it".
# Cracks are the binding constraint: an IRC-relevant 3 mm crack needs a few
# millimetres per pixel, which is only available very close to the vehicle.
_DEFAULT_GSD_BUDGET = {
    "pothole": 0.020,
    "longitudinal_crack": 0.005,
    "transverse_crack": 0.005,
    "alligator_crack": 0.008,
    "edge_damage": 0.020,
    "ravelling": 0.005,
    "rutting": 0.015,
    "drainage_issue": 0.030,
    "water_logging": 0.030,
}


@dataclass(frozen=True)
class AssessmentZone:
    """The ground range over which one class can be assessed."""

    cls_name: str
    required_gsd_m: float
    z_near_m: float
    z_far_m: float
    limited_by: str          # resolution | field_of_view | config
    achievable: bool = True  # False when the camera cannot resolve this class at all

    @property
    def depth_m(self) -> float:
        return max(0.0, self.z_far_m - self.z_near_m)

    def contains(self, z_m: float) -> bool:
        return self.z_near_m <= z_m <= self.z_far_m

    def summary(self) -> dict:
        return {
            "class": self.cls_name,
            "required_gsd_mm": round(1000 * self.required_gsd_m, 1),
            "zone_m": [round(self.z_near_m, 2), round(self.z_far_m, 2)],
            "depth_m": round(self.depth_m, 2),
            "limited_by": self.limited_by,
            "achievable": self.achievable,
        }


@dataclass
class ZoneSet:
    """Assessment zones for every configured class, plus their pixel masks."""

    zones: dict[str, AssessmentZone] = field(default_factory=dict)
    camera: CameraModel | None = None
    _masks: dict[str, object] = field(default_factory=dict, repr=False)

    def for_class(self, cls_name: str) -> AssessmentZone | None:
        return self.zones.get(cls_name)

    def widest(self) -> AssessmentZone | None:
        """The union band — the region worth running any detector on at all."""
        usable = [z for z in self.zones.values() if z.achievable]
        if not usable:
            return None
        return AssessmentZone(
            cls_name="__union__",
            required_gsd_m=max(z.required_gsd_m for z in usable),
            z_near_m=min(z.z_near_m for z in usable),
            z_far_m=max(z.z_far_m for z in usable),
            limited_by="union",
        )

    def mask(self, cls_name: str, width: int, height: int, z_map=None):
        """Boolean image mask of the pixels inside a class's zone.

        Cached per class: the zone depends only on calibration and frame size, so
        recomputing it every frame would be pure waste.
        """
        import numpy as np

        key = f"{cls_name}:{width}x{height}"
        cached = self._masks.get(key)
        if cached is not None:
            return cached

        zone = self.zones.get(cls_name)
        if zone is None or self.camera is None or not zone.achievable:
            out = np.zeros((height, width), dtype=bool)
        else:
            if z_map is None:
                _, z_map, _ = self.camera.ground_maps(width, height)
            with np.errstate(invalid="ignore"):
                out = (z_map >= zone.z_near_m) & (z_map <= zone.z_far_m)
            out = np.nan_to_num(out, nan=False).astype(bool)
        self._masks[key] = out
        return out

    def summary(self) -> dict:
        return {name: z.summary() for name, z in sorted(self.zones.items())}

    def unachievable(self) -> list[str]:
        return sorted(n for n, z in self.zones.items() if not z.achievable)


def build_zones(cfg, camera: CameraModel) -> ZoneSet:
    """Derive each class's assessment zone from the camera's resolution curve."""
    zc = cfg.get_path("geometry.zones", {}) or {}
    budget = {**_DEFAULT_GSD_BUDGET, **(zc.get("required_gsd_m") or {})}
    classes = [str(c) for c in (cfg.get_path("model.classes") or [])]

    near_visible, far_visible = camera.visible_range()
    # A margin past the nearest visible ground: the very bottom rows are usually
    # bonnet, bumper shadow or extreme obliquity, and are not worth trusting.
    z_min = max(near_visible, float(zc.get("min_distance_m", 0.0)) or near_visible)
    hard_far = float(zc.get("max_distance_m", 0.0)) or far_visible

    zones: dict[str, AssessmentZone] = {}
    for cls in classes:
        required = float(budget.get(cls, zc.get("default_gsd_m", 0.020)))
        z_far = camera.max_range_for_gsd(required, z_min=max(z_min, 0.5),
                                        z_max=min(hard_far, 150.0))

        limited_by = "resolution"
        if z_far >= min(hard_far, 150.0) - 1e-6:
            limited_by = "config" if hard_far < far_visible else "field_of_view"

        # If even the closest usable ground misses the budget, the camera simply
        # cannot resolve this class — say so instead of emitting a sliver of a zone.
        achievable = camera.gsd_at(max(z_min, 0.5)).worst <= required
        zones[cls] = AssessmentZone(
            cls_name=cls, required_gsd_m=required,
            z_near_m=round(z_min, 3), z_far_m=round(z_far, 3),
            limited_by=limited_by, achievable=achievable,
        )

    zs = ZoneSet(zones=zones, camera=camera)
    _log_zones(zs, camera)
    return zs


def _log_zones(zs: ZoneSet, camera: CameraModel) -> None:
    log.info("Assessment zones (visible ground %.1f–%.0f m):",
             *camera.visible_range())
    for name, z in sorted(zs.zones.items()):
        if not z.achievable:
            g = camera.gsd_at(max(z.z_near_m, 0.5))
            log.warning(
                "  %-20s NOT ACHIEVABLE — needs %.1f mm/px but the closest usable "
                "ground gives %.1f mm/px. This class cannot be assessed with this "
                "camera; higher resolution or a lower/steeper mount is the only fix.",
                name, 1000 * z.required_gsd_m, 1000 * g.worst)
        else:
            log.info("  %-20s %.1f–%.1f m  (needs %.1f mm/px, limited by %s)",
                     name, z.z_near_m, z.z_far_m, 1000 * z.required_gsd_m,
                     z.limited_by)
