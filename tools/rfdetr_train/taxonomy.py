"""Fixed 6-class road-defect taxonomy and name remapping."""
from __future__ import annotations

import re

CLASS_NAMES = [
    "alligator_crack",
    "drainage_issue",
    "longitudinal_crack",
    "pothole",
    "ravelling",
    "edge_damage",
]
CLASS_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}

CLASS_ALIASES: dict[str, str | None] = {
    "d00": "longitudinal_crack",
    "d10": "longitudinal_crack",
    "d20": "alligator_crack",
    "d40": "pothole",
    "longitudinal crack": "longitudinal_crack",
    "longitudinal_crack": "longitudinal_crack",
    "longitudinal cracking": "longitudinal_crack",
    "longitudinal-crack": "longitudinal_crack",
    "longitudinal transverse cracks": "longitudinal_crack",
    "longitudinal transverse crack": "longitudinal_crack",
    "longitudinal and transverse cracks": "longitudinal_crack",
    "longitudinal/transverse cracks": "longitudinal_crack",
    "longitudinal / transverse cracks": "longitudinal_crack",
    "lateral-crack": "longitudinal_crack",
    "lateral crack": "longitudinal_crack",
    "transverse crack": "longitudinal_crack",
    "transverse_crack": "longitudinal_crack",
    "transverse cracking": "longitudinal_crack",
    "tc": "longitudinal_crack",
    "lc": "longitudinal_crack",
    # Oblique crack (OC) — common 3rd crack orientation in UAV pavement-distress
    # datasets (UAPD, UAV-PDD2023). No dedicated class in this taxonomy; folded
    # into longitudinal_crack alongside transverse, same as D10.
    "oblique crack": "longitudinal_crack",
    "oblique_crack": "longitudinal_crack",
    "oblique-crack": "longitudinal_crack",
    "oblique cracking": "longitudinal_crack",
    "oc": "longitudinal_crack",
    # "line crack" — HighRPD's UAV term for a single linear (non-branching) crack.
    "line crack": "longitudinal_crack",
    "line_crack": "longitudinal_crack",
    "line": "longitudinal_crack",
    "alligator crack": "alligator_crack",
    "alligator_crack": "alligator_crack",
    "alligator cracking": "alligator_crack",
    "alligator and fatigue cracking": "alligator_crack",
    "alligator & fatigue cracking": "alligator_crack",
    "alligator fatigue cracking": "alligator_crack",
    "fatigue cracking": "alligator_crack",
    "alligator": "alligator_crack",
    "fatigue crack": "alligator_crack",
    "reticular crack": "alligator_crack",
    "reticular_crack": "alligator_crack",
    "rc": "alligator_crack",
    "ac": "alligator_crack",
    # "block crack" — HighRPD's UAV term for a rectangular fatigue-cracking cell;
    # same failure mode as alligator cracking, viewed top-down.
    "block crack": "alligator_crack",
    "block_crack": "alligator_crack",
    "block": "alligator_crack",
    "pothole": "pothole",
    "potholes": "pothole",
    "pot hole": "pothole",
    "pot-hole": "pothole",
    "high pothole": "pothole",
    "medium pothole": "pothole",
    "low pothole": "pothole",
    "ph": "pothole",
    # HighRPD calls a pothole a "pit" (top-down, it reads as a dark pit, not a rim).
    "pit": "pothole",
    "pits": "pothole",
    # Class 4 = rutting + ravelling / surface distress (merged into ravelling)
    "ravelling": "ravelling",
    "raveling": "ravelling",
    "high ravelling": "ravelling",
    "medium ravelling": "ravelling",
    "low ravelling": "ravelling",
    "high raveling": "ravelling",
    "medium raveling": "ravelling",
    "low raveling": "ravelling",
    "weathering & raveling": "ravelling",
    "weathering and raveling": "ravelling",
    "weathering raveling": "ravelling",
    "surface distress": "ravelling",
    "rutting": "ravelling",
    "rut": "ravelling",
    "medium rutting": "ravelling",
    "high rutting": "ravelling",
    "low rutting": "ravelling",
    "corrugations": "ravelling",
    "corrugation": "ravelling",
    # "loss of aggregate" / "hungry surface" — PWD/RD01's terms for ravelling:
    # aggregate stripped from the binder (loss of aggregate) leaves a lean,
    # dry-looking surface (hungry surface) — same distress, earlier-stage.
    "loss of aggregate": "ravelling",
    "loss_of_aggregate": "ravelling",
    "hungry surface": "ravelling",
    "hungry_surface": "ravelling",
    "edge_damage": "edge_damage",
    "edge damage": "edge_damage",
    "edgecrack": "edge_damage",
    "edge-crack": "edge_damage",
    "edge crack": "edge_damage",
    "edge cracking": "edge_damage",
    "edge-cracking": "edge_damage",
    "high edge cracking": "edge_damage",
    "medium edge cracking": "edge_damage",
    "low edge cracking": "edge_damage",
    "edge drop": "edge_damage",
    "edge-drop": "edge_damage",
    "edge_drop": "edge_damage",
    "edge breaking": "edge_damage",
    "edge_breaking": "edge_damage",
    "lane/shoulder drop-off": "edge_damage",
    "lane shoulder drop-off": "edge_damage",
    "shoulder drop-off": "edge_damage",
    "shoulder erosion": "edge_damage",
    "shoulder_erosion": "edge_damage",
    "drainage_issue": "drainage_issue",
    "drainage issue": "drainage_issue",
    "drainage": "drainage_issue",
    "culvert choke": "drainage_issue",
    "culvert": "drainage_issue",
    "water-stagnation": "drainage_issue",
    "water stagnation": "drainage_issue",
    "water_stagnation": "drainage_issue",
    "water-buildup": "drainage_issue",
    "water buildup": "drainage_issue",
    "waterlogging": "drainage_issue",
    "water logging": "drainage_issue",
    "water_logging": "drainage_issue",
    "wet surface": "drainage_issue",
    "drain hole": "drainage_issue",
    "drainage overflow": "drainage_issue",
    "drainage overflow(repair)": "drainage_issue",
    "open drainage(overflowing)": "drainage_issue",
    "open drainage(not overflowing)": "drainage_issue",
    "repair": None,
    "repaired": None,
    "rp": None,
    "patching": None,
    "patch": None,
    "other corruption": None,
    "other": None,
    "striping": None,
    "bump": None,
    "loose-gravel": None,
    "sand-buildup": None,
    "facecrack": None,
    "cleaning-required": None,
    "manhole": None,
    "sewer cover": None,
    "unpaved road": None,
    "depression": None,
    "slippage": None,
    "looseness": None,
    "uncertain": None,
    "null": None,
    "none": None,
    "n/a": None,
    "na": None,
    "background": None,
    "empty": None,
}


def _norm(name: str) -> str:
    s = str(name).strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    for pref in ("high ", "medium ", "med ", "low "):
        if s.startswith(pref):
            s = s[len(pref):]
            break
    return s.strip()


def resolve_class(name: str) -> str | None:
    key = _norm(name)
    if key in CLASS_ALIASES:
        return CLASS_ALIASES[key]
    if key in CLASS_TO_ID:
        return key
    snake = key.replace(" ", "_")
    if snake in CLASS_TO_ID:
        return snake
    if snake in CLASS_ALIASES:
        return CLASS_ALIASES[snake]
    for needle, dest in (
        ("alligator", "alligator_crack"),
        ("fatigue", "alligator_crack"),
        ("block crack", "alligator_crack"),
        ("oblique", "longitudinal_crack"),
        ("line crack", "longitudinal_crack"),
        ("pit", "pothole"),
        ("ravelling", "ravelling"),
        ("raveling", "ravelling"),
        ("rutting", "ravelling"),
        ("pothole", "pothole"),
        ("edge cracking", "edge_damage"),
        ("edge crack", "edge_damage"),
        ("edge-crack", "edge_damage"),
        ("edge drop", "edge_damage"),
        ("shoulder", "edge_damage"),
        ("longitudinal", "longitudinal_crack"),
        ("transverse", "longitudinal_crack"),
        ("water stagnation", "drainage_issue"),
        ("waterlogging", "drainage_issue"),
        ("water logging", "drainage_issue"),
        ("wet surface", "drainage_issue"),
        ("culvert", "drainage_issue"),
        ("drainage", "drainage_issue"),
    ):
        if needle in key:
            return dest
    if key in {"null", "none", "n/a", "na", "background", "empty"}:
        return None
    return None


def coco_categories(names: list[str] | None = None) -> list[dict]:
    names = names or CLASS_NAMES
    return [{"id": i + 1, "name": n, "supercategory": "road_defect"} for i, n in enumerate(names)]
