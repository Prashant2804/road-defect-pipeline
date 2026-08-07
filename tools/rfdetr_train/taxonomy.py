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
    "lateral-crack": "longitudinal_crack",
    "lateral crack": "longitudinal_crack",
    "transverse crack": "longitudinal_crack",
    "transverse_crack": "longitudinal_crack",
    "transverse cracking": "longitudinal_crack",
    "tc": "longitudinal_crack",
    "lc": "longitudinal_crack",
    "alligator crack": "alligator_crack",
    "alligator_crack": "alligator_crack",
    "alligator cracking": "alligator_crack",
    "alligator": "alligator_crack",
    "fatigue crack": "alligator_crack",
    "reticular crack": "alligator_crack",
    "reticular_crack": "alligator_crack",
    "rc": "alligator_crack",
    "pothole": "pothole",
    "potholes": "pothole",
    "pot hole": "pothole",
    "pot-hole": "pothole",
    "high pothole": "pothole",
    "medium pothole": "pothole",
    "low pothole": "pothole",
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
    "lane/shoulder drop-off": "edge_damage",
    "lane shoulder drop-off": "edge_damage",
    "shoulder drop-off": "edge_damage",
    "shoulder erosion": "edge_damage",
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
    "patching": None,
    "patch": None,
    "block crack": None,
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
    return None


def coco_categories(names: list[str] | None = None) -> list[dict]:
    names = names or CLASS_NAMES
    return [{"id": i + 1, "name": n, "supercategory": "road_defect"} for i, n in enumerate(names)]
