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
    "ravelling": "ravelling",
    "raveling": "ravelling",
    "high ravelling": "ravelling",
    "medium ravelling": "ravelling",
    "low ravelling": "ravelling",
    "high raveling": "ravelling",
    "medium raveling": "ravelling",
    "low raveling": "ravelling",
    "edge_damage": "edge_damage",
    "edge damage": "edge_damage",
    "edgecrack": "edge_damage",
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
    "drainage_issue": "drainage_issue",
    "drainage issue": "drainage_issue",
    "drainage": "drainage_issue",
    "water-stagnation": "drainage_issue",
    "water stagnation": "drainage_issue",
    "water_stagnation": "drainage_issue",
    "water-buildup": "drainage_issue",
    "water buildup": "drainage_issue",
    "waterlogging": "drainage_issue",
    "water logging": "drainage_issue",
    "repair": None,
    "repaired": None,
    "patching": None,
    "patch": None,
    "block crack": None,
    "other corruption": None,
    "other": None,
    "rutting": None,
    "medium rutting": None,
    "high rutting": None,
    "low rutting": None,
    "striping": None,
    "bump": None,
    "loose-gravel": None,
    "sand-buildup": None,
    "facecrack": None,
    "cleaning-required": None,
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
        ("pothole", "pothole"),
        ("edge cracking", "edge_damage"),
        ("edge crack", "edge_damage"),
        ("edge drop", "edge_damage"),
        ("longitudinal", "longitudinal_crack"),
        ("water stagnation", "drainage_issue"),
        ("waterlogging", "drainage_issue"),
        ("drainage", "drainage_issue"),
    ):
        if needle in key:
            return dest
    return None


def coco_categories(names: list[str] | None = None) -> list[dict]:
    names = names or CLASS_NAMES
    return [{"id": i + 1, "name": n, "supercategory": "road_defect"} for i, n in enumerate(names)]
