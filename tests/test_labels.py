"""Annotation validation and repair.

The case worth guarding hardest is mixed geometry. Ultralytics decides box-vs-segment
per FILE, so one polygon row makes it reinterpret that file's `x y w h` rows as two
points each — producing a box with the wrong centre and several times the area, with
no error raised. `test_mixed_geometry_*` pins both the detection and the repair, and
`test_ultralytics_misreads_mixed_file` documents the upstream behaviour that makes it
matter, so the check is not silently pointless if that behaviour ever changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))

import check_labels  # noqa: E402
import fix_labels  # noqa: E402


def _dataset(root: Path, splits: dict[str, list[tuple[str, list[str]]]],
             names: list[str], size=(64, 48)) -> Path:
    """Build a YOLO dataset. `splits` maps split -> [(stem, [label lines])]."""
    for split, items in splits.items():
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        for stem, lines in items:
            img = np.full((size[1], size[0], 3), 40, np.uint8)
            # Content keyed off the stem, not a per-split counter: the counter restarts
            # at each split, so the first image of train and of valid came out
            # identical and every dataset looked like it had leakage.
            x = (sum(map(ord, stem)) * 7) % 40
            cv2.rectangle(img, (x, 5), (x + 12, 30), (200, 200, 200), -1)
            cv2.imwrite(str(root / split / "images" / f"{stem}.jpg"), img)
            (root / split / "labels" / f"{stem}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
    (root / "data.yaml").write_text(yaml.safe_dump(
        {"train": "../train/images", "val": "../valid/images",
         "nc": len(names), "names": names}), encoding="utf-8")
    return root


BOX = "0 0.5 0.5 0.2 0.2"
POLY = "1 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3"
NAMES = ["pothole", "ravelling"]


def _check(root, **kw):
    rep = check_labels.Report()
    splits, ds_names = check_labels._discover(Path(root), rep)
    stats = {
        "per_class": __import__("collections").Counter(),
        "per_class_split": __import__("collections").Counter(), "counts": {},
        "bad_class": [], "degenerate": [], "oob": [], "dupes": [], "tiny": [],
        "areas": [], "boxes": 0, "polys": 0, "fixed": 0, "mixed_files": [],
        "sizes": __import__("collections").Counter(),
        "cross_dupes": [], "near_dupes": [], "split_sizes": {},
    }
    for name, i, l in splits:
        check_labels._check_split(name, i, l, ds_names or NAMES, rep,
                                  kw.get("fix_clip", False), stats)
    return rep, stats, ds_names


# --------------------------------------------------------------------------- geometry

def test_mixed_geometry_flagged_per_file(tmp_path):
    """A file holding both a box row and a polygon row is the corrupting case."""
    _dataset(tmp_path, {"train": [("a", [BOX, POLY]), ("b", [BOX])]}, NAMES)
    _, stats, _ = _check(tmp_path)
    assert stats["mixed_files"] == ["train/a.txt"]
    assert stats["boxes"] == 2 and stats["polys"] == 1


def test_geometry_split_across_files_is_not_a_mixed_file(tmp_path):
    """Boxes in one file and polygons in another still need normalising, but nothing
    is corrupted — the two cases get different severities and must stay distinct."""
    _dataset(tmp_path, {"train": [("a", [BOX]), ("b", [POLY])]}, NAMES)
    _, stats, _ = _check(tmp_path)
    assert stats["mixed_files"] == []
    assert stats["boxes"] == 1 and stats["polys"] == 1


def test_ultralytics_misreads_mixed_file():
    """Why the check above exists. Pinning upstream behaviour, not ours."""
    from ultralytics.data.utils import segments2boxes

    rows = [[0.0, 0.4, 0.8, 0.14, 0.29], [1.0] + [0.1, 0.1, 0.3, 0.1, 0.3, 0.3]]
    assert any(len(r) > 6 for r in rows), "the branch that triggers reinterpretation"

    segs = [np.array(r[1:], dtype=np.float32).reshape(-1, 2) for r in rows]
    out = segments2boxes(segs)
    cx, cy, w, h = out[0]
    # The box row said centre (0.4, 0.8), size 0.14x0.29. Read as the points
    # (0.4, 0.8) and (0.14, 0.29) it becomes something else entirely.
    assert not np.isclose(cx, 0.4, atol=0.05)
    assert w * h > 3 * (0.14 * 0.29), "and it inflates the area several-fold"


def test_polygon_converts_to_its_bounding_box():
    assert fix_labels._to_box([0.1, 0.2, 0.5, 0.2, 0.5, 0.6, 0.1, 0.6]) == \
        pytest.approx([0.3, 0.4, 0.4, 0.4])


def test_box_roundtrips_through_polygon_conversion():
    box = [0.5, 0.5, 0.2, 0.4]
    assert fix_labels._to_box(fix_labels._to_poly(box)) == pytest.approx(box)


def test_repair_normalises_to_one_geometry(tmp_path):
    src = _dataset(tmp_path / "src", {"train": [("a", [BOX, POLY])],
                                      "valid": [("c", [BOX])]}, NAMES)
    out = tmp_path / "out"
    assert fix_labels.main(["--labels", str(src), "--out", str(out), "--to", "box"]) == 0
    for line in (out / "train" / "labels" / "a.txt").read_text().splitlines():
        assert len(line.split()) == 5, "every row is now a box"
    _, stats, _ = _check(out)
    assert stats["polys"] == 0 and stats["mixed_files"] == []


# --------------------------------------------------------------------------- leakage

def test_duplicate_frame_across_splits_is_found_and_moved(tmp_path):
    src = tmp_path / "src"
    _dataset(src, {"train": [("a", [BOX]), ("b", [BOX])],
                   "valid": [("c", [BOX])]}, NAMES)
    # Make the valid image a byte-different but visually identical copy of a train one,
    # which is what a re-encoded export produces and what byte hashing would miss.
    img = cv2.imread(str(src / "train" / "images" / "a.jpg"))
    cv2.imwrite(str(src / "valid" / "images" / "c.jpg"), img,
                [cv2.IMWRITE_JPEG_QUALITY, 70])
    assert (src / "valid" / "images" / "c.jpg").read_bytes() != \
        (src / "train" / "images" / "a.jpg").read_bytes()

    stats = {"cross_dupes": [], "near_dupes": [], "split_sizes": {}}
    rep = check_labels.Report()
    splits, _ = check_labels._discover(src, rep)
    check_labels._check_leakage(splits, stats)
    assert stats["cross_dupes"] or stats["near_dupes"]

    out = tmp_path / "out"
    assert fix_labels.main(["--labels", str(src), "--out", str(out), "--to", "box"]) == 0
    assert (out / "train" / "images" / "c.jpg").exists(), "moved into train"
    assert not (out / "valid" / "images" / "c.jpg").exists()


def test_distinct_frames_are_not_called_duplicates(tmp_path):
    src = _dataset(tmp_path, {"train": [("a", [BOX]), ("b", [BOX])],
                              "valid": [("c", [BOX])]}, NAMES)
    stats = {"cross_dupes": [], "near_dupes": [], "split_sizes": {}}
    rep = check_labels.Report()
    splits, _ = check_labels._discover(src, rep)
    check_labels._check_leakage(splits, stats)
    assert not stats["cross_dupes"] and not stats["near_dupes"]


# --------------------------------------------------------------------------- taxonomy

def test_rename_maps_dataset_names_onto_the_taxonomy(tmp_path):
    src = _dataset(tmp_path / "src", {"train": [("a", [BOX])]},
                   ["Pothole", "Shoulder Erosion"])
    out = tmp_path / "out"
    assert fix_labels.main(["--labels", str(src), "--out", str(out),
                            "--to", "box", "--rename"]) == 0
    names = yaml.safe_load((out / "data.yaml").read_text())["names"]
    assert names == ["pothole", "edge_damage"]


def test_rename_refuses_rather_than_guessing(tmp_path):
    """An unrecognised class must stop the run. Silently dropping or mis-assigning it
    would mislabel the dataset in a way nothing downstream can detect."""
    src = _dataset(tmp_path / "src", {"train": [("a", [BOX])]},
                   ["Pothole", "Zebra Crossing"])
    out = tmp_path / "out"
    assert fix_labels.main(["--labels", str(src), "--out", str(out),
                            "--to", "box", "--rename"]) == 1
    assert not out.exists()


def test_class_ids_keep_their_positions_when_renaming(tmp_path):
    """Renaming must not renumber: ids in the label files are never rewritten."""
    src = _dataset(tmp_path / "src", {"train": [("a", ["1 0.5 0.5 0.2 0.2"])]},
                   ["Pothole", "Shoulder Erosion"])
    out = tmp_path / "out"
    fix_labels.main(["--labels", str(src), "--out", str(out), "--to", "box", "--rename"])
    assert (out / "train" / "labels" / "a.txt").read_text().split()[0] == "1"
    assert yaml.safe_load((out / "data.yaml").read_text())["names"][1] == "edge_damage"


# --------------------------------------------------------------------------- defects

def test_out_of_range_class_id_is_caught(tmp_path):
    _dataset(tmp_path, {"train": [("a", ["7 0.5 0.5 0.2 0.2"])]}, NAMES)
    _, stats, _ = _check(tmp_path)
    assert stats["bad_class"] and stats["bad_class"][0][2] == 7


def test_pixel_coordinates_are_caught_and_clamped(tmp_path):
    _dataset(tmp_path, {"train": [("a", ["0 320.0 240.0 64.0 48.0"])]}, NAMES)
    _, stats, _ = _check(tmp_path)
    assert stats["oob"]
    _check(tmp_path, fix_clip=True)
    vals = [float(v) for v in
            (tmp_path / "train" / "labels" / "a.txt").read_text().split()[1:]]
    assert all(0.0 <= v <= 1.0 for v in vals)


def test_zero_area_shapes_are_caught(tmp_path):
    _dataset(tmp_path, {"train": [("a", ["0 0.5 0.5 0.0 0.2"]),
                                  ("b", ["1 0.2 0.2 0.2 0.2 0.2 0.2"])]}, NAMES)
    _, stats, _ = _check(tmp_path)
    assert len(stats["degenerate"]) == 2


def test_degenerate_shapes_are_dropped_by_the_repair(tmp_path):
    src = _dataset(tmp_path / "src", {"train": [("a", ["0 0.5 0.5 0.0 0.2", BOX])]},
                   NAMES)
    out = tmp_path / "out"
    fix_labels.main(["--labels", str(src), "--out", str(out), "--to", "box"])
    kept = (out / "train" / "labels" / "a.txt").read_text().strip().splitlines()
    assert len(kept) == 1, "the zero-width box is gone, the valid one stays"


def test_duplicate_annotation_in_one_file(tmp_path):
    _dataset(tmp_path, {"train": [("a", [BOX, BOX])]}, NAMES)
    _, stats, _ = _check(tmp_path)
    assert stats["dupes"]


def test_empty_label_file_counts_as_a_negative_not_an_error(tmp_path):
    _dataset(tmp_path, {"train": [("a", [])]}, NAMES)
    rep, stats, _ = _check(tmp_path)
    assert stats["counts"]["train"]["empty"] == 1
    assert not rep.errors


# --------------------------------------------------------------------------- task

def test_task_is_read_from_the_labels_not_the_config(tmp_path):
    """config.model.arch says '-seg'; boxes must still train as detect."""
    from rdd.model.train import _infer_task

    box_ds = _dataset(tmp_path / "b", {"train": [("a", [BOX])]}, NAMES)
    poly_ds = _dataset(tmp_path / "p", {"train": [("a", [POLY])]}, NAMES)
    for ds in (box_ds, poly_ds):
        doc = yaml.safe_load((ds / "data.yaml").read_text())
        doc.update({"path": str(ds), "train": "train/images"})
        (ds / "data.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")

    assert _infer_task(box_ds / "data.yaml") == "detect"
    assert _infer_task(poly_ds / "data.yaml") == "segment"


# --------------------------------------------------------------------------- device

def test_explicit_cuda_request_fails_loudly_when_strict(monkeypatch):
    """A fine-tune must not quietly become a ten-hour CPU run.

    This happened: `--device cuda` on a CUDA-less laptop logged one warning, which
    scrolled past inside ultralytics' startup banner, and training ran on CPU at
    22 s/iteration.
    """
    import torch

    from rdd.utils.device import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="no CUDA-capable GPU"):
        resolve_device("cuda", strict=True)


def test_non_strict_callers_still_fall_back(monkeypatch):
    """Inference is short enough that CPU is a reasonable answer, so the default
    behaviour must not change."""
    import torch

    from rdd.utils.device import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("cuda") == "cpu"
    assert resolve_device("auto", strict=True) == "cpu", "auto never errors"
    assert resolve_device("cpu", strict=True) == "cpu"
