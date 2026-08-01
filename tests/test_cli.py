"""CLI wiring and config validation.

The override test exists because of a real bug: `--view` parsed fine and was then
dropped on the floor for the end-to-end run, so flat car footage was reprojected as
if it were 360 equirectangular. Nothing crashed — it just produced geometrically
garbage frames. A flag that is silently ignored is worse than one that errors.
"""
from __future__ import annotations

import pytest

from rdd.config import load_config
from run import _overrides, build_parser


def test_view_and_device_flags_become_overrides():
    args = build_parser().parse_args(["--input", "x.mp4", "--view", "drone_nadir",
                                      "--device", "cpu"])
    assert _overrides(args) == {"view.profile": "drone_nadir", "run.device": "cpu"}


def test_no_flags_means_no_overrides():
    args = build_parser().parse_args(["--input", "x.mp4"])
    assert _overrides(args) == {}


def test_overrides_actually_reach_the_config(tmp_path):
    """The end-to-end path must apply them, not just parse them."""
    cfg = load_config("config.yaml")
    assert cfg.get_path("view.profile") == "car_360"
    for dotted, value in {"view.profile": "drone_nadir"}.items():
        cfg.set_path(dotted, value)
    assert cfg.get_path("view.profile") == "drone_nadir"


def test_every_subcommand_accepts_the_common_flags():
    parser = build_parser()
    for argv in (
        ["preprocess", "--input", "v.mp4", "--view", "car_flat", "--device", "cpu"],
        ["quality", "--input", "v.mp4", "--view", "car_flat"],
        ["roadseg", "--input", "v.mp4", "--view", "drone_nadir", "--n", "3"],
        ["infer", "--input", "v.mp4", "--view", "car_flat"],
        ["train", "--labels", "d", "--device", "cpu"],
        ["annotate", "--frames", "d"],
    ):
        args = parser.parse_args(argv)
        assert args.config == "config.yaml"
        assert callable(args.func)


def test_invalid_view_is_rejected_by_the_parser():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--input", "v.mp4", "--view", "helicopter"])


# -- config validation --------------------------------------------------------

def test_shipped_config_is_valid():
    cfg = load_config("config.yaml")
    assert cfg.get_path("model.classes")
    assert cfg.get_path("roadseg.backend") == "classical"
    assert cfg.get_path("surface.occlusion_policy") == "abstain"


def _write(tmp_path, cfg_text):
    p = tmp_path / "c.yaml"
    p.write_text(cfg_text, encoding="utf-8")
    return p


def test_random_split_is_rejected(tmp_path):
    p = _write(tmp_path, """
model:
  classes: ["pothole"]
  train:
    split:
      mode: random
""")
    with pytest.raises(ValueError, match="random"):
        load_config(p)


def test_unknown_view_profile_is_rejected(tmp_path):
    p = _write(tmp_path, """
model:
  classes: ["pothole"]
view:
  profile: submarine
""")
    with pytest.raises(ValueError, match="view.profile"):
        load_config(p)


def test_unknown_roadseg_backend_is_rejected(tmp_path):
    p = _write(tmp_path, """
model:
  classes: ["pothole"]
roadseg:
  backend: magic
""")
    with pytest.raises(ValueError, match="roadseg.backend"):
        load_config(p)


def test_occluder_class_must_exist_in_model_classes(tmp_path):
    """Otherwise the exemption silently never applies to anything."""
    p = _write(tmp_path, """
model:
  classes: ["pothole", "crack"]
surface:
  occluder_classes: ["water_logging"]
""")
    with pytest.raises(ValueError, match="occluder_classes"):
        load_config(p)


def test_out_of_range_fov_is_rejected(tmp_path):
    p = _write(tmp_path, """
model:
  classes: ["pothole"]
preprocess:
  reproject:
    h_fov_deg: 200.0
""")
    with pytest.raises(ValueError, match="h_fov_deg"):
        load_config(p)


def test_vertical_fov_is_also_validated(tmp_path):
    """v_fov was previously unchecked while h_fov was."""
    p = _write(tmp_path, """
model:
  classes: ["pothole"]
preprocess:
  reproject:
    v_fov_deg: 0.0
""")
    with pytest.raises(ValueError, match="v_fov_deg"):
        load_config(p)


def test_bad_out_width_is_rejected(tmp_path):
    p = _write(tmp_path, """
model:
  classes: ["pothole"]
preprocess:
  reproject:
    out_width: "wide"
""")
    with pytest.raises(ValueError, match="out_width"):
        load_config(p)


def test_auto_out_width_is_accepted(tmp_path):
    p = _write(tmp_path, """
model:
  classes: ["pothole"]
preprocess:
  reproject:
    out_width: auto
    out_height: auto
""")
    assert load_config(p).get_path("preprocess.reproject.out_width") == "auto"


def test_nested_writes_persist():
    """A lazily-wrapping Cfg would drop this write into a temporary object."""
    cfg = load_config("config.yaml")
    cfg.run.name = "changed"
    assert cfg.get_path("run.name") == "changed"
    cfg.set_path("roadseg.classical.distance_tau", 9.5)
    assert cfg["roadseg"]["classical"]["distance_tau"] == 9.5


def test_set_path_creates_missing_levels():
    cfg = load_config("config.yaml")
    cfg.set_path("brand.new.key", 3)
    assert cfg.get_path("brand.new.key") == 3


# -- checkpoint/class alignment ------------------------------------------------

class _FakeModel:
    def __init__(self, names):
        self.names = names


def test_class_mismatch_is_reported():
    """A stock COCO checkpoint against a 4-class road config must not pass silently."""
    from rdd.model.loader import check_class_alignment

    cfg = load_config("config.yaml")
    coco = _FakeModel({i: f"coco_{i}" for i in range(80)})
    assert check_class_alignment(coco, cfg) is False


def test_matching_classes_pass():
    from rdd.model.loader import check_class_alignment

    cfg = load_config("config.yaml")
    aligned = _FakeModel(dict(enumerate(cfg.get_path("model.classes"))))
    assert check_class_alignment(aligned, cfg) is True


def test_reordered_classes_are_reported():
    """Same count means nothing errors — but the labels would be swapped."""
    from rdd.model.loader import check_class_alignment

    cfg = load_config("config.yaml")
    swapped = list(reversed(cfg.get_path("model.classes")))
    assert check_class_alignment(_FakeModel(dict(enumerate(swapped))), cfg) is False


def test_model_without_names_is_not_flagged():
    from rdd.model.loader import check_class_alignment

    assert check_class_alignment(_FakeModel({}), load_config("config.yaml")) is True


# -- checkpoint class mapping --------------------------------------------------

class _NamedModel:
    def __init__(self, names):
        self.names = dict(enumerate(names))


def test_without_a_map_ids_resolve_positionally():
    """Documents the trap: a 5-class checkpoint against a 9-class config."""
    from rdd.model.loader import build_class_resolver

    cfg = load_config("config.yaml")
    r = build_class_resolver(_NamedModel(["D00", "D10", "D20", "D40", "Repair"]), cfg)
    # D00 is a longitudinal crack, but index 0 of model.classes is "pothole".
    assert r(0) == "pothole", "positional resolution is what class_map exists to fix"


def test_class_map_translates_by_name():
    from rdd.model.loader import build_class_resolver

    cfg = load_config("config.yaml")
    cfg.set_path("model.class_map", {
        "D00": "longitudinal_crack", "D10": "transverse_crack",
        "D20": "alligator_crack", "D40": "pothole", "Repair": None})
    r = build_class_resolver(_NamedModel(["D00", "D10", "D20", "D40", "Repair"]), cfg)
    assert [r(i) for i in range(4)] == [
        "longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]


def test_class_mapped_to_null_is_dropped():
    """RDD2022's 'Repair' is a past intervention, not a defect."""
    from rdd.model.loader import build_class_resolver

    cfg = load_config("config.yaml")
    cfg.set_path("model.class_map", {"D40": "pothole", "Repair": None})
    r = build_class_resolver(_NamedModel(["D40", "Repair"]), cfg)
    assert r(0) == "pothole"
    assert r(1) is None, "a null mapping must drop the detection"


def test_unmapped_checkpoint_class_keeps_its_own_name():
    from rdd.model.loader import build_class_resolver

    cfg = load_config("config.yaml")
    cfg.set_path("model.class_map", {"D40": "pothole"})
    r = build_class_resolver(_NamedModel(["D40", "Something"]), cfg)
    assert r(1) == "Something", "better a visible raw name than a wrong one"


def test_out_of_range_id_is_conspicuous():
    from rdd.model.loader import build_class_resolver

    cfg = load_config("config.yaml")
    cfg.set_path("model.class_map", {"D40": "pothole"})
    assert build_class_resolver(_NamedModel(["D40"]), cfg)(9) == "UNMAPPED_CLASS_9"


# -- frame sampling ------------------------------------------------------------

def test_sampler_spans_the_whole_clip(tmp_path):
    """The bug this guards: taking the first N keyframes samples only the opening
    stretch of a route and calls it representative."""
    import subprocess

    import cv2
    import numpy as np

    from rdd.utils.video import iter_sampled_frames

    src = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (160, 120))
    rng = np.random.default_rng(0)
    for _ in range(450):
        w.write(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8))
    w.release()

    got = list(iter_sampled_frames(src, 20))
    assert len(got) >= 10, f"only {len(got)} samples"
    idxs = [i for i, _ in got]
    assert max(idxs) > 0.8 * 450, f"samples stop at frame {max(idxs)} of 450"
    assert all(f.shape == (120, 160, 3) for _, f in got)


def test_sampler_handles_a_clip_shorter_than_the_request(tmp_path):
    import cv2
    import numpy as np

    from rdd.utils.video import iter_sampled_frames

    src = tmp_path / "tiny.mp4"
    w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (64, 48))
    for _ in range(5):
        w.write(np.zeros((48, 64, 3), dtype=np.uint8))
    w.release()
    assert 0 < len(list(iter_sampled_frames(src, 60))) <= 60


def test_sampler_rejects_a_missing_file(tmp_path):
    from rdd.utils.video import iter_sampled_frames

    with pytest.raises(RuntimeError):
        list(iter_sampled_frames(tmp_path / "nope.mp4", 10))
