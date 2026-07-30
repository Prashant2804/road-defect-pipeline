"""Shared pytest fixtures. Scene builders live in `tests/scenes.py`."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))


@pytest.fixture
def cfg():
    """A fresh copy of the shipped config for each test."""
    from rdd.config import load_config

    return load_config(_ROOT / "config.yaml")


@pytest.fixture
def car_view(cfg):
    from rdd.viewpoint import resolve_view

    from tests.scenes import H, W

    cfg.set_path("view.profile", "car_flat")
    return resolve_view(cfg, W, H)


@pytest.fixture
def drone_view(cfg):
    from rdd.viewpoint import resolve_view

    from tests.scenes import H, W

    cfg.set_path("view.profile", "drone_nadir")
    return resolve_view(cfg, W, H)
