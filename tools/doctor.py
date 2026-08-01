#!/usr/bin/env python
"""Environment and pipeline health check.

Lives in a file rather than inline in the shell wrappers for two reasons: PowerShell
mangles quotes when passing a `-c` snippet to a native command (so any inline Python
containing a string literal silently becomes a syntax error), and both wrappers plus
the installer would otherwise carry three copies of the same checks.

Exit code 0 = usable, 1 = something is broken that will stop the pipeline running.
"""
from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

STAGES = [
    "rdd.config", "rdd.viewpoint", "rdd.geometry", "rdd.validity", "rdd.detect",
    "rdd.eval", "rdd.roadseg", "rdd.surface", "rdd.quality", "rdd.preprocess.scale",
    "rdd.inference.detect_track", "rdd.report.irc", "rdd.pipeline",
]
LIBS = ["ultralytics", "cv2", "numpy", "pandas", "yaml", "supervision"]


def _row(label: str, value: str, ok: bool = True) -> None:
    mark = "  " if ok else "! "
    print(f"    {mark}{label:<20}{value}")


def check_environment() -> list[str]:
    """Report interpreter, libraries and external binaries. Returns fatal problems."""
    problems: list[str] = []
    print("\n  environment")
    _row("python", sys.version.split()[0])

    try:
        import torch

        cuda = torch.cuda.is_available()
        _row("torch", f"{torch.__version__}   cuda={cuda}")
        if cuda:
            _row("gpu", torch.cuda.get_device_name(0))
        else:
            _row("gpu", "none - running on CPU (slower, still correct)")
    except Exception as e:
        _row("torch", f"MISSING ({e})", ok=False)
        problems.append("torch is not installed")

    for name in LIBS:
        try:
            mod = importlib.import_module(name)
            _row(name, str(getattr(mod, "__version__", "?")))
        except Exception:
            _row(name, "MISSING", ok=False)
            problems.append(f"{name} is not installed")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        _row("ffmpeg", "MISSING - needed for 360 reprojection and video output",
             ok=False)
        problems.append("ffmpeg is not on PATH")
    else:
        try:
            out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                                 capture_output=True, text=True, timeout=20)
            has_v360 = "v360" in (out.stdout or "")
        except Exception:
            has_v360 = False
        if has_v360:
            _row("ffmpeg", f"{ffmpeg}   v360=yes")
        else:
            _row("ffmpeg", f"{ffmpeg}   v360=NO - 360 input will fail", ok=False)
            print("       a fuller build is needed: https://www.gyan.dev/ffmpeg/builds/")
    return problems


def check_pipeline() -> list[str]:
    """Import every stage and read the config. Returns fatal problems."""
    problems: list[str] = []
    print("\n  pipeline")
    broken = []
    for name in STAGES:
        try:
            importlib.import_module(name)
        except Exception as e:
            broken.append(f"{name}: {e}")
    if broken:
        for b in broken:
            _row("import", b, ok=False)
        problems.append(f"{len(broken)} pipeline stage(s) failed to import")
    else:
        _row("stages", f"all {len(STAGES)} import cleanly")

    try:
        from rdd.config import load_config

        cfg = load_config(ROOT / "config.yaml")
        classes = cfg.get_path("model.classes") or []
        _row("classes", f"{len(classes)}  ({', '.join(classes[:4])}...)")
        _row("viewpoint", str(cfg.get_path("view.profile")))
        _row("road seg", str(cfg.get_path("roadseg.backend")))
        _row("validity", "on" if cfg.get_path("validity.enabled") else "OFF")
        _row("target prec", f"{float(cfg.get_path('eval.target_precision', 0.9)):.0%}")
        cam = cfg.get_path("geometry.camera", {}) or {}
        _row("camera", f"height {cam.get('height_m')} m, hfov {cam.get('h_fov_deg')} deg")
    except Exception as e:
        _row("config.yaml", f"INVALID ({e})", ok=False)
        problems.append(f"config.yaml is invalid: {e}")
    return problems


def check_assessment_zones() -> None:
    """Show what this camera configuration can actually resolve.

    Printed by default because it is the most commonly surprising result: the
    achievable crack range on a typical dashcam is a couple of metres, not tens, and
    that is a sensor limit no model can move.
    """
    try:
        from rdd.config import load_config
        from rdd.geometry.calibration import build_camera
        from rdd.geometry.zones import build_zones
    except Exception:
        return

    try:
        cfg = load_config(ROOT / "config.yaml")
        width = int(cfg.get_path("geometry.doctor_probe_width", 1920))
        height = int(cfg.get_path("geometry.doctor_probe_height", 1080))
        cam = build_camera(cfg, width, height)
        zones = build_zones(cfg, cam)
    except Exception:
        return

    print(f"\n  what a {width}x{height} camera can resolve, at the configured mount")
    for name, z in sorted(zones.zones.items()):
        if z.achievable:
            _row(name, f"{z.z_near_m:.1f} - {z.z_far_m:.1f} m")
        else:
            _row(name, f"not achievable (needs {1000 * z.required_gsd_m:.0f} mm/px)",
                 ok=False)
    print("       narrowing the field of view extends these; raising the mount does not")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify", action="store_true",
                   help="terse output; used by the installer")
    p.add_argument("--no-zones", action="store_true")
    args = p.parse_args(argv)

    problems = check_environment() + check_pipeline()
    if not args.verify and not args.no_zones:
        check_assessment_zones()

    print()
    if problems:
        print("  NOT READY:")
        for prob in problems:
            print(f"    - {prob}")
        print("\n  Re-run setup, or install the missing pieces above.\n")
        return 1
    print("  Ready.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
