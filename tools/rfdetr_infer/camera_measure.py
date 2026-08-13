"""Record measured GoPro mount parameters for defect area in m².

You still have to take the tape measurements on the vehicle. This writes them
into a camera.json that inference loads via --camera-json.

Typical GoPro dashcam (Linear / rectilinear mode — prefer this over Wide):

  1. Height: tape from pavement to the lens centre, vehicle on level ground.
  2. Pitch: phone inclinometer on the camera body, or measure how far the
     optical axis hits the road ahead (z) then pitch ≈ atan(height / z).
  3. HFOV: use Linear mode. Do NOT use the marketing diagonal (often 170°).
     If unknown, start from the Linear 16:9 preset (86°) and refine with the
     GPS+flow GSD check after a drive.
  4. Distortion: Linear ≈ k1=0. Wide/SuperView needs k1 (negative) or a
     proper checkerboard calibration.

Example::

    .venv/bin/python -m tools.rfdetr_infer.camera_measure \\
      --height-m 1.32 --pitch-deg 8 --mode linear --out data/camera.json
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from .camera import GOPRO_HFOV_16_9
from .config import repo_root


CHECKLIST = """
Measure these on the parked survey car (level ground):

  [ ] Camera height_m     tape, pavement → lens centre
  [ ] pitch_deg           downward tilt (phone inclinometer, or atan(h/z_hit))
  [ ] GoPro FOV mode      Linear (recommended) | Wide | Narrow
  [ ] Resolution / aspect  e.g. 1920x1080 16:9
  [ ] Distortion          Linear: leave k1=k2=0; Wide: calibrate or set k1

Then run inference with:
  --camera-json <this file> --camera-height-m ...  (json fills the rest)

Undistort Wide footage with --k1 (Brown-Conrady); identity when k1=k2=0.
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Write camera.json from measured GoPro dashcam parameters."
    )
    p.add_argument("--height-m", type=float, required=True,
                   help="Lens height above the pavement (metres). Must be measured.")
    p.add_argument("--pitch-deg", type=float, default=5.0,
                   help="Downward tilt in degrees (default 5).")
    p.add_argument("--yaw-deg", type=float, default=0.0)
    p.add_argument(
        "--hfov-deg",
        type=float,
        default=None,
        help="True horizontal FOV. If omitted, uses the --mode preset.",
    )
    p.add_argument(
        "--mode",
        choices=tuple(GOPRO_HFOV_16_9),
        default="linear",
        help="GoPro digital lens preset for HFOV if --hfov-deg is omitted.",
    )
    p.add_argument("--k1", type=float, default=0.0, help="Radial distortion k1")
    p.add_argument("--k2", type=float, default=0.0, help="Radial distortion k2")
    p.add_argument("--notes", type=str, default="")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Default: data/camera.json",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    hfov = args.hfov_deg if args.hfov_deg is not None else GOPRO_HFOV_16_9[args.mode]
    out = args.out or (repo_root() / "data" / "camera.json")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "height_m": float(args.height_m),
        "pitch_deg": float(args.pitch_deg),
        "yaw_deg": float(args.yaw_deg),
        "h_fov_deg": float(hfov),
        "gopro_mode": args.mode,
        "k1": float(args.k1),
        "k2": float(args.k2),
        "measured_on": str(date.today()),
        "notes": args.notes or (
            "Prefer GoPro Linear. Marketing 170° is diagonal, not HFOV. "
            "Area scales with height² — a 10% height error is ~21% area error."
        ),
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(CHECKLIST.strip())
    print(f"\nWrote {out}")
    print(json.dumps(payload, indent=2))
    print(
        "\nInference:\n"
        f"  python -m tools.rfdetr_infer.run --video VIDEO --weights WEIGHTS "
        f"--camera-json {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
