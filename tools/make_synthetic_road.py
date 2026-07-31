#!/usr/bin/env python
"""Generate synthetic road footage for end-to-end testing and threshold tuning.

The repo previously shipped `data/raw/synthetic_equirect.mp4`, which is a colour
test pattern — saturated magenta/blue/yellow wedges. It validates the v360
geometry and nothing else: run the detector stack on it and the numbers are
noise, because there is no road in it to segment.

This produces footage with the properties the pipeline actually reasons about,
and with known ground truth:

  * **Perspective-correct geometry.** Ground points are projected through a real
    pinhole model (`u = cx + f·x/z`, `v = cy + f·h/z`), so a defect's apparent
    size falls off with range exactly as it does in real footage. That is what
    makes it a fair test of the IPM ground-area correction — a defect of fixed
    real size must measure the same m² near and far.
  * **Texture separation.** Road is smoother than the verge, which is the cue
    the classical segmenter keys on.
  * **Physically distinct contaminants.** Water is specular (texture destroyed,
    brighter); mud is a warm chroma shift; shadow is *multiplicative* so its
    brightness-relative texture is preserved. That last one is the decoy that
    separates a real occlusion detector from one that just finds dark pixels.
  * **Forward motion**, so tracking and unique-counting have something to do.

Usage:
    python tools/make_synthetic_road.py --out data/raw
    python tools/make_synthetic_road.py --out data/raw --only car --degraded
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

RNG = np.random.default_rng(20260729)

ROAD_BGR = (118, 124, 130)
VERGE_BGR = (40, 95, 45)
SKY_BGR = (205, 180, 145)
POTHOLE_BGR = (56, 58, 60)
WATER_BGR = (208, 203, 194)
MUD_BGR = (42, 72, 118)

ROAD_SIGMA = 6.0
VERGE_SIGMA = 26.0


@dataclass
class Defect:
    kind: str          # pothole | water | mud
    x: float           # lateral offset, metres from centreline
    z: float           # forward distance, metres (decreases as we drive)
    half_w: float      # semi-axis across the road, metres
    half_l: float      # semi-axis along the road, metres


@dataclass
class CarCamera:
    width: int = 960
    height: int = 540
    focal_px: float = 780.0
    height_m: float = 1.5
    horizon_row: float = 190.0
    road_half_w: float = 2.6
    max_z: float = 60.0

    @property
    def cx(self) -> float:
        return self.width / 2.0

    def z_for_row(self, v: float) -> float:
        """Ground distance imaged at row v (below the horizon)."""
        dv = v - self.horizon_row
        if dv <= 0:
            return float("inf")
        return self.focal_px * self.height_m / dv

    def project(self, x: float, z: float) -> tuple[float, float]:
        if z <= 0.05:
            return float("nan"), float("nan")
        return (self.cx + self.focal_px * x / z,
                self.horizon_row + self.focal_px * self.height_m / z)


def _noise(shape, sigma):
    return RNG.normal(0.0, sigma, size=(*shape, 3)).astype(np.float32)


def _clip(a):
    return np.clip(a, 0, 255).astype(np.uint8)


def _filled(shape, bgr, sigma):
    """A full-size uint8 layer of one colour plus texture noise."""
    return _clip(np.asarray(bgr, dtype=np.float32) + _noise(shape, sigma))


def car_road_mask(cam: CarCamera) -> np.ndarray:
    """Exact road region, filled row by row from the ground-plane projection."""
    mask = np.zeros((cam.height, cam.width), dtype=bool)
    for v in range(int(cam.horizon_row) + 2, cam.height):
        z = cam.z_for_row(v)
        if z > cam.max_z:
            continue
        half_u = cam.focal_px * cam.road_half_w / z
        u0 = max(0, int(round(cam.cx - half_u)))
        u1 = min(cam.width, int(round(cam.cx + half_u)))
        if u1 > u0:
            mask[v, u0:u1] = True
    return mask


def render_car_frame(cam: CarCamera, defects: list[Defect],
                     shadow_z: float | None = None) -> np.ndarray:
    frame = np.zeros((cam.height, cam.width, 3), dtype=np.float32)

    # Sky above the horizon, textured verge below it.
    frame[:] = np.asarray(VERGE_BGR, dtype=np.float32)
    frame += _noise((cam.height, cam.width), VERGE_SIGMA)
    sky = slice(0, int(cam.horizon_row))
    frame[sky] = np.asarray(SKY_BGR, dtype=np.float32)
    frame[sky] += _noise((int(cam.horizon_row), cam.width), 4.0)

    road = car_road_mask(cam)
    road_layer = np.asarray(ROAD_BGR, dtype=np.float32) + _noise(
        (cam.height, cam.width), ROAD_SIGMA)
    frame[road] = road_layer[road]

    # Contaminants and defects, drawn only where they fall on the road.
    for d in defects:
        if not (1.5 < d.z < cam.max_z):
            continue
        u, v = cam.project(d.x, d.z)
        if not np.isfinite(u) or not np.isfinite(v):
            continue
        ru = cam.focal_px * d.half_w / d.z
        # Along-road extent foreshortens with the square of range.
        rv = cam.focal_px * cam.height_m * d.half_l / (d.z ** 2)
        if ru < 1.5 or rv < 1.0:
            continue

        blob = np.zeros((cam.height, cam.width), dtype=np.uint8)
        cv2.ellipse(blob, (int(round(u)), int(round(v))),
                    (int(round(ru)), max(1, int(round(rv)))), 0, 0, 360, 1, -1)
        m = blob.astype(bool) & road
        if not m.any():
            continue

        if d.kind == "pothole":
            layer = np.asarray(POTHOLE_BGR, dtype=np.float32) + _noise(
                (cam.height, cam.width), 7.0)
        elif d.kind == "water":
            layer = np.asarray(WATER_BGR, dtype=np.float32) + _noise(
                (cam.height, cam.width), 0.6)
        else:
            layer = np.asarray(MUD_BGR, dtype=np.float32) + _noise(
                (cam.height, cam.width), 2.5)
        frame[m] = layer[m]

    # A multiplicative shadow band: the decoy the surface stage must not call mud.
    if shadow_z is not None and 2.0 < shadow_z < cam.max_z:
        _, v = cam.project(0.0, shadow_z)
        if np.isfinite(v):
            band = np.zeros((cam.height, cam.width), dtype=np.uint8)
            half = max(4, int(cam.focal_px * cam.height_m * 2.0 / (shadow_z ** 2)))
            cv2.rectangle(band, (0, int(v) - half), (cam.width, int(v) + half), 1, -1)
            m = band.astype(bool)
            frame[m] *= 0.45

    return _clip(frame)


def make_car_video(path: Path, n_frames: int = 120, fps: float = 15.0,
                   speed_mps: float = 4.0) -> Path:
    cam = CarCamera()
    dt = 1.0 / fps

    defects = [
        Defect("pothole", x=-0.9, z=14.0, half_w=0.35, half_l=0.30),
        Defect("pothole", x=1.1, z=26.0, half_w=0.45, half_l=0.40),
        Defect("water", x=0.2, z=20.0, half_w=1.3, half_l=1.1),
        Defect("mud", x=-1.4, z=34.0, half_w=1.0, half_l=1.4),
        Defect("pothole", x=0.5, z=44.0, half_w=0.30, half_l=0.28),
        Defect("water", x=-0.6, z=52.0, half_w=1.1, half_l=0.9),
    ]
    shadow_z0 = 40.0

    writer = _open_writer(path, cam.width, cam.height, fps)
    try:
        for i in range(n_frames):
            travelled = speed_mps * dt * i
            frame_defects = [
                Defect(d.kind, d.x, d.z - travelled, d.half_w, d.half_l)
                for d in defects
            ]
            frame = render_car_frame(cam, frame_defects,
                                     shadow_z=shadow_z0 - travelled)
            writer.write(frame)
    finally:
        writer.close()
    print(f"  car view      -> {path} ({cam.width}x{cam.height}, {n_frames} frames)")
    return path


def make_drone_video(path: Path, n_frames: int = 120, fps: float = 15.0,
                     width: int = 900, height: int = 600,
                     gsd: float = 0.02, speed_mps: float = 5.0) -> Path:
    """Nadir view: orthographic, so scale is uniform and known exactly.

    With gsd metres per pixel supplied to the pipeline, defect areas here have an
    exact expected value in m² — this is the clip to check absolute severity on.
    """
    dt = 1.0 / fps
    road_half_m = 3.0
    road_half_px = int(road_half_m / gsd)
    cx = width // 2

    defects = [
        Defect("pothole", x=-1.0, z=6.0, half_w=0.40, half_l=0.35),
        Defect("water", x=0.3, z=14.0, half_w=1.4, half_l=1.2),
        Defect("mud", x=1.2, z=22.0, half_w=1.1, half_l=1.6),
        Defect("pothole", x=0.0, z=30.0, half_w=0.55, half_l=0.50),
    ]

    writer = _open_writer(path, width, height, fps)
    try:
        for i in range(n_frames):
            travelled = speed_mps * dt * i
            frame = np.asarray(VERGE_BGR, dtype=np.float32) + _noise(
                (height, width), VERGE_SIGMA)

            road = np.zeros((height, width), dtype=bool)
            road[:, cx - road_half_px:cx + road_half_px] = True
            road_layer = np.asarray(ROAD_BGR, dtype=np.float32) + _noise(
                (height, width), ROAD_SIGMA)
            frame[road] = road_layer[road]

            for d in defects:
                z = d.z - travelled
                # Scroll up the frame; bottom of frame is the current position.
                v = height - (z / gsd)
                if not (-50 < v < height + 50):
                    continue
                u = cx + d.x / gsd
                ru, rv = max(1, int(d.half_w / gsd)), max(1, int(d.half_l / gsd))

                blob = np.zeros((height, width), dtype=np.uint8)
                cv2.ellipse(blob, (int(u), int(v)), (ru, rv), 0, 0, 360, 1, -1)
                m = blob.astype(bool) & road
                if not m.any():
                    continue
                if d.kind == "pothole":
                    layer = np.asarray(POTHOLE_BGR, dtype=np.float32) + _noise((height, width), 7.0)
                elif d.kind == "water":
                    layer = np.asarray(WATER_BGR, dtype=np.float32) + _noise((height, width), 0.6)
                else:
                    layer = np.asarray(MUD_BGR, dtype=np.float32) + _noise((height, width), 2.5)
                frame[m] = layer[m]

            writer.write(_clip(frame))
    finally:
        writer.close()
    print(f"  drone nadir   -> {path} ({width}x{height}, {n_frames} frames, "
          f"gsd {gsd} m/px)")
    return path


def _open_writer(path: Path, w: int, h: int, fps: float):
    """Write via ffmpeg at near-lossless quality.

    OpenCV's `mp4v` writer was the first version of this and it quietly ruined
    the fixture: its compression smoothed the fine road texture away completely,
    leaving a surface with *zero* measured texture variance. Since texture is the
    main cue separating road from verge and water from dry surface, the generated
    "road" no longer tested what it was built to test.
    """
    from rdd.utils.ffmpeg import VideoWriter

    path.parent.mkdir(parents=True, exist_ok=True)
    return VideoWriter(path, w, h, fps, crf=8, preset="medium")


def make_scenario_video(path: Path, scenario: str, n_frames: int = 60,
                        fps: float = 15.0, speed_mps: float = 4.0) -> Path:
    """Footage the validity gate is *supposed* to refuse.

    Each scenario reproduces one of the conditions the requirement calls out, so the
    gates can be verified against video rather than only against hand-built contexts:

      flooded    — water covering essentially the whole carriageway
      muddy      — mud covering the whole carriageway
      off_track  — the vehicle has left the road; no road ahead of the bonnet
      no_road    — pointed at open ground, no carriageway at all
      stationary — parked; every frame is a near-duplicate
      glare      — blown-out sun flare
      night      — too dark to assess
    """
    cam = CarCamera()
    dt = 1.0 / fps
    writer = _open_writer(path, cam.width, cam.height, fps)
    # A genuinely parked vehicle produces near-identical frames: the scene texture is
    # fixed and only sensor noise changes. Re-rendering the texture each frame would
    # fake motion that isn't there and the stationary gate would (correctly) not fire.
    frozen = (render_car_frame(cam, [Defect("pothole", -0.8, 12.0, 0.4, 0.35)])
              .astype(np.float32) if scenario == "stationary" else None)
    try:
        for i in range(n_frames):
            travelled = 0.0 if scenario == "stationary" else speed_mps * dt * i
            defects = [Defect("pothole", -0.8, 12.0 - travelled, 0.4, 0.35)]
            if frozen is not None:
                frame = frozen + RNG.normal(0.0, 1.5, frozen.shape).astype(np.float32)
            else:
                frame = render_car_frame(cam, defects).astype(np.float32)

            shape = (cam.height, cam.width)
            if scenario in ("flooded", "muddy"):
                road = car_road_mask(cam)
                bgr = WATER_BGR if scenario == "flooded" else MUD_BGR
                sigma = 0.6 if scenario == "flooded" else 2.5
                cover = _filled(shape, bgr, sigma)
                frame[road] = cover[road]
            elif scenario == "off_track":
                # Verge fills the lower frame: the road is off to one side and does
                # not reach the vehicle.
                cut = int(0.62 * cam.height)
                frame[cut:, :] = _filled(shape, VERGE_BGR, VERGE_SIGMA)[cut:, :]
            elif scenario == "no_road":
                frame = _filled(shape, VERGE_BGR, VERGE_SIGMA).astype(np.float32)
                frame[: int(cam.horizon_row), :] = np.asarray(SKY_BGR, np.float32)
            elif scenario == "glare":
                frame *= 3.2
            elif scenario == "night":
                frame *= 0.06

            writer.write(_clip(frame))
    finally:
        writer.close()
    print(f"  {scenario:<11} -> {path}")
    return path


def degrade(src: Path, dst: Path) -> Path:
    """A deliberately poor copy: soft, noisy, low contrast, some blown frames.

    Exercises the quality stage — this clip should produce a visibly lower
    sharpness median and some dropped frames.
    """
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = _open_writer(dst, w, h, fps)
    i = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            f = frame.astype(np.float32)
            f = 0.55 * f + 60.0                       # crush contrast
            f = cv2.GaussianBlur(f, (0, 0), 1.6)      # soften
            f += RNG.normal(0, 9.0, f.shape)          # sensor noise
            if i % 17 == 0:
                f = cv2.GaussianBlur(f, (0, 0), 6.0)  # motion-blurred frame
            if i % 23 == 0:
                f *= 2.6                              # blown-out frame
            writer.write(_clip(f))
    finally:
        cap.release()
        writer.close()
    print(f"  degraded copy -> {dst}")
    return dst


def to_equirect(src: Path, dst: Path, h_fov: float = 110.0,
                v_fov: float = 70.0, out_w: int = 2560) -> Path | None:
    """Wrap a flat render into an equirectangular frame via ffmpeg v360.

    Lets the same road content exercise the 360 ingest + reprojection path, so
    that path is tested against real road pixels rather than a colour chart.
    """
    import shutil

    if not shutil.which("ffmpeg"):
        print("  (ffmpeg not found — skipping equirect variant)")
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
        "-vf", (f"v360=input=flat:output=equirect:ih_fov={h_fov}:iv_fov={v_fov}"
                f":pitch=30:w={out_w}:h={out_w // 2}:interp=lanczos"),
        "-c:v", "libx264", "-crf", "16", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-an", str(dst),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  (equirect conversion failed: {res.stderr[-300:]})")
        return None
    print(f"  equirect 360  -> {dst} ({out_w}x{out_w // 2})")
    return dst


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/raw", help="output directory")
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--only", choices=["car", "drone", "equirect", "scenarios"],
                   default=None)
    p.add_argument("--degraded", action="store_true",
                   help="also write a low-quality copy of the car view")
    p.add_argument("--scenarios", action="store_true",
                   help="write clips the validity gate should refuse to assess")
    args = p.parse_args(argv)

    out = Path(args.out)
    print(f"Generating synthetic road footage in {out}/")

    car = out / "synthetic_road_car.mp4"
    if args.only in (None, "car", "equirect"):
        make_car_video(car, n_frames=args.frames, fps=args.fps)
    if args.only in (None, "drone"):
        make_drone_video(out / "synthetic_road_drone.mp4",
                         n_frames=args.frames, fps=args.fps)
    if args.only in (None, "equirect"):
        to_equirect(car, out / "synthetic_road_equirect.mp4")
    if args.degraded:
        degrade(car, out / "synthetic_road_car_degraded.mp4")

    if args.scenarios or args.only == "scenarios":
        print("Scenarios the validity gate should refuse:")
        for name in ("flooded", "muddy", "off_track", "no_road", "stationary",
                     "glare", "night"):
            make_scenario_video(out / f"scenario_{name}.mp4", name,
                                n_frames=min(60, args.frames), fps=args.fps)

    print("\nRun the pipeline on these with the matching viewpoint:")
    print("  python run.py --input data/raw/synthetic_road_car.mp4 --view car_flat")
    print("  python run.py --input data/raw/synthetic_road_drone.mp4 --view drone_nadir")
    print("  python run.py --input data/raw/synthetic_road_equirect.mp4 --view car_360")
    return 0


if __name__ == "__main__":
    sys.exit(main())
