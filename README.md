# Road Defect Detection Pipeline (360° rural-road video)

Detect **potholes, water-logging, ruts/erosion, and cracks** in 360° video shot
from a survey vehicle on unpaved rural roads, and produce:

1. an **annotated video** (segmentation masks + track IDs + a live *unique*-count HUD), and
2. a **report** (CSV per unique defect, JSON counts, HTML/PDF summary) with severity
   and GPS location when available.

The pipeline is **config-driven** (`config.yaml`) with a single entrypoint
(`run.py`) and runs **without GPS** and **without the optional depth stage**.

---

## Why this pipeline is shaped the way it is

- **Unpaved roads have no clean "normal" surface.** Defect-vs-background is fuzzy,
  so we use **segmentation masks**, not boxes, and fine-tune on *your* labels.
- **360° footage must be flattened first.** We reproject equirectangular → a
  virtual "road camera" (pitched down) with FFmpeg's `v360` filter before detection.
- **Moving-camera video is hugely redundant.** We sample frames by **distance
  traveled** (GPS or optical-flow odometry) and **count unique tracked objects**,
  never per-frame detections.
- **Public models are asphalt-trained.** Use them only as a **warm start**, then
  fine-tune. Expect a real domain gap.

---

## Install

```bash
python -m venv .venv && . .venv/Scripts/activate    # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**External binary required:** [FFmpeg](https://www.gyan.dev/ffmpeg/builds/) **with the
`v360` filter** (needed for 360→flat and `.insv` conversion).

```powershell
winget install Gyan.FFmpeg          # Windows
# macOS: brew install ffmpeg   |   Linux: sudo apt install ffmpeg
ffmpeg -hide_banner -filters | findstr v360     # verify (unix: | grep v360)
```

Optional: `exiftool` (richer embedded GPS), and Video-Depth-Anything for the depth
stage (installed from source — see *Depth* below).

---

## Quick start (end-to-end)

```bash
python run.py --input path/to/video.mp4 --config config.yaml --output out/
```

Outputs land in `out/<run.name>/`:
- `annotated.mp4` — masks + track IDs + running unique-count overlay
- `defects.csv` — one row per **unique** defect (id, class, frames, area, severity, lat/lon)
- `summary.json` — unique counts per class + totals (and raw per-frame count, clearly labeled)
- `report.html` (or `report.pdf`) — counts, sample crops, GPS note
- `manifest.json` — config, package/tool versions, git commit, per-stage outputs
- `run.log`

### Insta360 `.insv` / `.insp` input
Raw dual-fisheye files aren't stitched. Two options:
- **Best quality:** open in **Insta360 Studio**, export **equirectangular** `.mp4`
  (2:1, e.g. 5760×2880), run the pipeline on that.
- **Automatic (lower quality):** set `ingest.auto_convert_insv: true` — the pipeline
  attempts an FFmpeg `v360=dfisheye→equirect` remap (tune `ih_fov` in
  `ingest/video.py` for your lens).

---

## Run stages independently

```bash
python run.py preprocess --input video.mp4        # 360->flat + distance sampling
python run.py annotate   --frames data/rectified  # active-learning: which frames to label first
python run.py train      --labels data/labels     # fine-tune (segment-safe split)
python run.py infer      --input out/default/preprocess/rectified.mp4 --weights best.pt
```

---

## Labeling workflow (SAM-assisted)

1. `python run.py preprocess ...` → rectified frames in `data/rectified/`.
2. `python run.py annotate --frames data/rectified` → `out/frames_to_label.txt`
   (most diverse/uncertain frames first — label these to get the most signal per click).
3. Use `rdd.annotate.sam_label` to point/box-prompt SAM on a frame, propagate the
   mask, and export YOLO-seg polygons:
   ```python
   from rdd.config import load_config
   from rdd.annotate.sam_label import load_sam, segment_frame, masks_to_yolo_seg, write_label
   cfg = load_config("config.yaml"); sam = load_sam(cfg)
   res = segment_frame(sam, "data/rectified/frame_0000123.jpg", boxes=[[x1,y1,x2,y2]])
   write_label(masks_to_yolo_seg(res, cls_id=0), "data/rectified/frame_0000123.jpg", Path("data/labels/labels"))
   ```
4. Lay out labels as `data/labels/images/*.jpg` + `data/labels/labels/*.txt`.

---

## Training (no frame leakage!)

`model/split.py` splits **by road segment / time range**, never randomly — adjacent
frames are near-identical and a random split leaks them across train/val/test,
inflating metrics. Config forbids `split.mode: random`.

Warm-start from a road-damage checkpoint via `model.warm_start_weights`
(RDD2022 India subset or a Roboflow road-defect model), then fine-tune.

---

## Depth & severity (optional)

`depth.enabled: false` by default. When enabled with Video-Depth-Anything installed,
severity = `f(mask_area, estimated_depth)`; otherwise severity degrades gracefully
to mask-area-only. Water-logging depth is unknowable under reflections — it's a
separate class and flagged conservatively, never depth-scored.

```bash
pip install git+https://github.com/DepthAnything/Video-Depth-Anything.git
# then set depth.enabled: true  (wire the model call in depth/estimator.py)
```

---

## Config cheatsheet (`config.yaml`)

| Key | Meaning |
|---|---|
| `preprocess.reproject.pitch_deg` | virtual camera tilt (−30° looks down at road) |
| `preprocess.reproject.h_fov_deg` / `v_fov_deg` | flat-view field of view |
| `preprocess.sampling.mode` | `distance` (recommended) / `time` / `every_n` |
| `model.arch` / `fallback_arch` | `yolo26-seg` → falls back to `yolo11-seg` if unavailable |
| `model.classes` | `pothole, water_logging, rut_erosion, crack` |
| `model.train.split.mode` | `segment` / `time` (**never** `random`) |
| `inference.tracker` | `botsort` / `bytetrack` |
| `inference.min_track_len` | frames a track must persist to count as a real defect |
| `depth.enabled` | toggle depth/severity |
| `report.format` | `html` / `pdf` |

---

## Module map

```
run.py                     CLI entrypoint (end-to-end + per-stage subcommands)
config.yaml                all tunables
src/rdd/
  config.py                load + validate config
  pipeline.py              end-to-end orchestration
  utils/                   logging, device, ffmpeg wrapper, geo, run manifest
  ingest/                  format detect, .insv->equirect, GPS/telemetry
  preprocess/              360->flat (v360), IPM bird's-eye, distance sampling
  annotate/                SAM-assisted labeling + active-learning frame picker
  model/                   YOLO loader (fallback), segment split, training
  depth/                   optional depth backend + severity scoring
  inference/               detect+track, unique counting, annotated render
  report/                  CSV / JSON / HTML / PDF
```

## Reproducibility
Seeds are set (`run.seed`); `manifest.json` records config, package/binary versions,
and git commit for every run.

## Reference repos
ultralytics · oracl4/RoadDamageDetection · FarzadNekouee/YOLOv8_Pothole_Segmentation ·
DepthAnything/Video-Depth-Anything · NitishMutha/equirectangular-toolbox · roboflow/supervision
