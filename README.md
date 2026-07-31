# Road Defect Detection Pipeline

Detect **potholes, water-logging, ruts/erosion, and cracks** in road survey video
from a **vehicle (360° or flat) or a drone (nadir)**, on unpaved rural roads, and
produce:

1. an **annotated video** — road outline, hatched unassessable areas, defect masks,
   track IDs, live *unique*-count HUD;
2. a **report** (CSV per unique defect, JSON, HTML/PDF) with severity, GPS where
   available, and — critically — **how much of the road could not be assessed**.

Config-driven (`config.yaml`), single entrypoint (`run.py`), runs without GPS and
without the optional depth stage.

---

## The three problems this pipeline is built around

**1. Survey footage is not clear.** Detail is lost long before the detector runs.
The dominant cause is geometric, not photographic: an equirectangular frame
spreads 360° across its width, so a 110° view carries only `src_width × 110/360`
pixels of real angular detail. Downscaling below that throws information away
permanently. `out_width: auto` matches the source instead. Beyond that, quality is
*measured* per clip and frames too poor to trust are excluded from analysis rather
than quietly producing confident nonsense.

**2. The road is covered in mud and water.** This is not a detection problem, it
is an **observability** problem: *you cannot inspect a surface you cannot see.* A
pothole under muddy water is invisible, and a detector reporting "nothing there"
is not observing an intact road — it is failing to observe anything. So water and
mud are treated as an **occlusion mask**, defects under it are marked
`indeterminate` instead of being severity-scored, and every report states the
fraction of road surface that could not be assessed.

**3. Two very different viewpoints.** A survey car and a nadir drone share almost
no geometry: road shape in frame, perspective, and how ground scale is recovered
all differ. A `ViewProfile` centralises those differences so stages ask for what
they need instead of branching on camera type.

**Everything starts with segmenting the road.** Defects are only meaningful
*on the road surface*. Without that constraint a roadside puddle, a dark bush or a
reprojection artifact in the sky are all candidate potholes. The road mask also
provides the denominator for "what fraction of this road is damaged / unassessable",
which is what a condition survey actually reports.

---

## Pipeline

```
ingest ─→ viewpoint ─→ reproject ─→ quality ─→ sampling ─→ geometry ─→ scale
                                                                         │
   ┌─────────────────────────────────────────────────────────────────────┘
   ↓
inference ─→ ROAD SEG ─→ SURFACE ─→ VALIDITY GATE ─→ detect + track + gate
                                          │                    │
                          (blocked: no detections)             │
                                             severity (abstains) ─→ report
```

| Stage | What it does | Key output |
|---|---|---|
| `ingest` | format detect, `.insv`→equirect, GPS | equirect/flat video |
| `viewpoint` | resolve camera profile + road prior + ground scale | `ViewProfile` |
| `reproject` | 360→flat at native angular resolution (skipped if already flat) | `rectified.mp4` |
| `quality` | learn the clip's sharpness/noise distribution; derive enhancement | `QualityProfile`, `EnhanceSpec` |
| `sampling` | frames for labeling, by distance travelled; skips unusable frames | `data/rectified/*.jpg` |
| `geometry` | calibrate the camera from the vanishing point; derive per-class **assessment zones** | `CameraModel`, `ZoneSet` |
| `scale` | metres per pixel (drone GSD, or IPM Jacobian for car) | `AreaScaler` |
| `roadseg` | segment the drivable surface | road mask + appearance baseline |
| `surface` | classify water / mud / dry → **occlusion mask** + plausibility | `SurfaceMap` |
| `validity` | **decide whether the frame may be assessed at all** | `FrameVerdict`, route coverage |
| `detect_track` | YOLO-seg + BoT-SORT, **gated to the road and to each class's zone** | tracks, occlusion flags |
| `severity` | absolute m² bins where scale is known; **abstains** under occlusion | `SeverityReport` |
| `report` | CSV / JSON / HTML / PDF | `defects.csv`, `report.html` |

---

## Install

```bash
python -m venv .venv && . .venv/Scripts/activate    # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**FFmpeg is required** — for the `v360` filter (360→flat, `.insv`) *and* as the
H.264 encoder for output video.

```powershell
winget install Gyan.FFmpeg          # Windows
# macOS: brew install ffmpeg   |   Linux: sudo apt install ffmpeg
ffmpeg -hide_banner -filters | findstr v360     # verify (unix: | grep v360)
```

Optional: `exiftool` (richer embedded GPS), Video-Depth-Anything (depth stage).

---

## Quick start

```bash
# Vehicle, 360 equirectangular
python run.py --input road.mp4 --view car_360

# Vehicle, ordinary forward camera
python run.py --input road.mp4 --view car_flat

# Drone, straight down — with ground scale, so severity is absolute m²
python run.py --input flight.mp4 --view drone_nadir \
              --set view.drone.gsd_m_per_px=0.02
```

`--set KEY=VALUE` overrides any config key (repeatable, values parsed as YAML so
types are preserved). Useful for sweeping a threshold without editing
`config.yaml`:

```bash
python run.py roadseg --input road.mp4 --n 6 \
  --set roadseg.classical.distance_tau=3.0 \
  --set surface.mud.min_warmer_z=3.5
```

Outputs land in `out/<run.name>/`:

| File | Contents |
|---|---|
| `annotated.mp4` | road outline, hatched water/mud, masks, IDs, HUD; unassessed frames banner-marked |
| `defects.csv` | one row per **unique** defect, incl. `area_m2`, `assessable`, `occluded_frac` |
| `summary.json` | unique counts, indeterminate counts, full per-stage pipeline summary |
| `report.html` | headline counts **and** unassessable-% banner, severity basis, coverage |
| `manifest.json` | config, versions, git commit, enhancement fingerprint, per-stage records |
| `run.log` | |

### No real footage yet? Generate some

```bash
python tools/make_synthetic_road.py --out data/raw --degraded
```

Writes perspective-correct car, nadir-drone, equirect and degraded clips with
known ground truth — potholes, puddles, mud, and a *multiplicative shadow* decoy
that a naive "dark pixels = mud" detector fails on. Use these to tune thresholds
before your own footage arrives.

> The pre-existing `data/raw/synthetic_equirect.mp4` is a **colour test pattern**,
> not a road. It exercises the v360 geometry and nothing else — detector numbers
> from it are meaningless.

---

## Tuning the road & surface masks (do this first)

```bash
python run.py roadseg --input video.mp4 --view car_flat --n 8
```

Writes annotated previews to `out/roadseg_preview/` and prints per-frame road
coverage, mask confidence, and water/mud fractions. **Green outline = road;
hatched blue = water, brown = mud.** If the road outline is wrong, nothing
downstream can be right — fix `view.road_prior` before touching anything else.

```bash
python run.py quality --input video.mp4 --csv out/quality.csv
```

Reports the clip's sharpness/contrast/noise distribution, which frames would be
dropped and why, and the enhancement that would be applied.

---

## How road segmentation works

Default backend is `classical` — colour+texture similarity seeded by the viewpoint's
geometric prior. Chosen because it is real-time on CPU with no extra dependencies,
and because on unpaved roads there is no fixed "road colour": laterite, gravel and
dust all differ, and all shift with light. So nothing is compared against an
absolute threshold:

1. Measure what the road looks like in a small high-confidence region (the eroded
   prior) — median/MAD, not mean/std, because that region deliberately contains
   the outliers we are hunting.
2. Blend that into a **running clip baseline**. Necessary because the seed is a
   geometric guess and can be dominated by the very thing we are looking for — a
   puddle filling the near field makes the seed describe *water*, after which dry
   road becomes the outlier. Road material is a property of the clip; contamination
   is transient.
3. Grow to nearby pixels resembling the baseline.
4. **Fill enclosed holes.** Load-bearing, not cosmetic: potholes and puddles are
   exactly the pixels that fail step 3, and the road mask must contain the defects
   sitting on it or gating rejects everything it exists to keep.
5. Temporally smooth. A flickering road mask breaks defect tracks and inflates the
   unique count.

It reports when it fails. If the appearance model separated nothing, coverage is
implausible, or too little of the seed survived, the frame falls back to the pure
geometric prior with `fell_back=True` and low confidence — and **gating is
suspended below `roadseg.gating.min_confidence`**, so a low-confidence guess never
discards real defects.

| Backend | Use when |
|---|---|
| `classical` | default; CPU-friendly, no extra deps |
| `geometric` | prior only; crude but cannot fail |
| `sam` | most accurate; **seconds per frame on CPU** — spot checks, not full runs |
| `none` | no road constraint; expect off-road false positives |

---

## Refusing to assess: the validity gate

The rule is that when the road cannot be seen, the pipeline produces **no
detections** and says why. A defect list from a frame where the road is buried is
worse than nothing, because a reader cannot tell "inspected and clean" from "never
inspected".

This is also the cheapest **precision** lever available, and the only one that needs
no training data: almost all false positives come from degraded frames. Tightening
the gates raises precision and lowers coverage — both numbers are reported, because
a precision figure over an undisclosed subset of the route means nothing.

| Gate | Blocks when | Action |
|---|---|---|
| `road_found` | road mask unconfident or implausibly small | BLOCK (prior-only → DEGRADE) |
| `surface_plausible` | the whole surface reads as water / mud / vegetation | BLOCK |
| `road_buried` | water+mud over >60% of the road | BLOCK (>25% → DEGRADE) |
| `off_track` | road doesn't reach the bonnet, or is far off centre | BLOCK |
| `ego_motion` | stationary, or reversing (flow contracting toward the VP) | BLOCK (sharp turn / pitching → DEGRADE) |
| `traffic` | vehicles cover >55% of the assessment zone | BLOCK (otherwise MASK them out) |
| `image_condition` | sun glare, night, tunnel | BLOCK (dusk → DEGRADE) |
| `windscreen` | dirt / rain / wiper across the lens | BLOCK (small → MASK) |
| `assessment_zone` | too little assessable road left after all exclusions | BLOCK |

Three actions, deliberately distinguished: **BLOCK** refuses the frame, **DEGRADE**
assesses it but marks results low-confidence, **MASK** removes a region (a car ahead)
and keeps the rest. Blanket-blocking any frame containing a vehicle would discard
most of a real survey.

`surface_plausible` closes a structural blind spot in `road_buried`. That gate
measures water and mud *relative to the road baseline*, which works for a puddle on
visible road — but when a contaminant covers the **entire** carriageway it *becomes*
the baseline, every pixel matches it, and the relative measurement reports a clean,
dry road. So this one gate uses **absolute** appearance, the single deliberate
exception to the relative-statistics rule used everywhere else. It is set loose and
only ever used to catch the catastrophic case, never to grade severity.

```bash
python run.py validity --input dashcam.mp4 --view car_flat --no-traffic --stride 4
```

Prints the per-frame verdict timeline and a route summary. Verified on purpose-built
clips (`tools/make_synthetic_road.py --scenarios`): flooded, mud-covered, off-track
and no-road footage all yield **0% assessable**, while clean footage yields 100%.

## Assessment zones: where each class can actually be seen

A hairline crack is a few millimetres wide. At 25 m ahead one pixel of a 1080p
dashcam covers several centimetres of road *along* the direction of travel, so the
crack is not faint — it is **absent**, below the sampling limit. Recorded as "no
defect" that is a false negative dressed as a clean road.

So each class gets a distance band derived from a required ground resolution, and
anything outside it is reported **not assessed**. The bands are computed from the
camera model, not typed in — change the mount height or the resolution and they move
on their own. The same bands raise precision, since far-field pixels are where
aliasing and compression noise invent thin-line detections.

Ground resolution is strongly anisotropic in a forward view: at 15 m a pixel may span
12 mm across the road but 90 mm along it. Both are computed and the **worst** is used
for the budget, because averaging them would hide the direction that actually limits
transverse crack detection.

Camera pitch and yaw are recovered per clip from the **vanishing point** (fitted from
the road-mask edges), because dashcams get knocked and remounted and a 2° pitch error
is a large range error. Camera **height must be measured** — it does not affect the
vanishing point and cannot be recovered this way, and it linearly scales every
distance and area.

## How mud & water detection works

Every cue is an **illumination invariant**, because the hardest false positive in
road inspection is shade being read as contamination.

| Cue | Feature | Why |
|---|---|---|
| smoothness | `rtex` = texture ÷ brightness | A shadow scales detail and brightness together, so the *ratio* survives. Water is specular and destroys detail, so it collapses. Keying on absolute texture reads deep shade as standing water. |
| warmth | `cr` = red chromaticity in **linear** RGB | Under a shadow the reflected spectrum is unchanged and linear RGB scales by one factor, so chromaticity is exactly invariant. Mud is a different material, so it shifts. |
| shadow | dark **and** both invariants preserved | Explicitly detected and excluded — otherwise every shadow becomes "mud" and the unassessable figure is junk. |

LAB `a`/`b` are deliberately *not* used for this. They are computed from
gamma-encoded values, so merely darkening a neutral grey moves them several units
— measured at 8 units of `b` on shadowed road, larger than a real mud signal. That
is a colour-space artifact wearing the costume of evidence.

Detection uses region-averaged z-scores (per-pixel scores would flag a third of a
clean road on sensor noise alone), then recovers true extent by growing confident
cores into a lightly-smoothed map — **hysteresis**, because detection confidence
and measured area are different jobs and the measured area drives the headline
number.

**Stated limitation:** if the *entire* road is uniformly covered, the covering
becomes the baseline and relative detection finds nothing. This case is detected
heuristically and warned about, but it cannot be resolved from relative statistics
alone. Check the annotated video.

---

## Severity: absolute where possible, and it says which

With ground scale available, severity uses **fixed physical thresholds in m²** and
is comparable between videos.

| Viewpoint | How to supply scale |
|---|---|
| `drone_nadir` | `view.drone.gsd_m_per_px`, or `altitude_m` + `camera.focal_mm` + `camera.sensor_width_mm` |
| `car_360` / `car_flat` | `preprocess.ipm.enabled: true` + `preprocess.ipm.ground_extent_m: [width, length]` — the real-world size of the `src_points` trapezoid |

Without it, severity falls back to min-max normalisation **across the run only**,
which means the largest defect present is always "high" even if trivial, and the
same road shot twice can score differently. The report states this explicitly
rather than letting relative numbers look physical.

IPM does **not** warp the video (that would resample away fine detail and produce
a bird's-eye output no reviewer can sanity-check). It is used analytically: the
homography's local area Jacobian, `|det H| / w³`, gives ground area per pixel, so
defects are measured in m² while still being detected and drawn in the natural
camera view. Without this, a pothole 30 m away covers a fraction of the pixels of
an identical one at 5 m — pixel area is size *confounded with range*.

---

## Video quality

Measurement first. Thresholds are learned **per clip** (relative to its own median,
with an absolute floor as backstop) because variance-of-Laplacian scales with
resolution, texture and contrast — a cutoff tuned on one camera silently rejects
every frame from another.

Frames judged unusable are **not detected on**, but are still written to the
annotated video with a banner. The output stays a faithful record of the survey
including where it could not see, and the report counts what was skipped and why.

Enhancement (`white balance → denoise → CLAHE → upscale → unsharp`) is a pure
function of `(frame, EnhanceSpec)`, applied identically to the labeling frames and
to inference input. The spec's **fingerprint is recorded in the manifest**: if you
label with one setting and infer with another, that mismatch becomes visible rather
than looking like a modelling problem.

---

## Run stages independently

```bash
python run.py preprocess --input video.mp4              # reproject + quality + sampling
python run.py quality    --input video.mp4 --csv q.csv  # measure only
python run.py roadseg    --input video.mp4 --n 8        # mask previews for tuning
python run.py validity   --input video.mp4 --no-traffic # per-frame assessability
python run.py annotate   --frames data/rectified        # which frames to label first
python run.py train      --labels data/labels           # fine-tune (segment-safe split)
python run.py infer      --input rectified.mp4 --weights best.pt
```

`--view`, `--device`, `--config` and `--output` work on every subcommand.

---

## Labeling workflow (SAM-assisted)

1. `python run.py preprocess ...` → enhanced frames in `data/rectified/`.
   **Label these, not the raw video** — they match what the detector will see.
2. `python run.py annotate --frames data/rectified` → `out/frames_to_label.txt`,
   most diverse/uncertain first.
3. Point/box-prompt SAM and export YOLO-seg polygons:
   ```python
   from rdd.config import load_config
   from rdd.annotate.sam_label import load_sam, segment_frame, masks_to_yolo_seg, write_label
   cfg = load_config("config.yaml"); sam = load_sam(cfg)
   res = segment_frame(sam, "data/rectified/frame_0000123.jpg", boxes=[[x1,y1,x2,y2]])
   write_label(masks_to_yolo_seg(res, cls_id=0), "data/rectified/frame_0000123.jpg",
               Path("data/labels/labels"))
   ```
4. Lay out as `data/labels/images/*.jpg` + `data/labels/labels/*.txt`.

## Training (no frame leakage)

`model/split.py` splits **by road segment / time range**, never randomly — adjacent
frames are near-identical, and a random split leaks them across train/val/test and
inflates metrics. `split.mode: random` is rejected by config validation.

Warm-start from a road-damage checkpoint via `model.warm_start_weights` (RDD2022
India subset, or a Roboflow road-defect model), then fine-tune.

---

## Config cheatsheet

| Key | Meaning |
|---|---|
| `view.profile` | `car_360` / `car_flat` / `drone_nadir` |
| `view.road_prior` | where the road is expected — trapezoid (car) or band (drone) |
| `view.drone.*` | altitude/optics → GSD → **absolute m² severity** |
| `preprocess.reproject.out_width` | `auto` preserves the source's angular detail |
| `preprocess.reproject.preserve_aspect` | derive height from FOVs for square pixels |
| `preprocess.reproject.lossless` | zero-loss intermediate (large files) |
| `preprocess.ipm.ground_extent_m` | real size of the trapezoid → m² measurement |
| `quality.assess.drop_unusable` | skip detection on frames too poor to trust |
| `quality.enhance.*` | **must match between labeling and inference** |
| `roadseg.backend` | `classical` / `geometric` / `sam` / `none` |
| `roadseg.stride` | recompute the road mask every N frames (speed) |
| `roadseg.gating.mode` | `gate` (post-filter) / `mask` (pre-mask) / `off` |
| `geometry.camera.height_m` | **measure this** — it scales every distance and area |
| `geometry.camera.auto_pitch_from_vp` | recover pitch/yaw per clip from the vanishing point |
| `geometry.zones.required_gsd_m` | resolution each class needs → its assessment range |
| `validity.enabled` | master switch for refusing unassessable frames |
| `validity.road_buried.block_above_frac` | water/mud coverage that blocks a frame |
| `validity.traffic.enabled` | mask vehicles out of the road region (COCO, no labels) |
| `surface.occlusion_policy` | `abstain` / `flag` / `exclude` |
| `surface.occluder_classes` | classes that *are* the occluder (`water_logging`) |
| `severity.absolute_bins_m2` | physical severity thresholds when scale is known |
| `inference.imgsz` | raise toward native width for small cracks |
| `inference.min_track_len` | frames a track must persist to count as a defect |

---

## Reading the report

- **Unique defects** — one physical defect per confirmed track. Never a per-frame
  count; the raw per-frame total is shown separately and clearly labelled.
- **Indeterminate** — detected but hidden under water/mud. Real defects, not
  measurable. Amber-outlined in the crops, `assessable=no` in the CSV.
- **Unassessable %** — the fraction of road surface obscured. If this is high, the
  low defect count is a statement about visibility, not about road condition.
- **Route coverage** — the fraction of *frames* that were assessed at all. Excluded
  frames were never inspected; they are not evidence of intact road.
- **Assessment zones** — the range over which each class was actually assessable.
  Beyond it, that class was not assessed.
- **Severity basis** — `absolute_m2` is comparable between runs; `relative_px` is
  not, and says so.

---

## Tests

```bash
python -m pytest tests/ -q
```

205 tests. The ones that matter most encode the design invariants: hole-filling
keeps defects inside the road mask; a multiplicative shadow is not mud; a clean
road reports ~0% occluded; water-logging is never occluded by itself; a defect
under water is never severity-scored; absolute severity is comparable across runs
while relative severity is not; flooded/mud-covered/off-track clips yield zero
detections with a stated reason; a fully-covered road is caught by absolute
plausibility where relative measurement is blind; and clean footage stays 100%
assessable, so the gates are not merely blocking everything.

## Reproducibility

Seeds are set (`run.seed`). `manifest.json` records config, package/binary
versions, git commit, ground-scale basis, and the enhancement fingerprint for
every run.

## Module map

```
run.py                     CLI (end-to-end + per-stage subcommands)
config.yaml                all tunables
tools/make_synthetic_road.py  test-footage generator with known ground truth
src/rdd/
  config.py                load + validate (rejects silently-wrong geometry)
  viewpoint.py             ViewProfile: geometry, road prior, ground scale
  pipeline.py              end-to-end orchestration
  utils/                   logging, device, ffmpeg (+VideoWriter), geo, manifest
  ingest/                  format detect, .insv->equirect, GPS/telemetry
  preprocess/              reproject (360->flat), ipm (area Jacobian), scale, sampling
  geometry/                calibration (ground model, GSD), autocal (VP), zones
  quality/                 metrics (adaptive thresholds), enhance (shared spec)
  roadseg/                 ops, geometric prior, classical, temporal, sam
  surface/                 water/mud condition -> occlusion mask + plausibility
  validity/                frame verdict, gates, ego-motion, traffic occlusion
  annotate/                SAM-assisted labeling + active-learning frame picker
  model/                   YOLO loader (fallback), segment split, training
  depth/                   optional depth backend + severity (with abstention)
  inference/               detect+track+gate, unique counting, render
  report/                  CSV / JSON / HTML / PDF
```

## Reference repos
ultralytics · oracl4/RoadDamageDetection · FarzadNekouee/YOLOv8_Pothole_Segmentation ·
DepthAnything/Video-Depth-Anything · NitishMutha/equirectangular-toolbox · roboflow/supervision
