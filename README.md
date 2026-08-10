# Road Defect Detection Pipeline

Detect and classify **potholes, longitudinal & transverse cracks, alligator/fatigue
cracking, ravelling, rutting, edge damage/shoulder erosion, drainage issues and
water-logging** in dashcam or drone road-survey video, graded to IRC/PMGSY, and
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

## Install — one command

The setup script installs **everything**: Python (if missing), FFmpeg with the
`v360` filter, a virtualenv, PyTorch (CUDA build if you have an NVIDIA GPU, CPU
build otherwise), all requirements — then verifies it by importing every stage and
running the test suite. Safe to re-run; each step is skipped if already satisfied.

**Windows (PowerShell):**
```powershell
.\setup.ps1                 # add -Cpu to force the CPU build of PyTorch
```
If PowerShell blocks the script: `powershell -ExecutionPolicy Bypass -File setup.ps1`

**Linux / macOS:**
```bash
chmod +x setup.sh rdd.sh
./setup.sh                  # add --cpu to force the CPU build
```

Then check it worked:
```
.\rdd.ps1 doctor        # Windows
./rdd.sh doctor         # Linux/macOS
```
`doctor` prints versions, whether CUDA is live, whether FFmpeg has `v360`, and —
usefully — **what your configured camera can actually resolve**, per class.

<details><summary>Manual install, if you'd rather</summary>

```bash
python -m venv .venv && . .venv/bin/activate   # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
# FFmpeg (required, needs the v360 filter):
#   Windows: winget install Gyan.FFmpeg
#   macOS:   brew install ffmpeg
#   Linux:   sudo apt install ffmpeg
```
Optional: `exiftool` (richer embedded GPS), Video-Depth-Anything (depth stage).
</details>

---

## If a run is slow

The GPU is usually **not** the bottleneck, so raising `imgsz` makes things slower, not
faster. The per-frame cost is dominated by CPU work — road segmentation, surface
analysis, optical flow — with the GPU idle in between. The lever that matters is doing
that work on fewer frames:

```bash
./rdd.sh detect source=road.mp4 preset=fast     # ~2.5x, detects every 3rd frame
./rdd.sh detect source=road.mp4 preset=turbo    # first look at long footage
```

At 30 fps and 30 km/h the vehicle advances ~28 cm per frame, so consecutive frames
re-inspect the same tarmac — `frame_stride: 3` costs very little real coverage. Skipped
frames are still written to the annotated video, so the output stays a complete record.

`min_track_len` scales automatically with the stride. It is a requirement about
persistence *in time* but is counted in processed frames, so without scaling, raising
the stride would quietly make it stricter and drop short-lived defects from the report.

## Using a public checkpoint

There is no fine-tuned model in this repo yet. The closest public fit is
[rezzzq/yolo12s-road-damage-rdd2022](https://huggingface.co/rezzzq/yolo12s-road-damage-rdd2022)
(MIT), whose RDD2022 classes cover four of the nine categories:

```yaml
model:
  class_map:            # translate the checkpoint's OWN names onto our taxonomy
    D00: longitudinal_crack
    D10: transverse_crack
    D20: alligator_crack
    D40: pothole
    Repair: null        # a past repair is not a defect
```

**`class_map` is not optional for a third-party checkpoint.** Detections carry an
integer class id, so without it ids resolve *positionally*: a 5-class RDD2022 model
against these 9 classes reports its `D00` (a longitudinal crack) as `pothole`, purely
because both sit at index 0 — silently, with no error.

Note it is a **detection** model, not segmentation. The pipeline handles boxes, but
defect area becomes the box area, which overestimates badly for a thin diagonal crack.
Alligator cracking also can't be confirmed by counting enclosed cells in a rectangle,
so the model's own `D20` label is trusted instead. For real m² areas you want a
segmentation model — e.g.
[keremberke/yolov8n-pothole-segmentation](https://huggingface.co/keremberke/yolov8n-pothole-segmentation)
for potholes alone — or your own `rdd train` run.

## Try it on Colab (GPU, no local install)

`notebooks/colab_inference.ipynb` runs a trial inference on real footage with GPU
acceleration: pulls a video from Google Drive, loads weights (Drive / URL / upload),
lets you set camera geometry, previews the road mask before committing, then produces
the annotated video, `defects.csv`, `segments.csv` and `report.html` — all downloadable
or written back to Drive.

For **RF-DETR Medium** training on Colab, see `notebooks/colab_rfdetr_train.ipynb`.
On a dedicated GPU VM, prefer the headless scripts in **RF-DETR Stage 1 (headless VM)** below.

Open it from GitHub with *Open in Colab*, or upload the `.ipynb` directly.

> The repo is private, so the notebook clones with a GitHub token from Colab Secrets
> (`GITHUB_TOKEN`). There is a zip-upload fallback if you would rather not use one.

Mid-session, the **Update** cell (right after Clone) pulls new commits in place rather
than re-cloning, so the downloaded video, weights and form settings survive. It also
clears cached `rdd.*` modules — without that, in-process cells keep running the old
code even though the files on disk changed.

## Everyday commands

Ultralytics/Roboflow-style `key=value` flags. The wrapper finds the virtualenv
itself — no activation needed. Windows uses `.\rdd.ps1`, Linux/macOS `./rdd.sh`;
the tasks and keys are identical.

```bash
./rdd.sh demo                                   # end-to-end on generated footage
./rdd.sh doctor                                 # is my environment sane?

./rdd.sh check    source=road.mp4 view=car_flat # road mask previews  <- DO THIS FIRST
./rdd.sh detect   source=road.mp4 view=car_flat # full pipeline -> video + report
./rdd.sh validity source=road.mp4               # which frames are assessable, and why not
./rdd.sh quality  source=road.mp4               # sharpness / exposure / noise

./rdd.sh preprocess source=road.mp4             # rectify + sample frames to label
./rdd.sh label    frames=data/rectified         # which frames to label first
./rdd.sh train    data=data/labels epochs=100   # fine-tune on your labels
./rdd.sh val      truth=ground_truth.csv        # measure precision, pick thresholds
```

| Key | Meaning |
|---|---|
| `source=` | input video |
| `view=` | `car_flat` (dashcam) / `car_360` / `drone_nadir` |
| `weights=` | trained `.pt` |
| `conf=` `imgsz=` `device=` | confidence, inference size, `cpu`/`cuda` |
| `preset=` | `fast` (~2.5x) / `turbo` (first look) / `accurate` |
| `name=` `project=` | output goes to `<project>/<name>/` |

**Any key in `config.yaml` works as an override** — anything with a dot passes
straight through, so nothing needs editing to experiment:

```bash
./rdd.sh detect source=road.mp4 geometry.camera.height_m=1.35 geometry.camera.h_fov_deg=95
./rdd.sh check  source=road.mp4 roadseg.classical.distance_tau=3.0 surface.mud.min_warmer_z=3.5
```

> **Measure your camera height and field of view.** They are what convert pixels to
> metres, so a wrong height silently rescales every area, and the FOV sets how far
> ahead each class can be assessed at all. `doctor` shows the resulting ranges.

---

## Quick start (direct `run.py`)

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
| `segments.csv` | per-100 m chainage rollup with a condition grade and coverage |
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

## Capture geometry: what actually limits crack detection

Worth knowing before buying cameras, because the binding constraint is physical and
no amount of modelling moves it.

Ground resolution in a forward view is strongly **anisotropic**, and the *longitudinal*
direction is the limit. It degrades as

```
dz/dv  ≈  z² / (h · f)        z = range, h = camera height, f = focal length in px
```

so it worsens with the **square** of distance. At 1080p / 78° / 1.3 m, a 5 mm/px budget
is met only out to about 2.6 m. Measured with `run.py validity`/the zone table:

| Change | Crack band (5 mm/px) | Verdict |
|---|---|---|
| 1080p → 4K | 2.6 m → 3.8 m | helps, sub-linearly |
| 78° → 35° FOV | 2.6 m → 4.3 m | **the effective lever** |
| 1.3 m → 3 m mount | band **vanishes** | counter-productive |

Mounting higher improves resolution at a given range but pushes the *nearest visible
ground* away faster, so "can see it" and "can resolve it" stop overlapping — the
resolvable band ends up below the bottom of the frame. The pipeline detects that and
reports the class unachievable rather than emitting a zero-depth zone.

Practical guidance: **narrow the field of view** (or add a second, longer-focal camera
aimed at the near road), keep the mount low, and accept that hairline cracks are
assessed in a short band close to the vehicle. Potholes and edge damage, needing only
20 mm/px, get a much longer band.

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
python run.py evaluate   --defects out/default/defects.csv --truth gt.csv
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

### RF-DETR Stage 1 (headless VM)

For **RFDETRMedium** Stage-1 training over SSH (e.g. RTX 5090 32GB), use the
scripts under `scripts/` — no Colab/Drive required. Defaults: CRRI-only COCO,
`batch_size=16`, `grad_accum=1`, `epochs=50`, early stopping, checkpoints in
`runs/rfdetr_stage1`.

```bash
cp .env.example .env          # paste ROBOFLOW_API_KEY
./scripts/setup_rfdetr_vm.sh  # creates .venv + installs rfdetr[train], etc.

# Keep the job alive across SSH disconnects:
tmux new -s rfdetr
./scripts/run_stage1.sh
# detach: Ctrl-b d   |  reattach: tmux attach -t rfdetr
```

Or step by step:

```bash
.venv/bin/python -m tools.rfdetr_train.download
.venv/bin/python -m tools.rfdetr_train.train --batch 24 --epochs 50
# resume:  ... train --resume runs/rfdetr_stage1/checkpoint.pth
```

Optional downloads: `EXTRA_DOWNLOAD_ARGS="--bharatpothole --road-crack" ./scripts/run_stage1.sh`.
Colab path remains `notebooks/colab_rfdetr_train.ipynb` (points here for VM use).

### RF-DETR Medium on built 6-class (50 epochs, high VRAM)

Train **RFDETRMedium** on the prepared 6-class COCO (`data/rfdetr/stage2` preferred,
else `stage1`) for **50 epochs** with a large batch so the run uses **>20 GiB** GPU
RAM on a 32GB card. Writes to a **new** run dir (does not overwrite `runs/rfdetr_stage1`).

```bash
# Data must already exist (Stage-2 merge or Stage-1 COCO):
ls data/rfdetr/stage2/train/_annotations.coco.json

tmux new -s rfdetr_medium_50
./scripts/run_rfdetr_medium_50ep.sh
# detach: Ctrl-b d   |  reattach: tmux attach -t rfdetr_medium_50

# OOM: BATCH=24 ./scripts/run_rfdetr_medium_50ep.sh
# More VRAM: BATCH=32 ./scripts/run_rfdetr_medium_50ep.sh
# Custom data: DATASET_DIR=/path/to/coco ./scripts/run_rfdetr_medium_50ep.sh
```

Defaults: `batch=20`, `grad_accum=1`, `workers=4`, `epochs=50`, `--no-early-stop`,
`runs/rfdetr_medium_6class_50ep/`. If the job dies with `Killed` (no traceback),
that is usually host-RAM OOM from DataLoader workers — retry
`WORKERS=2 BATCH=16 ./scripts/run_rfdetr_medium_50ep.sh`.

### RF-DETR Stage 2 (RFDETRLarge, multi-source, overnight)

Stage 1 was pothole-heavy on real GoPro video. Stage 2 merges **India + rare-class**
sources, maps **rutting → ravelling**, caps pothole-only images, and trains
**RFDETRLarge** for **100 epochs**.

Sources (soft-skip on failure): CRRI, RDD2022 India, BharatPotHole, Crack-2,
pavement-distress, road-crack, road_damage_2, water-logging, drain-overflow, PWD.

```bash
# Needs ROBOFLOW_API_KEY + Kaggle creds in .env for BharatPotHole
tmux new -s rfdetr_stage2
./scripts/run_stage2.sh
# detach: Ctrl-b d   |  reattach: tmux attach -t rfdetr_stage2

# OOM: EXTRA_TRAIN_ARGS="--batch 2 --grad-accum 8" ./scripts/run_stage2.sh
# Data already prepared: SKIP_DOWNLOAD=1 ./scripts/run_stage2.sh
```

Outputs: `data/rfdetr/stage2/` (merged COCO), `runs/rfdetr_stage2/checkpoint_best_total.pth`.

### Parallel companion: Ultralytics RT-DETR-l (same Stage-2 data)

While RF-DETR Large is running (often ~7GB on a 32GB card), train **RT-DETR-l**
on a second tmux using the rest of the GPU. Same 6-class Stage-2 merge, faster
epochs, fair DETR-family A/B. Do **not** stop the Large job.

```bash
cd ~/road-defect-pipeline && git pull

# One-time export (shared COCO → YOLO); then train in a 2nd session:
.venv/bin/python -m tools.rfdetr_train.export_yolo \
  --coco-dir data/rfdetr/stage2 \
  --out-dir data/rfdetr/stage2_yolo

tmux new -s rtdetr_stage2
./scripts/run_rtdetr_parallel.sh
# Ctrl-b d

# If RF-DETR OOMs: EXTRA_TRAIN_ARGS="--batch 8 --memory-fraction 0.40 --workers 6" ./scripts/run_rtdetr_parallel.sh
# Faster (defaults now batch=16 workers=8 cache=ram):
#   SKIP_EXPORT=1 EXTRA_TRAIN_ARGS="--batch 24 --workers 12 --memory-fraction 0.60" ./scripts/run_rtdetr_parallel.sh
# Re-export skipped: SKIP_EXPORT=1 ./scripts/run_rtdetr_parallel.sh
```

Outputs: `data/rfdetr/stage2_yolo/`, `runs/rtdetr_stage2/weights/best.pt`.
Workers prefetch on **CPU RAM**; `--batch` / `--memory-fraction` use **GPU VRAM**;
`--cache ram` keeps images in host RAM so the GPU stays busier.

**Report training metrics** (precision, recall, mAP50, mAP50-95, losses):

```bash
./scripts/check_rtdetr_run.sh
# Fresh val on best.pt + per-class breakdown:
./scripts/check_rtdetr_run.sh --val
# JSON dump:
./scripts/check_rtdetr_run.sh --json-out runs/rtdetr_stage2/metrics_report.json
```

Also inspect plots under `runs/rtdetr_stage2/` (`results.png`, `confusion_matrix.png`,
`BoxPR_curve.png`).

### RT-DETR near-field inference (test Stage-2 weights)

Same near-field overlay / GPS / dashboard pipeline as RF-DETR, with `--backend rtdetr`.
Always use a **new** `--out-dir` so earlier POC runs stay untouched.

**Set Google Maps API key** (for `dashboard/index.html`):

```bash
cd ~/road-defect-pipeline
# Append to .env (do not commit the real key):
echo 'GOOGLE_MAPS_API_KEY=AIza...your_key...' >> .env
# Or one shell session:
export GOOGLE_MAPS_API_KEY='AIza...your_key...'
```

Enable **Maps JavaScript API** in Google Cloud; restrict the key (HTTP referrers).

**Infer ROAD-1 with `best.pt` (stricter gates — new out-dir):**

`run_rtdetr_infer.sh` injects `--conf 0.5`, `--min-overlap 0.50`, `--nms-iou 0.5`
unless you override them. Python also enables box-center-in-mask + clip for
`--backend rtdetr`. Do not overwrite prior `ROAD-1-Gopro-rtdetr*` outputs.

```bash
cd ~/road-defect-pipeline && git pull
ls runs/rtdetr_stage2/weights/

tmux new -s rtdetr_infer_v2
./scripts/run_rtdetr_infer.sh \
  --video 'https://drive.google.com/drive/folders/1rhnvLoPFv87-vecmMhN-G2FJMbqYpJbj' \
  --srt   'https://drive.google.com/drive/folders/1rhnvLoPFv87-vecmMhN-G2FJMbqYpJbj' \
  --weights runs/rtdetr_stage2/weights/best.pt \
  --z-far 5 \
  --out-dir 'runs/rfdetr_infer/ROAD-1-Gopro-rtdetr-v2'
```

**Upload** to a new Drive subfolder under the existing parent:

```bash
./scripts/upload_infer_results.sh \
  --run-dir 'runs/rfdetr_infer/ROAD-1-Gopro-rtdetr-v2' \
  --folder  'https://drive.google.com/drive/folders/1gFw80e4fMdL3ztDlUxVdQinNQlskpoz-' \
  --subfolder 'ROAD-1-Gopro-rtdetr-v2' \
  --client-secret ~/secrets/drive_oauth_client.json
```

### RF-DETR near-field inference (Phase 1)

Standalone dashcam inference: detect only in a **near-field trapezoid** (default
assess ≤ ~5 m ahead, both lanes), fill that polygon with a light green wash,
draw boxes, attach **timeline + GPS from an SRT sidecar**, and write a Leaflet
**map trail**.

```bash
# After Stage 1 weights exist — prefer tmux for long videos:
tmux new -s rfdetr_infer
./scripts/run_rfdetr_infer.sh \
  --video /path/to/dashcam.mp4 \
  --weights runs/rfdetr_stage1/checkpoint_best_total.pth \
  --z-far 5
# optional: --srt /path/to/dashcam.srt  (else uses <video>.srt next to the file)

# Or download from GCS / HTTPS (needs gcloud/gsutil for gs://):
./scripts/run_rfdetr_infer.sh \
  --video gs://YOUR_BUCKET/path/clip.mp4 \
  --srt   gs://YOUR_BUCKET/path/clip.srt \
  --weights runs/rfdetr_stage1/checkpoint_best_total.pth \
  --z-far 5

# Both-lane corridor + RF-DETR defaults (--conf 0.15, --min-overlap 0.15, --nms-iou 0.5):
./scripts/run_rfdetr_infer.sh \
  --video 'https://drive.google.com/drive/folders/FOLDER_ID' \
  --srt   'https://drive.google.com/drive/folders/FOLDER_ID' \
  --weights runs/rfdetr_stage1/checkpoint_best_total.pth \
  --z-far 5 \
  --out-dir 'runs/rfdetr_infer/ROAD-1-Gopro-v4'
# Still missing shoulder: --road-top-half-w 0.55 --road-bottom-half-w 0.85 --road-center-x 0.55
# Quieter / fewer overlaps: --conf 0.30 (NMS already on by default)
```

**ROAD-1 re-run (Stage-1 Medium @ conf 0.30 + NMS — new folder):**

Collapses stacked same-patch boxes (`--nms-iou 0.5`) and raises conf to cut low-score duplicates. Does not overwrite `*-c020` / earlier POCs.

```bash
cd ~/road-defect-pipeline && git pull
ls runs/rfdetr_stage1/checkpoint_best_total.pth

tmux new -s rfdetr_infer_stage1_c030
./scripts/run_rfdetr_infer.sh \
  --video 'https://drive.google.com/drive/folders/1rhnvLoPFv87-vecmMhN-G2FJMbqYpJbj' \
  --srt   'https://drive.google.com/drive/folders/1rhnvLoPFv87-vecmMhN-G2FJMbqYpJbj' \
  --weights runs/rfdetr_stage1/checkpoint_best_total.pth \
  --z-far 5 \
  --conf 0.30 \
  --nms-iou 0.5 \
  --out-dir 'runs/rfdetr_infer/ROAD-1-Gopro-medium-stage1-c030'
```

```bash
./scripts/upload_infer_results.sh \
  --run-dir 'runs/rfdetr_infer/ROAD-1-Gopro-medium-stage1-c030' \
  --folder  'https://drive.google.com/drive/folders/1gFw80e4fMdL3ztDlUxVdQinNQlskpoz-' \
  --subfolder 'ROAD-1-Gopro-medium-stage1-c030' \
  --client-secret ~/secrets/drive_oauth_client.json
```

Outputs in `runs/rfdetr_infer/<video_stem>/`:

| File | Contents |
|------|----------|
| `annotated.mp4` | Near-field green wash + outline + boxes + HUD |
| `defects.csv` / `.json` | Unique defects with `t_start/end`, lat/lon, chainage |
| `map_trail.html` | Synced dashboard: stats + annotated video + Google Maps/Leaflet |
| `summary.json` | Counts and run metadata |

Rebuild dashboard for an existing run into a **new sibling folder** (POC run stays
untouched — no edits to annotated.mp4 / map_trail.html / defects.*):

```bash
# In .env: GOOGLE_MAPS_API_KEY=your_key
.venv/bin/python -m tools.rfdetr_infer.rebuild_dashboard \
  --run-dir 'runs/rfdetr_infer/ROAD-1-Gopro-v3' \
  --srt /path/to/gopro.SRT
# writes → runs/rfdetr_infer/ROAD-1-Gopro-v3_dashboard/

# Serve the NEW folder only:
cd runs/rfdetr_infer/ROAD-1-Gopro-v3_dashboard && python3 -m http.server 8765
# open http://localhost:8765/index.html

# Upload to a NEW Drive subfolder under the existing parent (does not overwrite POC):
./scripts/upload_infer_results.sh \
  --run-dir 'runs/rfdetr_infer/ROAD-1-Gopro-v3_dashboard' \
  --folder  'https://drive.google.com/drive/folders/1gFw80e4fMdL3ztDlUxVdQinNQlskpoz-' \
  --subfolder 'ROAD-1-Gopro-v3-dashboard' \
  --dashboard \
  --client-secret ~/secrets/drive_oauth_client.json
```

Use `--copy-video` on rebuild if you need a fully self-contained folder (instead of a
symlink to the POC video).

Defaults: wide trapezoid (`bottom_half_w=0.78`, `top_half_w=0.50`), green wash
inside the assess polygon (far corridor tint off), RF-DETR Medium recall gate
(`--conf 0.15`, `--min-overlap 0.15`, bottom-center), plus **`--nms-iou 0.5`** to
collapse overlapping boxes. RT-DETR via `run_rtdetr_infer.sh` keeps stricter gates
(`conf 0.5`, `min-overlap 0.50`, center+clip, NMS). Classical road grow is off
(it was dropping cracked asphalt). Note: Stage-1 taxonomy has no **rutting**
class (labels were dropped at train time) — lowering conf helps cracks/potholes,
not rutting until you retrain.

Tune the trapezoid with `--road-top-y`, `--road-bottom-half-w`, `--road-center-x`.
For metric depth instead of the trapezoid proxy, pass `--camera-height-m` and
`--vfov-deg`.

**Upload results to Google Drive** (use your own Desktop OAuth client — not
`gcloud auth application-default`, which Google blocks for Drive):

1. GCP Console → OAuth consent **Testing** → add your Gmail as test user  
2. Create **OAuth client ID → Desktop app** → download JSON to  
   `~/secrets/drive_oauth_client.json` on the VM  
3. Share the destination Drive folder with that Gmail (Editor)  
4. Run:

```bash
# First-time auth: from your laptop open an SSH tunnel, then on the VM:
ssh -L 8090:localhost:8090 ubuntu@YOUR_VM_IP

./scripts/upload_infer_results.sh \
  --run-dir 'runs/rfdetr_infer/ROAD-1-Gopro-v3' \
  --folder  'https://drive.google.com/drive/folders/1gFw80e4fMdL3ztDlUxVdQinNQlskpoz-' \
  --client-secret ~/secrets/drive_oauth_client.json
```

Open the printed URL in your **laptop** browser (same Gmail as OAuth test user).  
Later runs reuse `~/.config/rfdetr_drive/token_*.json` (no browser).  
Fallback: `scp` the run folder to your laptop and drag into drive.google.com.

**Phase 2 (not in this pass):** `inference.backend: rfdetr` in the full `rdd`
pipeline (`detect_track` + IRC report), and optional 3-panel dashboard (telemetry
sidebar + overlay + north-up map) matching the inspiration UI.

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
| `detect.linear.*` | ground-plane angle bands for L/T; cell density for alligator |
| `detect.boundary.min_inset_m` | edge loss threshold, in metres |
| `detect.texture.cell_m` | ravelling grid cell size **in metres** |
| `detect.confusers.*` | rejection rules for shadow/tar/marking/manhole/joint |
| `detect.tiling.enabled` | sliced inference — turn on once a crack model is trained |
| `eval.target_precision` | the precision contract (default 0.90) |
| `eval.exclude_from_target` | classes outside the guarantee (rutting) |
| `report.irc.*` | IRC severity bands per measured quantity |
| `report.segments.length_m` | chainage segment length (default 100 m) |
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

267 tests. The ones that matter most encode the design invariants: hole-filling
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
  detect/                  linear (crack geometry), boundary (edge), texture
                           (ravelling/rutting), confusers, tiling, aggregate
  eval/                    per-unique-defect precision, threshold calibration
  annotate/                SAM-assisted labeling + active-learning frame picker
  model/                   YOLO loader (fallback), segment split, training
  depth/                   optional depth backend + severity (with abstention)
  inference/               detect+track+gate, unique counting, render
  report/                  CSV / JSON / HTML / PDF
```

## Reference repos
ultralytics · oracl4/RoadDamageDetection · FarzadNekouee/YOLOv8_Pothole_Segmentation ·
DepthAnything/Video-Depth-Anything · NitishMutha/equirectangular-toolbox · roboflow/supervision
