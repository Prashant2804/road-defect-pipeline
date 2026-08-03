# Handoff

For the next agent picking this up. `README.md` documents the architecture; this
covers what the README cannot: where things actually stand, why several
odd-looking decisions are deliberate, and which traps have already been paid for.

Last updated: 2026-08-03. Head: `22a929e`. 297 tests pass.

---

## 1. What this is meant to do

Detect road distress from **dashcam and drone video of Indian rural roads** and
grade it to IRC/PMGSY, at **precision ≥ 90%**, with an explicit refusal mode: when
the road is buried under water or mud, cannot be located, or the vehicle has left the
carriageway, the system must **report nothing rather than guess**.

Nine classes are configured; they are not all detected the same way, and that
distinction is the core design idea:

| Family | Classes | How |
|---|---|---|
| Instance objects | `pothole` | YOLO detector |
| Thin linear | `longitudinal_crack`, `transverse_crack`, `alligator_crack` | detector finds cracks class-agnostically; **orientation measured on the ground plane** decides which |
| Area / texture | `ravelling`, `rutting` | statistical, label-free |
| Boundary geometry | `edge_damage` | derived from the road mask edge, label-free |
| Hydrology | `drainage_issue`, `water_logging` | water mask + road geometry, label-free |

Only `pothole` and the crack classes need a trained model at all.

### Decisions the user made — do not relitigate these

- Drone camera is **rectilinear nadir**, not fisheye.
- Road segmentation default is **classical + geometric prior** (CPU-friendly, no new
  dependencies), not a learned segmenter.
- Mud/water policy is **abstain and report unassessable %**, not best-effort detection.
- Precision is measured **per unique defect, multi-frame confirmed** — not per frame.
- Labelling strategy: **public datasets + minimal labelling**.
- Standard: **IRC/PMGSY**.
- **`rutting` is excluded from the 90% target** by explicit decision. Report it as
  indicative only.

---

## 2. Honest status

Working and tested:

- Road segmentation, with fallback reporting and confidence.
- Surface condition (mud/water) using illumination invariants.
- Validity gating — verified in **both** directions: flooded/muddy/off-track/no-road
  clips yield 0% assessable; clean clips yield 100%.
- Camera calibration, assessment zones, vanishing-point auto-calibration.
- Label-free edge damage, ravelling, drainage, rutting proxy.
- Precision harness with Wilson bounds and per-class threshold calibration.
- Annotation validation/repair, training, and held-out testing.
- Colab notebook and standalone fine-tune cell; PowerShell/bash installers.

**Not achieved, and central to the brief:**

> **No class is certified at 90% precision.** Nothing has been measured against the
> agreed contract (per unique multi-frame-confirmed defect on real video), because
> that needs a labelled *validation set of video*, which does not exist. The
> annotation dataset is per-image and cannot answer this question. Until then, 90%
> is an aspiration in this repo, not a result. Do not let it be reported as achieved.

Also unresolved:

- **Thresholds are tuned on synthetic footage only** (`tools/make_synthetic_road.py`).
  Every threshold in the `roadseg` and `surface` sections is provisional.
- **Auto-calibrated pitch was 19.63° against 8° configured** on the user's clip, with
  a vanishing-point row spread of ±26.7px. It squeaked under the 12° sanity limit. If
  that estimate is wrong, every distance, zone and area is wrong. Unverified.
- Phase 5 of the plan (targeted labelling driven by measurement) is not started.

---

## 3. The blocking problem

The RDD2022 checkpoint produces **zero raw detections** on the user's footage.

The evidence that matters: `rejected off-road 0`, `rejected out of zone 0`,
`rejected as confusers 0`. Every filter reports what it removed and they all removed
nothing — so nothing ever arrived. This is not a pipeline tuning problem.

**Near-certain cause: surface type.** RDD2022 is trained overwhelmingly on sealed
bituminous roads, where a defect is a fracture in asphalt. The footage is unpaved
earth/gravel village road. There is no asphalt to crack, and a gravel depression looks
nothing like the dark, sharp-edged asphalt pothole the model learned.

`tools/diagnose_model.py` runs a checkpoint **raw** — no gating, zones, confusers or
tracking — across a conf/imgsz/enhancement sweep, and settles the question definitively.
**It has still not been run.** Run it before assuming anything:

```
python tools/diagnose_model.py --input <video> --weights <ckpt> --frames 6
```

The response is fine-tuning, which is why sections 4–5 below exist.

---

## 4. The annotation dataset

Roboflow: workspace `nidhis-workspace-zyeyu`, project `mp_road_annotation_poc`, **v3**.
493 images, 6 classes. Source images are Windows screen captures
(`ApplicationFrameHost_*`) — recordings of video playback, not extracted frames.

`tools/check_labels.py` found, and `tools/fix_labels.py` repairs:

1. **10 files mixed a box row with a polygon row.** This is the one that silently
   corrupts. Ultralytics picks box-vs-segment **per file**, so in those files every
   `x y w h` row is reinterpreted as two polygon points. Verified: a box at
   (0.43, 0.83) size 0.14×0.29 came back as (0.28, 0.56) size 0.28×0.53 — wrong
   centre, 3.6× the area, no error. Pinned by
   `tests/test_labels.py::test_ultralytics_misreads_mixed_file`.
2. **Class order differs from `model.classes`**, and detections resolve by index.
3. **4 eval frames near-duplicate training frames** (perceptual hash; byte hashing
   found none, because exports recompress).

Instances after repair — `ravelling` 256, `pothole` 147, `edge_damage` 118,
`longitudinal_crack` 95, `alligator_crack` 28, `drainage_issue` 21.

**Not fixable in software, needs a re-export:**

- **512×512 "Stretch".** If the source was 16:9 this squashed every frame. It matters
  twice: the model trains on ellipse-shaped potholes it will never meet at inference,
  and anisotropic scaling *changes angles* while longitudinal-vs-transverse is an angle
  measurement. Re-export with **Resize = Fit (letterbox), 1280px**.
- 512px is below the resolution at which hairline cracks exist at all.
- The **test split contains zero `alligator_crack` and zero `drainage_issue`**, so it
  cannot score those two. `run.py val` names untested classes rather than omitting the
  rows, because an omitted row reads as a pass.

---

## 5. Commands

```bash
# Annotations
python tools/check_labels.py --labels <export>
python tools/fix_labels.py   --labels <export> --out data/mp_road --to box --rename

# Train / test  (needs CUDA; training errors out rather than falling back to CPU)
python run.py train --data data/mp_road/data.yaml --device cuda \
  --set run.name=finetune --set model.size=s \
  --set model.train.epochs=100 --set model.train.imgsz=512 --set model.train.batch=16
python run.py val --weights out/finetune/train/weights/best.pt \
  --data data/mp_road/data.yaml --split test --set run.name=finetune

# Diagnose a checkpoint that finds nothing
python tools/diagnose_model.py --input <video> --weights <ckpt>

# Inspect stages
python run.py roadseg  --input <video> --n 8
python run.py validity --input <video>
python tools/doctor.py
```

GPU work happens in Colab: `notebooks/colab_inference.ipynb` section 5b, or
`notebooks/colab_finetune_cell.py` as a single paste-able cell.

---

## 6. Non-obvious decisions — reasons, so they are not "fixed" back

Each of these looks wrong at a glance and is not.

**LAB a/b was rejected for chromaticity.** Gamma artifacts shift `b` by 8 units on
shadowed road — larger than a real mud signal. Linear-RGB `cr`/`cg` is used instead
because it is genuinely invariant under multiplicative shading.

**Road masks are hole-filled.** Potholes and puddles are appearance outliers, so
similarity-based segmentation carves them out. Without filling them back, road gating
rejects exactly the defects it exists to preserve.

**The appearance baseline is a temporal EMA, not the geometric seed.** A near-field
puddle can dominate the seed region, making water the definition of "normal road" —
that produced 60% of clean road being read as mud.

**`gate_surface_plausible` uses absolute thresholds**, unlike everything else here
which is relative. A fully flooded road is uniform, so relative statistics call it
clean. This is the one place relative measurement is structurally blind.

**Vibration is measured as `pitch_px` at the horizon.** The obvious metric — spatial
spread of vertical flow — is always large under forward motion, so it conflated
driving with shaking.

**The road mask is eroded before texture analysis.** A box filter mixes in the verge
at the boundary; that caused 46% false ravelling on clean road.

**L/T crack classification is geometric, not learned.** In a perspective view the
distinction is confounded by camera geometry, and RDD2022's own D00/D10 split is noisy
and camera-dependent. On the rectified ground plane it is a direct measurement.
Alligator is decided by enclosed-cell density, which distinguishes true fatigue
cracking from several parallel cracks without labels.

**IPM is used analytically, never for warping** — the area Jacobian `|det H| / w³`.
Warping would resample and lose detail for no benefit.

**`model.class_map` is required even when names look identical.** The checkpoint has 6
classes in its own order; `config.yaml` declares 9. Resolution is positional, so
without the map id 0 reads as `pothole` when it is `alligator_crack`. The identity map
looks redundant and is load-bearing — it forces resolution *by name*.

**Train/val splitting must never be random.** `model.train.split.mode: random` raises
rather than warns. Adjacent video frames are near-duplicates; a random split puts the
same physical defect in both sides and turns validation into a memorisation test.

**Training resolves the device *before* loading the model**, and `strict=True` makes an
unsatisfiable `cuda` request an error. Inference keeps the fallback, where CPU is slow
rather than wrong.

---

## 7. Traps already paid for — do not rediscover these

- `dict.get(key, default)` returns **`None`**, not the default, for a key explicitly
  set to `null` in YAML. Use the `_num()` helper in `geometry/calibration.py`.
- Ultralytics 8.4 renamed `half` → `quantize` **and changed its type**: it takes
  `'fp16'`, not `True`. Checking that a key exists is not enough; check what it accepts.
- Keyframe decoding (`ffmpeg -skip_frame nokey`) is far faster than seeking. Measured:
  keyframes 8.2s, grab 19.9s, read 27.9s, seek 62.0s, ffmpeg-ss 143.5s. **Seeking is
  slower than reading.**
- When subsampling keyframes use `step = max(1, round(len(times)/n))` and stride.
  Flooring then truncating keeps only the first N keyframes — i.e. the first few
  seconds of the video.
- Dense Farneback at 1920×1080 took 12 of 13 minutes on a 30s clip. It runs at 480px
  now, and `preprocess.sampling.enabled` turns it off for pure inference runs.
- PowerShell strips quotes from native command arguments; put logic in `tools/doctor.py`
  rather than inline. `$ErrorActionPreference='Stop'` plus any stderr output is fatal —
  hence the `Try-Exec` helper.
- In bash, `exit` inside `$(...)` exits only the subshell.
- `importlib.reload(torch)` raises `TORCH_LIBRARY` errors in Colab. Probe with a
  subprocess instead.
- **A `git pull` in a Colab runtime cannot update the open notebook.** Colab holds its
  own copy of the `.ipynb`. A Drive copy is frozen at save time. This cost a full
  round trip; `notebooks/colab_finetune_cell.py` exists so fine-tuning does not depend
  on it.
- Ultralytics prepends `runs/<task>/` to a **relative** `project` path, so `out/<name>`
  silently becomes `runs/detect/out/<name>`. Resolve it to absolute.
- Avoid `git add -A` here; it has twice swept in generated data (989 dataset files
  once, ~150 MLflow files another). `runs/`, `out/`, `data/mp_road/` are now ignored.

### Process notes

Two failures in this project came from *not verifying*, not from bad reasoning:
a string-replacement patch that silently did not apply (the benchmark still moved, so
it looked fine), and `quantize=True` which crashed the user's run after 13 minutes of
setup. Both would have been caught by running the thing. Since then: run pytest **and**
the notebook cell-syntax check on every notebook change, and execute new commands
before putting them in front of the user.

---

## 8. Environment

- Windows 11, PowerShell primary; Git Bash available.
- **`torch` here has no CUDA.** All training must happen in Colab.
- GitHub: `gh` defaults to the user's BCG work account; this repo is under the
  personal account **Prashant2804** (private). Use `gh auth login --web` if pushes
  target the wrong account.
- Repo: https://github.com/Prashant2804/road-defect-pipeline (private, `master`).

---

## 9. What to do next, in order

1. **Run `tools/diagnose_model.py`** on the user's footage. It is one command and it
   settles whether the domain gap is real or the pipeline is hiding detections. Every
   other decision depends on the answer.
2. **Re-export the dataset** at Fit/letterbox, 1280px. Cheap, and it changes the
   ceiling on everything trained afterwards.
3. **Fine-tune and test** (section 5 above). Read per-class mAP, not the `all` row.
4. **Verify the camera calibration** against something known — a measured lane width or
   a known object. Every metric output rides on it and it is currently unchecked.
5. **Build the video validation set** the 90% claim actually requires: ~250–400 frames
   labelled for *unique multi-frame-confirmed* defects. Until this exists, no class can
   be certified, and the deliverable should say so plainly rather than quoting per-image
   mAP as if it answered the question.
6. Only then Phase 5: targeted labelling for whichever classes measurably missed.

Be straight with the user about (5). They have been consistently receptive to bad news
delivered with evidence, and the precision target is the thing most at risk of being
quietly assumed rather than measured.
