# Drone/UAV Stage-1 dataset — sources, licensing, gaps

Companion to the dashcam Stage-1 pipeline (`tools/rfdetr_train/download.py`).
This one prepares the **top-down** counterpart: `tools/rfdetr_train/download_drone.py`
merges public nadir/UAV pavement-distress datasets into the same fixed
6-class COCO layout (`alligator_crack, drainage_issue, longitudinal_crack,
pothole, ravelling, edge_damage`) so RFDETRMedium can be fine-tuned on either
viewpoint with the same training code.

Run it: `python -m tools.rfdetr_train.download_drone` or `./scripts/run_drone_stage1.sh`.
Two ground-level "nearest match" sources for `ravelling`/`edge_damage` are on
by default (`college_pavement_distress`, `rd01_pwd` — disable with
`--no-college-pavement-distress`/`--no-rd01-pwd`); `--cqu-bpdd-ravelling` adds
a third, non-commercial-licensed one, off by default. `--extra-local
edge_damage=/path/to/coco_or_yolo` (repeatable) merges in your own
hand-labeled bootstrap once you have one — the only real fix for
`drainage_issue`. See "`ravelling`/`edge_damage` sourcing" below.

## Why no perspective/road-segmentation stage

The dashcam pipeline segments the road plane first (`src/rdd/roadseg/`,
`src/rdd/preprocess/ipm.py`) because a windshield camera sees the road at a
shallow, receding angle — segmentation + IPM is what turns that into a
scale-consistent top-down-like image before detection. A drone looking
straight down **already is** that top-down image; there is no perspective to
correct. Point the trained RFDETRMedium checkpoint at drone frames directly —
skip roadseg/IPM for this pipeline. What top-down capture does still need,
that dashcam doesn't, is **GSD consistency** across merged sources (below),
because there's no perspective distortion to hide a mismatched capture scale.

## Sources used

| Source | Raw images | Annotated images (kept classes) | Classes shipped | Format | License | Altitude (documented) |
|---|---|---|---|---|---|---|
| [UAV-PDD2023](https://zenodo.org/records/8429208) (Zenodo record 8429208) | 2,440 | 2,404 | longitudinal/transverse/oblique/alligator crack, repair, pothole | VOC XML | CC BY 4.0 | 30 m, nadir, hover or 0.8 m/s |
| [UAPD](https://github.com/tantantetetao/UAPD-Pavement-Distress-Dataset) ([raw file](https://drive.google.com/file/d/1yQ0GMXFwwM5qdYY_5HzJBQqqjNtWJxEc/view)) | 3,151 (512×512 crops) | 2,147 | transverse/longitudinal/oblique/alligator/block crack, pothole, repair | VOC XML | Released for public research use (cite Zhu et al., *Automation in Construction* 2022) | not specified |
| [HighRPD](https://data.mendeley.com/datasets/sywswj7djj/1) (Mendeley DOI 10.17632/sywswj7djj.1) | 11,696 | 11,696 | line crack, block crack, pit (pothole) | YOLO txt, fixed ids (0/1/2) | CC BY 4.0 | 50 m, DJI M300 + Zenmuse P1 |
| [Roboflow: Pothole detection (by Drone)](https://universe.roboflow.com/drone-zh0ho/pothole-detection-zdizt) | 465 | 465 | pothole only | Roboflow COCO/YOLO export | MIT | not specified |
| [Roboflow: Pavement Distress Datasets (COLLEGE)](https://universe.roboflow.com/college-7qowe/pavement-distress-datasets) — **ground-level, not drone** | 1,120 | up to 1,120 | Edge Cracking / Pothole / Ravelling × {High,Medium,Low}, Medium Rutting | Roboflow COCO/YOLO export | **Public Domain** | not a drone — handheld close-range |
| [Roboflow: RD01 (RCDRD01)](https://universe.roboflow.com/rcdrd01/rd01) — **ground-level, not drone** | 1,362 | up to 1,362 | 31 classes incl. Edge_Breaking/Edge_Cracks, Ravelling, Loss_of_Aggregate, Hungry_Surface, Corrugations, severity-graded Indian-PWD style | Roboflow COCO/YOLO export | MIT | not a drone — mixed ground-level/oblique |
| [CQU-BPDD](https://huggingface.co/datasets/Ggggcs/CQU-BPDD) — **opt-in, off by default** | 477 (ravelling only, of 60,056) | 477 | ravelling (whole-image label → full-frame box) | Nested ZIP, HF-hosted | **CC BY-NC 4.0 (non-commercial)** | not a drone — in-vehicle camera, ~2×3m patch |

`tools/rfdetr_train/drone_sources.py` holds this same table in code
(`DRONE_SOURCES`) plus the license/URL each downloader cites.

**Every number above (except the Roboflow row) was computed directly from the
real archive**, not copied from the paper abstract — the full UAPD zip was
downloaded and parsed, and UAV-PDD2023/HighRPD were parsed via HTTP range
reads against their real ZIP central directories (no full download needed:
Zenodo and the Mendeley→S3 redirect both serve `Accept-Ranges: bytes`). The
instance totals matched each paper's published aggregate exactly (UAV-PDD2023:
11,158; HighRPD: 12,365/8,239/1,412), which cross-checks the parse. Roboflow's
465 is read off the dataset page instead — exact instance count needs a
Roboflow API key.

### Class-wise breakdown (after remapping to the 6-class taxonomy)

Instances = individual bounding boxes. Images = images containing **at least
one** box of that class (an image with both a pothole and a crack counts in
both rows, so column sums exceed each source's annotated-image count above).

| Class | UAV-PDD2023 | UAPD | HighRPD | RF pothole-drone | **Total instances** | **Total images** |
|---|---|---|---|---|---|---|
| `longitudinal_crack` | 10,078 inst / 2,377 img | 2,840 inst / 1,843 img | 12,365 inst / 6,853 img | — | **25,283** | **11,073** |
| `alligator_crack` | 603 inst / 408 img | 417 inst / 365 img | 8,239 inst / 5,677 img | — | **9,259** | **6,450** |
| `pothole` | 195 inst / 132 img | 94 inst / 68 img | 1,412 inst / 997 img | ≥465 inst / 465 img | **≥2,166** | **1,662** |
| `drainage_issue` | 0 | 0 | 0 | 0 | **0** | **0** |
| `ravelling` | 0 | 0 | 0 | 0 | **0** | **0** |
| `edge_damage` | 0 | 0 | 0 | 0 | **0** | **0** |

`longitudinal_crack` folds in transverse + oblique + line crack; `alligator_crack`
folds in block crack — see the mapping table below for why. `repair` instances
(1,904 combined, UAV-PDD2023 + UAPD) are dropped, not miscounted into another
class.

Raw pre-mapping counts, for reference — UAV-PDD2023: Transverse 5,398 / Longitudinal
2,994 / Oblique 1,686 / Alligator 603 / Repair 282 / Pothole 195. UAPD: Repair 1,622 /
Longitudinal 1,340 / Transverse 1,327 / Alligator 414 / Oblique 173 / Pothole 94 /
Block 3. HighRPD: Line 12,365 / Block 8,239 / Pit 1,412.

Merged dataset size before any train/valid/test split: **16,712 images**
from the 4 nadir sources (2,404 + 2,147 + 11,696 + 465), plus up to 1,120 +
1,362 more from `college_pavement_distress`/`rd01_pwd` (ground-level; exact
per-class split needs a Roboflow API key to inspect, shows up in the
histogram after your first run). The `pothole` row is the thin one among the
nadir sources — 1,662 images vs. 6,450–11,073 for the crack classes — worth
watching for class imbalance; `coco_io.cap_majority_class_images` (already
used by the dashcam Stage-2 merge) is available if crack classes end up
crowding out pothole instead of the other way round — check the histogram
`download_drone.py` prints after each run.

### Ruled out

- **Roboflow "Road Damage Dataset" by JURIS DRONE** (4,915 images, 8 classes) —
  workspace name says "drone" but the actual images are street-level oblique
  shots (verified by opening the image browser — see screenshot check during
  research). Do not trust dataset names on Roboflow Universe; always open the
  image grid before relying on a "drone"/"aerial" label.
- **Roboflow "Pavement cracks from UAV imagery" by nimi** (1,282 images, 11
  classes) — genuinely nadir crops and a taxonomy that lines up suspiciously
  exactly with UAV-PDD2023/UAPD (alligator/block/longitudinal/oblique/repair/
  transverse crack + pothole), but no license is published on the listing and
  it's very likely an unattributed re-upload of one of the two academic sets
  above. Skipped in the default script to avoid double-counting images under
  an unclear license; the URL is left here in case you want to fork it
  yourself and compare hashes against `uav_pdd2023`/`uapd` first.
- **Mendeley `csd32bm8zx`** (DJI Mini 2, 386 m test road, Quebec) — real
  drone-based crack inspection data, but it's a small single-road survey
  focused on crack length/width measurement rather than a labeled multi-class
  detection set, and altitude/class format weren't confirmed. Not wired into
  the downloader; worth a manual look if you need more crack-only diversity.
- **209 Roboflow results tagged "ravelling"** — checked the class lists on all
  of them; essentially every one is an Indian PWD-style road-inspection
  workspace (`pwd3601`–`pwd3606`, `RCD*`/`RD*` workspaces) with class lists
  like `CLEANING-REQUIRED, DAMAGED-KERB, HOTSPOT, TRENCH, RAVELLING` — the same
  family of ground-level/oblique inspection-vehicle surveys the dashcam
  pipeline already draws from (`download.py`'s `use_pwd` source), not drone
  imagery. None had "UAV"/"drone"/"aerial" in the title, unlike the
  pothole-specific ones that do exist. No genuinely nadir ravelling source
  found on Roboflow.

## `ravelling` / `edge_damage` sourcing (why they're still thin)

Both classes are structurally hard to see from directly overhead, which is
likely *why* no drone dataset labels them, not just an oversight:

- **`ravelling`** (loss of surface aggregate) is a **texture-scale** distress —
  visible at cm-level resolution, close to the pavement. Wide-area survey
  drones flying at 30–50 m (the altitude every source above actually used)
  don't resolve it; the classes that *do* get labeled at that altitude
  (crack, pothole) are all things still visible at a few cm/px GSD.
- **`edge_damage`** (shoulder drop-off, lane-edge erosion) is fundamentally a
  **3D elevation cue** — a drop-off reads clearly at an oblique angle (which
  is exactly why the dashcam pipeline can see it) but is easy to miss in
  pure monocular nadir RGB without a shadow or a stereo/depth cue.

Given that, no drone-native match exists for either class — but two
ground-level "nearest match" sources are wired in and **on by default**
(the user explicitly chose a viewpoint-mismatched real dataset over zero
data for these two classes):

1. **`college_pavement_distress`** — Roboflow ["Pavement Distress Datasets" by
   COLLEGE](https://universe.roboflow.com/college-7qowe/pavement-distress-datasets),
   1,120 images, **Public Domain**. Handheld/close-range shots pointed roughly
   down at the pavement surface (not drone altitude, but closer to nadir than
   a dashcam) — the preview grid is texture-level crops, not street scenes.
   Ships `High/Medium/Low Edge Cracking`, `.../Pothole`, `.../Ravelling`, and
   `Medium Rutting`; the taxonomy's existing severity-prefix stripping
   resolves all of these with zero new aliases needed. Already used by the
   dashcam Stage-2 pipeline (`download.py`'s `use_pavement_distress`), so
   it's a previously-vetted source, not a new unknown.
2. **`rd01_pwd`** — Roboflow ["RD01" by RCDRD01](https://universe.roboflow.com/rcdrd01/rd01),
   1,362 images, MIT. Mixed ground-level/oblique — one preview frame is a
   narrow tree-lined rural road, consistent with an Indian PWD road-condition
   survey. 31 raw classes in a severity-graded taxonomy that reads as
   official Indian pavement-engineering terminology; added 4 new taxonomy
   aliases for this source specifically: `Edge_Breaking` → `edge_damage`,
   `Loss_of_Aggregate` / `Hungry_Surface` / `Corrugations` → `ravelling`
   (aggregate stripped from the binder and a lean/dry-looking surface are
   both standard descriptions of the same early-stage ravelling; corrugation
   folds in the same way rutting already does).

Both are excluded from `_apply_gsd` — their capture geometry (handheld,
unknown/variable distance) has no altitude to normalize against, so mixing
them into the GSD-normalized nadir sources means `ravelling`/`edge_damage`
will see a wider range of apparent scale than the other four classes. Watch
the per-class histogram after a run; if these two dominate the merged
`ravelling`/`edge_damage` counts and validation looks worse on genuine drone
frames than on these ground-level crops, that scale mismatch is the likely
cause — `--no-college-pavement-distress`/`--no-rd01-pwd` drop them.

Two more options if you want to go further:

3. **CQU-BPDD, opt-in (`--cqu-bpdd-ravelling`), off by default.** The only
   *near-nadir-altitude* ravelling source (in-vehicle camera pointed straight
   down, not handheld) — see below for the full non-commercial/weak-label
   tradeoff before enabling.
4. **Hand-labeled bootstrap, `--extra-local NAME=PATH`.** Still the only way
   to get `drainage_issue` any coverage, and the only way to get tighter,
   true-drone-GSD boxes for `edge_damage`/`ravelling` than any of the above.
   Fly your own drone, run the checkpoint trained on what's here (it'll miss
   drainage/edge cases — expected), then use this repo's existing
   SAM-assisted labeling workflow (README "Labeling workflow (SAM-assisted)")
   to box-label the frames where they appear. Export as COCO or YOLO and pass
   `--extra-local edge_damage=/path/to/export` — merged in via the same
   `ingest_to_coco` auto-detector the dashcam pipeline uses for `--local-dir`.
   Even 200–400 instances is normally enough for RF-DETR to pick up a
   visually distinct class.

### CQU-BPDD detail

The [Chongqing University Bituminous Pavement Disease Detection Dataset](https://huggingface.co/datasets/Ggggcs/CQU-BPDD)
is captured by an **in-vehicle inspection camera pointed straight down** (not
a drone; 2×3 m patch per image, ~1,200×900 px) rather than altitude. It has
**956 ravelling images** (477 train / 479 test) among 60,056 total, **CC BY-NC
4.0 — non-commercial only**, and ships **whole-image classification labels,
not boxes** — `download_cqu_bpdd_ravelling()` + `ingest_classification_folder()`
convert each into one full-frame weak box, which is real signal for "is this
patch ravelling" but not a tight localization the way the other sources'
boxes are. `_apply_gsd` will under- or over-scale it badly relative to your
drone's actual altitude until you've measured its GSD too (see "Calibrating
GSD" — a 2×3 m patch at ~1,200 px wide implies roughly 0.25 cm/px, several
times finer than the 30–50 m sources). Enable it only if the non-commercial
license is acceptable for how this model will be used.

## Class mapping into the 6-class taxonomy

All of the above use crack-orientation taxonomies (Chinese pavement-engineering
convention: longitudinal/transverse/oblique/alligator/block) rather than this
repo's names. `tools/rfdetr_train/taxonomy.py` now resolves these directly —
extend that file, not the ingestion code, if a new source uses different labels:

| Source label | → taxonomy class | Rationale |
|---|---|---|
| longitudinal crack (LC), transverse crack (TC), oblique crack (OC), line crack | `longitudinal_crack` | Same merge the dashcam pipeline already does for D00/D10 — this repo doesn't split crack orientation into separate classes. |
| alligator crack (AC), block crack | `alligator_crack` | Block cracking is alligator/fatigue cracking's rectangular-cell form, same failure mode viewed top-down. |
| pothole (PH), pit | `pothole` | Direct match. |
| repair (RP) | dropped (`None`) | Already-fixed pavement — matches the dashcam pipeline's existing `"repair": None`. |
| Edge_Breaking | `edge_damage` | RD01's term for edge/shoulder deterioration. |
| Loss_of_Aggregate, Hungry_Surface, Corrugations | `ravelling` | PWD terms for the same distress at different stages, plus corrugation folding in the same way rutting already does. |

**`drainage_issue` has no public source at any camera angle; `edge_damage`
likewise; `ravelling` has exactly one, CQU-BPDD.** See "`ravelling` /
`edge_damage` sourcing" above for what's actually available for each and why.
If you're tempted to paste in the dashcam-side `drainage_issue`/`edge_damage`
crops (CRRI, PWD drainage — see `download.py`) as a placeholder: don't, at
least not silently — they're oblique, not top-down, and will teach the model
the wrong appearance prior for that class. A 4-class-strong, 2-class-absent
checkpoint is more useful than one quietly trained on a bad prior for 2 of
its 6 heads. If you do it anyway, say so in the dataset `info.description`
field so it's visible later.

## Calibrating GSD (why "same capture height" actually means same cm/px)

Altitude alone doesn't fix ground sampling distance (GSD, cm covered by one
pixel) — a Zenmuse P1 at 50 m and a Mavic-class camera at 30 m can land on a
similar or wildly different GSD depending on sensor size and focal length,
neither of which the source papers publish in full. So none of the altitudes
in the table above are used to compute GSD automatically — `SOURCE_GSD_CM_PX`
in `tools/rfdetr_train/drone_sources.py` ships with every value `None`, and
`download_drone.py` skips normalization for any source left unmeasured
(better to merge at mismatched scale *visibly* than silently on a guessed
number).

To fill it in:

1. Pick 5–10 sample images per source (`data/rfdetr_drone/stage1_parts/<source>/train/`).
2. Measure a known real-world feature in pixels — a painted lane-edge stripe
   (~10 cm wide) or the paved carriageway width (rural single-lane India,
   IRC SP:20 guidance, is commonly ~3.0–3.75 m) is easiest to eyeball in a
   nadir image.
3. `cm_per_px = real_world_cm / measured_px`, averaged across your samples.
4. Fill it into `SOURCE_GSD_CM_PX["<source_key>"]` in `drone_sources.py`, and
   set `SOURCE_GSD_CM_PX["_target"]` to the GSD **your own drone** will fly
   at (same formula, using your actual flight altitude + camera once you've
   flown a calibration pass).
5. Re-run `download_drone.py` — each source's images/boxes are rescaled to
   match `_target` before merge, via `drone_sources.resize_split_to_target`.

Until you've flown your own drone and know its real altitude/camera, leave
`_target` unset — training on the raw, un-normalized mix is still reasonable
for a first checkpoint, since HighRPD (50 m) and UAV-PDD2023 (30 m) aren't
wildly apart. Normalize before the real 50-epoch production run once you have
your own flight numbers, not before.

## Rural-India-specific coverage notes

None of the 4 sources are India-specific (China: UAV-PDD2023, UAPD, HighRPD;
source country unconfirmed for the Roboflow set) or rural-specific — they're
mostly captured on paved arterial/highway pavement, cleaner surface and less
foliage/shadow occlusion than a rural road survey. Expect a domain gap on:

- **Tree-shadow clutter and canopy occlusion** — none of the sources document
  deliberate shadow/foliage diversity. Your own bootstrap footage (see above)
  should specifically include shadow-heavy and partially-occluded segments,
  not just clean daylight passes, or the model will under-detect on exactly
  the frames that matter most in a rural canopy-lined road.
- **Mud/unpaved shoulder and seasonal surface change** — same gap; add
  monsoon-season and dry-season passes from the same road segment if you can,
  since mud coverage changing pothole/edge appearance is a known failure mode
  for the dashcam pipeline's `edge_damage`/`drainage_issue` classes already.
- **Consistent flight height** — beyond the GSD-calibration reason above,
  keeping altitude fixed across your own capture flights keeps object pixel
  footprint (and therefore anchor/query scale) consistent for RF-DETR,
  independent of the merged public data.

## Citing

If you publish results, cite the two academic sources per their listings:

- UAV-PDD2023 — Zenodo record [10.5281/zenodo.8429208](https://zenodo.org/records/8429208)
- HighRPD — Mendeley Data [10.17632/sywswj7djj.1](https://data.mendeley.com/datasets/sywswj7djj/1)
- UAPD — Zhu et al., *Automation in Construction* (2022); GitHub [tantantetetao/UAPD-Pavement-Distress-Dataset](https://github.com/tantantetetao/UAPD-Pavement-Distress-Dataset)
