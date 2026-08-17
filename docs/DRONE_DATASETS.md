# Drone/UAV Stage-1 dataset — sources, licensing, gaps

Companion to the dashcam Stage-1 pipeline (`tools/rfdetr_train/download.py`).
This one prepares the **top-down** counterpart: `tools/rfdetr_train/download_drone.py`
merges public nadir/UAV pavement-distress datasets into the same fixed
6-class COCO layout (`alligator_crack, drainage_issue, longitudinal_crack,
pothole, ravelling, edge_damage`) so RFDETRMedium can be fine-tuned on either
viewpoint with the same training code.

Run it: `python -m tools.rfdetr_train.download_drone` or `./scripts/run_drone_stage1.sh`.

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

| Source | Images | Instances | Classes shipped | Format | License | Altitude (documented) |
|---|---|---|---|---|---|---|
| [UAV-PDD2023](https://zenodo.org/records/8429208) | 2,440 | 11,158 | longitudinal/transverse/oblique/alligator crack, repair, pothole | VOC XML | CC BY 4.0 | 30 m, nadir, hover or 0.8 m/s |
| [UAPD](https://github.com/tantantetetao/UAPD-Pavement-Distress-Dataset) | 3,151 (512×512 crops) | — | transverse/longitudinal/oblique/alligator crack, pothole, repair | VOC XML | Released for public research use (cite Zhu et al., *Automation in Construction* 2022) | not specified |
| [HighRPD](https://data.mendeley.com/datasets/sywswj7djj/1) | 11,696 | 22,016 (line 12,365 / block 8,239 / pit 1,412) | line crack, block crack, pit (pothole) | YOLO txt, fixed ids (0/1/2) | CC BY 4.0 | 50 m, DJI M300 + Zenmuse P1 |
| [Roboflow: Pothole detection (by Drone)](https://universe.roboflow.com/drone-zh0ho/pothole-detection-zdizt) | 465 | — | pothole only | Roboflow COCO/YOLO export | MIT | not specified |

`tools/rfdetr_train/drone_sources.py` holds this same table in code
(`DRONE_SOURCES`) plus the license/URL each downloader cites.

### Ruled out

- **CQU-BPDD** (60,056 images) — captured by an in-vehicle inspection vehicle's
  downward camera, not a drone. Geometrically closer to nadir than a dashcam
  but at road height, not altitude; also non-commercial-only license. Skipped.
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

**`drainage_issue` and `edge_damage` have no public drone source.** No dataset
above (or found in research) labels waterlogging/culvert-choke/shoulder
drop-off from an aerial view — these are comparatively rare even in
ground-level datasets and effectively unannotated at altitude. Two ways to
close this before the 50-epoch run:

1. **Bootstrap from your own drone footage.** Fly a short pass over a few km
   of known-bad rural road, run the RFDETRMedium checkpoint trained on the 4
   sources above (it'll miss drainage/edge cases, but that's fine), then
   hand-correct/add boxes for drainage_issue and edge_damage on the frames
   where they appear. Even 200–400 hand-labeled instances per class, mixed
   into Stage-1 at merge time, is normally enough for RF-DETR to pick up a
   visually distinct class — waterlogging and shoulder drop-off are both very
   legible top-down.
2. If you want a placeholder before real drone footage exists, the
   dashcam-side `drainage_issue`/`edge_damage` crops (CRRI, PWD drainage,
   BharatPotHole-adjacent sources — see `download.py`) are oblique, not
   top-down, so pasting them in as-is will teach the wrong appearance prior.
   Don't do this silently; if you go this route, say so in the dataset
   `info.description` field so it's visible later.

Skip drainage_issue/edge_damage from the first drone run rather than fake
data for them — a 4-class-strong, 2-class-absent checkpoint is more useful
than one quietly trained on a bad appearance prior for 2 of its 6 heads.

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
