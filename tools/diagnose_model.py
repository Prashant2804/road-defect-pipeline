#!/usr/bin/env python
"""Is the detector finding nothing, or is the pipeline hiding it?

When a run reports zero defects there are two very different causes, and they need
opposite responses:

  * the model produced no detections at all      -> the model is wrong for this footage
  * the model detected things that were filtered -> the pipeline is too strict

The full pipeline cannot tell you which, because everything it does downstream of the
detector is designed to remove things. This runs the checkpoint **raw** — no road
gating, no assessment zones, no confuser rejection, no tracking, no minimum track
length — across a sweep of confidence thresholds, input sizes and with/without the
pipeline's image enhancement.

If the raw sweep is empty at conf 0.01, no amount of pipeline tuning will help: the
checkpoint does not recognise this surface, and the answer is different weights or
fine-tuning. If the raw sweep finds things the pipeline dropped, the numbers below tell
you which stage to loosen.

    python tools/diagnose_model.py --input road.mp4 --weights best.pt --frames 6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="video to sample frames from")
    p.add_argument("--weights", required=True, help="checkpoint to interrogate")
    p.add_argument("--frames", type=int, default=6, help="frames to sample")
    p.add_argument("--conf", default="0.01,0.05,0.10,0.25",
                   help="confidence thresholds to sweep")
    p.add_argument("--imgsz", default="640,960,1280", help="input sizes to sweep")
    p.add_argument("--config", default=str(ROOT / "config.yaml"))
    p.add_argument("--out", default="out/diagnose", help="where to write annotated frames")
    p.add_argument("--no-enhance", action="store_true",
                   help="skip the enhanced-vs-raw comparison")
    args = p.parse_args(argv)

    import cv2
    from ultralytics import YOLO

    from rdd.config import load_config
    from rdd.quality.enhance import enhance_frame, resolve_spec
    from rdd.quality.metrics import assess_video
    from rdd.utils.video import iter_sampled_frames

    cfg = load_config(args.config)
    confs = [float(c) for c in args.conf.split(",")]
    sizes = [int(s) for s in args.imgsz.split(",")]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading {args.weights}")
    model = YOLO(args.weights)
    names = (model.names if isinstance(model.names, dict)
             else dict(enumerate(model.names or [])))
    print(f"  task    : {getattr(model, 'task', '?')}")
    cls = [names[k] for k in sorted(names)]
    print(f"  classes : {cls if len(cls) <= 12 else cls[:12] + ['...']}")

    frames = [(i, f) for i, f in iter_sampled_frames(args.input, args.frames)]
    if not frames:
        print("No frames readable")
        return 1
    h, w = frames[0][1].shape[:2]
    print(f"  frames  : {len(frames)} sampled from {args.input} ({w}x{h})")

    variants = [("raw", None)]
    if not args.no_enhance:
        # The pipeline enhances before detecting. A checkpoint trained on untouched
        # camera frames can react badly to CLAHE, so the comparison is worth making.
        spec = resolve_spec(cfg, assess_video(args.input, cfg))
        if spec.enabled:
            variants.append(("enhanced", spec))
            print(f"  enhance : {spec.describe()}")

    print(f"\n{'variant':<10}{'imgsz':>7}{'conf':>7}{'dets':>7}  by class")
    print("-" * 78)

    best = (0, None)
    for label, spec in variants:
        imgs = [f if spec is None else enhance_frame(f, spec) for _, f in frames]
        for size in sizes:
            for conf in confs:
                counts: dict[str, int] = {}
                total = 0
                for img in imgs:
                    r = model.predict(img, conf=conf, imgsz=size, verbose=False)[0]
                    b = getattr(r, "boxes", None)
                    if b is None or not len(b):
                        continue
                    for cid in b.cls.int().cpu().tolist():
                        nm = names.get(int(cid), str(cid))
                        counts[nm] = counts.get(nm, 0) + 1
                        total += 1
                shown = dict(sorted(counts.items(), key=lambda kv: -kv[1])[:5])
                print(f"{label:<10}{size:>7}{conf:>7.2f}{total:>7}  "
                      f"{shown if shown else '-'}")
                if total > best[0]:
                    best = (total, (label, spec, size, conf))

    print("-" * 78)
    if best[0] == 0:
        print("""
  ZERO detections anywhere, including at conf 0.01.

  The checkpoint does not recognise anything in this footage. That is a model/domain
  problem, not a pipeline one — no threshold, zone or gate change will produce
  detections that were never made.

  The usual cause is surface type. RDD2022 models are trained overwhelmingly on SEALED
  BITUMINOUS roads; their crack classes are fractures in asphalt. On an unpaved gravel
  or earth road there is no asphalt to crack, and a gravel depression looks nothing
  like the dark sharp-edged asphalt pothole the model learned.

  Options, in order of effort:
    1. Try a checkpoint trained on unpaved / Indian rural roads.
    2. Fine-tune on your own footage. ~200-400 labelled frames is enough to start:
         python run.py preprocess --input <video>      # writes frames to label
         python run.py annotate   --frames data/rectified
         python run.py train      --labels data/labels
    3. Meanwhile, rely on the label-free stages, which need no model at all:
       edge damage, drainage pooling, surface texture and the rutting proxy.
""")
    else:
        total, (label, _, size, conf) = best
        print(f"""
  Best raw result: {total} detections at imgsz {size}, conf {conf:.2f} ({label}).

  So the detector DOES fire on this footage, and a zero-defect pipeline run means
  something downstream removed them. Check, in this order:
    * inference.conf         — must be at or below {conf:.2f}
    * assessment zones       — 'rejected out of zone' in summary.json; a class is only
                               assessed within its resolvable range
    * road gating            — 'rejected off-road'; the detection must overlap the mask
    * confusers              — 'rejected as confusers'
    * inference.min_track_len — a defect must persist across frames to be counted
""")

    # Write annotated frames at the most permissive setting so the result is visible
    # rather than only tabulated.
    lo = min(confs)
    hi = max(sizes)
    for idx, frame in frames:
        r = model.predict(frame, conf=lo, imgsz=hi, verbose=False)[0]
        cv2.imwrite(str(out_dir / f"frame_{idx:07d}_conf{lo}.jpg"), r.plot())
    print(f"  Annotated frames (conf {lo}, imgsz {hi}) -> {out_dir}")

    (out_dir / "diagnose.json").write_text(json.dumps({
        "weights": str(args.weights), "task": getattr(model, "task", None),
        "classes": [names[k] for k in sorted(names)],
        "max_detections": best[0],
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
