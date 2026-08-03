"""One paste-able Colab cell: upload a dataset zip, repair it, fine-tune, test.

Exists because updating the pipeline code inside a Colab runtime does NOT update the
notebook document open in the browser — Colab holds its own copy of the .ipynb, so a
`git pull` in the runtime leaves the cells untouched and new sections never appear.
Rather than making that the only route to training, this is the whole flow in a single
cell that can be pasted into any notebook, however stale.

Copy everything below the marker into one Colab cell and run it.
Runtime -> Change runtime type -> T4 GPU first; it stops rather than falling back.
"""

# --------------------------- PASTE FROM HERE ---------------------------------
CELL = r'''
#@title Fine-tune on your own dataset (upload a zip) { display-mode: "form" }
EPOCHS      = 100  #@param {type:"integer"}
TRAIN_IMGSZ = 512  #@param {type:"integer"}
BATCH       = 16  #@param {type:"integer"}
MODEL_SIZE  = "s"  #@param ["n", "s", "m", "l"]
GEOMETRY    = "box"  #@param ["box", "polygon"]
#@markdown TRAIN_IMGSZ should match your EXPORTED image size, not your video size.
#@markdown Training at 640 on a 512px export only upsamples blur and costs ~55% more
#@markdown time per epoch for nothing.

import os, subprocess, sys, time, zipfile
from pathlib import Path

REPO = Path("/content/road-defect-pipeline")

def sh(cmd, **kw):
    """Run a command, streaming output — a silent cell looks hung."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, **kw)
    for line in p.stdout:
        print(line, end="")
    p.wait()
    return p.returncode

# ---- 1. code -----------------------------------------------------------------
if REPO.exists():
    print("Updating the pipeline ...")
    sh(["git", "-C", str(REPO), "fetch", "-q", "origin"])
    sh(["git", "-C", str(REPO), "reset", "-q", "--hard", "origin/master"])
else:
    print("Cloning the pipeline ...")
    sh(["git", "clone", "-q",
        "https://github.com/Prashant2804/road-defect-pipeline.git", str(REPO)])
os.chdir(REPO)
print(subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True,
                     text=True).stdout)

print("Installing dependencies (about a minute) ...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "ultralytics", "opencv-python-headless", "pyyaml"], check=True)

# Fail now, not after the upload and the repair, if the runtime has no GPU.
import torch
if not torch.cuda.is_available():
    raise SystemExit(
        "No GPU in this runtime. Runtime -> Change runtime type -> T4 GPU, "
        "then run this cell again. Training on Colab's CPU takes many hours.")
print(f"GPU: {torch.cuda.get_device_name(0)}\n")

# ---- 2. dataset --------------------------------------------------------------
RAW = Path("/content/dataset/raw")
if not RAW.exists():
    from google.colab import files
    print("Choose your dataset .zip ...")
    up = files.upload()
    RAW.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(next(iter(up))) as z:
        z.extractall(RAW)
    # A zip usually holds one wrapper folder; descend so we see train/ valid/ test/.
    kids = [p for p in RAW.iterdir() if not p.name.startswith((".", "__"))]
    if len(kids) == 1 and kids[0].is_dir():
        RAW = kids[0]
else:
    kids = [p for p in RAW.iterdir() if not p.name.startswith((".", "__"))]
    if len(kids) == 1 and kids[0].is_dir():
        RAW = kids[0]
    print(f"Reusing the dataset already uploaded at {RAW}")
print(f"Dataset: {RAW}\n  " + ", ".join(sorted(p.name for p in RAW.iterdir())))

# ---- 3. check, then repair ---------------------------------------------------
print("\n" + "=" * 78 + "\nCHECKING THE ANNOTATIONS\n" + "=" * 78)
sh([sys.executable, "tools/check_labels.py", "--labels", str(RAW)])

print("\n" + "=" * 78 + "\nREPAIRING\n" + "=" * 78)
CLEAN = Path("/content/dataset/clean")
if sh([sys.executable, "tools/fix_labels.py", "--labels", str(RAW),
       "--out", str(CLEAN), "--to", GEOMETRY, "--rename"]) != 0:
    raise SystemExit("Repair failed — read the message above.")
sh([sys.executable, "tools/check_labels.py", "--labels", str(CLEAN)])
DATA_YAML = CLEAN / "data.yaml"

# ---- 4. train ----------------------------------------------------------------
print("\n" + "=" * 78 + "\nTRAINING\n" + "=" * 78)
t0 = time.time()
sh([sys.executable, "run.py", "train", "--data", str(DATA_YAML),
    "--device", "cuda", "--output", "out",
    "--set", "run.name=finetune",
    "--set", f"model.size={MODEL_SIZE}",
    "--set", f"model.train.epochs={EPOCHS}",
    "--set", f"model.train.imgsz={TRAIN_IMGSZ}",
    "--set", f"model.train.batch={BATCH}"])
print(f"\nTraining took {(time.time() - t0) / 60:.1f} min")

TRAINED = REPO / "out" / "finetune" / "train" / "weights" / "best.pt"
if not TRAINED.exists():
    raise SystemExit("Training did not produce weights — read the log above.")

# ---- 5. test on the held-out split -------------------------------------------
print("\n" + "=" * 78 + "\nTESTING ON THE HELD-OUT SPLIT\n" + "=" * 78)
sh([sys.executable, "run.py", "val", "--weights", str(TRAINED),
    "--data", str(DATA_YAML), "--split", "test",
    "--imgsz", str(TRAIN_IMGSZ), "--device", "cuda",
    "--set", "run.name=finetune"])

# ---- 6. hand over ------------------------------------------------------------
import yaml
names = yaml.safe_load(DATA_YAML.read_text())["names"]
WEIGHTS = TRAINED
# Identity, but load-bearing: it forces resolution BY NAME rather than by index,
# which is the only thing standing between a 6-class checkpoint and a 9-class config.
CLASS_MAP = {n: n for n in names}

print(f"\n{'=' * 78}\nDONE\n{'=' * 78}")
print(f"  weights   {WEIGHTS}")
print(f"  classes   {names}")
print(f"  plots     {REPO / 'out' / 'finetune' / 'train' / 'results.png'}")
print("\nWEIGHTS and CLASS_MAP are now set, so the inference cells will use them.")
print("Save the weights before the runtime recycles:")
print("    from google.colab import files; files.download(str(WEIGHTS))")
print("  or copy to Drive:")
print("    from google.colab import drive; drive.mount('/content/drive')")
print(f"    !cp '{WEIGHTS}' /content/drive/MyDrive/")

from IPython.display import Image, display
for n in ("results.png", "confusion_matrix_normalized.png"):
    p = REPO / "out" / "finetune" / "train" / n
    if p.exists():
        print(f"\n{n}")
        display(Image(str(p), width=900))
'''
# ---------------------------- TO HERE ----------------------------------------

if __name__ == "__main__":
    print(CELL)
