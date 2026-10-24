# %% [markdown]
# # ⚽ Step 7c: choose the confidence threshold for the pitch homography (CPU only, no GPU quota)
#
# The minimap needs **at least 4 confident landmarks** per frame to fit its homography. "Confident" has meant
# confidence ≥ 0.5. After training session 3, the landmarks got *more accurate* (median error 13 → 12 px) but the model
# also got more cautious: frames with 4 landmarks above 0.5 fell from 71% to 66%, and frames without a minimap count as
# failures. So the threshold, not the model, may be what holds the score back.
#
# ## The honest way to pick it
# A threshold chosen by looking at the **test** score would make that test score look better than it really is.
# So we choose on the **held-out validation clips** (the 10 valid clips no session trained on), for both the session 2
# and session 3 weights, then score the test frames **once** at that choice.
#
# The model runs once per frame here; the predictions are saved, and every threshold / RANSAC tolerance is scored from
# them afterwards (no re-running).
#
# ## The bigger catch: frames where the pitch itself isn't "detected"
# The first tuning run found that 193 of the 195 test frames without a minimap had **no prediction at all**: YOLO-pose
# first detects the object (here, the pitch) and only reports landmarks for detections above 0.25 confidence, its
# default. Every broadcast frame shows the pitch, so we keep the best detection however unsure it is
# (`--box-conf 0.01`) and let validation choose the cut-off (0.01 / 0.05 / 0.1 / 0.25) along with the others.

# %%
import glob, json, os, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)
subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/hamdotexe/football.git", "/tmp/football"], check=True)
print(subprocess.run(["git", "-C", "/tmp/football", "log", "--oneline", "-1"], capture_output=True, text=True).stdout)
sys.path.insert(0, "/tmp/football/scripts")
from pathlib import Path
from train_pitch import split_holdout

images_root = [d for d in glob.glob("/kaggle/input/**/soccernet_yolo", recursive=True) if os.path.isdir(f"{d}/valid/images")]
labels_root = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/keypoints.json", recursive=True)
               if os.path.isdir(os.path.join(os.path.dirname(p), "valid", "labels"))]
weights = {f"session{n}": glob.glob(f"/kaggle/input/**/pitch_session{n}.pt", recursive=True) for n in (2, 3)}
assert images_root and labels_root and all(weights.values()), f"inputs not found: {images_root} {labels_root} {weights}"
print("images:", images_root[0], "\nlabels:", labels_root[0], "\nweights:", weights)

# %%
# the same 712 validation frames training used: every 6th valid clip held out, every 2nd frame of those clips
labelled = {p.stem for p in Path(labels_root[0], "valid", "labels").glob("*.txt")}
valid = sorted(p for p in Path(images_root[0], "valid", "images").glob("*.jpg") if p.stem in labelled)
_, held = split_holdout(valid, 6)
val_frames = held[::2]
for sub in ("images", "labels"):
    os.makedirs(f"/tmp/valheld/{sub}", exist_ok=True)
for p in val_frames:
    os.symlink(p, f"/tmp/valheld/images/{p.name}")
    os.symlink(Path(labels_root[0], "valid", "labels", f"{p.stem}.txt"), f"/tmp/valheld/labels/{p.stem}.txt")
print(len(val_frames), "validation frames from clips", sorted({p.name.split("_")[0] for p in val_frames}))
assert len(val_frames) == 712, "not the validation frames training used"

# %%
splits = {"val": ("/tmp/valheld/images", "/tmp/valheld/labels", "1"),
          "test": (f"{images_root[0]}/test/images", f"{labels_root[0]}/test/labels", "8")}
for name, (w,) in weights.items():
    for split, (imgs, labs, stride) in splits.items():
        out = f"/kaggle/working/tune/{name}/{split}"
        code = subprocess.run([sys.executable, "-u", "/tmp/football/scripts/eval_pitch_keypoints.py", "--weights", w,
                               "--mapping", "ours", "--images", imgs, "--labels", labs, "--out", out, "--stride", stride,
                               "--imgsz", "1280", "--box-conf", "0.01", "--max-minutes", "40"]).returncode
        assert code == 0, f"evaluation of {name} on {split} stopped with exit code {code}: see the ERROR line above"
        assert json.load(open(f"{out}/metrics.json"))["complete"], f"{name} {split}: time budget ran out"

# %% [markdown]
# ## Choose on validation, score test once

# %%
runs = sum([["--run", n, f"/kaggle/working/tune/{n}/val/predictions.npz", f"/kaggle/working/tune/{n}/test/predictions.npz"]
            for n in weights], [])
subprocess.run([sys.executable, "/tmp/football/scripts/tune_pitch_conf.py", *runs, "--out", "/kaggle/working/tune/tune.json"], check=True)
