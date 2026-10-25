# %% [markdown]
# # ⚽ Full matches: events, score and highlight clips (Google Colab)
#
# For each match below, this notebook:
# 1. downloads both halves (720p) from SoccerNet on Hugging Face (needs your `HF_TOKEN` Colab secret)
# 2. learns the two teams' shirt colours from frames across the whole match
# 3. runs the pipeline at **5 frames per second** over both halves and keeps the **game state**: where the ball and
#    every player are on the pitch (from the minimap's homography), their team and role, who holds the ball
# 4. finds **events by rules** on that state (no extra model): kick-offs, corners, goal kicks, penalties and
#    throw-ins from how play restarts, and **goals** from the kick-offs that follow them, which gives the **score**
# 5. checks them against SoccerNet's labels of the same match and the real final score
# 6. renders a ~14 s clip around each event (banner + running score + minimap) to pick highlights from
#
# Everything is saved to `MyDrive/football/matches/<match>/`. If Colab disconnects, run all cells again: finished
# steps are read back from Drive and skipped.
#
# **Before running:** Runtime → Change runtime type → **T4 GPU**; the 🔑 secret `HF_TOKEN` with notebook access;
# `MyDrive/football/models/` holds `football_yolo11m_1280.pt` and `pitch_keypoints.pt`.

# %%
MATCHES = [
    "europe_uefa-champions-league/2016-2017/2017-03-08 - 22-45 Barcelona 6 - 1 Paris SG",
    "germany_bundesliga/2016-2017/2017-03-04 - 17-30 Dortmund 6 - 2 Bayer Leverkusen",
    "italy_serie-a/2016-2017/2016-10-02 - 19-00 AC Milan 4 - 3 Sassuolo",
]
DRIVE = "/content/drive/MyDrive/football"
RENDER_CLIPS = True

# %%
import json, os, shutil, subprocess, sys, time
from pathlib import Path
from google.colab import drive, userdata

drive.mount("/content/drive")
token = userdata.get("HF_TOKEN")
models = {name: Path(DRIVE) / "models" / name for name in ("football_yolo11m_1280.pt", "pitch_keypoints.pt")}
missing = [str(p) for p in models.values() if not p.exists()]
assert not missing, f"upload these to Google Drive first: {missing}"

import torch
assert torch.cuda.is_available(), "no GPU: Runtime -> Change runtime type -> T4 GPU"
print("GPU:", torch.cuda.get_device_name(0))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.178", "supervision==0.31.0",
                "trackers==2.6.1", "umap-learn", "sentencepiece"], check=True)
shutil.rmtree("/content/football", ignore_errors=True)
subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/hamdotexe/football.git", "/content/football"],
               check=True)
print(subprocess.run(["git", "-C", "/content/football", "log", "--oneline", "-1"], capture_output=True, text=True).stdout)
sys.path.insert(0, "/content/football/src")

from huggingface_hub import hf_hub_download
from footy import match as M
from footy.calibration import PitchCalibrator
from footy.pipeline import Detector

detector = Detector(str(models["football_yolo11m_1280.pt"]), device=0, half=True)
calibrator = PitchCalibrator(str(models["pitch_keypoints.pt"]), device=0, half=True)

# %%
summaries = []
for game in MATCHES:
    name = game.rsplit("/", 1)[1]
    t0 = time.time()
    halves = [hf_hub_download("SoccerNet/SoccerNet_raw_HQ", f"{game}/{h}_720p.mkv", repo_type="dataset",
                              revision="videos-720p", local_dir="/content/videos", token=token) for h in (1, 2)]
    labels = json.load(open(hf_hub_download("SoccerNet/SN-Labels", f"{game}/Labels-v2.json", repo_type="dataset")))
    print(f"{name}: downloaded in {time.time() - t0:.0f} s")
    summaries.append(M.process_match(name, halves, Path(DRIVE) / "matches" / name, detector, calibrator, "cuda",
                                     labels=labels, render=RENDER_CLIPS))
    shutil.rmtree("/content/videos", ignore_errors=True)
    print(f"{name}: done in {(time.time() - t0) / 60:.0f} min\n")

# %% [markdown]
# ## How the rules did, against SoccerNet's labels (visible events)

# %%
for s in summaries:
    print(f"\n{s['match']}: our score {s['score_ours'][0]}-{s['score_ours'][1]}, real {s['score_true']}")
    for kind, r in s["vs_soccernet_labels"].items():
        print(f"  {kind:10s} found {r['found']:3d} | labelled {r['labelled_visible']:3d} | matched {r['matched']:3d} | "
              f"precision {r['precision']} | recall {r['recall']}")
