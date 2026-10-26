# %% [markdown]
# # ⚽ Full matches: events, score and highlight clips (Kaggle, 2 × T4)
#
# For each of 3 full SoccerNet matches (both halves, 720p):
# 1. learns the two teams' shirt colours from frames across the whole match
# 2. runs the pipeline at **5 frames per second** over both halves and keeps the **game state**: where the ball
#    (both ball models merged: the detector on one frame, BallNet-L on three) and every player are on the pitch (from
#    the minimap's homography), their team and role, who holds the ball
# 3. finds **events by rules** on that state (no extra model): kick-offs, corners, goal kicks, penalties and
#    throw-ins from how play restarts, and **goals** from the kick-offs that follow them, which gives the **score**
# 4. checks them against SoccerNet's labels of the same match and the real final score
# 5. (run 2, `RENDER_CLIPS = True`) renders a ~14 s clip around each event: banner + running score + minimap
#
# One worker per GPU (`scripts/process_matches.py`); each first runs every step on 20 s pieces, so a broken step
# fails in minutes. Inputs are private datasets: the 3 models, and the videos + labels (downloaded from SoccerNet's
# Hugging Face repos, which need a token, so they are uploaded rather than fetched here).

# %%
RENDER_CLIPS = True    # run 1: game state + events + evaluation; run 2 (rules tuned on run 1's states): True
HOURS = 2.0            # time budget: the workers stop with an error after this (finished halves and clips are kept)
MATCHES = {            # file prefix in the videos dataset -> SoccerNet game folder name (it holds the real score)
    "barcelona_psg": "2017-03-08 - 22-45 Barcelona 6 - 1 Paris SG",
    "dortmund_leverkusen": "2017-03-04 - 17-30 Dortmund 6 - 2 Bayer Leverkusen",
    "milan_sassuolo": "2016-10-02 - 19-00 AC Milan 4 - 3 Sassuolo",
}

# %%
import glob, json, os, shutil, subprocess, sys, time
from pathlib import Path
START = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.178", "supervision==0.31.0",
                "trackers==2.6.1", "umap-learn", "sentencepiece"], check=True)
subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/hamdotexe/football.git", "/tmp/football"],
               check=True)
print(subprocess.run(["git", "-C", "/tmp/football", "log", "--oneline", "-1"], capture_output=True, text=True).stdout)
sys.path.insert(0, "/tmp/football/src")
import footy.match, footy.ball, footy.calibration   # every pipeline import works before any GPU work


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert len(hits) == 1, f"{name}: expected exactly 1 input file, found {hits}"
    return hits[0]


models = {"detector": find("football_yolo11m_1280.pt"), "pitch": find("pitch_keypoints.pt"),
          "ballnet": find("ballnet_l_run8.pt")}
jobs = [{"name": name, "halves": [find(f"{key}_1.mkv"), find(f"{key}_2.mkv")], "labels": find(f"{key}_labels.json")}
        for key, name in MATCHES.items()]
OUT = Path("/kaggle/working/matches")
if RENDER_CLIPS:   # run 2: game state + team colours from run 1 (uploaded as a dataset), so only the clips are new
    import gzip
    for key, name in MATCHES.items():
        (OUT / name).mkdir(parents=True, exist_ok=True)
        shutil.copy(find(f"{key}_teams.pkl"), OUT / name / "teams.pkl")
        for h in (1, 2):               # Kaggle unpacks .gz files of a dataset: <key>_state_1.jsonl/<inner>.jsonl
            hits = (glob.glob(f"/kaggle/input/**/{key}_state_{h}.jsonl.gz", recursive=True)
                    + glob.glob(f"/kaggle/input/**/{key}_state_{h}.jsonl/*.jsonl", recursive=True))
            assert len(hits) == 1, f"state of {key} half {h}: expected exactly 1 input file, found {hits}"
            if hits[0].endswith(".gz"):
                shutil.copy(hits[0], OUT / name / f"state_{h}.jsonl.gz")
            else:
                with open(hits[0], "rb") as src, gzip.open(OUT / name / f"state_{h}.jsonl.gz", "wb") as dst:
                    shutil.copyfileobj(src, dst)
            assert len(footy.match.load_states(OUT / name / f"state_{h}.jsonl.gz")) > 10000, f"{key} half {h}: too short"
print(json.dumps(models, indent=1))

# %%
import torch
n_gpu = torch.cuda.device_count()
assert n_gpu >= 1, "no GPU: this notebook needs the GPU T4 x2 accelerator"
print(n_gpu, "x", torch.cuda.get_device_name(0))
logs = Path("/kaggle/working/logs")
logs.mkdir(parents=True, exist_ok=True)
workers = []
for g in range(n_gpu):
    mine = jobs[g::n_gpu]
    if not mine:
        continue
    Path(f"/tmp/jobs_{g}.json").write_text(json.dumps(mine))
    cmd = [sys.executable, "-u", "/tmp/football/scripts/process_matches.py", "--jobs", f"/tmp/jobs_{g}.json",
           "--out", str(OUT), "--detector", models["detector"], "--pitch", models["pitch"],
           "--ballnet", models["ballnet"], "--device", "0", "--stop-at", str(START + HOURS * 3600)]
    cmd += ["--render"] if RENDER_CLIPS else []
    log = open(logs / f"gpu{g}.log", "w")
    workers.append((g, log, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                             env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)})))
    print(f"GPU {g}:", [j["name"] for j in mine])


def last_line(g):
    lines = (logs / f"gpu{g}.log").read_text(errors="replace").strip().splitlines()
    return lines[-1][-200:] if lines else ""


# fail fast: if one worker stops with an error, the other is stopped too (finished halves and clips stay saved)
ticks = 0
while any(p.poll() is None for _, _, p in workers):
    time.sleep(30)
    ticks += 1
    if any(p.poll() not in (None, 0) for _, _, p in workers):
        for _, _, p in workers:
            if p.poll() is None:
                p.terminate()
        break
    if ticks % 20 == 0:
        print(f"{(time.time() - START) / 60:.0f} min |", " | ".join(f"GPU {g}: {last_line(g)}" for g, _, _ in workers),
              flush=True)
for g, log, p in workers:
    p.wait()
    log.close()
    print(f"\n===== GPU {g} (exit code {p.returncode}) =====")
    print("\n".join(l for l in (logs / f"gpu{g}.log").read_text(errors="replace").splitlines()
                    if "it/s" not in l and "s/it" not in l)[-6000:])
failed = [g for g, _, p in workers if p.returncode != 0]
if failed:
    raise SystemExit(f"worker(s) on GPU {failed} stopped with an error: see the ERROR above")

# %% [markdown]
# ## How the rules did, against SoccerNet's labels (visible events)

# %%
for name in MATCHES.values():
    s = json.load(open(OUT / name / "events.json"))
    print(f"\n{name}: our score {s['score_ours'][0]}-{s['score_ours'][1]}, real {s['score_true']}")
    for kind, r in s["vs_soccernet_labels"].items():
        print(f"  {kind:10s} found {r['found']:3d} | labelled {r['labelled_visible']:3d} | matched {r['matched']:3d} | "
              f"precision {r['precision']} | recall {r['recall']}")
