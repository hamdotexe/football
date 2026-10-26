# %% [markdown]
# # ⚽ One whole match through the full pipeline (Kaggle, 2 × T4)
#
# Every frame of both halves of one match, with everything on:
# - players, goalkeepers and referees **detected and tracked** (YOLO11m at 1280 px + ByteTrack), ringed in their
#   **team's colour** (shirt colours: SigLIP → UMAP → 2 clusters) with their track id
# - the **ball** from both ball models merged (the detector on one frame, BallNet-L on three) and a physically
#   plausible pick, with a **trail** only where the ball's path makes sense (moved with the camera)
# - the **minimap** (top left) from the pitch keypoint model's homography, held steady over time
# - **event banners** and the **running score** from the rules on the game state (the same as in the clips)
# - the half's clock (bottom left) and the broadcast's sound
#
# One half per GPU (`scripts/render_full_match.py`), each first rendering 20 s as a quick check. The team colours,
# events and score come from run 1 of the match notebook (its game state, uploaded as a dataset).

# %%
MATCH = "milan_sassuolo"                          # file prefix in the videos / states datasets
NAME = "2016-10-02 - 19-00 AC Milan 4 - 3 Sassuolo"  # SoccerNet game folder name
HOURS = 7.0            # time budget: past it each half stops with an error after joining what it rendered

# %%
import glob, gzip, json, os, shutil, subprocess, sys, time
from pathlib import Path
START = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.178", "supervision==0.31.0",
                "trackers==2.6.1", "umap-learn", "sentencepiece"], check=True)
subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/hamdotexe/football.git", "/tmp/football"],
               check=True)
print(subprocess.run(["git", "-C", "/tmp/football", "log", "--oneline", "-1"], capture_output=True, text=True).stdout)
sys.path.insert(0, "/tmp/football/src")
import footy.match   # every pipeline import works before any GPU work


def find(pattern):
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    assert len(hits) == 1, f"{pattern}: expected exactly 1 input file, found {hits}"
    return hits[0]


models = {"detector": find("football_yolo11m_1280.pt"), "pitch": find("pitch_keypoints.pt"),
          "ballnet": find("ballnet_l_run8.pt")}
halves = {h: find(f"{MATCH}_{h}.mkv") for h in (1, 2)}
labels = json.load(open(find(f"{MATCH}_labels.json")))
state = Path("/tmp/state")                      # run 1's game state + team colours of this match
state.mkdir(exist_ok=True)
shutil.copy(find(f"{MATCH}_teams.pkl"), state / "teams.pkl")
for h in (1, 2):                                # Kaggle unpacks .gz files of a dataset: <name>.jsonl/<inner>.jsonl
    hits = (glob.glob(f"/kaggle/input/**/{MATCH}_state_{h}.jsonl.gz", recursive=True)
            + glob.glob(f"/kaggle/input/**/{MATCH}_state_{h}.jsonl/*.jsonl", recursive=True))
    assert len(hits) == 1, f"state of half {h}: expected exactly 1 input file, found {hits}"
    if hits[0].endswith(".gz"):
        shutil.copy(hits[0], state / f"state_{h}.jsonl.gz")
    else:
        with open(hits[0], "rb") as src, gzip.open(state / f"state_{h}.jsonl.gz", "wb") as dst:
            shutil.copyfileobj(src, dst)
    assert len(footy.match.load_states(state / f"state_{h}.jsonl.gz")) > 10000, f"half {h}: state too short"
print(json.dumps({**models, **{f"half {h}": v for h, v in halves.items()}}, indent=1))

# %%
import torch
n_gpu = torch.cuda.device_count()
assert n_gpu >= 1, "no GPU: this notebook needs the GPU T4 x2 accelerator"
print(n_gpu, "x", torch.cuda.get_device_name(0))
OUT = Path("/kaggle/working") / NAME
logs = Path("/kaggle/working/logs")
logs.mkdir(parents=True, exist_ok=True)


def worker(h, gpu):
    cmd = [sys.executable, "-u", "/tmp/football/scripts/render_full_match.py", "--video", halves[h], "--half", str(h),
           "--match-dir", str(state), "--out", str(OUT / f"{h}H_full.mp4"), "--detector", models["detector"],
           "--pitch", models["pitch"], "--ballnet", models["ballnet"], "--device", "0",
           "--stop-at", str(START + HOURS * 3600)]
    log = open(logs / f"half{h}.log", "w")
    return h, log, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})


def last_line(h):
    lines = [l for l in (logs / f"half{h}.log").read_text(errors="replace").splitlines() if l[:2].isdigit()]
    return lines[-1][-160:] if lines else ""


def wait(workers):
    """Fail fast: if one worker stops with an error, the other is stopped too."""
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
            print(f"{(time.time() - START) / 60:.0f} min |", " | ".join(f"half {h}: {last_line(h)}" for h, _, _ in workers),
                  flush=True)
    for h, log, p in workers:
        p.wait()
        log.close()
        print(f"\n===== half {h} (exit code {p.returncode}) =====")
        print("\n".join(l for l in (logs / f"half{h}.log").read_text(errors="replace").splitlines()
                        if l[:2].isdigit() or "Error" in l or "File " in l)[-4000:])
    return [h for h, _, p in workers if p.returncode != 0]


# both halves side by side on 2 GPUs; on 1 GPU one after the other
failed = wait([worker(1, 0), worker(2, 1)]) if n_gpu >= 2 else wait([worker(1, 0)]) + wait([worker(2, 0)])
if failed:
    raise SystemExit(f"half {failed} stopped with an error: see the ERROR above")

# %% [markdown]
# ## Where the events are in the two videos (checked against SoccerNet's labels)

# %%
from footy import events as ev
lines = [f"{NAME}: events in 1H_full.mp4 / 2H_full.mp4 (video time = the half's clock, bottom left)",
         "REAL = SoccerNet labels an event of that kind within 10 s (60 s for goals); the others are false alarms", ""]
for h in (1, 2):
    for e in ev.detect(footy.match.load_states(state / f"state_{h}.jsonl.gz")):
        tol = 60 if e.kind == ev.GOAL else 10
        near = [t for t, _ in ev.soccernet_events(labels, h, e.kind) if abs(t - e.t) <= tol]
        lines.append(f"{h}H {int(e.t // 60):02d}:{int(e.t % 60):02d}  {e.kind:10s} {'REAL' if near else '-'}")
(OUT / "events.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
for f in sorted(OUT.iterdir()):
    print(f"{f.stat().st_size / 1e9:6.2f} GB  {f.name}")
