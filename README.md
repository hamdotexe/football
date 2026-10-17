# ⚽ Football Match Analysis

A multi-model computer vision pipeline that watches a football (soccer) broadcast and produces an annotated video:

- **Players, goalkeepers, referees** detected and tracked with persistent IDs
- **Team assignment** by unsupervised jersey-colour clustering
- **Ball detection and tracking**
- **Top-down minimap** via pitch keypoints and homography
- **Match events**: goals, corners, throw-ins, goal kicks

> 🚧 Work in progress, built step by step.

## Pipeline

```
video → detector (players/GK/refs) ─┐
      → ball detector (multi-frame) ─┼→ tracker → team classifier → pitch homography
                                     │                                   ↓
                                     └──────────────→ game state → event engine → annotated video + minimap
```

## Project layout

| Folder | Contents |
|---|---|
| `src/footy/` | Pipeline code |
| `notebooks/` | Training notebooks (run on Kaggle GPUs) |
| `configs/` | Settings |
| `scripts/` | Entry points |
| `data/`, `models/`, `outputs/` | Videos, weights, results (not committed) |

## Data

- [SoccerNet](https://www.soccer-net.org/): Game State Reconstruction and Tracking (research, non-commercial use)
- [Roboflow Universe](https://universe.roboflow.com/): football player detection
- Bundesliga sample clips (DFL), used as held-out test footage

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```
