"""Unsupervised team assignment: torso crops -> SigLIP embeddings -> UMAP -> KMeans(k=2)."""
from collections import Counter, defaultdict

import numpy as np
import supervision as sv
import torch
from sklearn.cluster import KMeans
from transformers import AutoProcessor, SiglipVisionModel
from umap import UMAP

SIGLIP = "google/siglip-base-patch16-224"


def torso_crop(frame, xyxy):
    """The shirt area of a player box: middle 60% horizontally, 15-55% from the top (no grass, no legs)."""
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    return sv.crop_image(frame, [x1 + 0.2 * w, y1 + 0.15 * h, x2 - 0.2 * w, y1 + 0.55 * h])


class TeamClassifier:
    def __init__(self, device="cpu", batch_size=32):
        self.device = device
        self.batch_size = batch_size
        self.processor = AutoProcessor.from_pretrained(SIGLIP)
        self.model = SiglipVisionModel.from_pretrained(SIGLIP).to(device).eval()
        self.reducer = UMAP(n_components=3, random_state=0)
        self.kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)

    @torch.no_grad()
    def embed(self, crops):
        crops = [sv.cv2_to_pillow(c) for c in crops]
        out = []
        for i in range(0, len(crops), self.batch_size):
            inputs = self.processor(images=crops[i:i + self.batch_size], return_tensors="pt").to(self.device)
            out.append(self.model(**inputs).pooler_output.cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 768))

    def fit(self, crops):
        self.kmeans.fit(self.reducer.fit_transform(self.embed(crops)))
        return self

    def predict(self, crops):
        if len(crops) == 0:
            return np.zeros(0, dtype=int)
        return self.kmeans.predict(self.reducer.transform(self.embed(crops)))


class TrackTeamVotes:
    """Majority vote of team predictions per tracker ID, so a player can't flicker between teams."""

    def __init__(self, votes_needed=5):
        self.votes = defaultdict(Counter)
        self.votes_needed = votes_needed

    def needs_vote(self, track_id):
        return sum(self.votes[track_id].values()) < self.votes_needed

    def add(self, track_id, team):
        self.votes[track_id][int(team)] += 1

    def team(self, track_id):
        v = self.votes.get(track_id)
        return v.most_common(1)[0][0] if v else -1


def goalkeeper_teams(goalkeepers: sv.Detections, players: sv.Detections, player_teams):
    """Assign each goalkeeper to the team whose players' centre (on screen) is closest."""
    gk_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pl_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    centroids = [pl_xy[player_teams == t].mean(0) if (player_teams == t).any() else np.full(2, np.inf) for t in (0, 1)]
    return np.array([int(np.argmin([np.linalg.norm(xy - c) for c in centroids])) for xy in gk_xy], dtype=int)
