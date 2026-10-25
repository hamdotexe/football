"""Match events from the game state (pitch positions per frame, see footy.pipeline.frame_state), by rules.

Restarts are recognised by how play restarts: it takes seconds, the camera shows it, and the ball is on the ground
(the minimap places the ball as if on the grass, which is only right when it is):
- kick-off: the ball sits still on the centre spot (and, when teams are known, each team is in its own half)
- corner: the ball sits still at a corner
- goal kick: the ball sits still in a goal area
- penalty: the ball sits still on a penalty spot
- throw-in: a player whose feet are at a touchline holds the ball at or above the head
Goals: kick-offs only start a half or restart play after a goal, so every later kick-off follows a goal, and the
team taking it is the team that conceded.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from footy.pitch import GOAL_AREA_DEPTH, GOAL_AREA_HALF_WIDTH, HL, HW, PENALTY_SPOT

KICK_OFF, CORNER, GOAL_KICK, PENALTY, THROW_IN, GOAL = "Kick-off", "Corner", "Goal kick", "Penalty", "Throw-in", "Goal"
PLAYER = 2                      # role ids of the detector (footy.soccernet.CLASS_NAMES)
GOAL_HALF_WIDTH = 3.66


@dataclass
class Event:
    t: float                    # seconds into the half (for a restart: when play restarts)
    kind: str
    x: float                    # pitch metres
    y: float
    team: int | None = None     # 0 / 1: the team taking the restart, or the team that scored
    note: str = ""

    def to_dict(self):
        return asdict(self)


def still_spans(states, min_s=1.0, radius_m=1.0, max_gap_s=1.0):
    """Spans where the ball sits still on the pitch: each sighting within radius_m of the span's median position,
    no gap longer than max_gap_s, lasting at least min_s. -> [(t_start, t_end, x, y)]"""
    pts = [(s["t"], *s["ball"]) for s in states
           if s.get("ball") and abs(s["ball"][0]) <= HL + 3 and abs(s["ball"][1]) <= HW + 3]
    spans, i = [], 0
    while i < len(pts):
        j, xs, ys = i, [pts[i][1]], [pts[i][2]]
        while (j + 1 < len(pts) and pts[j + 1][0] - pts[j][0] <= max_gap_s
               and np.hypot(pts[j + 1][1] - np.median(xs), pts[j + 1][2] - np.median(ys)) <= radius_m):
            j += 1
            xs.append(pts[j][1])
            ys.append(pts[j][2])
        if pts[j][0] - pts[i][0] >= min_s:
            spans.append((pts[i][0], pts[j][0], float(np.median(xs)), float(np.median(ys))))
            i = j + 1
        else:
            i += 1
    return spans


def place(x, y):
    """Which restart a still ball at (x, y) means, or None (the projection is good to ~1-2 m, hence the margins)."""
    ax, ay = abs(x), abs(y)
    if np.hypot(x, y) <= 2.0:
        return KICK_OFF
    if np.hypot(ax - (HL - PENALTY_SPOT), y) <= 1.5:
        return PENALTY
    if ax >= HL - 2.5 and ay >= HW - 2.5:
        return CORNER
    if HL - GOAL_AREA_DEPTH - 1.0 <= ax <= HL - 0.3 and ay <= GOAL_AREA_HALF_WIDTH + 1.0:   # not in the net
        return GOAL_KICK
    return None


def _window(states, t0, t1):
    return [s for s in states if t0 <= s["t"] <= t1]


def kick_off_sides(states, t0, t1, min_seen=3, share=0.7):
    """At a kick-off each team stands in its own half. -> (True/False, {team: side -1/+1}), or (None, {}) when too
    few players with a known team were seen to tell."""
    xs = {0: [], 1: []}
    for s in _window(states, t0, t1):
        for _, role, team, x, _ in s["people"]:
            if role == PLAYER and team in xs and abs(x) >= 1.0:
                xs[team].append(np.sign(x))
    if min(len(v) for v in xs.values()) < min_seen:
        return None, {}
    sides = {t: float(np.sign(np.mean(v)) or 1.0) for t, v in xs.items()}
    own = all(np.mean(np.array(v) == sides[t]) >= share for t, v in xs.items())
    return own and sides[0] != sides[1], sides


def formation(state, min_players=4, share=0.85, ball_m=12.0):
    """Kick-off formation in one frame: each team (>= min_players seen) stands in its own half, on opposite sides,
    and the ball, when seen, is near the centre spot. -> {team: side -1/+1}, or None."""
    xs = {0: [], 1: []}
    for _, role, team, x, _ in state["people"]:
        if role == PLAYER and team in xs and abs(x) >= 1.0:
            xs[team].append(np.sign(x))
    if min(len(v) for v in xs.values()) < min_players:
        return None
    ball = state.get("ball")
    if ball is not None and np.hypot(*ball) > ball_m:
        return None
    sides = {t: float(np.sign(np.mean(v)) or 1.0) for t, v in xs.items()}
    if sides[0] == sides[1] or any(np.mean(np.array(v) == sides[t]) < share for t, v in xs.items()):
        return None
    return sides


def formation_kick_offs(states, min_s=0.6, max_gap_s=0.6, **kw):
    """Kick-offs from the formation (see `formation`) lasting min_s: after a goal the broadcast often returns from the
    replay just after the kick, too late to see the ball still on the spot, while the teams are still lined up.
    The team kicking off plays the ball back into its own half, so it is the team whose half the ball is in."""
    events, run = [], []

    def close():
        if run and run[-1][0] - run[0][0] >= min_s:
            sides = run[0][1]
            bx = [s["ball"][0] for _, _, s in run if s.get("ball") and abs(s["ball"][0]) >= 1.0]
            team = next((t for t, side in sides.items() if bx and side == np.sign(np.median(bx))), None)
            events.append(Event(run[0][0], KICK_OFF, 0.0, 0.0, team, "formation"))

    for s in states:
        sides = formation(s, **kw)
        if sides is not None and run and (s["t"] - run[-1][0] > max_gap_s or sides != run[-1][1]):
            close()
            run = []
        if sides is not None:
            run.append((s["t"], sides, s))
    close()
    return events


def taker(states, t0, t1, x, y):
    """Team of the player nearest the ball in the last second of a still span (the one about to play it)."""
    best = None
    for s in _window(states, t1 - 1.0, t1):
        for _, role, team, px, py in s["people"]:
            if role in (1, PLAYER) and team in (0, 1):
                d = np.hypot(px - x, py - y)
                if best is None or d < best[0]:
                    best = (d, team)
    return best[1] if best and best[0] <= 5.0 else None


def throw_ins(states, min_s=0.6, max_gap_s=0.6, head=0.2, line_m=2.0):
    """Runs of frames where the ball is at or above the head of a player standing at a touchline."""
    events, run = [], []

    def close():
        if run and run[-1][0] - run[0][0] >= min_s:
            h = run[-1][1]
            events.append(Event(run[-1][0], THROW_IN, h[3], h[4], h[2] if h[2] in (0, 1) else None))

    for s in states:
        h = s.get("holder")
        hit = h is not None and h[5] < head and abs(h[4]) >= HW - line_m and abs(h[3]) <= HL
        if hit and run and (s["t"] - run[-1][0] > max_gap_s or np.sign(h[4]) != np.sign(run[-1][1][4])):
            close()
            run = []
        if hit:
            run.append((s["t"], h))
    close()
    return events


def restarts(states):
    events = []
    for t0, t1, x, y in still_spans(states):
        kind = place(x, y)
        if kind is None:
            continue
        ev = Event(t1, kind, x, y, taker(states, t0, t1, x, y))
        if kind == KICK_OFF:
            own_halves, _ = kick_off_sides(states, t0, t1)
            if own_halves is False:
                continue                            # players of both teams in both halves: not a kick-off
            ev.note = "teams in own halves" if own_halves else "teams not seen"
        events.append(ev)
    return dedupe(events + throw_ins(states) + formation_kick_offs(states))


def dedupe(events, within_s=8.0, kick_off_within_s=60.0):
    """One event per restart: the same kind again within within_s is the same restart (keep the later one). Kick-offs
    are a minute apart at the very least (a goal, its celebration and replays come between), and the formation before
    one can start long before the kick."""
    out = []
    for ev in sorted(events, key=lambda e: e.t):
        window = kick_off_within_s if ev.kind == KICK_OFF else within_s
        if out and out[-1].kind == ev.kind and ev.t - out[-1].t <= window:
            out[-1] = ev
        else:
            out.append(ev)
    return out


def goals(states, events, half_start_within_s=600.0, lookback_s=(20.0, 240.0)):
    """A goal before every kick-off except the one starting the half (the earliest, if within half_start_within_s:
    a half's video may begin a little before it). The goal's time and end come from the last sighting of the ball in
    a goal mouth before the kick-off; the scorer is the team that didn't take the kick-off."""
    kick_offs = sorted((e for e in events if e.kind == KICK_OFF), key=lambda e: e.t)
    if kick_offs and kick_offs[0].t <= half_start_within_s:
        kick_offs = kick_offs[1:]
    out = []
    for ko in kick_offs:
        mouth = [s for s in states if s.get("ball") and ko.t - lookback_s[1] <= s["t"] <= ko.t - lookback_s[0]
                 and abs(s["ball"][0]) >= HL - 0.5 and abs(s["ball"][1]) <= GOAL_HALF_WIDTH + 1.5]
        scorer = 1 - ko.team if ko.team in (0, 1) else None
        if mouth:
            m = mouth[-1]
            out.append(Event(m["t"], GOAL, m["ball"][0], m["ball"][1], scorer, "ball seen in the goal mouth"))
        else:
            out.append(Event(ko.t - 60.0, GOAL, 0.0, 0.0, scorer, "time estimated from the kick-off"))
    return out


def detect(states):
    """All events of one half, in time order."""
    events = restarts(states)
    return sorted(events + goals(states, events), key=lambda e: e.t)


# --- scoring against SoccerNet's labels (Labels-v2.json) --------------------------------------------------------
SOCCERNET_LABEL = {KICK_OFF: "Kick-off", CORNER: "Corner", THROW_IN: "Throw-in", PENALTY: "Penalty", GOAL: "Goal",
                   GOAL_KICK: "Clearance"}


def soccernet_events(labels: dict, half: int, kind: str):
    """Times (s) of one kind of labelled event in one half, with their visibility."""
    want = SOCCERNET_LABEL[kind]
    return [(int(a["position"]) / 1000, a.get("visibility", "visible")) for a in labels["annotations"]
            if a["label"] == want and a["gameTime"].strip().startswith(str(half))]


def match_events(ours, truth, tolerance_s):
    """Greedy one-to-one matching in time. -> (matched, len(ours), len(truth))"""
    used, matched = set(), 0
    for t in sorted(ours):
        cands = [(abs(t - g), i) for i, g in enumerate(truth) if i not in used and abs(t - g) <= tolerance_s]
        if cands:
            used.add(min(cands)[1])
            matched += 1
    return matched, len(ours), len(truth)
