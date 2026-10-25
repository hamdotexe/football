"""Tests for the event rules on hand-made game states. Run: python -m pytest tests -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footy import events as ev  # noqa: E402

FPS = 5


def frames(t0, t1, ball=None, people=(), holder=None):
    """States every 1/FPS s from t0 to t1 (inclusive) with the same ball / people / holder."""
    n = int(round((t1 - t0) * FPS))
    return [{"t": round(t0 + i / FPS, 2), "calibrated": True, "ball": ball, "people": [list(p) for p in people],
             "holder": holder} for i in range(n + 1)]


def kickoff_line_up(taker_team=0):
    """Team 0 in the left half, team 1 in the right half, one player of `taker_team` at the ball."""
    people = [(i, ev.PLAYER, 0, -10.0 - i, i - 5.0) for i in range(5)] + \
             [(10 + i, ev.PLAYER, 1, 10.0 + i, i - 5.0) for i in range(5)]
    people.append((99, ev.PLAYER, taker_team, -0.5 if taker_team == 0 else 0.5, 0.3))
    return people


def test_still_ball_at_each_spot_is_the_right_restart():
    for (x, y), kind in [((51.8, 33.4), ev.CORNER), ((-50.0, 2.0), ev.GOAL_KICK), ((41.3, 0.5), ev.PENALTY)]:
        states = frames(0, 3, ball=[x, y]) + frames(3.2, 5, ball=[x - 20 * (1 if x > 0 else -1), 0.0])
        found = ev.restarts(states)
        assert [e.kind for e in found] == [kind] and abs(found[0].t - 3.0) < 0.01, (kind, found)


def test_a_moving_ball_is_no_restart():
    states = [{"t": i / FPS, "calibrated": True, "ball": [-40 + i * 0.8, 0.0], "people": [], "holder": None}
              for i in range(60)]
    assert ev.restarts(states) == []


def test_kick_off_needs_each_team_in_its_own_half():
    ok = ev.restarts(frames(0, 2, ball=[0.2, 0.1], people=kickoff_line_up(taker_team=1)))
    assert [e.kind for e in ok] == [ev.KICK_OFF] and ok[0].team == 1 and ok[0].note == "teams in own halves"
    mixed = [(i, ev.PLAYER, i % 2, -10.0 + 5 * i, 0.0) for i in range(8)]    # both teams on both sides
    assert ev.restarts(frames(0, 2, ball=[0.2, 0.1], people=mixed)) == []


def test_throw_in_is_a_ball_overhead_at_the_touchline():
    thrower = [7, ev.PLAYER, 1, 12.0, -33.9, 0.05]
    found = ev.restarts(frames(0, 1.2, holder=thrower))
    assert [(e.kind, e.team) for e in found] == [(ev.THROW_IN, 1)]
    dribbler = [7, ev.PLAYER, 1, 12.0, -33.9, 0.9]                            # ball at the feet: no throw-in
    assert ev.restarts(frames(0, 1.2, holder=dribbler)) == []


def test_every_kick_off_after_the_first_follows_a_goal_scored_by_the_other_team():
    states = (frames(0, 2, ball=[0.0, 0.0], people=kickoff_line_up(0))       # the half's kick-off
              + frames(2.2, 500, ball=[10.0, 5.0 + 0])[::50]                 # play (sparse)
              + frames(540, 541, ball=[53.0, 1.0])                            # ball in the right goal
              + frames(600, 602, ball=[0.1, 0.0], people=kickoff_line_up(1)))  # team 1 kicks off: team 0 scored
    found = ev.detect(states)
    goals = [e for e in found if e.kind == ev.GOAL]
    assert len(goals) == 1 and goals[0].team == 0 and abs(goals[0].t - 541) < 0.01
    assert sorted(e.kind for e in found) == [ev.GOAL, ev.KICK_OFF, ev.KICK_OFF]   # the ball in the net isn't a goal kick


def test_matching_is_one_to_one_within_the_tolerance():
    assert ev.match_events([10, 11, 50], [10.5, 49, 90], tolerance_s=2) == (2, 3, 3)
