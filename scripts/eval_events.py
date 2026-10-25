"""Events and score from saved game states (footy.events rules), against SoccerNet's labels, without any GPU:
the rules can be changed and re-checked in seconds.

    python scripts/eval_events.py outputs/matches                # every match folder with state_*.jsonl.gz + labels.json
    python scripts/eval_events.py outputs/matches --details Corner  # + each miss and each false alarm, with its time
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footy import events as ev  # noqa: E402
from footy.match import evaluate, load_states, score_at, true_score  # noqa: E402


def mmss(t):
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def unmatched(ours, truth, tolerance_s):
    """The same greedy matching as footy.events.match_events -> (our times without a label, labels without ours)."""
    used, extra = set(), []
    for t in sorted(ours):
        cands = [(abs(t - g), i) for i, g in enumerate(truth) if i not in used and abs(t - g) <= tolerance_s]
        if cands:
            used.add(min(cands)[1])
        else:
            extra.append(t)
    return extra, [g for i, g in enumerate(truth) if i not in used]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("matches", help="folder of match folders")
    p.add_argument("--details", nargs="*", default=[], help="event kinds to list misses / false alarms for")
    a = p.parse_args()
    totals = {}
    for d in sorted(Path(a.matches).iterdir()):
        if not (d / "state_1.jsonl.gz").exists() or not (d / "labels.json").exists():
            continue
        labels = json.loads((d / "labels.json").read_text())
        states = {h: load_states(d / f"state_{h}.jsonl.gz") for h in (1, 2) if (d / f"state_{h}.jsonl.gz").exists()}
        events = {h: ev.detect(s) for h, s in states.items()}
        goals = [(h, e.t, e.team) for h, es in events.items() for e in es if e.kind == ev.GOAL]
        final = score_at(goals, 2, 1e9)
        print(f"\n{d.name}: our score {final[0]}-{final[1]}, real {true_score(d.name)}")
        for kind, r in evaluate(events, labels).items():
            print(f"  {kind:10s} found {r['found']:3d} | labelled {r['labelled_visible']:3d} | matched {r['matched']:3d} "
                  f"| precision {r['precision']} | recall {r['recall']}")
            t = totals.setdefault(kind, [0, 0, 0, 0])
            t[0], t[1], t[2], t[3] = (t[0] + r["matched_any"], t[1] + r["found"], t[2] + r["matched"],
                                      t[3] + r["labelled_visible"])
        for kind in a.details:
            tol = 60.0 if kind == ev.GOAL else 5.0
            for h, es in events.items():
                labelled, ours = ev.soccernet_events(labels, h, kind), [e.t for e in es if e.kind == kind]
                missed = unmatched(ours, [t for t, vis in labelled if vis == "visible"], tol)[1]
                extra = unmatched(ours, [t for t, _ in labelled], tol)[0]
                print(f"  {kind} half {h}: missed {[mmss(t) for t in missed]}")
                print(f"  {kind} half {h}: false  {[mmss(t) for t in extra]}")
    print("\nall matches:")
    for kind, (m_any, n_ours, m, n_truth) in totals.items():
        print(f"  {kind:10s} precision {m_any / n_ours if n_ours else float('nan'):.2f} ({m_any}/{n_ours}) | "
              f"recall {m / n_truth if n_truth else float('nan'):.2f} ({m}/{n_truth} shown)")


if __name__ == "__main__":
    main()
