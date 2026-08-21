#!/usr/bin/env python3
"""Ablate candidate false-event rules on an EXISTING events.json. No video.

WHY THIS EXISTS
    tools/score_false_events.py compares two runs that already happened. To
    decide WHICH rule to ship you have to re-run the 20-minute video once per
    candidate, and the video lives on an EC2 box. But every field a post-filter
    can read -- t, dir, area, x -- is already in the events file, so the rules
    can be applied and scored offline, in a second, before anyone pays for a run.

    Only rules expressible from those four fields are testable here. LINE_MAX_W
    (blob width) is NOT in the events file, so it cannot be ablated offline;
    it needs a re-run.

THE RITUAL (from slit_count.py, unchanged)
    One change at a time. Held-out windows scored SEPARATELY from the tuned one.
    A rule that removes false events but costs held-out recall is REJECTED --
    a false positive on an empty doorway and a missed guest are both failures,
    and the operator flagged the first while the ground truth catches the second.

USAGE
    python tools/ablate_slit_rules.py [events.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# eval/gt_entries_305_318.json is the window slit thresholds were TUNED on.
# Scoring a rule on it and reporting one number is how the six-person group
# quietly went 6 -> 4 -> 3 while held-out recall read 100%.
TUNED_WINDOW = "gt_entries_305_318.json"


def load_truth():
    """-> (false_timestamps, tolerance, [(name, (t0,t1), truth_count, is_tuned)])."""
    neg = json.loads((ROOT / "eval/gt_false_events_20min.json").read_text())
    windows = []
    for path in sorted((ROOT / "eval").glob("gt_entries_*.json")):
        g = json.loads(path.read_text())
        windows.append((path.name, tuple(g["window_s"]),
                        g.get("truth_count", len(g.get("entries", []))),
                        path.name == TUNED_WINDOW))
    return neg["false_events_s"], neg["tolerance_s"], windows, neg["impossible_reversals_s"]


def score(events, false_ts, tol, windows, reversals):
    """-> dict of the four numbers a rule is judged on.

    `held_out` is the only recall number that means anything; `tuned` is
    reported beside it so a rule that wrecks the group window is visible
    rather than averaged away.
    """
    near = lambda t: any(abs(e["t"] - t) <= tol for e in events)
    held_emitted = held_truth = tuned_emitted = tuned_truth = 0
    for _name, (t0, t1), truth, is_tuned in windows:
        n = sum(1 for e in events if t0 <= e["t"] <= t1)
        if is_tuned:
            tuned_emitted += n
            tuned_truth += truth
        else:
            held_emitted += n
            held_truth += truth
    still_reversed = sum(1 for a, b in reversals if near(a) and near(b))
    return {
        "events": len(events),
        "false_present": sum(1 for t in false_ts if near(t)),
        "held_emitted": held_emitted,
        "held_truth": held_truth,
        "tuned_emitted": tuned_emitted,
        "tuned_truth": tuned_truth,
        "reversals": still_reversed,
    }


def drop_reversals(events, min_gap_s=3.0):
    """Remove the SECOND of any two opposite-direction events closer than
    min_gap_s. A 0.3s IN->OUT is one person moving near the line, not two
    completed crossings -- the operator's core verdict in new.txt.

    The second is dropped, not the first: the first event is the one with a
    real approach behind it, and the reversal is the trailing edge of the same
    body leaving the slit.
    """
    out = []
    for e in sorted(events, key=lambda z: z["t"]):
        if out and e["dir"] != out[-1]["dir"] and e["t"] - out[-1]["t"] < min_gap_s:
            continue
        out.append(e)
    return out


def _x_cut(events, limit=181):
    """Keep events whose blob sits before the marble/LED end of the line.

    Raises when the field is missing so the caller reports N/A rather than a
    silent pass-through: `.get("x", 0)` would make every event satisfy the cut
    and print the rule as "no change", which is the same lie as an
    unmeasurable regression case reading green.
    """
    if any("x" not in e for e in events):
        raise KeyError("x")
    return [e for e in events if e["x"] < limit]


# Each rule declares which event fields it needs, so a run that predates a
# field reports N/A instead of a fabricated verdict.
RULES = {
    "R0 baseline (no rule)":
        lambda e: e,
    "R1 x < 181  (VALID_X_MAX: marble + LED strip end)":
        _x_cut,
    "R2 drop reversal < 3.0s":
        lambda e: drop_reversals(e, 3.0),
    "R3 area >= 500  (KNOWN BAD, kept as a control)":
        lambda e: [x for x in e if x["area"] >= 500],
    "R1+R2":
        lambda e: drop_reversals(_x_cut(e), 3.0),
    "R1+R2+R3 (all three)":
        lambda e: drop_reversals(
            [x for x in _x_cut(e) if x["area"] >= 500], 3.0),
}


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "output/slit20c_events.json"
    events = json.loads(src.read_text())
    false_ts, tol, windows, reversals = load_truth()

    print(f"source   {src.relative_to(ROOT) if src.is_relative_to(ROOT) else src}")
    print(f"truth    {len(false_ts)} operator-flagged false (+/-{tol}s) · "
          f"{len(reversals)} impossible reversals")
    for name, (t0, t1), truth, is_tuned in windows:
        print(f"         {name:32s} {t0:>6.1f}-{t1:<6.1f} truth={truth} "
              f"{'[TUNED — not a score]' if is_tuned else '[held-out]'}")
    print()
    head = (f"{'RULE':<48} {'ev':>4} {'FALSE':>6} {'held-out':>9} {'tuned':>7} {'rev':>4}")
    print(head)
    print("-" * len(head))
    for label, rule in RULES.items():
        try:
            filtered = rule(events)
        except KeyError as missing:
            print(f"{label:<48} {'N/A — events file has no ' + str(missing.args[0]):>28}")
            continue
        s = score(filtered, false_ts, tol, windows, reversals)
        print(f"{label:<48} {s['events']:>4} "
              f"{s['false_present']:>2}/{len(false_ts):<3} "
              f"{s['held_emitted']:>4}/{s['held_truth']:<4} "
              f"{s['tuned_emitted']:>3}/{s['tuned_truth']:<3} "
              f"{s['reversals']:>2}/{len(reversals)}")
    print()
    print("READ IT AS: fewer FALSE is better; held-out must NOT fall; tuned is")
    print("context only. A rule that trades held-out recall for false events is")
    print("rejected -- see MIN_AREA in tools/slit_count.py for why R3 is a control.")
    return 0


def _selftest():
    """One runnable check: the reversal filter must drop the trailing event
    of a close opposite pair and keep a well-separated pair."""
    close = [{"t": 10.0, "dir": "IN", "area": 1}, {"t": 10.3, "dir": "OUT", "area": 1}]
    assert [e["t"] for e in drop_reversals(close, 3.0)] == [10.0]
    far = [{"t": 10.0, "dir": "IN", "area": 1}, {"t": 20.0, "dir": "OUT", "area": 1}]
    assert [e["t"] for e in drop_reversals(far, 3.0)] == [10.0, 20.0]
    same = [{"t": 10.0, "dir": "IN", "area": 1}, {"t": 10.2, "dir": "IN", "area": 1}]
    assert len(drop_reversals(same, 3.0)) == 2, "same-direction pairs are not reversals"
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        raise SystemExit(main())
