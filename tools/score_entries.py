#!/usr/bin/env python3
"""Score door counting against hand-read ground truth windows.

Reports per window and in total, and keeps HELD-OUT windows separate from
the one the parameters were tuned on -- a score on tuned data is not a score.
"""
import json, glob, gzip, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.doorcount import door_events, dedupe, _spans

TUNED = {"eval/gt_entries_305_318.json"}


def main(run):
    frames = json.load(gzip.open(f"output/{run}/debug/CAM.112_frames.json.gz", "rt"))
    kept = dedupe(door_events(frames), spans=_spans(frames))
    rows = []
    for path in sorted(glob.glob("eval/gt_entries_*.json")):
        gt = json.load(open(path))
        t0, t1 = gt["window_s"]
        got = sum(1 for e in kept if t0 <= e["t"] <= t1 and e["dir"] == "IN")
        truth = gt["truth_count"]
        rows.append((path, gt.get("kind", "?"), t0, t1, truth, got,
                     path in TUNED))
    print(f"run = {run}\n")
    print(f"  {'window':>12} {'kind':>6} {'truth':>6} {'counted':>8} "
          f"{'error':>6}   set")
    for path, kind, t0, t1, truth, got, tuned in rows:
        err = got - truth
        print(f"  {t0:5.0f}-{t1:<6.0f} {kind:>6} {truth:>6} {got:>8} "
              f"{err:>+6}   {'TUNED (not a score)' if tuned else 'held-out'}")
    # REGRESSION CANARY on the tuned window.
    #
    # Excluding the tuned window from the SCORE is right -- you cannot grade
    # yourself on data you tuned. But it is the ONLY window containing a GROUP
    # arrival (6 guests in 13s); the held-out set is 3 arrivals and 0 arrivals.
    # So while held-out recall read a clean 100%/0FP, the false-positive work
    # took the group from 6 -> 4 -> 3 completely invisibly:
    #
    #     window            truth  slit20  slit20b  slit20c  slit20d
    #     026-039 quiet         0     0       0        0        0
    #     221-234 busy          3     3       3        3        3
    #     305-318 TUNED         6     6       4        4        3   <-- halved
    #
    # Deleted: 307.7 IN(364), 308.8 IN(420), 311.4 IN(479), 312.8 IN(402) --
    # all below MIN_AREA=500, all inside a hand-labelled 6-arrival window.
    #
    # So the tuned window is still not a score, but it IS a canary: if it drops
    # while held-out looks fine, we are deleting real people and cannot see it.
    for path, kind, t0, t1, truth, got, tuned in rows:
        if tuned and got < truth:
            print(f"\n  !! CANARY: tuned window lost {truth - got} of {truth} "
                  f"real arrival(s). Held-out cannot see this -- it is the only "
                  f"window with a GROUP arrival. Do not read the score below as "
                  f"progress until this is 0.")
    ho = [r for r in rows if not r[6]]
    if ho:
        # Hits and false positives are counted PER WINDOW and never pooled.
        # Pooling let a false positive in the quiet window cancel a miss in the
        # busy one and print "33% recall" when there were zero real detections.
        hits = sum(min(r[5], r[4]) for r in ho)
        truth = sum(r[4] for r in ho)
        fp = sum(max(0, r[5] - r[4]) for r in ho)
        print(f"\n  HELD-OUT (per-window, never pooled)")
        print(f"    truth                 {truth}")
        print(f"    hits             {hits}")
        print(f"    misses           {truth - hits}")
        print(f"    false positives  {fp}")
        print(f"    recall           {hits/truth:5.0%}" if truth else "")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "covis")
