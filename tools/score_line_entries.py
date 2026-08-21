#!/usr/bin/env python3
"""Score the PIPELINE's entry-line crossings against hand-read GT windows.

tools/score_entries.py grades tools/doorcount.py -- a separate slit counter.
Nothing graded kevacv's own entry line, which is the thing that actually feeds
the guest number, so its recall had never been measured against truth at all.

Usage:  python3 tools/score_line_entries.py <run> [<run> ...]
"""
import glob
import json
import os
import sys

# The 305-318 window was used to tune the SLIT counter's MIN_AREA. Treat it as
# tuned for anything derived from it. A score on tuned data is not a score.
TUNED = {"eval/gt_entries_305_318.json"}
# Tolerance: a hand-read "first_seen" and a line crossing are different
# instants -- the labeller marks when a person becomes visible in the doorway
# crop, the line fires when their foot point crosses. Count per WINDOW, not
# per event, and allow the window a little slack at each end.
SLACK_S = 3.0


def score(run, line_name="entry line"):
    path = f"output/{run}/debug/CAM.112_crossings.json"
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    rows = d if isinstance(d, list) else d.get("crossings", d)
    ins = sorted(float(r["t"]) for r in rows
                 if r.get("line") == line_name
                 and str(r.get("direction", "")).lower() == "in")
    out = []
    for gt_path in sorted(glob.glob("eval/gt_entries_*.json")):
        gt = json.load(open(gt_path))
        t0, t1 = gt["window_s"]
        got = sum(1 for t in ins if t0 - SLACK_S <= t <= t1 + SLACK_S)
        truth = gt.get("truth_count", len(gt.get("entries", [])))
        out.append(dict(window=os.path.basename(gt_path), kind=gt.get("kind", "-"),
                        t0=t0, t1=t1, truth=truth, got=got,
                        tuned=gt_path in TUNED))
    return ins, out


def main(argv):
    runs = argv or ["p0v4"]
    for run in runs:
        r = score(run)
        if r is None:
            print(f"{run}: no crossings file")
            continue
        ins, rows = r
        print(f"\n=== {run} ===   entry-line IN events: {len(ins)}")
        print(f"    {'window':28s} {'kind':6s} {'truth':>6s} {'got':>5s} {'':>4s}")
        held_t = held_g = 0
        for x in rows:
            tag = "TUNED" if x["tuned"] else ""
            print(f"    {x['window']:28s} {x['kind']:6s} {x['truth']:6d} "
                  f"{x['got']:5d}  {tag}")
            if not x["tuned"]:
                held_t += x["truth"]; held_g += x["got"]
        print(f"    {'HELD-OUT TOTAL':28s} {'':6s} {held_t:6d} {held_g:5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
