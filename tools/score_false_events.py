#!/usr/bin/env python3
"""Score a run against the operator's flagged FALSE events (new.txt).

Recall alone cannot see over-firing. This is the other half: the events a
human watched and judged to be nothing.
"""
import json, sys

gt = json.load(open("eval/gt_false_events_20min.json"))
old = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "output/slit20/events.json"))
new = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "output/slit20b/events.json"))
tol = gt["tolerance_s"]


def near(evs, t):
    return any(abs(e["t"] - t) <= tol for e in evs)


fl = gt["false_events_s"]
o = sum(1 for t in fl if near(old, t))
n = sum(1 for t in fl if near(new, t))
print(f"OPERATOR-FLAGGED FALSE EVENTS ({len(fl)} timestamps, +/-{tol}s)")
print(f"   present BEFORE : {o}")
print(f"   present AFTER  : {n}")
print(f"   removed        : {o - n}")
print()
print("IMPOSSIBLE REVERSALS")
for a, b in gt["impossible_reversals_s"]:
    # TWO DISTINCT events, not one event matching both halves. The pairs are
    # 0.3-1.6s apart and tol is 0.6s, so `near(a) and near(b)` reported
    # 482.5 <-> 482.8 as STILL PRESENT when only the single 482.8 OUT
    # survived -- a reversal that no longer exists, scored as unfixed.
    hits = {id(e) for e in new if abs(e["t"] - a) <= tol}
    hits |= {id(e) for e in new if abs(e["t"] - b) <= tol}
    still = len(hits) >= 2
    print(f"   {a} <-> {b}: {'STILL PRESENT' if still else 'gone'}")
print()
under = lambda e: sum(1 for x in e if x["area"] < 500)
rev = lambda e: sum(1 for i in range(len(e) - 1)
                    if e[i]["dir"] != e[i + 1]["dir"]
                    and e[i + 1]["t"] - e[i]["t"] < 3.0)
print(f"total events        {len(old):>4} -> {len(new)}")
print(f"IN                  {sum(1 for x in old if x['dir']=='IN'):>4} -> "
      f"{sum(1 for x in new if x['dir']=='IN')}")
print(f"OUT                 {sum(1 for x in old if x['dir']=='OUT'):>4} -> "
      f"{sum(1 for x in new if x['dir']=='OUT')}")
print(f"area < 500          {under(old):>4} -> {under(new)}")
print(f"reversals < 3s      {rev(old):>4} -> {rev(new)}")
