#!/usr/bin/env python3
"""check_closure.py — measure IN/OUT accuracy WITHOUT any ground truth.

THE PROBLEM THIS SOLVES
    We cannot currently score entry counting at all. The 100 hand-labelled
    frames span 12.5 seconds and contain zero entries, so there is nothing to
    compare against — not "it scores badly", literally no measurement. Every
    change to the counting path has therefore been unverifiable, which is how
    six separate fixes got shipped and later retracted.

    Labelling entry events would fix that, and costs a human an hour of video.
    But there is a constraint that costs nothing, because it comes from physics
    rather than from labels:

        OVER A CLOSED PERIOD, EVERYONE WHO ENTERED ALSO LEFT.

    So IN must equal OUT, and occupancy must return to zero. Any gap is our own
    error, measured exactly, with nobody labelling anything. This is what
    commercial counting vendors use to self-audit, and it is principle 4 of the
    four that separate a 95% system from a 99% one.

WHAT IT CHECKS
    closure      |IN - OUT| over the period. Should be ~0 for a full night.
                 A persistent gap is systematic bias, and its SIGN says which
                 way: more INs than OUTs means exits are being missed.
    negative     occupancy below zero is impossible. It means an OUT was
                 counted for somebody who was never counted IN — a phantom, a
                 reflection, or a track that entered during a blind spot.
    ceiling      occupancy above plausible venue capacity means double-counting
                 (one person split into two ids at the line).
    per door     each door separately, because a single door's bias can hide
                 inside a total that happens to balance.

HOW TO READ IT
    A 10-minute chunk is NOT a closed period — people legitimately remain
    inside when it ends, so a gap there is expected and means nothing. Run it
    over a full night, open to close. That is the only interval where the
    physics applies.

USAGE
    python3 tools/check_closure.py output/hour/debug/CAM.112_crossings.json
    python3 tools/check_closure.py output/hour/SUMMARY.txt --from-summary
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def load_crossings(p):
    """[{t, direction, line, track_id}] from whatever the run wrote."""
    p = Path(p)
    txt = p.read_text(encoding="utf-8", errors="replace")
    if p.suffix == ".json":
        data = json.loads(txt)
        if isinstance(data, dict):
            data = data.get("crossings") or data.get("events") or []
        return [c for c in data if isinstance(c, dict) and c.get("direction")]
    # fall back to scraping a log/summary: "dining entry  IN 2 | OUT 1"
    out = []
    for m in re.finditer(r"([\w .]+?)\s+IN\s+(\d+)\s*\|\s*OUT\s+(\d+)", txt):
        line, i, o = m.group(1).strip(), int(m.group(2)), int(m.group(3))
        out += [{"line": line, "direction": "in", "t": 0.0}] * i
        out += [{"line": line, "direction": "out", "t": 0.0}] * o
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--capacity", type=int, default=0,
                    help="plausible max people inside. 0 = skip that check.")
    ap.add_argument("--closed-period", action="store_true",
                    help="assert this covers open->close, so IN must equal OUT. "
                         "On a partial chunk the gap is expected and means "
                         "nothing.")
    a = ap.parse_args()

    cr = load_crossings(a.path)
    if not cr:
        print("no crossings found in that file")
        return 1

    per = defaultdict(lambda: {"in": 0, "out": 0})
    for c in cr:
        per[c.get("line") or "?"][c["direction"]] += 1
    tin = sum(v["in"] for v in per.values())
    tout = sum(v["out"] for v in per.values())

    print("=" * 70)
    print(f"  OCCUPANCY CLOSURE · {Path(a.path).name}")
    print("=" * 70)
    print(f"\n{'door':<22}{'IN':>7}{'OUT':>7}{'gap':>7}")
    for line, v in sorted(per.items()):
        print(f"{line:<22}{v['in']:>7}{v['out']:>7}{v['in']-v['out']:>+7}")
    print(f"{'TOTAL':<22}{tin:>7}{tout:>7}{tin-tout:>+7}")

    # occupancy trace over time — only meaningful if crossings carry timestamps
    timed = [c for c in cr if c.get("t")]
    lo = hi = 0
    if timed:
        occ = 0
        for c in sorted(timed, key=lambda c: c["t"]):
            occ += 1 if c["direction"] == "in" else -1
            lo, hi = min(lo, occ), max(hi, occ)
        print(f"\noccupancy trace   min {lo}   max {hi}   final {occ}")

    print("\nWHAT THIS MEANS")
    bad = False
    if lo < 0:
        print(f"  IMPOSSIBLE: occupancy went to {lo}. An OUT was counted for")
        print(f"     somebody never counted IN — a phantom, a reflection, or")
        print(f"     someone who entered through a blind spot.")
        bad = True
    if a.capacity and hi > a.capacity:
        print(f"  IMPLAUSIBLE: peak {hi} exceeds capacity {a.capacity}.")
        print(f"     Suggests double-counting — one person split into two ids")
        print(f"     at the line.")
        bad = True
    if a.closed_period:
        gap = tin - tout
        rate = 100.0 * abs(gap) / max(1, tin)
        print(f"  CLOSURE: {gap:+d} over {tin} entries = {rate:.1f}% error")
        if abs(gap) == 0:
            print("     Perfect closure. Note this does NOT prove the count is")
            print("     right — equal numbers of missed INs and OUTs also close.")
        elif gap > 0:
            print("     More INs than OUTs: EXITS are being missed.")
        else:
            print("     More OUTs than INs: ENTRIES are being missed, or exits")
            print("     are double-counted.")
        # per-door, because a balanced total can hide two opposite biases
        for line, v in sorted(per.items()):
            g = v["in"] - v["out"]
            if v["in"] + v["out"] >= 4 and abs(g) > 0.25 * max(1, v["in"]):
                print(f"     door '{line}' is {g:+d} on {v['in']} entries — "
                      f"biased on its own")
        bad = bad or abs(gap) > 0
    else:
        print("  Partial chunk: IN != OUT is EXPECTED (people are still inside).")
        print("     Re-run with --closed-period over a full night for the real")
        print("     closure number. That is the only interval where the physics")
        print("     applies.")
    if not bad:
        print("  No impossible states found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
