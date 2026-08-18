"""test_check_closure.py — measure IN/OUT accuracy with no ground truth.

WHY THIS EXISTS
    Entry counting is currently UNMEASURABLE here. The 100 hand-labelled frames
    span 12.5 seconds and contain zero entries, so there is nothing to score
    against — not a bad score, no score. That is how six fixes to the counting
    path got shipped and later retracted: nothing could tell a fix from a
    regression.

    Physics gives one constraint for free: over a closed period, everyone who
    entered also left. IN must equal OUT and occupancy must end at zero. Any
    gap is our own error, quantified, with nobody labelling anything.

    Two impossible states are worth as much as the gap itself:
      occupancy < 0        an OUT for somebody never counted IN
      occupancy > capacity double-counting at the line
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "check_closure.py"
fail = 0


def check(ok, what, detail=""):
    global fail
    print(f"  {'PASS' if ok else 'FAIL'}  {what}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        fail = 1


def run(crossings, *args):
    p = Path(tempfile.mktemp(suffix=".json"))
    p.write_text(json.dumps(crossings))
    r = subprocess.run([sys.executable, str(TOOL), str(p), *args],
                       capture_output=True, text=True)
    return r.stdout


def ev(t, d, line="entry line"):
    return {"t": t, "direction": d, "line": line}


print("=" * 74)
print("  occupancy closure — IN/OUT error without any labels")
print("=" * 74)

# 1. a perfectly balanced night
bal = [ev(i, "in") for i in range(1, 11)] + [ev(100 + i, "out") for i in range(10)]
out = run(bal, "--closed-period")
check("+0 over 10 entries = 0.0% error" in out or "0.0% error" in out,
      "balanced night reports 0% closure error")
# the message wraps, so match on a fragment that cannot straddle the
# line break — an earlier version failed on correct output for this.
check("does NOT prove the count is" in out,
      "and says plainly that closure alone does not prove correctness",
      "equal numbers of missed INs and OUTs also close")

# 2. missed exits -> positive gap, named correctly
miss_out = [ev(i, "in") for i in range(1, 11)] + [ev(100 + i, "out") for i in range(7)]
out = run(miss_out, "--closed-period")
check("EXITS are being missed" in out, "surplus INs is diagnosed as missed exits",
      "+3 of 10")

# 3. missed entries -> negative gap AND an impossible occupancy
miss_in = [ev(1, "in")] + [ev(10 + i, "out") for i in range(4)]
out = run(miss_in)
check("IMPOSSIBLE" in out and "occupancy went to" in out,
      "occupancy below zero is flagged as impossible")

# 4. double counting -> exceeds capacity
dbl = [ev(i, "in") for i in range(1, 41)]
out = run(dbl, "--capacity", "10")
check("IMPLAUSIBLE" in out and "double-counting" in out,
      "peak above capacity is flagged as double-counting")

# 5. a partial chunk must NOT be treated as a closure failure
part = [ev(i, "in") for i in range(1, 6)]
out = run(part)
check("EXPECTED" in out and "full night" in out,
      "a partial chunk is not scored as an error",
      "people are legitimately still inside")

# 6. one door's bias must not hide inside a balanced total
hidden = ([ev(i, "in", "door A") for i in range(1, 11)]
          + [ev(50 + i, "out", "door A") for i in range(5)]
          + [ev(60 + i, "in", "door B") for i in range(5)]
          + [ev(70 + i, "out", "door B") for i in range(10)])
out = run(hidden, "--closed-period")
check("biased on its own" in out,
      "per-door bias is surfaced even when the TOTAL balances",
      "door A +5, door B -5, total 0")

print()
print("  ALL PASS" if not fail else "  FAILURES ABOVE")
sys.exit(fail)
