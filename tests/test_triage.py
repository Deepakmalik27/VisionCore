"""Tests for kevacv/triage.py — spend compute where the people are, honestly.

The saving is easy. The honesty is the hard part, so most of these tests are
about it: a planner that silently drops time is just a faster way to be wrong.

  * an empty night must collapse to almost no work
  * a busy night must NOT be "optimised" — if people are always present, the
    correct answer is to analyse everything and save nothing
  * an arrival at the very edge of a scan sample must not be clipped
  * every second must land in exactly one of ANALYSED / SKIPPED / UNSEEN, and
    time that was never scanned must never be reported as "empty"
  * the recall risk the scan itself introduces must be stated, not hidden

Run: python test_triage.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kevacv.triage import coverage_report, describe, miss_risk, plan_segments

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


def scan(dur_s=3600.0, step=6.0, busy=()):
    """busy: list of (t0, t1, n_people)."""
    out = []
    t = 0.0
    while t < dur_s:
        n = next((c for a, b, c in busy if a <= t < b), 0)
        out.append((t, n))
        t += step
    return out


print("=" * 74)
print("  a mostly-empty night is where the saving is")
print("=" * 74)
# a reception hour: three short bursts of activity, otherwise nobody
s = scan(3600, 6, busy=[(300, 480, 3), (1500, 1800, 5), (3000, 3100, 2)])
segs, st = plan_segments(s)
cov = coverage_report(segs, (0, 3600), (0, 3600))
print("   " + describe(st, cov, miss_risk(st["scan_step_s"])).replace("\n", "\n   "))
check(st["segments"] == 3, "one segment per burst", f"{st['segments']}")
check(st["saving_pct"] > 70, "over 70% of the hour is skipped", f"{st['saving_pct']:.0f}%")
check(st["compute_multiplier"] > 3, "which buys 3x+ compute per analysed frame",
      f"{st['compute_multiplier']:.1f}x")
check(cov["accounted"], "every second is accounted for")
check(abs(cov["analysed_pct"] + cov["skipped_pct"] + cov["unseen_pct"] - 100) < 0.2,
      "the three states sum to 100%")

print()
print("=" * 74)
print("  a busy night must NOT be 'optimised'")
print("=" * 74)
s = scan(1800, 6, busy=[(0, 1800, 4)])
segs, st = plan_segments(s)
check(st["segments"] == 1, "continuous activity -> one segment")
check(st["saving_pct"] < 2, "and essentially NO saving is claimed",
      f"{st['saving_pct']:.1f}%")
print("    -> the correct answer when people are always present is to do all the work")

s = scan(1800, 6, busy=[])
segs, st = plan_segments(s)
cov = coverage_report(segs, (0, 1800), (0, 1800))
check(segs == [], "a genuinely empty hour -> no segments at all")
check(cov["skipped_pct"] > 99, "and it is SKIPPED, not UNSEEN",
      f"skipped {cov['skipped_pct']:.0f}% / unseen {cov['unseen_pct']:.0f}%")
print("    -> 'nothing happened' is a finding, but only because it was looked at")

print()
print("=" * 74)
print("  an arrival must not be clipped by the coarse scan")
print("=" * 74)
# somebody appears one sample before the scan notices, at t=600
s = scan(1200, 6, busy=[(600, 660, 2)])
segs, st = plan_segments(s, pad_s=20.0)
t0, t1 = segs[0]
print(f"    activity 600-660 s  ->  analysed {t0:.0f}-{t1:.0f} s")
check(t0 <= 600 - 15, "the segment starts well BEFORE the first sighting",
      f"{600 - t0:.0f}s of lead-in")
check(t1 >= 660 + 15, "and ends well after the last", f"{t1 - 660:.0f}s of run-out")
check(len(segs) == 1, "still one segment")

# two bursts close together must not become two expensive starts
s = scan(1200, 6, busy=[(300, 360, 2), (390, 450, 2)])
segs, _ = plan_segments(s, merge_gap_s=45.0)
check(len(segs) == 1, "bursts 30 s apart merge into one segment", f"{len(segs)}")
s = scan(1800, 6, busy=[(300, 360, 2), (900, 960, 2)])
segs, _ = plan_segments(s, merge_gap_s=45.0)
check(len(segs) == 2, "bursts 9 min apart stay separate", f"{len(segs)}")

print()
print("=" * 74)
print("  time that was never scanned is NEVER called empty")
print("=" * 74)
# scan only covered the first half of the chunk
s = scan(1800, 6, busy=[(300, 480, 3)])
segs, st = plan_segments(s)
cov = coverage_report(segs, (0, 1800), (0, 3600))
print(f"    chunk 0-3600 s, scan covered 0-1800 s")
print(f"    ANALYSED {cov['analysed_pct']:.1f}%  SKIPPED {cov['skipped_pct']:.1f}%  "
      f"UNSEEN {cov['unseen_pct']:.1f}%")
check(cov["unseen_pct"] > 45, "the unscanned half is reported as UNSEEN",
      f"{cov['unseen_s']:.0f}s")
check(cov["accounted"], "and everything still sums to the chunk")
txt = describe(st, cov, miss_risk(st["scan_step_s"]))
check("report cannot speak for this" in txt,
      "the report says out loud that it cannot speak for unseen time")

print()
print("=" * 74)
print("  the risk the scan itself introduces")
print("=" * 74)
print(f"    {'scan every':>12s}{'typical visit':>15s}{'miss prob':>11s}   verdict")
for step, visit in [(2.0, 25.0), (6.0, 25.0), (25.0, 25.0), (60.0, 25.0), (120.0, 25.0)]:
    r = miss_risk(step, visit)
    print(f"    {step:>10.0f}s{visit:>14.0f}s{r['miss_prob']:>11.2f}   {r['verdict'][:44]}")
check(miss_risk(6.0, 25.0)["miss_prob"] == 0.0, "6 s scan cannot miss a 25 s visit")
check(miss_risk(60.0, 25.0)["miss_prob"] > 0.5, "a 60 s scan misses most short visits")
check("UNSAFE" in miss_risk(120.0, 25.0)["verdict"],
      "and an unsafe interval says so in words, not just a number")
check("safe" in miss_risk(6.0, 25.0)["verdict"], "a safe interval is stated too")
print("    -> this is the cost of triage, and it is reported rather than absorbed")

print()
print("=" * 74)
print("  edge cases")
print("=" * 74)
check(plan_segments([])[0] == [], "no scan data -> no segments, no crash")
check(plan_segments([])[1]["reason"] == "no scan data", "and it says why")
check(plan_segments([(0.0, 5)])[0] != [] or True, "a single sample does not crash")
segs, st = plan_segments(scan(600, 6, busy=[(100, 104, 1)]), min_segment_s=300)
check(segs == [], "a burst shorter than min_segment_s is not worth a segment")
segs, _ = plan_segments(scan(600, 6, busy=[(100, 200, 1)]), min_people=3)
check(segs == [], "min_people is respected (1 person ignored when 3 required)")
cov = coverage_report([], (0, 0), (0, 0))
check(cov["total_s"] == 0 and cov["accounted"], "a zero-length chunk is safe")
segs, _ = plan_segments(scan(600, 6, busy=[(0, 600, 2)]))
check(segs[0][0] >= 0.0, "segments never start before the chunk does")

print()
print("=" * 74)
if FAILED:
    print(f"  {len(FAILED)} FAILED:")
    for f in FAILED:
        print(f"    - {f}")
    sys.exit(1)
print("  ALL PASS")
print("=" * 74)
