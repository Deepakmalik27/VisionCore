"""Tests for kevacv/threshold.py — pick a threshold by cost, not by error count.

Built from the CAM.112 calibration: same-person p50=0.435, different-person
p50=0.370 / p90=0.573, best balanced accuracy 0.658, and an EER sweep that
suggested 0.340 against a live threshold of 0.60. Every test asks: would this
have stopped us adopting 0.340?

Run: python test_threshold.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kevacv.threshold import (compare, cost_weighted_threshold, describe,
                              verdict)

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


random.seed(7)


def sims(median, spread, n):
    return [max(0.0, min(1.0, random.gauss(median, spread))) for _ in range(n)]


# Mirrors the real measured distributions
SAME = sims(0.435, 0.09, 400)
DIFF = sims(0.370, 0.12, 1400)

print("=" * 74)
print("  cost weighting moves the threshold UP, away from fusing strangers")
print("=" * 74)
t_bal, _ = cost_weighted_threshold(SAME, DIFF, fa_cost=1.0, fr_cost=1.0)
t_cost, rep = cost_weighted_threshold(SAME, DIFF)          # default 8:1
check(t_cost > t_bal, "8:1 cost picks a HIGHER threshold than 1:1",
      f"{t_bal} -> {t_cost}")
check(rep["false_accept_rate"] < 0.20,
      "and the false-ACCEPT rate is held low", str(rep["false_accept_rate"]))

t_low, rep_low = cost_weighted_threshold(SAME, DIFF, fa_cost=1.0, fr_cost=8.0)
check(t_low < t_cost, "inverting the costs moves it back down",
      f"{t_low} vs {t_cost}")
check(rep_low["false_reject_rate"] < rep["false_reject_rate"],
      "...and trades in the expected direction")

print()
print("=" * 74)
print("  it refuses to let a tuned number imply a solved problem")
print("=" * 74)
v = verdict(rep)
check(rep["degenerate_no_merge"],
      "on CAM.112-shaped data the optimum is literally 'merge nothing'",
      f"t={rep['threshold']} FR={rep['false_reject_rate']}")
check("MERGE NOTHING is cheapest" in v, "and that is stated, not disguised",
      v[:40])
check("hand-off, stationary and topology" in v,
      "and it names the evidence to use instead")
print("    -> this is the arithmetic reaching the same conclusion as the physics")

# a mildly-overlapping signal is NOT degenerate and must still merge
soft_same = sims(0.62, 0.10, 300)
soft_diff = sims(0.42, 0.10, 900)
t_soft, rep_soft = cost_weighted_threshold(soft_same, soft_diff)
check(not rep_soft["degenerate_no_merge"],
      "a merely-overlapping signal still yields a working threshold",
      f"t={t_soft} FR={rep_soft['false_reject_rate']}")
check("NOT SEPARABLE" in verdict(rep_soft) or "usable" in verdict(rep_soft),
      "and gets an overlap verdict rather than the degenerate one")

CLEAN_SAME = sims(0.85, 0.04, 200)
CLEAN_DIFF = sims(0.30, 0.05, 600)
t_clean, rep_clean = cost_weighted_threshold(CLEAN_SAME, CLEAN_DIFF)
check(rep_clean["separable"], "a genuinely separable signal is recognised")
check("SEPARABLE" in verdict(rep_clean), "and reported as a real boundary")
check(rep_clean["false_accept_rate"] == 0.0,
      "with no false accepts at the chosen point")

MID_SAME = sims(0.70, 0.12, 400)      # tails cross, but the signal is strong
MID_DIFF = sims(0.40, 0.12, 1200)
rep_mid = cost_weighted_threshold(MID_SAME, MID_DIFF)[1]
check(not rep_mid["separable"], "distributions genuinely overlap here",
      f"same_p10={rep_mid['same_p10']} diff_p90={rep_mid['diff_p90']}")
check("usable" in verdict(rep_mid),
      "an overlapping-but-good signal gets the middle verdict",
      verdict(rep_mid)[:38])
check(rep_mid["best_balanced_accuracy"] > rep_mid["balanced_accuracy"],
      "signal quality is judged separately from the chosen operating point",
      f"signal {rep_mid['best_balanced_accuracy']} vs "
      f"chosen {rep_mid['balanced_accuracy']}")
print("    -> a cautious cost policy must not make good data look bad")

print()
print("=" * 74)
print("  compare() informs, it never auto-applies")
print("=" * 74)
c = compare(SAME, DIFF, current=0.60)
check(set(c) >= {"current", "suggested", "cost_delta", "direction", "verdict"},
      "reports both options and the delta between them")
check("apply" not in str(c).lower(), "and never says 'apply this'")
check(c["current"]["threshold"] == 0.60, "current threshold is echoed back")
check(c["direction"] in ("more conservative", "more aggressive", "unchanged"),
      "direction is stated in words", c["direction"])
check(c["current"]["false_accept_rate"] <= c["suggested"]["false_accept_rate"]
      or c["suggested"]["threshold"] >= 0.60,
      "a lower suggestion always means MORE false accepts, and says so")

print()
print("=" * 74)
print("  ties break toward the safer (higher) threshold")
print("=" * 74)
flat_same = [0.9] * 10
flat_diff = [0.1] * 10
t_flat, _ = cost_weighted_threshold(flat_same, flat_diff)
check(0.1 < t_flat <= 0.9, "a wide zero-cost basin resolves inside the gap",
      str(t_flat))
t_a, _ = cost_weighted_threshold(flat_same, flat_diff, lo=0.0, hi=1.0)
t_b, _ = cost_weighted_threshold(flat_same, flat_diff, lo=0.0, hi=1.0)
check(t_a == t_b, "and the choice is deterministic across runs")

print()
print("=" * 74)
print("  degenerate input never crashes and never guesses")
print("=" * 74)
t, r = cost_weighted_threshold([], DIFF)
check(t is None and "insufficient data" in r["note"],
      "no same-person pairs -> None, not a default")
t, r = cost_weighted_threshold(SAME, [])
check(t is None, "no different-person pairs -> None")
check(verdict(r) == "NO DATA — cannot choose a threshold", "verdict says NO DATA")
check(describe(r) == r["note"], "describe() degrades to the note")
t, r = cost_weighted_threshold([0.5], [0.4])
check(t is not None, "a single pair each still produces an answer", str(t))

print()
print("=" * 74)
print("  the curve is printed so a flat basin is visible")
print("=" * 74)
txt = describe(rep)
check("COST CURVE" in txt and "<- chosen" in txt, "curve renders and marks the pick")
check(txt.count("\n") >= 10, "and shows the shape, not just one number")
print("\n".join("   " + l for l in txt.split("\n")[:8]))

print()
print("=" * 74)
# PYTEST VISIBILITY. This file's checks run at IMPORT and reported only via
# an exit code. A module-level sys.exit aborts pytest COLLECTION, which hid
# 59 of 74 test files -- and simply guarding the exit would have hidden the
# FAILURES instead. So the same condition is also asserted as a real test.
def test_script_level_checks_passed():
    assert not (FAILED), "module-level checks in this file failed"


if __name__ == "__main__":
    if FAILED:
        print(f"  {len(FAILED)} FAILED:")
        for f in FAILED:
            print(f"    - {f}")
        sys.exit(1)
print("  ALL PASS")
print("=" * 74)
