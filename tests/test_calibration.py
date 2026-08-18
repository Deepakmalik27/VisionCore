"""Tests for kevacv/calibration.py — measure appearance without using appearance.

Built from the circularity in calibrate_appearance_threshold: its same-person
set is admitted only when `_cosine(va, vb) >= HANDOFF_VETO_SIM`, and its
stationary clause has no gap bound at all. Every test asks whether the
corrected version removes strangers on physics rather than on appearance.

Run: python test_calibration.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kevacv.reid_calibration import (calibrate, compare_to_legacy, cosine,
                                describe, percentile)

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


# Orthogonal-ish vectors: LIKE pairs match, UNLIKE pairs do not.
LIKE_A = [1.0, 0.0, 0.0]
LIKE_B = [0.98, 0.2, 0.0]          # cos ~0.98
UNLIKE = [0.0, 1.0, 0.0]           # cos 0.0 against LIKE_A

print("=" * 74)
print("  the same-person set is NOT filtered by appearance any more")
print("=" * 74)
# one genuine hand-off whose appearance happens to disagree — the exact pair
# the legacy rule deleted, and the exact pair we most need to see
W = {1: (0, 10), 2: (11, 20)}
P = {1: ((100, 100), (100, 100)), 2: ((105, 100), (105, 100))}
E = {1: LIKE_A, 2: UNLIKE}          # spatially certain, appearance says no
rep = calibrate(W, P, E)
check(rep["same_n"] == 1, "a low-similarity hand-off IS kept", str(rep["same_n"]))
check(abs(rep["same_p50"]) < 0.01, "and its low score lands in the distribution",
      str(rep["same_p50"]))
check(rep["appearance_independent"], "the report says it is appearance-independent")
check(rep["excluded"]["appearance_veto"] == 0, "nothing vetoed on appearance")

legacy = calibrate(W, P, E, appearance_veto_sim=0.30,
                   legacy_stationary_unbounded=True)
check(legacy["same_n"] == 0, "the LEGACY rule deletes that same pair", "circular")
check(legacy["excluded"]["appearance_veto"] == 1, "and counts the veto")
check(not legacy["appearance_independent"], "and admits it is not independent")
check("CIRCULAR" in describe(legacy), "describe() refuses to let that pass quietly")

print()
print("=" * 74)
print("  same spot + long gap is the NEXT customer, not the same person")
print("=" * 74)
# identical position, 40 minutes apart — the queue-spot stranger
W2 = {1: (0, 10), 2: (2400, 2410)}
P2 = {1: ((100, 100), (100, 100)), 2: ((100, 100), (100, 100))}
E2 = {1: LIKE_A, 2: LIKE_B}
rep2 = calibrate(W2, P2, E2)
check(rep2["same_n"] == 0, "excluded on the TIME bound", str(rep2["same_n"]))
check(rep2["excluded"]["stale_stationary"] == 1, "and counted as stale_stationary")
check("queue-spot strangers" in describe(rep2), "and named in plain words")

old2 = calibrate(W2, P2, E2, legacy_stationary_unbounded=True)
check(old2["same_n"] == 1, "the LEGACY unbounded rule counts it as one person")

# the same pair a few seconds apart IS a dropout and must survive
W3 = {1: (0, 10), 2: (25, 35)}
check(calibrate(W3, P2, E2)["same_n"] == 1,
      "the same spot 15s later is a tracker dropout and IS kept")
check(calibrate(W3, P2, E2, stationary_gap_s=5.0)["same_n"] == 0,
      "and the bound is tunable")

print()
print("=" * 74)
print("  the two appearance-INDEPENDENT guards are kept")
print("=" * 74)
rep3 = calibrate(W, P, E, role_hint={1: "staff", 2: "customer"})
check(rep3["same_n"] == 0 and rep3["excluded"]["role_conflict"] == 1,
      "opposite earned roles still exclude a spatially-close pair")
check(calibrate(W, P, E, role_hint={1: "staff", 2: "staff"})["same_n"] == 1,
      "the same role does not")

# duplicate-track guard on the different-person side
W4 = {1: (0, 100), 2: (10, 90)}                       # co-visible
P4 = {1: ((50, 50), (60, 60)), 2: ((52, 51), (61, 62))}   # glued at both ends
rep4 = calibrate(W4, P4, {1: LIKE_A, 2: LIKE_B})
check(rep4["diff_n"] == 0 and rep4["excluded"]["duplicate_track"] == 1,
      "one body with two ids is NOT different-person ground truth")
P5 = {1: ((50, 50), (60, 60)), 2: ((500, 500), (600, 600))}
check(calibrate(W4, P5, {1: LIKE_A, 2: LIKE_B})["diff_n"] == 1,
      "two genuinely separate co-visible bodies ARE")

print()
print("=" * 74)
print("  compare_to_legacy makes the size of the correction visible")
print("=" * 74)
WB = {1: (0, 10), 2: (11, 20), 3: (2400, 2410), 4: (5, 15)}
PB = {1: ((100, 100), (100, 100)), 2: ((105, 100), (105, 100)),
      3: ((100, 100), (100, 100)), 4: ((900, 900), (900, 900))}
EB = {1: LIKE_A, 2: UNLIKE, 3: LIKE_B, 4: UNLIKE}
cmp_ = compare_to_legacy(WB, PB, EB, appearance_veto_sim=0.30)
check(set(cmp_) == {"legacy", "corrected", "delta"}, "both reports and a delta")
check(cmp_["legacy"]["excluded"]["appearance_veto"] > 0,
      "legacy shows its circular exclusions")
check(cmp_["corrected"]["excluded"]["appearance_veto"] == 0,
      "corrected shows none")
check(cmp_["corrected"]["excluded"]["stale_stationary"] > 0,
      "corrected shows what the time bound removed")
check(cmp_["delta"]["same_n"] == cmp_["corrected"]["same_n"] - cmp_["legacy"]["same_n"],
      "the delta is arithmetic, not narrative")
print("   legacy   :", describe(cmp_["legacy"]).split("\n")[1].strip())
print("   corrected:", describe(cmp_["corrected"]).split("\n")[1].strip())

print()
print("=" * 74)
print("  helpers and degenerate input")
print("=" * 74)
check(abs(cosine([1, 0], [1, 0]) - 1.0) < 1e-9, "cosine of identical vectors is 1")
check(abs(cosine([1, 0], [0, 1])) < 1e-9, "orthogonal is 0")
check(cosine([0, 0], [1, 0]) == 0.0, "a zero vector does not divide by zero")
check(cosine(None, [1, 0]) == 0.0, "a missing vector is 0, not a crash")
check(percentile([], 0.5) is None, "percentile of nothing is None, not 0")
check(percentile([1, 2, 3, 4], 0.5) == 2.5, "and interpolates")

empty = calibrate({}, {}, {})
check(empty["same_n"] == 0 and empty["diff_n"] == 0, "no tracks -> no crash")
check(empty["same_p10"] is None, "and percentiles are None, not 0.0")
check(empty["separable"] is False, "separable is False, never None-ish truthy")
check(calibrate(W, {}, E)["same_n"] == 0, "no positions -> no same-person pairs")

print()
print("=" * 74)
if FAILED:
    print(f"  {len(FAILED)} FAILED:")
    for f in FAILED:
        print(f"    - {f}")
    sys.exit(1)
print("  ALL PASS")
print("=" * 74)
