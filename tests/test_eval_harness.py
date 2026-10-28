"""Validate eval_harness.py against cases whose answers are known BY CONSTRUCTION.

This matters more than it looks. The harness is about to become the thing that
decides whether every future change ships. A metric nobody tested is not a
measurement, it is a second opinion from the same brain that wrote the bug.

Each case below has an answer derivable on paper, written next to it:

  perfect            pred == gt                  -> HOTA 1, DetA 1, AssA 1
  relabelled         ids all +100                -> still 1.0 everywhere
                     (association is about CONSISTENCY, not the label value)
  swapped halfway    ids traded at the midpoint  -> DetA 1, AssA 0.5, HOTA .707
  half missed        every 2nd frame dropped     -> DetA 0.5, recall 0.5
  duplicated         every box emitted twice     -> DetA 0.5, precision 0.5
  fragmented into K  each person split K ways    -> AssA ~ 1/K
  over-merged        two people share one id     -> AssA drops, n_pr_ids halves
  empty              no predictions              -> all zero, no crash

Run: python test_eval_harness.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kevacv.eval_harness import (compare, dump_errors_csv, iou_matrix, load_mot,
                          score_sequence, write_mot)

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


def near(a, b, tol=0.02):
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# a synthetic sequence: 3 people walking across 40 frames
# ---------------------------------------------------------------------------
N_FRAMES, N_PEOPLE = 40, 3


def make_gt():
    gt = {}
    for f in range(1, N_FRAMES + 1):
        gt[f] = [(pid, 100 + pid * 150 + f * 4, 200 + pid * 20, 60, 160)
                 for pid in range(1, N_PEOPLE + 1)]
    return gt


GT = make_gt()
N_BOXES = N_FRAMES * N_PEOPLE

print("=" * 74)
print("  geometry")
print("=" * 74)
M = iou_matrix([(0, 0, 10, 10)], [(0, 0, 10, 10), (5, 0, 10, 10), (100, 100, 10, 10)])
check(near(M[0, 0], 1.0, 1e-9), "identical boxes -> IoU 1.0", f"{M[0,0]:.4f}")
check(near(M[0, 1], 1 / 3, 1e-6), "half-overlap -> IoU 1/3", f"{M[0,1]:.4f}")
check(M[0, 2] == 0.0, "disjoint boxes -> IoU 0")
check(iou_matrix([], [(0, 0, 1, 1)]).shape == (0, 1), "empty input is safe")

print()
print("=" * 74)
print("  metric cases with known answers")
print("=" * 74)

# 1 — perfect
r = score_sequence(GT, GT)
check(near(r["HOTA"], 1.0), "perfect tracker -> HOTA 1.0", f"{r['HOTA']:.4f}")
check(near(r["DetA"], 1.0) and near(r["AssA"], 1.0), "  DetA and AssA both 1.0")
check(near(r["IDF1"], 1.0) and near(r["MOTA"], 1.0), "  IDF1 and MOTA 1.0")
check(r["ID_switches"] == 0, "  zero ID switches")
check(r["FP"] == 0 and r["FN"] == 0, "  no FP, no FN")

# 2 — relabelled: association is about consistency, not label values
relabelled = {f: [(pid + 100, *b) for pid, *b in v] for f, v in GT.items()}
r = score_sequence(GT, relabelled)
check(near(r["HOTA"], 1.0), "ids all +100 -> still HOTA 1.0", f"{r['HOTA']:.4f}")
check(r["ID_switches"] == 0, "  renaming every id is not an ID switch")

# 3 — swapped halfway: detection perfect, association halved
half = N_FRAMES // 2
swapped = {}
for f, v in GT.items():
    if f <= half:
        swapped[f] = list(v)
    else:                      # rotate the identities among the same boxes
        ids = [b[0] for b in v]
        rot = ids[1:] + ids[:1]
        swapped[f] = [(rot[i], *v[i][1:]) for i in range(len(v))]
r = score_sequence(GT, swapped)
check(near(r["DetA"], 1.0), "swapped ids -> DetA still 1.0 (boxes are right)",
      f"{r['DetA']:.4f}")
# AssA = 1/3 here, NOT 1/2 — hand-derived, and the distinction is the point.
# For a swap, gt person A is matched to pred id 1 for half the frames (TPA=N/2,
# FNA=N/2), but pred id 1 ALSO stays alive on someone else's box for the other
# half, so FPA=N/2 too:  A = (N/2)/(N/2+N/2+N/2) = 1/3.
# For a FRAGMENT the second id exists nowhere else, so FPA=0 and A = 1/2.
# => an ID SWAP costs association MORE than a fragmentation. That is correct
#    and worth knowing: swaps corrupt two identities, fragments only split one.
check(near(r["AssA"], 1 / 3, 0.05), "  AssA ~ 1/3 (a swap corrupts BOTH identities)",
      f"{r['AssA']:.4f}")
check(near(r["HOTA"], 0.577, 0.05), "  HOTA ~ sqrt(1 x 1/3) = 0.577", f"{r['HOTA']:.4f}")
check(r["ID_switches"] >= N_PEOPLE - 1, "  ID switches detected",
      f"{r['ID_switches']}")
check(r["DetA"] > r["AssA"], "  diagnosis points at ASSOCIATION, not detection")

# 4 — half the detections missing
missed = {f: v for f, v in GT.items() if f % 2 == 1}
r = score_sequence(GT, missed)
check(near(r["DetA"], 0.5, 0.03), "half the frames missed -> DetA 0.5", f"{r['DetA']:.4f}")
check(near(r["recall"], 0.5, 0.03), "  recall 0.5", f"{r['recall']:.4f}")
check(near(r["precision"], 1.0, 0.02), "  precision still 1.0 (nothing invented)",
      f"{r['precision']:.4f}")

# 5 — every box emitted twice (the duplicate-box / shadow failure)
dup = {f: list(v) + [(pid + 500, *b) for pid, *b in v] for f, v in GT.items()}
r = score_sequence(GT, dup)
check(near(r["DetA"], 0.5, 0.03), "every box duplicated -> DetA 0.5", f"{r['DetA']:.4f}")
check(near(r["precision"], 0.5, 0.03), "  precision 0.5", f"{r['precision']:.4f}")
check(r["n_pr_ids"] == 2 * r["n_gt_ids"], "  identity count doubled (inflates any count)",
      f"{r['n_pr_ids']} vs {r['n_gt_ids']}")

# 6 — fragmentation: each person split into K identities
for K in (2, 4):
    frag = {}
    for f, v in GT.items():
        seg = min(K - 1, (f - 1) * K // N_FRAMES)
        frag[f] = [(pid * 100 + seg, *b) for pid, *b in v]
    r = score_sequence(GT, frag)
    check(near(r["AssA"], 1.0 / K, 0.10), f"split into {K} -> AssA ~ 1/{K}",
          f"{r['AssA']:.4f}")
    check(near(r["DetA"], 1.0, 0.02), f"  DetA unaffected by fragmentation",
          f"{r['DetA']:.4f}")

# 7 — over-merge: two different people share one identity
merged = {f: [(1 if pid in (1, 2) else pid, *b) for pid, *b in v]
          for f, v in GT.items()}
r = score_sequence(GT, merged)
check(r["n_pr_ids"] < r["n_gt_ids"], "over-merge -> fewer identities than people",
      f"{r['n_pr_ids']} vs {r['n_gt_ids']}")
check(r["AssA"] < 0.95, "  AssA penalises over-merging too", f"{r['AssA']:.4f}")

# 8 — empty predictions must not crash
r = score_sequence(GT, {})
check(r["HOTA"] == 0.0 and r["DetA"] == 0.0, "empty predictions -> all zero, no crash")
check(r["FN"] == N_BOXES, "  every gt box counted as a miss", f"{r['FN']}")

print()
print("=" * 74)
print("  MOT file round-trip")
print("=" * 74)
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    rows = [(f, pid, *b) for f, v in GT.items() for pid, *b in v]
    p = write_mot(td / "gt.txt", rows)
    back = load_mot(p)
    check(len(back) == N_FRAMES, "frame count survives round-trip", f"{len(back)}")
    check(sum(len(v) for v in back.values()) == N_BOXES, "box count survives")
    check(near(score_sequence(GT, back)["HOTA"], 1.0),
          "round-tripped file still scores 1.0 against the original")

    # MOT gt "ignore" flag (column 7 == 0) must be honoured, or a gt file
    # exported from CVAT would silently score against boxes we must not count
    (td / "ign.txt").write_text("1,1,10,10,20,40,0,-1,-1,-1\n1,2,60,10,20,40,1,-1,-1,-1")
    ig = load_mot(td / "ign.txt")
    check(len(ig.get(1, [])) == 1, "MOT ignore-flag rows are dropped",
          f"{len(ig.get(1, []))} kept of 2")

    # malformed lines must be reported, not silently swallowed
    (td / "bad.txt").write_text("1,1,10,10,20,40,1,-1,-1,-1\nGARBAGE LINE\n"
                                "2,1,10,10,-5,40,1,-1,-1,-1")
    bad = load_mot(td / "bad.txt")
    check(sum(len(v) for v in bad.values()) == 1,
          "garbage and non-positive boxes rejected", f"{sum(len(v) for v in bad.values())} kept")

    err = dump_errors_csv(GT, missed, score_sequence(GT, missed), td / "e.csv")
    n_err = len(err.read_text().splitlines()) - 1
    check(n_err == N_BOXES // 2, "error CSV lists every FN with its box",
          f"{n_err} rows")

print()
print("=" * 74)
print("  A/B comparison")
print("=" * 74)
before = {"metrics": {k: v for k, v in score_sequence(GT, swapped).items()
                      if not k.startswith("_")},
          "config": {"ANALYSIS_FPS": 4, "CALIBRATION_AUTO_APPLY": True}}
after = {"metrics": {k: v for k, v in score_sequence(GT, GT).items()
                     if not k.startswith("_")},
         "config": {"ANALYSIS_FPS": 8, "CALIBRATION_AUTO_APPLY": True}}
d = compare(before, after, "v55 (4fps)", "v56 (8fps)")
check(d["delta_HOTA"] > 0.2, "improvement is detected", f"{d['delta_HOTA']:+.4f}")
check("ANALYSIS_FPS" in d["config_changed"], "config diff attributes the change")
check("CALIBRATION_AUTO_APPLY" not in d["config_changed"], "unchanged keys not listed")
d2 = compare(after, before, "good", "bad")
check(d2["delta_HOTA"] < -0.2, "regression is detected as a regression")

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
print("  ALL PASS — the harness measures what it claims to measure")
print("=" * 74)
