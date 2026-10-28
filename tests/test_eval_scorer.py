"""test_eval_scorer.py — the HOTA scorer must be trustworthy BEFORE anyone
spends an afternoon labelling.

WHY THIS EXISTS
    Every SUMMARY.txt this project has ever produced carries the line

        "HOTA NOT MEASURED — every accuracy claim below is an estimate"

    and the 90% accuracy target is therefore not a target anyone can hit or
    miss. The fix is ground truth, and ground truth costs hours of human
    clicking in CVAT.

    Spending those hours against a scorer nobody has verified is the expensive
    version of this mistake. A scorer that silently returns 1.0, or that is
    blind to identity switches, would turn a real afternoon into a number that
    means nothing — and it would be believed, because it came with decimals.

    So: plant errors of a KNOWN size and check the scorer reports that size.
    No labelling required to run this.

WHAT IT PROVES
    self-score      gt == predictions must be a perfect 1.0
    missed people   drop 10% of boxes -> recall ~0.90, precision untouched
    hallucinations  add phantom boxes -> precision falls, recall untouched
    id switch       swap two ids MID-SEQUENCE -> AssA falls, DetA untouched

    That last one matters twice over. A GLOBAL relabel of id 1<->2 is NOT an
    error in MOT metrics — identity names are arbitrary, only consistency
    counts — and an earlier version of this test asserted it was, then read the
    correct answer (AssA 1.000) as the scorer being broken. The error has to be
    a switch partway through, which is what the real tracker does.
"""
import copy
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kevacv.eval_harness import score_sequence  # noqa: E402

fail = 0


def check(ok, what, detail=""):
    global fail
    print(f"  {'PASS' if ok else 'FAIL'}  {what}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        fail = 1


def synth(n_frames=300, n_people=4):
    """A clean synthetic sequence. Self-contained: this test must run on a
    laptop with no run output and no GPU."""
    random.seed(3)
    seq = {}
    for f in range(1, n_frames + 1):
        boxes = []
        for pid in range(1, n_people + 1):
            x = 100 + pid * 220 + f * 1.5
            y = 300 + (pid % 2) * 180
            boxes.append((pid, x, y, 150.0, 380.0))
        seq[f] = boxes
    return seq


print("=" * 74)
print("  HOTA scorer — verified against planted errors, before labelling")
print("=" * 74)

gt = synth()
n_boxes = sum(len(v) for v in gt.values())

s = score_sequence(gt, copy.deepcopy(gt))
check(abs(s["HOTA"] - 1.0) < 1e-6 and s["FP"] == 0 and s["FN"] == 0,
      "gt == predictions scores a perfect 1.0",
      f"HOTA {s['HOTA']:.4f} FP {s['FP']} FN {s['FN']}")

# --- missed people: the failure mode this project actually has -------------
random.seed(11)
pred = copy.deepcopy(gt)
dropped = 0
for f, bs in pred.items():
    keep = [b for b in bs if random.random() > 0.10]
    dropped += len(bs) - len(keep)
    pred[f] = keep
s = score_sequence(gt, pred)
want = 1.0 - dropped / n_boxes
check(abs(s["recall"] - want) < 0.02,
      "dropping 10% of boxes lands recall where it should",
      f"recall {s['recall']:.3f}, expected {want:.3f}")
check(s["precision"] > 0.999,
      "...and does NOT move precision", f"{s['precision']:.3f}")

# --- hallucinated people ---------------------------------------------------
pred = copy.deepcopy(gt)
added = 0
for f, bs in list(pred.items()):
    if bs and f % 5 == 0:
        pred[f] = bs + [(9999, 50.0, 50.0, 140.0, 360.0)]
        added += 1
s = score_sequence(gt, pred)
check(s["FP"] == added and s["precision"] < 0.999,
      "phantom boxes cost precision",
      f"FP {s['FP']} of {added} planted, precision {s['precision']:.3f}")
check(s["recall"] > 0.999, "...and do NOT move recall", f"{s['recall']:.3f}")

# --- identity switch, the thing AssA exists for ----------------------------
frames = sorted(gt)
mid = frames[len(frames) // 2]
pred = copy.deepcopy(gt)
for f in frames:
    if f < mid:
        continue
    pred[f] = [((2 if x[0] == 1 else 1 if x[0] == 2 else x[0]),) + tuple(x[1:])
               for x in pred[f]]
s = score_sequence(gt, pred)
check(s["AssA"] < 0.999,
      "a MID-SEQUENCE id swap costs association", f"AssA {s['AssA']:.3f}")
check(s["DetA"] > 0.999,
      "...while detection is untouched (only ids changed)",
      f"DetA {s['DetA']:.3f}")

# A global relabel is NOT an error. Recorded because asserting otherwise is an
# easy mistake that makes a correct scorer look broken.
pred = {f: [((2 if x[0] == 1 else 1 if x[0] == 2 else x[0]),) + tuple(x[1:])
            for x in bs] for f, bs in gt.items()}
s = score_sequence(gt, pred)
check(s["AssA"] > 0.999,
      "a GLOBAL relabel is correctly NOT an error (names are arbitrary)",
      f"AssA {s['AssA']:.3f}")

print()
print("  ALL PASS — safe to spend labelling time" if not fail
      else "  FAILURES ABOVE — do NOT label until these are fixed")
# PYTEST VISIBILITY. This file's checks run at IMPORT and reported only via
# an exit code. A module-level sys.exit aborts pytest COLLECTION, which hid
# 59 of 74 test files -- and simply guarding the exit would have hidden the
# FAILURES instead. So the same condition is also asserted as a real test.
def test_script_level_checks_passed():
    assert not (fail), "module-level checks in this file failed"


if __name__ == "__main__":
    sys.exit(fail)
