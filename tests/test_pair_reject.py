"""test_pair_reject.py — every rejected merge must name the gate that killed it.

WHY THIS EXISTS
    Identity fragmentation (408 track fragments becoming 31 "people" in one
    hour) is the largest open defect in this pipeline, and five separate fixes
    for it have been designed and shipped and measured, and all five did
    nothing:

        duplicate detections  98 pairs in 8,219 boxes — too few to matter
        ReID bar 0.37 -> 0.61 A/B moved 2 merges, 0 downstream numbers
        short analysis window the FULL HOUR made staff inflation worse
        phantom patience      A/B: nothing suppressed, 0 effect
        IR colour veto        6,868 pairs waived, 0 of them would have blocked

    Every one was inferred from AGGREGATE diagnostics — "356 blocked by
    overlap", "712 merges HSV-disputed". Aggregates prove a population is
    unhealthy and never say which gate killed the patient in front of you. The
    log could already point at the patient:

        DEBUG unmerged near-pair: 1066->1118  dist=2.7px  gap=128.1s

    Two fragments 2.7 PIXELS apart, left as different people, and nothing
    anywhere said why.

WHAT IS ENFORCED
    A pair rejected for a known reason reports THAT reason — not a generic
    failure, and not silence. Each case below is a rejection whose cause is
    unambiguous by construction.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kevacv.analytics import merge_fragmented_tracks  # noqa: E402

fail = 0


def check(ok, what, detail=""):
    global fail
    print(f"  {'PASS' if ok else 'FAIL'}  {what}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        fail = 1


def reject_reason(windows, positions, **kw):
    res = merge_fragmented_tracks(windows, {}, positions=positions,
                                  sim_threshold=0.37, **kw)
    return (res[-1].get("pair_reject") or {}).get((1, 2), "")


print("=" * 74)
print("  every rejected merge names its gate")
print("=" * 74)

# CO-VISIBLE: both tracks on screen at once — genuinely two people.
r = reject_reason({1: (0.0, 100.0), 2: (50.0, 150.0)},
                  {1: [(900.0, 700.0)] * 2, 2: [(902.0, 700.0)] * 2})
check("overlap" in r.lower(), "co-visible pair blames time overlap", r[:60])

# TOO FAR: nothing spatial to stand on.
r = reject_reason({1: (0.0, 100.0), 2: (200.0, 300.0)},
                  {1: [(100.0, 700.0)] * 2, 2: [(1500.0, 700.0)] * 2})
check("apart" in r.lower(), "distant pair blames distance", r[:60])

# GAP TOO LONG: beyond max_gap_s.
r = reject_reason({1: (0.0, 100.0), 2: (5000.0, 5100.0)},
                  {1: [(900.0, 700.0)] * 2, 2: [(901.0, 700.0)] * 2})
check("gap" in r.lower(), "long-gap pair blames the gap", r[:60])

# ROLE CONFLICT: one staff, one customer.
r = reject_reason({1: (0.0, 100.0), 2: (200.0, 300.0)},
                  {1: [(900.0, 700.0)] * 2, 2: [(901.0, 700.0)] * 2},
                  role_hint={1: "staff", 2: "customer"})
check("role" in r.lower(), "staff/customer pair blames the role conflict", r[:60])

# And a pair that SHOULD merge must have no rejection recorded at all.
res = merge_fragmented_tracks({1: (0.0, 100.0), 2: (228.0, 400.0)}, {},
                              positions={1: [(900.0, 700.0)] * 2,
                                         2: [(903.0, 700.0)] * 2},
                              sim_threshold=0.37)
merged = res[0].get(1) == res[0].get(2)
check(merged and not (res[-1].get("pair_reject") or {}).get((1, 2)),
      "a pair that merges records no rejection", "merged cleanly")

print()
print("  ALL PASS" if not fail else "  FAILURES ABOVE")
# PYTEST VISIBILITY. This file's checks run at IMPORT and reported only via
# an exit code. A module-level sys.exit aborts pytest COLLECTION, which hid
# 59 of 74 test files -- and simply guarding the exit would have hidden the
# FAILURES instead. So the same condition is also asserted as a real test.
def test_script_level_checks_passed():
    assert not (fail), "module-level checks in this file failed"


if __name__ == "__main__":
    sys.exit(fail)
