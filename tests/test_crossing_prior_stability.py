"""test_crossing_prior_stability.py — a crossing needs a track that was already there.

WHY THIS EXISTS
    The published spec for a valid line crossing has six conditions. This
    pipeline implemented four; the missing first one is that a person must have
    been stably visible BEFORE the crossing counts.

    Without it, a track BORN near the line registers a transit though nobody
    walked anywhere. On CAM.112 that is not hypothetical: 16.3% of detections
    cannot start a track at all and ids churn constantly, so tracks appear near
    lines all the time. The operator's frame-by-frame audit of the annotated
    video found exactly this signature:

        "dining entry IN 2" while only ONE staff member is visible
        "staff entry OUT 6" while those staff are still in the room

    counts accumulating with no corresponding movement — the opposite failure
    to the main entrance, which stays silent while people walk through it.

WHAT IS ENFORCED
    a track born ON the line          dropped, with the reason recorded
    a track that approached first     kept
    OFF (0.0)                         byte-identical to the old behaviour
    no first-seen data                check SKIPPED, not half-applied
    u-turn filtering                  still works, and stays separable
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kevacv.analytics import confirm_crossings  # noqa: E402

fail = 0


def check(ok, what, detail=""):
    global fail
    print(f"  {'PASS' if ok else 'FAIL'}  {what}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        fail = 1


def cross(tid, t, direction="in", line="dining entry"):
    return {"track_id": tid, "t": t, "direction": direction, "line": line}


print("=" * 74)
print("  crossing prior-stability — a transit needs a track that already existed")
print("=" * 74)

# track 1 approached for 4s; track 2 was born ON the line
crossings = [cross(1, 10.0), cross(2, 10.0)]
first_seen = {1: 6.0, 2: 9.9}

kept, dropped = confirm_crossings(crossings, confirm_s=5.0,
                                  track_first_seen=first_seen, min_prior_s=1.5)
kept_ids = {c["track_id"] for c in kept}
check(1 in kept_ids, "a track that approached the line is KEPT")
check(2 not in kept_ids, "a track BORN at the line is dropped")
check(any("born at the line" in (d.get("dropped_reason") or "") for d in dropped),
      "and the reason is recorded, not silently discarded",
      next((d.get("dropped_reason") for d in dropped if d.get("dropped_reason")), ""))

# OFF must change nothing
kept0, dropped0 = confirm_crossings(crossings, confirm_s=5.0,
                                    track_first_seen=first_seen, min_prior_s=0.0)
check(len(kept0) == 2 and not dropped0,
      "OFF (0.0) is exactly the previous behaviour")

# Missing first-seen data must SKIP the check, never half-apply it
keptn, droppedn = confirm_crossings(crossings, confirm_s=5.0,
                                    track_first_seen=None, min_prior_s=1.5)
check(len(keptn) == 2, "without first-seen data the check is skipped, not guessed")

# The u-turn filter must still work, and the two reasons must stay separable
uturn = [cross(3, 20.0, "in"), cross(3, 21.0, "out")]
kept2, dropped2 = confirm_crossings(uturn, confirm_s=5.0,
                                    track_first_seen={3: 5.0}, min_prior_s=1.5)
check(not kept2 and len(dropped2) == 2,
      "u-turn (crossed and came straight back) still cancels BOTH events")
check(all(not d.get("dropped_reason") for d in dropped2),
      "u-turn drops are NOT mislabelled as born-at-the-line",
      "the two causes must stay distinguishable in the log")

# a brisk but real transit must survive a sensible threshold
brisk = [cross(4, 30.0)]
kept3, _ = confirm_crossings(brisk, confirm_s=5.0,
                             track_first_seen={4: 28.0}, min_prior_s=1.5)
check(len(kept3) == 1, "a 2s approach still counts as a real transit",
      "the bar must not eat brisk interior traffic")

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
