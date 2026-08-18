"""test_phantom_fast_patience.py — the live suppressor must catch furniture
EARLY without ever catching a person who stands still.

WHY THIS EXISTS
    On the 18:30 CAM.112 chunk the live phantom stage reported:

        "funnel: live phantom suppress removed nothing all chunk — it is
         either unnecessary here or misconfigured"

    Misconfigured, in a defensible way. The plant and the mirror sit inside the
    reception/seating polygons, and those zones get 240 s of patience BECAUSE
    PEOPLE LEGITIMATELY STAND STILL THERE. Deleting the receptionist is far
    worse than keeping a plant, so the patience is right. But on a 600 s run
    the phantom then lives the entire chunk, holding a canonical id and
    blocking real people from resolving to it, and only the end-of-chunk pass
    removes it — which cannot undo an id break that already happened.

THE MARGIN THIS RELIES ON
    "Stands still" and "is furniture" are different claims. A detector fed
    identical pixels returns an identical box; a real body's box breathes.

        statue / mirror   size cv ~0.004
        person standing   size cv ~0.080        19x

    That is the only signal in this problem with room to stand in. For
    comparison, the alternatives that were tried and rejected separate a wall
    phantom from a near-field guest by 0.02 of frame height (flat cap) and by
    28 pixels (top-anchor bound) — both fitted to two examples.

WHAT IS ENFORCED
    1. with the feature OFF, behaviour is exactly as before
    2. a rigid statue is caught EARLY, well inside the zone patience
    3. a person standing still at the desk is NEVER caught, however long
    4. the fast path skips the CLOCK, never the jitter/size evidence
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import kevacv.phantoms as ph  # noqa: E402

fail = 0


def check(ok, what, detail=""):
    global fail
    print(f"  {'PASS' if ok else 'FAIL'}  {what}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        fail = 1


def run(kind, seconds, fast_ratio, fps=8.0, patience=240.0):
    """Feed one location for `seconds` and report when (if) it was suppressed."""
    ph.PHANTOM_FAST_CV_RATIO = fast_ratio
    ph.PHANTOM_FAST_MIN_S = 30.0
    s = ph.OnlineStaticSuppressor((1920, 1080), min_life_for=lambda x, y: patience,
                                  default_life_s=patience)
    random.seed(5)
    caught_at = None
    n = int(seconds * fps)
    for i in range(n):
        t = i / fps
        if kind == "statue":
            # identical pixels -> identical box, sub-pixel wobble only
            cx, cy, w, h = 900.0 + (i % 2) * 0.4, 700.0, 60.0, 119.0 + (i % 2) * 0.4
        else:
            # a receptionist HOLDING POSITION: centre barely moves, but the
            # box breathes as they turn, lean and gesture
            cx = 900.0 + random.uniform(-6, 6)
            cy = 700.0 + random.uniform(-5, 5)
            w = 180.0 + random.uniform(-14, 14)
            h = 430.0 + random.uniform(-30, 30)
        box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        if s.observe(t, box) and caught_at is None:
            caught_at = t
    return caught_at, s


print("=" * 74)
print("  live phantom suppressor — evidence-scaled patience")
print("=" * 74)

# 1. OFF = unchanged behaviour
at, _ = run("statue", 120, fast_ratio=0.0)
check(at is None,
      "feature OFF: statue survives 120s inside 240s patience (old behaviour)",
      "not suppressed")

# 2. ON = statue caught early
at, s = run("statue", 120, fast_ratio=0.30)
check(at is not None and at < 240,
      "feature ON: statue caught EARLY, inside the 240s zone patience",
      f"suppressed at {at:.0f}s" if at else "never")
check(s.n_fast_tracked == 1,
      "...and it is counted as a fast-track, not a normal catch",
      f"n_fast_tracked={s.n_fast_tracked}")

# 3. the receptionist is never touched — the whole point
at, _ = run("person", 600, fast_ratio=0.30, patience=240.0)
check(at is None,
      "a person standing still for 600s is NEVER suppressed",
      "not suppressed" if at is None else f"SUPPRESSED at {at:.0f}s")

# 4. an aggressive ratio must still not reach a person
at, _ = run("person", 600, fast_ratio=1.00, patience=240.0)
check(at is None,
      "even at ratio 1.00 the person survives the fast path",
      "not suppressed" if at is None else f"SUPPRESSED at {at:.0f}s")

# 5. the floor holds: nothing is suppressed below PHANTOM_FAST_MIN_S
at, _ = run("statue", 20, fast_ratio=0.30)
check(at is None,
      "20s of evidence is not enough, whatever the rigidity (30s floor)",
      "not suppressed")

print()
print("  ALL PASS" if not fail else "  FAILURES ABOVE")
sys.exit(fail)
