"""test_foot_anchor.py — the foot anchor must sit where the foot is.

WHY THIS EXISTS
    Every zone test, entry-line crossing and ground-plane sample in this
    pipeline is decided by one point: the bottom-centre of the detection box.
    That assumed the box ends at the person's feet.

    Measured against 600 hand-labelled boxes on CAM.112 (Delilah Dallas), it
    does not. The detector is trained on CrowdHuman — street-level footage
    where a standing body is h/w ~2.5 — and this is a ceiling camera where a
    foreshortened person is h/w ~1.14. The model stretches its learned shape
    onto them:

        pipeline box   194 x 542 px   h/w 2.80
        hand-labelled  290 x 330 px   h/w 1.14

    so the bottom edge lands a MEDIAN 260 px below the real feet. Everything
    downstream has been asking about a point roughly half a body underneath
    the person, which is why main_entrance registered 0 hits, why reception
    read 0 with staff at the desk, and why the entry lines needed extending by
    60% to catch anything.

    True foot position inside the predicted box, n=442:
        p25 0.501    MEDIAN 0.590    p75 0.951
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kevacv import config  # noqa: E402
from kevacv.helpers import anchor_point  # noqa: E402

fail = 0


def check(ok, what, detail=""):
    global fail
    print(f"  {'PASS' if ok else 'FAIL'}  {what}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        fail = 1


print("=" * 74)
print("  foot anchor")
print("=" * 74)

BOX = (100.0, 200.0, 300.0, 742.0)      # 200 x 542, the measured median size

# Default must be byte-identical to the old behaviour.
config.FOOT_ANCHOR_FRAC = 1.0
x, y = anchor_point(BOX)
check((x, y) == (200.0, 742.0), "frac 1.0 = the old bottom-edge behaviour", f"{x},{y}")

# Calibrated value lifts the anchor to where the foot measurably is.
config.FOOT_ANCHOR_FRAC = 0.59
x, y = anchor_point(BOX)
check(x == 200.0, "x is untouched (median 0.474 of width — already correct)")
check(abs(y - (200.0 + 542.0 * 0.59)) < 1e-6,
      "y moves to 0.59 of the way down the box", f"{y:.1f}")
# 542 * 0.59 = 320px DOWN FROM THE TOP, which is 542 - 320 = 222px ABOVE the
# old bottom-edge anchor. The first version of this assertion compared the
# "above" distance against the "down from the top" number and failed on correct
# code — the two are easy to transpose and mean opposite things.
check(abs((742.0 - y) - 222.0) < 1.0,
      "which is ~222px ABOVE the old anchor on a median box",
      f"{742.0 - y:.0f}px higher")

# centre anchor (staff zones, where the counter clips the body) is unaffected
cx, cy = anchor_point(BOX, centre=True)
check((cx, cy) == (200.0, 471.0), "centre anchor is unchanged by this knob")

# A degenerate box must not produce a point outside itself.
config.FOOT_ANCHOR_FRAC = 0.59
x, y = anchor_point((0.0, 0.0, 10.0, 0.0))
check(y == 0.0, "zero-height box stays in bounds", f"{y}")

config.FOOT_ANCHOR_FRAC = 1.0           # leave the module as we found it
print()
print("  ALL PASS" if not fail else "  FAILURES ABOVE")
sys.exit(fail)
