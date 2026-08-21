"""test_absurd_size.py — the D0 cap, and why it must not be relaxable.

The audit's "huge box around the doorway", "P3 over the whole right-side
background" and "P11 over a large mostly empty area" survived every filter the
pipeline had: their ASPECT is inside MIN/MAX_BODY_ASPECT, and D1 -- the only
size filter -- is geometry-based, so a bad ground fit made it DOUBLE its own
tolerance (2.5x -> 5.0x) and wave them through. The guard that was supposed to
stop giant boxes made them worse.

D0 needs no geometry, so it cannot be fooled by a bad one.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kevacv.config import (ENABLE_ABSURD_SIZE_CAP, MAX_BOX_AREA_FRAC,  # noqa
                           MAX_BOX_HEIGHT_FRAC, MIN_BODY_ASPECT,
                           MAX_BODY_ASPECT, SIZE_FILTER_TOL)

W, H = 1920, 1080
_fail = False


def check(ok, what, detail=""):
    global _fail
    print(f"  {'PASS' if ok else 'FAIL'}  {what}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        _fail = True


def kept(x1, y1, x2, y2):
    """The D0 predicate, exactly as engine._filter_chain applies it."""
    h = (y2 - y1) / H
    a = ((x2 - x1) * (y2 - y1)) / float(W * H)
    return h <= MAX_BOX_HEIGHT_FRAC and a <= MAX_BOX_AREA_FRAC


print("=" * 74)
print("  the boxes from the audit")
print("=" * 74)

# P3 @ 07:59 -- the blue box over the entire right-hand column of frame_60s.
P3 = (1310, 88, 1920, 1160 - 88)
w, h = P3[2] - P3[0], P3[3] - P3[1]
check(MIN_BODY_ASPECT <= h / w <= MAX_BODY_ASPECT,
      "P3 PASSES the aspect filter — which is why it was never caught",
      f"aspect {h/w:.2f} inside [{MIN_BODY_ASPECT}, {MAX_BODY_ASPECT}]")
# 2026-08-15 -- D0 CATCHES THESE, AND IT IS NOT FREE. READ BEFORE TUNING.
#
# These assertions hold at MAX_BOX_HEIGHT_FRAC 0.70, which is what ships. But
# instrumenting the run showed the price: in 600 seconds D0 dropped 8,246
# boxes, EVERY ONE on the height bound and none on area, and 7,999 (97%) were
# within 1.35x the height the scene geometry expects at their own foot
# position. The same run's F2 fit puts a person standing at the frame bottom --
# where the main entrance is -- at 804px = 0.744 of frame, ABOVE the cap. D0
# buys these two phantoms with real guests at the door.
#
# Raising the cap to 0.95 was TRIED AND REVERTED. The funnel improved (+3,716
# detections past D0) while the frame log got worse (869 frames seeing fewer
# people against 417 seeing more), because the phantoms D0 stopped removing
# entered F2's sample pool, moved the fit, and D1 then rejected the small
# distant people instead. Never judge a recall change on the funnel alone --
# tools/audit_frames.py diffs the frame logs.
#
# There is no repair by tuning this number. Measured, phantoms and people
# INTERLEAVE on both bounds:
#     P11 phantom    h 0.854  area 0.218
#     leaning guest  h 0.833  area 0.200
# Nine hundredths apart. A flat single-frame threshold cannot hold that line.
# The candidate that might is a TOP-anchor bound -- a guest whose feet are at
# the frame bottom has their head near y=276, while P3's top is y=88 and P11's
# is y=150. Unmeasured, so not shipped.
# See tests/test_absurd_cap_vs_geometry.py and config/cam112.yaml.
check(not kept(*P3), "D0 REJECTS P3",
      f"{h/H:.0%} of frame height; a person leaning is {900/H:.0%} — only 8 apart")

# P11 -- the left-hand column box over the mirror/wall.
P11 = (0, 150, 490, 1160 - 88)
check(not kept(*P11), "D0 REJECTS P11 (whole left column)")

# The doorway box. Still rejected: at full frame height it IS physically
# impossible, which is the one question D0 is entitled to answer.
check(not kept(200, 100, 900, 1000),
      "D0 still REJECTS a full-height doorway box (physically impossible)")

print()
print("=" * 74)
print("  real people are untouched")
print("=" * 74)
for name, box in [("guest mid-floor", (700, 500, 850, 900)),
                  ("staff at the desk", (350, 200, 470, 560)),
                  ("person near camera", (600, 150, 900, 850)),
                  ("small guest at the door", (1500, 700, 1560, 860))]:
    check(kept(*box), f"D0 KEEPS {name}",
          f"h={(box[3]-box[1])/H:.0%} a={((box[2]-box[0])*(box[3]-box[1]))/(W*H):.1%}")

print()
print("=" * 74)
print("  it is not relaxable")
print("=" * 74)
check(ENABLE_ABSURD_SIZE_CAP is True, "D0 is on by default")
# D1's guard doubles SIZE_FILTER_TOL on a bad fit; D0 has no tolerance to
# double. That is still true and still worth keeping -- what changed is only
# WHERE the bound sits, not that it is unrelaxable.
check(not kept(200, 100, 900, 1000),
      "the impossible stays rejected regardless of SIZE_FILTER_TOL",
      f"D1 would relax {SIZE_FILTER_TOL}x -> {SIZE_FILTER_TOL*2}x; D0 has no such knob")

print()
print("=" * 74)
print("  ALL PASS" if not _fail else "  FAILURES ABOVE")
print("=" * 74)
# PYTEST VISIBILITY. This file's checks run at IMPORT and reported only via
# an exit code. A module-level sys.exit aborts pytest COLLECTION, which hid
# 59 of 74 test files -- and simply guarding the exit would have hidden the
# FAILURES instead. So the same condition is also asserted as a real test.
def test_script_level_checks_passed():
    assert not (1 if _fail else 0), "module-level checks in this file failed"


if __name__ == "__main__":
    sys.exit(1 if _fail else 0)
