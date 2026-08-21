"""Tests for kevacv/phantoms.py — kill the plant, keep the receptionist.

Built from the real CAM.112 frames. Two phantoms (P3 on the mirror, P8 on the
potted plant) dominate every annotated frame and survived D2 because their ids
churn. The single most dangerous thing this filter could do is delete the
receptionist, who also barely moves — so that case is tested first and hardest.

Run: python test_phantoms.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kevacv.phantoms import (phantom_regions, in_phantom, drop_phantom_dets,
                             describe)

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


def log_from(tracks, fps=7.5, n=2700):     # 2700 @ 7.5fps = 6 min, past MIN_SPAN_S
    """tracks: [(name, fn(t) -> (tid, x1,y1,x2,y2) or None)] -> frame_log"""
    out = []
    for i in range(n):
        t = i / fps
        boxes = [b for _n, f in tracks if (b := f(t)) is not None]
        out.append((i, t, boxes))
    return out


rnd = random.Random(7)

# ── the plant: identical box, but a NEW id every few seconds ────────────────
def plant(t):
    tid = f"plant_{int(t // 4)}"                   # id churns constantly
    j = lambda: rnd.uniform(-1.0, 1.0)             # sub-pixel detector noise
    return (tid, 690 + j(), 350 + j(), 880 + j(), 800 + j())


# ── the mirror phantom: large, static, present the whole hour ───────────────
def mirror(t):
    tid = f"mir_{int(t // 9)}"
    j = lambda: rnd.uniform(-1.5, 1.5)
    return (tid, 820 + j(), 140 + j(), 1270 + j(), 780 + j())


# ── the receptionist: ONE id, stands at her post the whole hour, but SHIFTS ─
# The plant and mirror get +-1px of box noise; she gets far more, and her box
# SIZE changes. That asymmetry is not a thumb on the scale, it is the physics:
# the detector sees identical pixels for a plant every frame and returns an
# identical box, while her arms, shoulders and facing genuinely change what is
# there to detect. A perfectly rigid person is not a conservative test case,
# it is an impossible one.
def receptionist(t):
    sway = 14 * (0.5 - abs((t / 40.0) % 1.0 - 0.5))   # leans, steps, turns
    bob = 6 * ((t / 7.0) % 1.0)
    arm = 11 * ((t / 3.0) % 1.0)          # reaches for the book, the phone
    n = lambda: rnd.uniform(-3.0, 3.0)    # real per-edge detector noise
    return ("sarah", 345 + sway - arm + n(), 170 + bob + n(),
            490 + sway + arm + n(), 430 + bob + n())


print("=" * 74)
print("  THE DANGEROUS CASE: a person who stands still is NOT furniture")
print("=" * 74)
regs = phantom_regions(log_from([("sarah", receptionist)]), frame_wh=(1280, 808))
check(regs == [], "the receptionist standing at her post for 6 min is NOT flagged",
      f"{len(regs)} region(s)")
print("    -> she sways/leans/steps; furniture does not. Jitter separates them.")

print()
print("=" * 74)
print("  THE REAL FAILURE: static phantoms whose IDS CHURN (D2 cannot see these)")
print("=" * 74)
fl = log_from([("plant", plant), ("mirror", mirror), ("sarah", receptionist)])
regs = phantom_regions(fl, frame_wh=(1280, 808))
cent = {r["centre"] for r in regs}
check(len(regs) == 2, "both phantoms found", f"{len(regs)}")
check(any(abs(c[0] - 785) < 60 and abs(c[1] - 575) < 60 for c in cent),
      "the PLANT is flagged", str(cent))
check(any(abs(c[0] - 1045) < 60 and abs(c[1] - 460) < 60 for c in cent),
      "the MIRROR phantom is flagged", str(cent))
check(all(r["ids"] > 1 for r in regs),
      "and each was many different ids — exactly what D2 cannot catch",
      f"{[r['ids'] for r in regs]}")
check(not any(abs(c[0] - 417) < 60 and abs(c[1] - 300) < 60 for c in cent),
      "the receptionist is STILL not flagged, with phantoms present too")
print()
print("   " + describe(regs).replace("\n", "\n   "))

print()
print("=" * 74)
print("  a real person walking IN FRONT of the plant must survive")
print("=" * 74)
walker = ("P9", 700, 300, 800, 780)             # overlaps the plant, taller/thinner
check(not in_phantom(walker[1:5], regs),
      "a person overlapping the plant is kept (IoU, not centre-containment)")
check(in_phantom((691, 351, 879, 799), regs), "the plant box itself is dropped")
kept, dropped = drop_phantom_dets([walker, ("plant_9", 690, 350, 880, 800)], regs)
check(len(kept) == 1 and kept[0][0] == "P9", "drop_phantom_dets keeps the person")
check(len(dropped) == 1, "and drops the phantom", f"{len(dropped)}")

print()
print("=" * 74)
print("  it must not fire on thin evidence")
print("=" * 74)
check(phantom_regions([]) == [], "no frames -> no regions, no crash")
short = log_from([("plant", plant)], n=40)       # ~5s only
check(phantom_regions(short, frame_wh=(1280, 808)) == [],
      "a few seconds of a static box is NOT enough to call it furniture")
check(phantom_regions(fl, frame_wh=(1280, 808),
                      protected={"plant_3"}) and
      not any(abs(r["centre"][0] - 785) < 60 for r in
              phantom_regions(fl, frame_wh=(1280, 808), protected={"plant_3"})),
      "a PROTECTED id vetoes its location entirely",
      "someone who crossed the line or matched a face is never furniture")


def moving(t):                                    # a person crossing the room
    return ("walker", 100 + 12 * t, 300, 180 + 12 * t, 640)


check(phantom_regions(log_from([("w", moving)]), frame_wh=(1280, 808)) == [],
      "someone walking across the room is never flagged")


def breather(t):                                  # static centre, changing SIZE
    s = 40 * ((t / 5.0) % 1.0)
    return ("b", 400 - s, 300 - s, 500 + s, 700 + s)


check(phantom_regions(log_from([("b", breather)]), frame_wh=(1280, 808)) == [],
      "a box that stays put but changes SIZE is not furniture",
      "size_cv veto — people turn and their box breathes")

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
