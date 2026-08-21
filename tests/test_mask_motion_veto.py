"""test_mask_motion_veto.py — a mask must suppress the plant, not the guest.

WHY THIS EXISTS
    A mask polygon is drawn around a static distractor, but it inevitably also
    covers floor that people walk on. On CAM.112 the "plant area mask" spans
    18.7% of the frame because the plant is tall — and the bottom half of that
    rectangle is the floor beside the MAIN ENTRANCE.

    Measured against hand labels: person 5's feet were inside the mask in 100
    of 100 frames and he was detected 29% of the time. The stage removed 18,501
    detections and emptied 212 frames. A polygon we drew ourselves was deleting
    a real guest at the door.

    Location cannot separate a plant from a person standing in front of it —
    they occupy the same polygon. MOTION can, and it is the discriminator
    phantoms.py already relies on: on this footage a statue holds size cv
    ~0.004 while a standing person holds ~0.080, a 19x margin.

WHAT IS ENFORCED
    plant (never moves)        suppressed
    guest (walks through)      kept
    OFF by default             byte-identical to the old delete-everything rule
    unknown history            kept — deleting a real guest is worse than
                               briefly keeping a plant the phantom pass will
                               remove anyway
"""
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

fail = 0


def check(ok, what, detail=""):
    global fail
    print(f"  {'PASS' if ok else 'FAIL'}  {what}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        fail = 1


def veto(track, need, static_s=30.0, window=60.0, rad=90.0):
    """The rule as implemented in engine._drop_masked, replayed here.

    A spot is deleted only when something has occupied it, essentially
    motionless (spread < need), for longer than static_s. Keying by grid cell
    and measuring displacement was the FIRST attempt and it was wrong: a
    walking person leaves a 64px cell before accumulating much displacement,
    so a guest looked as static as the plant. Radius + time-span is what
    phantoms.py already uses, and it separates them.
    """
    hist = defaultdict(list)
    out = []
    for (t, cx, cy) in track:
        if need <= 0:
            out.append(False)          # OFF: original delete-everything
            continue
        k = (int(cx // 128), int(cy // 128))
        near = [h for h in hist[k]
                if t - h[0] <= window and math.hypot(cx - h[1], cy - h[2]) <= rad]
        hist[k] = ([h for h in hist[k] if t - h[0] <= window] + [(t, cx, cy)])[-400:]
        if not near:
            out.append(True)
            continue
        span = t - min(h[0] for h in near)
        spread = (max(math.hypot(a[1] - b[1], a[2] - b[2])
                      for a in near for b in near) if len(near) > 1 else 0.0)
        out.append(not (span >= static_s and spread < need))
    return out


print("=" * 74)
print("  mask motion veto — suppress the plant, keep the guest")
print("=" * 74)

# A PLANT: fixed spot, a pixel or two of box jitter.
plant = [(i * 0.125, 1200 + (i % 2), 900 + (i % 2)) for i in range(400)]  # 50 s
# A GUEST walking through the same region at ~1 m/s.
guest = [(i * 0.125, 1050 + i * 6, 900 + i * 2) for i in range(60)]

kept_plant = sum(veto(plant, need=40.0))
kept_guest = sum(veto(guest, need=40.0))
check(kept_plant <= len(plant) * 0.65, "plant is suppressed once it proves static",
      f"{kept_plant} of {len(plant)} kept — the first 30 s are kept by design")
check(kept_guest >= len(guest) * 0.8, "walking guest is KEPT",
      f"{kept_guest} of {len(guest)} kept")

# OFF must be byte-identical to the old rule: everything in a mask dies.
check(sum(veto(guest, need=0.0)) == 0,
      "OFF (0.0) is exactly the old delete-everything behaviour")

# First sighting has no history. Keeping is the safe error: the phantom pass
# still removes a genuinely static object at end of chunk, but a deleted guest
# is gone for good.
check(veto([(0.0, 1200, 900)], need=40.0)[0] is True,
      "an unknown first sighting is KEPT, not deleted",
      "deleting a real guest is the worse error")

# A guest who pauses briefly must not be dropped mid-visit — the window
# remembers that they arrived by walking.
pause = ([(i * 0.125, 1050 + i * 6, 900) for i in range(20)]
         + [(2.5 + i * 0.125, 1170, 900) for i in range(80)])  # 10 s pause
kept_pause = sum(veto(pause, need=40.0))
check(kept_pause >= len(pause) * 0.8,
      "a guest who walks in then PAUSES stays kept",
      f"{kept_pause} of {len(pause)}")

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
