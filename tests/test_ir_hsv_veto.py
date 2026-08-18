"""test_ir_hsv_veto.py — a COLOUR veto must abstain across the IR boundary.

WHY THIS EXISTS
    merge_fragmented_tracks has a pure-spatial tier: a body that reappears
    within ~30px of where it vanished is the same person, "even with a
    weak/inconclusive embedding". That tier is what holds a receptionist
    together across the gaps where the detector loses them.

    Two vetoes can overrule it. One of them, _handoff_hsv_contradicts, compares
    torso HUE histograms and justifies itself with

        "within a hand-off-sized gap clothing color cannot change"

    which is true only if both crops came from the same imaging mode. 66% of
    CAM.112's frames are INFRARED — greyscale — so the same person photographed
    either side of a switch has two unrelated histograms, and the veto fires on
    physics rather than on identity.

    The module already knew this. Its cross-boundary guard _ir_mismatch says,
    140 lines earlier:

        "Only pairs on OPPOSITE sides of the boundary are blocked; face +
         hand-off/stationary tiers still bridge it (faces survive near-IR,
         physics doesn't care about colour)."

    The guard exempted those tiers. The veto inside them did not. Measured cost
    on the 18:30 hour: pairs 2.7px apart across a 128s gap left unmerged, 408
    fragments collapsing to only 31 identities, and the staff rule reading that
    scatter as 18 people working a one-receptionist desk.

WHAT IS ENFORCED
    1. same imaging mode  -> the colour veto still works (strangers stay apart)
    2. across the boundary -> the veto abstains and the spatial merge stands
    3. abstaining is not the same as inverting: a spatially IMPLAUSIBLE pair is
       still not merged, boundary or no boundary
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


# Two fragments of ONE stationary person: 3px apart, 128s gap — the exact
# shape of the unmerged pairs in the hour log.
# Integer ids ON PURPOSE. A non-numeric track id means an ENROLLED
# STAFF NAME to the stitcher, and two different names can never merge
# whatever the geometry says — an earlier version of this test used
# "a"/"b" and every case came back split, which looked exactly like
# the fix not working.
WINDOWS = {1: (0.0, 100.0), 2: (228.0, 400.0)}
POSITIONS = {1: [(900.0, 700.0), (900.0, 700.0)],
             2: [(903.0, 700.0), (903.0, 700.0)]}

# Torso crops a colour veto reads as "different people": a solid blue shirt and
# a solid red one. Under IR that difference is the SENSOR, not the clothing.
import numpy as np  # noqa: E402


def _solid(bgr):
    img = np.zeros((64, 32, 3), dtype=np.uint8)
    img[:, :] = bgr
    return img


CROPS_DIFFERENT = {1: [_solid((255, 0, 0))] * 3,     # blue
                   2: [_solid((0, 0, 255))] * 3}     # red


def merged(ir_hint, crops=CROPS_DIFFERENT, positions=POSITIONS):
    res = merge_fragmented_tracks(
        WINDOWS, {}, positions=positions, raw_crops=crops,
        ir_hint=ir_hint, sim_threshold=0.37)
    mapping = res[0] if isinstance(res, tuple) else res
    return mapping.get(1) == mapping.get(2)


print("=" * 74)
print("  colour veto vs the infrared boundary")
print("=" * 74)

# 1. BOTH IN COLOUR — the veto is meaningful and must still bite.
same_mode = {1: 0.0, 2: 0.0}
check(not merged(same_mode),
      "same imaging mode: clearly different torsos are NOT merged",
      "veto still works")

# 2. ACROSS THE BOUNDARY — colour says nothing; spatial evidence stands.
across = {1: 0.0, 2: 1.0}
check(merged(across),
      "across the IR boundary: 3px/128s stationary pair IS merged",
      "veto abstains, physics wins")

# 3. Abstaining must not become merging-anything. Same boundary, but the two
#    fragments are now 900px apart — no spatial claim to stand on.
far = {1: [(200.0, 700.0), (200.0, 700.0)],
       2: [(1100.0, 700.0), (1100.0, 700.0)]}
check(not merged(across, positions=far),
      "across the boundary, a spatially implausible pair stays separate",
      "abstain != invert")

print()
print("  ALL PASS" if not fail else "  FAILURES ABOVE")
sys.exit(fail)
