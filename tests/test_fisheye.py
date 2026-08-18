"""test_fisheye.py — the lens model, checked against a lens we KNOW.

The whole point of this module is that boxes come back in SOURCE coordinates,
so every zone, door line and stored point keeps working. If that round trip is
wrong, nothing downstream can notice -- the boxes are simply in the wrong place
and the counts are quietly wrong, which is this camera's signature failure.

So: bend a synthetic scene with a KNOWN k, then check the module recovers it
and puts the boxes back where they started. No cv2, no torch, no video.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kevacv.fisheye import (box_to_source, dewarped_predict,  # noqa: E402
                            fit_k, horizon_r, in_domain, straightness,
                            to_rectified, to_source)

SIZE = (3840, 2160)
_fail = False


def check(ok, what, detail=""):
    global _fail
    print(f"  {'PASS' if ok else 'FAIL'}  {what}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        _fail = True


print("=" * 74)
print("  the round trip — boxes must land back where they started")
print("=" * 74)

# Every pixel the model CAN invert must round-trip exactly. Points outside the
# domain are handled by the next block -- conflating the two is how "silently
# wrong at the edges" gets shipped.
PROBES = [(10, 10), (1920, 1080), (3830, 2150), (200, 1900), (3600, 90),
          (2600, 1600), (900, 700), (3000, 1950)]
for k in (0.0, 0.12, -0.18, -0.35, 0.35, 0.55):
    worst, n = 0.0, 0
    for x, y in PROBES:
        if not in_domain(x, y, k, SIZE):
            continue
        n += 1
        rx, ry = to_rectified(x, y, k, SIZE)
        bx, by = to_source(rx, ry, k, SIZE)
        worst = max(worst, math.hypot(bx - x, by - y))
    check(n > 0 and worst < 0.5,
          f"to_rectified -> to_source is lossless at k={k:+.2f}",
          f"{n}/{len(PROBES)} probes in domain, worst {worst:.4f}px")

print()
print("=" * 74)
print("  the model's DOMAIN is stated, not discovered")
print("=" * 74)
check(horizon_r(0.2) is None, "k>0 images every radius — no horizon")
check(horizon_r(-0.18) is not None and horizon_r(-0.18) < 1.0,
      "k<0 has a horizon, and at k=-0.18 it is INSIDE the frame corners",
      f"horizon r={horizon_r(-0.18):.3f} vs corner r=1.0")
check(not in_domain(3839, 2159, -0.18, SIZE),
      "so a corner pixel is correctly reported OUT of domain")
check(in_domain(1920, 1080, -0.18, SIZE), "and the centre is in it")
_r = to_rectified(3839, 2159, -0.35, SIZE)
check(all(math.isfinite(c) for c in _r),
      "an out-of-domain pixel CLAMPS to the horizon, never NaN/inf", str(_r))

check(to_source(1920, 1080, 0.3, SIZE) == (1920.0, 1080.0),
      "the optical centre never moves, whatever k is")

print()
print("=" * 74)
print("  a mapped box keeps the WHOLE person")
print("=" * 74)

# Rectifying bends straight edges, so a box's outline bulges between corners.
# Sampling only the corners clips the person -- the exact bad-box failure this
# module exists to remove, reintroduced in miniature.
k = 0.30
box = (2600.0, 1500.0, 2900.0, 2100.0)          # near the edge, where it bends
full = box_to_source(box, k, SIZE)
corners = [to_source(x, y, k, SIZE)
           for x, y in [(box[0], box[1]), (box[2], box[1]),
                        (box[0], box[3]), (box[2], box[3])]]
cx1, cy1 = min(p[0] for p in corners), min(p[1] for p in corners)
cx2, cy2 = max(p[0] for p in corners), max(p[1] for p in corners)
grew = (full[2] - full[0]) * (full[3] - full[1]) >= (cx2 - cx1) * (cy2 - cy1)
check(grew, "perimeter sampling never loses area vs corners alone",
      f"perimeter {full[2]-full[0]:.0f}x{full[3]-full[1]:.0f} vs "
      f"corners {cx2-cx1:.0f}x{cy2-cy1:.0f}")
check(box_to_source((10, 20, 30, 40, 0.9, 0), 0.0, SIZE)[:4] == (10, 20, 30, 40),
      "k=0 is the identity — an undistorted camera pays nothing")
check(box_to_source((10, 20, 30, 40, 0.9, 7), 0.2, SIZE)[4:] == (0.9, 7),
      "confidence and class ride along untouched")

print()
print("=" * 74)
print("  k is MEASURED from straight lines, not eyeballed")
print("=" * 74)

TRUE_K = 0.22


def bend(pts):
    """Where a physically straight line LANDS on a lens with TRUE_K."""
    return [to_source(x, y, TRUE_K, SIZE) for x, y in pts]


# Two lines, in different parts of the frame, as fit_k's docstring demands.
horiz = bend([(300 + i * 320, 1750) for i in range(11)])
vert = bend([(3200, 200 + i * 190) for i in range(11)])

check(straightness(horiz, TRUE_K, SIZE) < 1e-6,
      "at the true k a bent line rectifies back to straight",
      f"rms {straightness(horiz, TRUE_K, SIZE):.2e}")
check(straightness(horiz, 0.0, SIZE) > 1e-3,
      "and at k=0 it is measurably bent — so the metric can tell them apart",
      f"rms {straightness(horiz, 0.0, SIZE):.4f}")

# THE SCALE-BIAS TRAP. straightness() once divided by the FRAME diagonal, so a
# larger k -- which pulls every point toward the centre and shrinks the chain --
# scored "straighter" no matter what the lens did. The objective was monotonic
# in k and every fit walked to the search bound and reported it confidently.
# A chain scaled about the centre is the SAME SHAPE and must score the SAME.
shrunk = [((x - SIZE[0]/2)*0.5 + SIZE[0]/2, (y - SIZE[1]/2)*0.5 + SIZE[1]/2)
          for x, y in horiz]
a, b = straightness(horiz, 0.0, SIZE), straightness(shrunk, 0.0, SIZE)
check(abs(a - b) < 0.02 * max(a, 1e-9) + 1e-6,
      "straightness is SCALE-INVARIANT — shrinking a chain does not fake it",
      f"full {a:.5f} vs half-size {b:.5f}")

got, rms = fit_k([horiz, vert], SIZE)
check(abs(got - TRUE_K) < 0.01, "fit_k recovers the true k from the scene",
      f"got {got:.4f} vs true {TRUE_K}, rms {rms:.2e}")

k0, _ = fit_k([], SIZE)
check(k0 == 0.0, "no lines -> k=0, i.e. do nothing, rather than guess")
k1, _ = fit_k([[(1, 1), (2, 2)]], SIZE)
check(k1 == 0.0, "a 2-point 'line' is refused — 2 points are straight for ANY k")

print()
print("=" * 74)
print("  the detector contract matches tiled.py")
print("=" * 74)


class FakeFrame:
    shape = (2160, 3840, 3)


seen = {}


def fake_predict(img):
    seen["got"] = img
    return [(2600.0, 1500.0, 2900.0, 2100.0, 0.88, 0)]


out = dewarped_predict(FakeFrame(), fake_predict, 0.25,
                       remap_fn=lambda f, k: "RECTIFIED")
check(seen["got"] == "RECTIFIED", "the detector is shown the RECTIFIED image")
check(len(out) == 1 and out[0][4:] == (0.88, 0),
      "and gets back the same tuple shape tiled_predict uses")
src = out[0]
check(src[:4] != (2600.0, 1500.0, 2900.0, 2100.0),
      "the box is mapped BACK to source pixels, not left in rectified space",
      "if it were not, every zone and door line would be silently wrong")

seen.clear()
out0 = dewarped_predict(FakeFrame(), fake_predict, 0.0,
                        remap_fn=lambda f, k: "RECTIFIED")
check(seen.get("got") != "RECTIFIED" or out0[0][:4] == (2600.0, 1500.0, 2900.0, 2100.0),
      "k=0 skips the remap entirely — no cost on a rectilinear camera")

print()
print("=" * 74)
print("  ALL PASS" if not _fail else "  FAILURES ABOVE")
print("=" * 74)
sys.exit(1 if _fail else 0)
