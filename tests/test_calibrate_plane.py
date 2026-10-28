"""test_calibrate_plane.py — the check that makes the calibration trustworthy.

Four points ALWAYS produce a homography. Four BAD points produce a confident
wrong one, and a wrong plane is worse than no plane here: it makes the
implausible-size filter over-fire, which makes the D1 guard relax its own
tolerance, which makes the giant wall boxes bigger. So the property under test
is not "does it fit" -- it is "does it REFUSE to fit rubbish".

Synthetic camera, no frame and no cv2 window: points are projected through a
known homography, so the true answer is known and a failure is unambiguous.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.calibrate_plane import (enough_evidence, parse_point,  # noqa: E402
                                   residuals, verdict, world_of, FIT_TOL_M)
from kevacv.ground_plane import GroundPlane  # noqa: E402

_fail = False


def check(ok, what, detail=""):
    global _fail
    print(f"  {'PASS' if ok else 'FAIL'}  {what}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        _fail = True


print("=" * 74)
print("  parsing the operator's shorthand")
print("=" * 74)
check(parse_point("812,1490@0,3") == ((812.0, 1490.0), (0.0, 3.0)),
      "X,Y@col,row parses to image px and tile index")
try:
    parse_point("812-1490")
    check(False, "malformed input is rejected")
except Exception:
    check(True, "malformed input is rejected, not silently half-read")

check(world_of((3, 2), 0.30) == (0.8999999999999999, 0.6) or
      abs(world_of((3, 2), 0.30)[0] - 0.9) < 1e-9,
      "tile (col,row) x tile size -> metres", world_of((3, 2), 0.30))

print()
print("=" * 74)
print("  a GOOD set of points passes")
print("=" * 74)

# A synthetic floor: image points generated FROM known world points through a
# plausible perspective, so the correct plane is recoverable exactly.
TILE = 0.30
GRID = [(0, 0), (4, 0), (0, 4), (4, 4), (2, 2)]


def project(col, row):
    """A deliberately non-affine (perspective) mapping, so the test would fail
    for anything that quietly assumes a linear floor."""
    X, Z = col * TILE, row * TILE
    w = 1.0 + 0.12 * Z
    return (900.0 + 260.0 * X / w, 1400.0 + 240.0 * Z / w)


good = [(project(c, r), (c, r)) for c, r in GRID[:4]]
plane = GroundPlane.from_correspondences(
    [p for p, _ in good], [world_of(t, TILE) for _, t in good], (3840, 2160))
check(plane.ok, "4 well-spread points produce a plane", plane.note)
rows = residuals(plane, good, TILE)
ok, worst = verdict(rows, FIT_TOL_M)
check(ok, "and it reproduces the tile grid it was fitted on",
      f"worst {worst:.4f}m")

# The only honest test: a point the fit never saw.
held = good + [(project(*GRID[4]), GRID[4])]
hrows = [r for r in residuals(plane, held, TILE)
         if "p5" in (r[0], r[1])]
hok, hworst = verdict(hrows, FIT_TOL_M)
check(hok, "a HELD-OUT point lands where the grid says it must",
      f"worst {hworst:.4f}m")

print()
print("=" * 74)
print("  BAD points are refused, not fitted")
print("=" * 74)

# THE TRAP, and the reason enough_evidence() exists. A homography has 8 DOF
# and 4 correspondences give 8 equations, so it fits ANY four points exactly.
# Mislabel one tile index and the residual is STILL 0.00m -- the check is
# vacuous at n=4, which makes "4 measurements" the wrong thing to ask for.
mis4 = [good[0], (good[1][0], (1, 0)), good[2], good[3]]
m4 = GroundPlane.from_correspondences(
    [p for p, _ in mis4], [world_of(t, TILE) for _, t in mis4], (3840, 2160))
_, w4 = verdict(residuals(m4, mis4, TILE), FIT_TOL_M)
# 1e-3 m = 1mm, i.e. solver noise. The contrast is what matters: the SAME
# mislabelled point scores ~0.000m at n=4 and 0.281m at n=5.
check(w4 is not None and w4 < 1e-3,
      "4 points fit ANY labelling with ~zero error — the trap is real",
      f"worst {w4:.6f}m even though p2 is mislabelled")
check(enough_evidence(4, 0)[0] is False,
      "so 4 fitted points with no --check is refused as UNVERIFIED")
check(enough_evidence(5, 0)[0] is True, "5 fitted points is over-determined")
check(enough_evidence(4, 1)[0] is True, "or 4 fitted + 1 held out")

# With a 5th point the fit can no longer absorb the mistake, and it shows.
mis5 = mis4 + [(project(*GRID[4]), GRID[4])]
m5 = GroundPlane.from_correspondences(
    [p for p, _ in mis5], [world_of(t, TILE) for _, t in mis5], (3840, 2160))
mok, mworst = verdict(residuals(m5, mis5, TILE), FIT_TOL_M)
check(not mok, "with 5 points a mislabelled tile index FINALLY FAILS",
      f"worst {mworst:.3f}m > {FIT_TOL_M}m tolerance" if mworst else "no plane")

# Collinear points: a homography is not defined, and inventing one is exactly
# the "confident nonsense" this module exists to prevent.
line = [((900.0 + 50 * i, 1400.0), (i, 0)) for i in range(4)]
lplane = GroundPlane.from_correspondences(
    [p for p, _ in line], [world_of(t, TILE) for _, t in line], (3840, 2160))
lrows = residuals(lplane, line, TILE) if lplane.ok else []
lok, _ = verdict(lrows, FIT_TOL_M) if lrows else (False, None)
check(not lok, "four points in a straight line do not produce a usable plane",
      lplane.note)

check(verdict([("p1", "p2", 1.0, None, None)], FIT_TOL_M)[0] is False,
      "a pair the plane cannot map at all is a hard FAIL, not a skipped row")

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
