"""Validate ground_plane.py against a SYNTHETIC CAMERA with known truth.

This is the strongest validation available without footage: build a pinhole
camera with a known height and focal length, project people of known height
standing at known floor positions, fit the perspective line from those
projections exactly the way Phase 1 does, then check that the recovered
geometry matches the truth we started from.

It also does the thing a calibration paper usually skips: MEASURES how wrong
the auto mode gets when the camera is tilted (it assumes level), instead of
asserting it is fine. If the honest answer were "unusable", this file would say
so and Phase 3 would need the 4-point homography as a hard requirement.

Run: python test_ground_plane.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kevacv.ground_plane import PERSON_H_M, GroundPlane, synth_camera

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


def fit_line(pts, n_bins=12):
    """Same robust bin-median fit the notebook uses, so we validate the mapping
    against the estimator that actually feeds it — not an idealised one."""
    ys = [p[0] for p in pts]
    lo, hi = min(ys), max(ys)
    buckets = [[] for _ in range(n_bins)]
    for y, h in pts:
        k = min(n_bins - 1, int((y - lo) / (hi - lo) * n_bins))
        buckets[k].append(h)
    P = []
    for k, hs in enumerate(buckets):
        if len(hs) < 3:
            continue
        hs.sort()
        P.append((lo + (k + 0.5) * (hi - lo) / n_bins, hs[len(hs) // 2]))
    mx = sum(p[0] for p in P) / len(P)
    my = sum(p[1] for p in P) / len(P)
    den = sum((p[0] - mx) ** 2 for p in P)
    a = sum((p[0] - mx) * (p[1] - my) for p in P) / den
    return a, my - a * mx


def sample_people(project, zs, xs, person_h=PERSON_H_M):
    """-> (foot_y, pixel_height) pairs and the ground truth behind each."""
    pts, truth = [], []
    for Z in zs:
        for X in xs:
            f = project(X, Z, 0.0)
            hd = project(X, Z, person_h)
            if f is None or hd is None:
                continue
            pts.append((f[1], f[1] - hd[1]))
            truth.append((f, (X, Z)))
    return pts, truth


CAM_H, FOCAL, FRAME = 3.0, 1200.0, (1920, 1080)
ZS = [z / 2 for z in range(6, 60)]
XS = [-4, -2, 0, 2, 4]

print("=" * 74)
print("  AUTO mode vs a LEVEL synthetic camera (truth is known exactly)")
print("=" * 74)
proj = synth_camera(CAM_H, FOCAL, FRAME, tilt_deg=0.0)
pts, truth = sample_people(proj, ZS, XS)
a, b = fit_line(pts)
gp = GroundPlane.from_perspective(a, b, FRAME[0], FRAME[1], focal_px=FOCAL)
print(f"    {gp.describe()}")

check(gp.ok, "plane built from the fitted line")
check(abs(gp.camera_height_m() - CAM_H) < 0.05,
      f"recovers camera height {CAM_H} m", f"{gp.camera_height_m():.3f} m")
check(abs(gp.horizon_y() - FRAME[1] / 2) < 3,
      "recovers the horizon at the principal row",
      f"{gp.horizon_y():.1f} vs {FRAME[1]/2}")
check(not gp.sanity(FRAME[1]), "no sanity warnings on a clean fit",
      str(gp.sanity(FRAME[1])))

errs_x, errs_z = [], []
for (foot, (X, Z)) in truth:
    g = gp.to_ground(*foot)
    if g is None:
        continue
    errs_x.append(abs(g[0] - X))
    errs_z.append(abs(g[1] - Z) / Z)
check(max(errs_x) < 0.15, "lateral position recovered to <15 cm",
      f"max {max(errs_x)*100:.1f} cm")
check(max(errs_z) < 0.03, "depth recovered to <3% of range",
      f"max {max(errs_z)*100:.1f}%")

print()
print("=" * 74)
print("  the whole point: distances that pixels get badly wrong")
print("=" * 74)
# Two people 2.0 m apart, once near the camera and once far away. In PIXELS
# these look wildly different; in metres they must both read 2.0 m.
rows = []
for Z in (5.0, 12.0, 25.0):
    p = proj(-1.0, Z, 0.0)
    q = proj(1.0, Z, 0.0)
    px = math.hypot(p[0] - q[0], p[1] - q[1])
    m = gp.dist_m(p, q)
    rows.append((Z, px, m))
    print(f"    two people 2.00 m apart at {Z:5.1f} m depth -> "
          f"{px:7.1f} px   ground plane says {m:.2f} m")
px_spread = max(r[1] for r in rows) / min(r[1] for r in rows)
m_err = max(abs(r[2] - 2.0) for r in rows)
check(px_spread > 4.0, "the same 2 m reads 4x+ differently in PIXELS",
      f"{px_spread:.1f}x spread")
check(m_err < 0.10, "the same 2 m reads correctly in METRES everywhere",
      f"max error {m_err*100:.1f} cm")
print(f"    -> this ratio IS the bug: one px threshold cannot serve both ends")

s_near, s_far = gp.scale_at(1000), gp.scale_at(600)
check(s_far > s_near * 2, "metres-per-pixel varies strongly across the frame",
      f"{s_near:.4f} near vs {s_far:.4f} far m/px")
check(abs(gp.px_for_metres(2.0, 1000) - math.hypot(*(np.subtract(
    proj(-1.0, gp.to_ground(FRAME[0]/2, 1000)[1], 0.0),
    proj(1.0, gp.to_ground(FRAME[0]/2, 1000)[1], 0.0))))) < 12,
    "px_for_metres round-trips against the real projection")

print()
print("=" * 74)
print("  HOW WRONG does AUTO get on a TILTED camera? (it assumes level)")
print("=" * 74)
print(f"    {'tilt':>6s}{'cam height':>13s}{'lateral err':>14s}{'depth err':>12s}"
      f"{'2m dist err':>13s}")
tilt_results = {}
for tilt in (0, 5, 10, 15, 20, 30):
    pr = synth_camera(CAM_H, FOCAL, FRAME, tilt_deg=tilt)
    pp, tt = sample_people(pr, ZS, XS)
    if len(pp) < 40:
        continue
    aa, bb = fit_line(pp)
    g2 = GroundPlane.from_perspective(aa, bb, FRAME[0], FRAME[1], focal_px=FOCAL)
    if not g2.ok:
        continue
    ex, ez, ed = [], [], []
    for (foot, (X, Z)) in tt:
        gg = g2.to_ground(*foot)
        if gg is None:
            continue
        ex.append(abs(gg[0] - X))
        ez.append(abs(gg[1] - Z) / Z)
    for Z in (6.0, 12.0, 20.0):
        p, q = pr(-1.0, Z, 0.0), pr(1.0, Z, 0.0)
        d = g2.dist_m(p, q)
        if d:
            ed.append(abs(d - 2.0))
    tilt_results[tilt] = (g2.camera_height_m(), np.median(ex),
                          np.median(ez), np.median(ed) if ed else float("nan"))
    ch, mx, mz, md = tilt_results[tilt]
    print(f"    {tilt:>5d}d{ch:>12.2f}m{mx*100:>12.1f}cm{mz*100:>11.1f}%"
          f"{md*100:>11.1f}cm")

check(tilt_results[10][3] < 0.35, "at 10 deg tilt, a 2 m distance is still within 35 cm",
      f"{tilt_results[10][3]*100:.0f} cm")
check(tilt_results[20][3] < 1.0, "at 20 deg tilt, still within 1 m (degrades, not breaks)",
      f"{tilt_results[20][3]*100:.0f} cm")
check(tilt_results[30][3] > tilt_results[10][3],
      "error grows with tilt (the model is honest about its assumption)")
print("    -> AUTO degrades gracefully with tilt and never collapses. For exact")
print("       numbers on a steeply tilted camera, supply ground_points (below).")

print()
print("=" * 74)
print("  EXACT mode — four correspondences, no assumptions")
print("=" * 74)
pr = synth_camera(CAM_H, FOCAL, FRAME, tilt_deg=20.0)     # the hard case
world = [(-3.0, 6.0), (3.0, 6.0), (-3.0, 18.0), (3.0, 18.0)]
img = [pr(X, Z, 0.0) for X, Z in world]
gx = GroundPlane.from_correspondences(img, world, FRAME)
check(gx.ok and gx.mode == "exact", "homography built from 4 floor points")
ex = []
for Z in (5.0, 9.0, 14.0, 22.0):
    for X in (-2.0, 0.0, 2.5):
        p = pr(X, Z, 0.0)
        g = gx.to_ground(*p)
        ex.append(math.hypot(g[0] - X, g[1] - Z))
check(max(ex) < 0.05, "EXACT mode recovers floor positions to <5 cm even at 20 deg tilt",
      f"max {max(ex)*100:.2f} cm")
d = gx.dist_m(pr(-1.0, 11.0, 0.0), pr(1.0, 11.0, 0.0))
check(abs(d - 2.0) < 0.02, "exact 2 m distance", f"{d:.3f} m")
check(GroundPlane.from_correspondences([(0, 0), (1, 1), (2, 2), (3, 3)],
                                       world, FRAME).mode in ("none", "exact"),
      "collinear/degenerate input does not raise")

print()
print("=" * 74)
print("  refusing to produce confident nonsense")
print("=" * 74)
check(not GroundPlane.from_perspective(0.0, 100, 1920, 1080).ok,
      "a=0 (no perspective at all) -> plane refused")
check(not GroundPlane.from_perspective(-0.5, 100, 1920, 1080).ok,
      "a<0 (people shrink as they approach) -> plane refused")
check(not GroundPlane.none().ok, "explicit none() is not ok")
check(GroundPlane.none().to_ground(5, 5) is None, "no plane -> no coordinates")
check(GroundPlane.none().dist_m((0, 0), (1, 1)) is None, "no plane -> no distance")
check("NOT AVAILABLE" in GroundPlane.none().describe(),
      "no-plane state announces itself")
above = gp.to_ground(960, gp.horizon_y() - 5)
check(above is None, "a point ABOVE the horizon returns None, not a fabricated metre")
bad = GroundPlane.from_perspective(0.05, 0.0, 1920, 1080)   # implies 34 m camera
check(bad.sanity(1080), "implausible camera height raises a sanity warning",
      str(bad.sanity(1080))[:60])
check("implausible" in bad.describe(), "and says so in describe()")

print()
print("=" * 74)
print("  zone-config integration")
print("=" * 74)


class FakePersp:
    def __init__(self, a, b):
        self._f = (a, b)

    def _refit(self):
        return self._f


cfg_exact = {"ground_points": [{"image": list(img[i]), "world": list(world[i])}
                               for i in range(4)]}
g = GroundPlane.from_zone_config(cfg_exact, FRAME, persp=FakePersp(a, b))
check(g.mode == "exact", "ground_points in the zones JSON win over the auto fit")
g = GroundPlane.from_zone_config({}, FRAME, persp=FakePersp(a, b))
check(g.mode == "auto", "no ground_points -> falls back to the auto fit")
g = GroundPlane.from_zone_config({}, FRAME, persp=None)
check(g.mode == "none", "no ground_points and no fit -> honest 'none'")
g = GroundPlane.from_zone_config({"ground_points": [{"image": [1, 2]}]}, FRAME,
                                 persp=FakePersp(a, b))
check(g.mode == "auto", "malformed ground_points degrade to auto, not a crash")

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
