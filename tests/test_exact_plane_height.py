"""test_exact_plane_height.py — a measured floor must reach the filter it was sold to fix.

WHY THIS EXISTS
    kevacv/config.py tells an operator that supplying `ground_points` is "the
    real fix" for the implausible-size filter (D1). It was not, and could not
    be: D1 asks its geometry object for expected_h(), and GroundPlane returned
    None for anything except the automatic fit. A venue that measured its floor
    got byte-identical behaviour to one that had not.

    Found twice, independently: by measurement on 2026-08-26 (applying the
    exact plane moved D1's drop count by exactly zero) and in the code on
    2026-08-29 (`if self.mode != "auto": return None`).

WHAT THIS GUARDS
    That the wiring stays connected, and that the approximation stays honest
    about what it is.

Run: python tests/test_exact_plane_height.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


from kevacv.ground_plane import GroundPlane  # noqa: E402

# a synthetic floor: 4 image points -> a 2x2 metre square on the ground
IMG = [[400, 900], [1500, 900], [1300, 600], [600, 600]]
WORLD = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]

print()
print("=" * 76)
print("  an EXACT plane now answers the question D1 asks")
print("=" * 76)
g = GroundPlane.from_correspondences(IMG, WORLD, (1920, 1080))
check(g.ok and g.mode == "exact", "built an exact plane from 4 correspondences",
      f"mode={g.mode}")
h_near = g.expected_h(880)
h_far = g.expected_h(620)
check(h_near is not None,
      "expected_h returns a NUMBER in exact mode (it used to return None)",
      f"{h_near}")
check(h_far is not None, "and at a far row too", f"{h_far}")

print()
print("=" * 76)
print("  and the answer has the right SHAPE")
print("=" * 76)
if h_near and h_far:
    check(h_near > h_far,
          "a person NEARER the camera is predicted TALLER in pixels",
          f"near row 880 -> {h_near:.0f}px, far row 620 -> {h_far:.0f}px")
    check(50 < h_near < 2000,
          "the near prediction is a physically sane pixel height",
          f"{h_near:.0f}px")

print()
print("=" * 76)
print("  auto mode is unchanged — this must not disturb the existing path")
print("=" * 76)
ga = GroundPlane.from_perspective(0.8, 20.0, 1920, 1080)
check(ga.mode == "auto", "auto plane still builds", f"mode={ga.mode}")
check(ga.expected_h(900) is not None, "auto expected_h still answers")
check(abs(ga.expected_h(900) - (0.8 * 900 + 20.0)) < 1e-6,
      "auto still uses the linear fit exactly as before",
      "the regression path must be untouched")

print()
print("=" * 76)
print("  a plane that cannot say anything says None, not a wrong number")
print("=" * 76)
gn = GroundPlane.none("no geometry")
check(gn.expected_h(900) is None,
      "a 'none' plane returns None",
      "D1 flags nothing when expected_h is None — a filter that fires on "
      "missing information is worse than no filter")

print()
print("=" * 76)
print("  the wiring is present in the engine")
print("=" * 76)
ENG = (ROOT / "kevacv" / "engine.py").read_text(encoding="utf-8")
check("ENABLE_D1_EXACT_PLANE" in ENG,
      "D1 consults the flag")
check('getattr(_ground, "mode", "") == "exact"' in ENG,
      "and only uses the plane when it is genuinely EXACT",
      "an auto plane must keep going through _PerspectiveModel")
CFG = (ROOT / "kevacv" / "config.py").read_text(encoding="utf-8")
check('"analysis.enable_d1_exact_plane"' in CFG,
      "the flag is reachable from a yaml, so it can be A/B'd")

print()
print("=" * 76)


def test_script_level_checks_passed():
    assert not FAILED, f"exact-plane height checks failed: {FAILED}"


if __name__ == "__main__":
    if FAILED:
        print(f"  {len(FAILED)} FAILED:")
        for x in FAILED:
            print(f"    - {x}")
        sys.exit(1)
    print("  ALL PASS")
    print("=" * 76)
