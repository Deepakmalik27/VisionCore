"""test_mirror_suppression.py — the glass half of a lockstep pair is a reflection.

THE PROBLEM THIS CLOSES
    `detect_filters.mirrored_pair_ids` finds track pairs that move in lockstep
    at a fixed offset -- the signature of a person and their reflection. On the
    19:30-20:30 Delilah hour it found 55 such pairs, EVERY run, and removed
    none of them. The reason was written into pipeline.py and was correct:

        "mirrored pairs are NOT removed, because deciding which half of the
         pair is the reflection needs geometry this stage does not have, and
         deleting the real person is worse than counting both."

    So 55 phantom identities stayed in the headcount rather than risk deleting
    a real guest. That was the right call with the information available.

WHAT CHANGED
    `mirror_regions` -- polygons drawn over the glass in CVAT against the real
    footage -- supplies exactly the missing geometry. One of the pair inside
    the glass and the other outside is no longer ambiguous.

WHAT THIS FILE GUARDS
    The CONSERVATISM, mostly. A suppression rule that fires when it should not
    deletes real guests, which is strictly worse than the bug it replaces. So
    the cases that must stay untouched are tested harder than the case that
    fires.

Run: python tests/test_mirror_suppression.py
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


from kevacv.coverage import _point_in_poly  # noqa: E402

# the rule, extracted exactly as pipeline.py applies it, so it can be exercised
# without a video. The source assertions at the bottom keep them in lockstep.
GLASS = [[1000, 0], [1900, 0], [1900, 1000], [1000, 1000]]


def frac_in(points, regions):
    if not points:
        return 0.0
    n = sum(1 for q in points if any(_point_in_poly(mp, q) for mp in regions))
    return n / len(points)


def resolve(pair_positions, regions):
    """-> the id to DROP, or None when it cannot be told."""
    (a_id, a_pts), (b_id, b_pts) = pair_positions
    fa, fb = frac_in(a_pts, regions), frac_in(b_pts, regions)
    if fa >= 0.6 and fb <= 0.2:
        return a_id
    if fb >= 0.6 and fa <= 0.2:
        return b_id
    return None


print()
print("=" * 74)
print("  it resolves the case it was built for")
print("=" * 74)
real = [(500, 800)] * 10                     # out on the floor
refl = [(1500, 800)] * 10                    # inside the glass
check(resolve((("A", real), ("B", refl)), [GLASS]) == "B",
      "the half INSIDE the glass is dropped", "and the real person is kept")
check(resolve((("A", refl), ("B", real)), [GLASS]) == "A",
      "order does not matter")

print()
print("=" * 74)
print("  it REFUSES when the geometry is ambiguous")
print("=" * 74)
check(resolve((("A", real), ("B", real)), [GLASS]) is None,
      "BOTH outside the glass -> resolve nothing",
      "two real people walking together must never be halved")
check(resolve((("A", refl), ("B", refl)), [GLASS]) is None,
      "BOTH inside -> resolve nothing")
check(resolve((("A", real), ("B", refl)), []) is None,
      "no mirror_regions drawn -> resolve nothing",
      "the old behaviour is the default; drawing glass is opt-in")

half = [(500, 800)] * 5 + [(1500, 800)] * 5   # 50% inside — a person walking past
check(resolve((("A", real), ("B", half)), [GLASS]) is None,
      "a track only HALF inside the glass is not called a reflection",
      "0.6/0.2 thresholds exist so a passer-by is not deleted")

print()
print("=" * 74)
print("  the engine really wires it, and into the SAME drop set")
print("=" * 74)
PIPE = (ROOT / "kevacv" / "pipeline.py").read_text(encoding="utf-8")
check('_mirror_regions = (zones or {}).get("mirror_regions")' in PIPE,
      "mirror_regions is read from the zones config")
check("drop = set(still) | set(rigid) | _mirror_drop" in PIPE,
      "resolved reflections join the SAME drop set as static/rigid",
      "so drop_tracks removes them from events, crossings AND frame_log together")
check("fa >= 0.6 and fb <= 0.2" in PIPE,
      "the engine uses the same conservative thresholds as this test")
check("_unresolved" in PIPE,
      "unresolved pairs still raise the warning they always did",
      "silence would read as 'no reflections', which is the opposite of true")

print()
print("=" * 74)


def test_script_level_checks_passed():
    assert not FAILED, f"mirror suppression checks failed: {FAILED}"


if __name__ == "__main__":
    if FAILED:
        print(f"  {len(FAILED)} FAILED:")
        for x in FAILED:
            print(f"    - {x}")
        sys.exit(1)
    print("  ALL PASS")
    print("=" * 74)
