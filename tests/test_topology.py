"""Tests for kevacv/topology.py — the re-appearance gate that survives long gaps.

Built from the measured CAM.112 ceiling: same-person p50=0.435 against a 0.60
merge bar, and a velocity gate that allows ~1980 m once the gap reaches 900 s.
Every test asks: does this constraint still bite when appearance has stopped
discriminating?

Run: python test_topology.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kevacv.topology import (describe, doors_from_endpoints, doors_from_zones,
                             reappearance_verdict, veto_pairs)

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


WH = (1280, 720)          # diagonal ~1468 px -> door radius ~147, still ~88
DOOR = (1150, 600)        # main entrance, right of frame
DOORS = [DOOR]
MID = (400, 400)          # middle of the room, far from any door

print("=" * 74)
print("  the three physically possible shapes")
print("=" * 74)
v = reappearance_verdict(DOOR, DOOR, 600.0, DOORS, WH)
check(v["allow"] and v["shape"] == "door_to_door",
      "left at a door, returned at a door -> allowed at ANY gap", f"gap 600s")

v = reappearance_verdict(MID, (430, 420), 300.0, DOORS, WH)
check(v["allow"] and v["shape"] == "occlusion_recovery",
      "vanished and reappeared in the same place -> tracker lost a body",
      f"drift {v['drift_px']}px")

v = reappearance_verdict(DOOR, MID, 300.0, DOORS, WH)
check(not v["allow"] and v["shape"] == "impossible",
      "left through the door, reappeared mid-room -> VETO")
check("walk back in unobserved" in v["why"], "and says why in plain words")

v = reappearance_verdict(MID, DOOR, 300.0, DOORS, WH)
check(not v["allow"], "vanished mid-room, reappeared at the door -> VETO")
check("walk to the door unobserved" in v["why"], "and names that shape too")

v = reappearance_verdict(MID, (900, 200), 300.0, DOORS, WH)
check(not v["allow"], "two different mid-room places, no door -> VETO")
check("no path exists" in v["why"], "and explains there is no unseen route")

print()
print("=" * 74)
print("  it stays sharp exactly where the velocity gate goes blind")
print("=" * 74)
for gap in (60.0, 300.0, 900.0, 7200.0):
    v = reappearance_verdict(DOOR, MID, gap, DOORS, WH)
    check(not v["allow"], f"still vetoed at gap={gap:.0f}s", v["shape"])
print("    -> at 900s the velocity gate allows ~1980 m; this does not care")

v = reappearance_verdict(DOOR, DOOR, 7200.0, DOORS, WH)
check(v["allow"], "and a genuine door-to-door return is NOT punished for the gap")

print()
print("=" * 74)
print("  a gate with no information must not block anything")
print("=" * 74)
v = reappearance_verdict(DOOR, MID, 300.0, [], WH)
check(v["allow"] and v["shape"] == "abstain", "no doors -> abstain, never veto")
check("cannot judge" in v["why"], "and says it is abstaining, not approving")

v = reappearance_verdict(DOOR, MID, 5.0, DOORS, WH)
check(v["allow"] and v["shape"] == "continuous",
      "short gaps are left to the positional gate")
v = reappearance_verdict(DOOR, MID, None, DOORS, WH)
check(v["allow"], "an unknown gap is not treated as a long one")

print()
print("=" * 74)
print("  two doors, and the radius is a place not a point")
print("=" * 74)
TWO = [DOOR, (100, 650)]                       # main entrance + side door
v = reappearance_verdict(DOOR, (100, 650), 600.0, TWO, WH)
check(v["allow"] and v["shape"] == "door_to_door",
      "out one door, in another -> allowed (venues have more than one door)")
v = reappearance_verdict((1150 + 100, 600), DOOR, 600.0, DOORS, WH)
check(v["allow"], "100px from the door centre still counts as at the door",
      f"radius {v['door_radius_px']}px")
v = reappearance_verdict((1150 - 400, 600), MID, 600.0, DOORS, WH)
check(not v["allow"], "400px away does not")

print()
print("=" * 74)
print("  it only ever vetoes — it never invents a merge")
print("=" * 74)
pairs = [
    {"a": 1, "b": 2, "death_pos": DOOR, "birth_pos": DOOR, "gap_s": 400.0},
    {"a": 3, "b": 4, "death_pos": DOOR, "birth_pos": MID, "gap_s": 400.0},
    {"a": 5, "b": 6, "death_pos": MID, "birth_pos": (420, 410), "gap_s": 400.0},
]
kept, vetoed = veto_pairs(pairs, DOORS, WH)
check(len(kept) == 2 and len(vetoed) == 1, "one impossible pair removed",
      f"{len(kept)} kept / {len(vetoed)} vetoed")
check({p["a"] for p in vetoed} == {3}, "and it is the right one")
check(all("a" in p and "b" in p for p in kept),
      "caller fields are carried through untouched")
check(kept[0]["topology"]["shape"] == "door_to_door", "verdict is attached")
check(len(veto_pairs([], DOORS, WH)[0]) == 0, "no pairs -> no crash")
txt = describe(kept, vetoed)
check("VETO" in txt and "door_to_door" in txt, "describe() names shapes and vetoes")

print()
print("=" * 74)
print("  doors can come from drawn zones OR from learned endpoints")
print("=" * 74)
d = doors_from_zones({"main_entrance": [(0, 0), (100, 0), (100, 100), (0, 100)],
                      "waiting_area": [(500, 500), (600, 500), (600, 600)]},
                     {"main_entrance": ["entry"], "waiting_area": ["wait"]})
check(len(d) == 1 and d[0] == (50.0, 50.0), "entry polygon -> its centre", str(d))
check(doors_from_zones({}, {}) == [], "no zones -> no doors (abstain, not guess)")
check(doors_from_endpoints([{"centre": (10, 20)}]) == [(10.0, 20.0)],
      "learned zone with 'centre'")
check(doors_from_endpoints([{"center": (10, 20)}]) == [(10.0, 20.0)],
      "...or American spelling")
check(doors_from_endpoints([{"polygon": [(0, 0), (10, 0), (10, 10), (0, 10)]}])
      == [(5.0, 5.0)], "...or a polygon")
check(doors_from_endpoints([(7, 8)]) == [(7.0, 8.0)], "...or a bare pair")
check(doors_from_endpoints(None) == [], "...or nothing at all")

print()
print("=" * 74)
if FAILED:
    print(f"  {len(FAILED)} FAILED:")
    for f in FAILED:
        print(f"    - {f}")
    sys.exit(1)
print("  ALL PASS")
print("=" * 74)
