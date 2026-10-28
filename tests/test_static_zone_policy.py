"""test_static_zone_policy.py — phantom patience belongs to the zone, not the run.

WHY THIS EXISTS
    The static filter asks "has this box sat still long enough to be
    furniture?" against ONE global threshold, 120 s. That threshold trades two
    opposite errors, and the right trade differs across the same frame:

      * At the reception desk and in the waiting area people legitimately hold
        position for minutes. Too short a bar DELETES THEM.
      * In the corridors and the doorway nothing human holds position at all.
        Too long a bar lets a plant mint ids and pollute zone events for two
        full minutes before anything suppresses it.

    One number cannot satisfy both, so it was set to satisfy the desk and
    everywhere else paid. Operator confirmed the split on 2026-08-12; these
    tests pin it.

Run: python tests/test_static_zone_policy.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kevacv.detect_filters import (_point_in_poly, static_min_life_by_id,
                                   static_track_ids)

FAILED = []

DOOR = [(0, 0), (100, 0), (100, 100), (0, 100)]
DESK = [(200, 0), (300, 0), (300, 100), (200, 100)]
POLYS = {"main_entrance": DOOR, "reception": DESK}
ZROLES = {"main_entrance": ["entry"], "reception": ["staff"]}
BY_ROLE = {"entry": 30.0, "staff": 240.0}


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


def rigid_log(seconds, ids_at):
    """A frame log where every listed id sits at EXACTLY the same pixels for
    `seconds` — the definition of furniture."""
    return [(i, float(i), [(tid, x - 10, 0, x + 10, 50) for tid, x in ids_at])
            for i in range(seconds)]


def test_point_in_poly():
    check(_point_in_poly(DOOR, 50, 50), "a point inside is inside")
    check(not _point_in_poly(DOOR, 150, 50), "a point outside is outside")
    check(not _point_in_poly(DOOR, -5, 50), "a point left of the polygon is outside")


def test_patience_comes_from_the_zone():
    log = rigid_log(300, [("door_plant", 50), ("desk_person", 250)])
    m = static_min_life_by_id(log, POLYS, ZROLES, default_s=120.0, by_role=BY_ROLE)
    check(m["door_plant"] == 30.0, "a track in the doorway gets the short bar",
          str(m.get("door_plant")))
    check(m["desk_person"] == 240.0, "a track at the desk gets the long bar",
          str(m.get("desk_person")))


def test_the_case_the_global_bar_got_wrong():
    """99 seconds: under the old global 120 s NEITHER was flagged, so the
    doorway plant survived. Now the doorway plant dies and the desk is safe."""
    log = rigid_log(100, [("door_plant", 50), ("desk_person", 250)])
    m = static_min_life_by_id(log, POLYS, ZROLES, default_s=120.0, by_role=BY_ROLE)
    flagged = set(static_track_ids(log, min_life_by_id=m))
    check("door_plant" in flagged, "the doorway phantom is caught at 99 s")
    check("desk_person" not in flagged,
          "the person standing at the desk is NOT caught", str(flagged))
    old = set(static_track_ids(log, min_life_s=120.0))
    check(old == set(), "and the old global bar caught neither — the regression",
          str(old))


def test_long_enough_and_both_are_furniture():
    log = rigid_log(300, [("door_plant", 50), ("desk_plant", 250)])
    m = static_min_life_by_id(log, POLYS, ZROLES, default_s=120.0, by_role=BY_ROLE)
    flagged = set(static_track_ids(log, min_life_by_id=m))
    check(flagged == {"door_plant", "desk_plant"},
          "past 240 s even a desk-side rigid box is furniture", str(flagged))


def test_protected_ids_still_win():
    """Someone seen walking through the door is a human, and no statistic gets
    to overrule that — including the shortened doorway bar."""
    log = rigid_log(300, [("door_plant", 50)])
    m = static_min_life_by_id(log, POLYS, ZROLES, default_s=120.0, by_role=BY_ROLE)
    flagged = static_track_ids(log, min_life_by_id=m, protected={"door_plant"})
    check(flagged == {}, "a protected id survives the shortest bar", str(flagged))


def test_most_conservative_wins_on_overlap():
    """An over-long wait leaves a phantom alive; an over-short one deletes a
    person. Only one of those is recoverable, so overlap must round UP."""
    overlap = {"a": DOOR, "b": [(0, 0), (100, 0), (100, 100), (0, 100)]}
    roles = {"a": ["entry"], "b": ["wait"]}
    log = rigid_log(50, [("t", 50)])
    m = static_min_life_by_id(log, overlap, roles, default_s=120.0,
                              by_role={"entry": 30.0, "wait": 240.0})
    check(m["t"] == 240.0, "the larger of two overlapping zone bars is used",
          str(m.get("t")))


def test_track_outside_every_zone_gets_the_default():
    log = rigid_log(50, [("nowhere", 900)])
    m = static_min_life_by_id(log, POLYS, ZROLES, default_s=120.0, by_role=BY_ROLE)
    check(m["nowhere"] == 120.0, "a track in no zone falls back to the global bar",
          str(m.get("nowhere")))


def test_no_policy_is_backwards_compatible():
    """Callers that pass nothing must behave exactly as before."""
    log = rigid_log(300, [("x", 50)])
    check(set(static_track_ids(log)) == {"x"}, "global bar still works alone")
    check(set(static_track_ids(log, min_life_by_id={})) == {"x"},
          "an empty policy dict is the same as none")


def main():
    print(__doc__.strip().splitlines()[0])
    for fn in (test_point_in_poly,
               test_patience_comes_from_the_zone,
               test_the_case_the_global_bar_got_wrong,
               test_long_enough_and_both_are_furniture,
               test_protected_ids_still_win,
               test_most_conservative_wins_on_overlap,
               test_track_outside_every_zone_gets_the_default,
               test_no_policy_is_backwards_compatible):
        print(f"\n{fn.__name__}")
        fn()
    print()
    if FAILED:
        print(f"FAILED ({len(FAILED)}): " + ", ".join(FAILED))
        return 1
    print("all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
