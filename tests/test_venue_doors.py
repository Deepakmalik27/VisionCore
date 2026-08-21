"""test_venue_doors.py — a dining door is not an arrival.

Every crossing has carried {"line": <door name>} since multi-door support was
added, and BOTH arrival paths ignored it. So a staff member stepping out of the
staff room and a guest walking INTO the dining room were each counted as
"a guest arrived" -- inflating the headline business question by every interior
movement in the building. The 20s smoke run on 2026-08-13 reported arrivals=1
sourced from the DINING line.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kevacv.analytics import entered_count, tier_a_crossings, venue_entry_lines
from kevacv.derive import line_arrivals_by_id

_fail = False


def check(ok, what, detail=""):
    global _fail
    print(f"  {'PASS' if ok else 'FAIL'}  {what}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        _fail = True


print("=" * 74)
print("  which doors mean 'arrived at the venue'")
print("=" * 74)
LIVE = ["dining entry", "entry line", "staff entry"]        # CAM.112's real names
check(venue_entry_lines(LIVE) == {"entry line"},
      "CAM.112: only 'entry line' is an arrival", sorted(venue_entry_lines(LIVE)))
check(venue_entry_lines(["main_entrance", "dining", "staff_door"]) == {"main_entrance"},
      "a differently-named venue works too — nothing is hardcoded")
# classify_zones makes entry WIN over seating for a compound name, which is
# right for a ZONE and wrong for a DOOR. Guard the distinction.
check("dining entry" not in venue_entry_lines(LIVE),
      "'dining entry' is NOT an arrival despite containing 'entry'",
      "classify_zones would call it entry-only; doors need both roles")
check(venue_entry_lines(["dining door"]) == {"dining door"},
      "a venue whose ONLY door is oddly named still counts",
      "an empty result would silently report zero guests forever")

print()
print("=" * 74)
print("  both arrival paths honour it")
print("=" * 74)
X = [{"t": 1, "track_id": 1, "direction": "in", "line": "dining entry", "pos": (10, 10)},
     {"t": 2, "track_id": 2, "direction": "in", "line": "staff entry", "pos": (20, 20)},
     {"t": 3, "track_id": 3, "direction": "in", "line": "entry line", "pos": (30, 30)}]
check(line_arrivals_by_id({"crossings": X}) == {3: 3.0},
      "line_arrivals_by_id counts ONLY the venue door", line_arrivals_by_id({"crossings": X}))
n, _ = tier_a_crossings(X)
check(n == 1, "tier_a_crossings counts ONLY the venue door", f"got {n}, was 3")
check(entered_count(X, lines=venue_entry_lines(LIVE)) == 1,
      "entered_count counts ONLY the venue door")
check(entered_count(X) == 3,
      "and defaults to the OLD behaviour when no lines are given",
      "so an existing caller cannot silently change meaning")

print()
print("=" * 74)
print("  runs recorded before per-door tracking still work")
print("=" * 74)
OLD = [{"t": 5, "track_id": 9, "direction": "in"}]          # no "line" key
check(line_arrivals_by_id({"crossings": OLD}) == {9: 5.0},
      "a crossing with no door name is COUNTED, not dropped",
      "dropping it would turn an old run's guest count into zero")

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
