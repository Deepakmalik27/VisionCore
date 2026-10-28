"""test_crossing_state.py — one person cannot enter twice without leaving.

WHY THIS EXISTS
    The 20-minute CAM.112 run reported IN 61 / OUT 66 -- more unique exits than
    entries, which a closed room cannot produce. Measured cause: 127 crossing
    events from 15 distinct people (8.5 each), because the counting lines sit
    INSIDE the room, so a receptionist pacing generated an "arrival" per pass.

    Tier-A dedupe cannot help: it collapses one person crossing ONCE under
    several ids within 2.5 s / 140 px. Crossings minutes apart are separate
    events; the question is what they MEAN.

WHAT THIS GUARDS
    Mostly the CONSERVATIVE half. A rule that suppresses a real arrival is
    worse than the double-count it replaces, so the cases that must survive
    are tested harder than the cases that are suppressed.

Run: python tests/test_crossing_state.py
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


from kevacv.analytics import enforce_crossing_state as E  # noqa: E402


def C(t, tid, d, line="entry line"):
    return {"t": t, "track_id": tid, "direction": d, "line": line}


print()
print("=" * 74)
print("  the pacing case it was built for")
print("=" * 74)
pacing = [C(1, "A", "in"), C(5, "A", "out"), C(9, "A", "in"),
          C(13, "A", "out"), C(17, "A", "in")]
kept, d = E(pacing)
check(d["in"] == 3 and d["out"] == 2,
      "alternating crossings all survive — they are real transitions",
      f"in {d['in']} out {d['out']}")
repeat = [C(1, "A", "in"), C(3, "A", "in"), C(5, "A", "in"), C(7, "A", "in")]
kept, d = E(repeat)
check(d["in"] == 1 and d["suppressed"] == 3,
      "four INs in a row count as ONE arrival",
      "a person already inside cannot arrive again")

print()
print("=" * 74)
print("  conservation — the identity that was violated")
print("=" * 74)
kept, d = E([C(1, "A", "in"), C(2, "B", "in"), C(3, "A", "out"),
             C(4, "B", "out"), C(5, "C", "out")])
check(d["balances"],
      "exits never exceed entries + initial occupants",
      f"in {d['in']} out {d['out']} initial {d['initial_occupants']}")
check(d["initial_occupants"] == 1,
      "a person first seen LEAVING is an initial occupant, not a phantom exit",
      "C left without ever entering -> was already here when the clip opened")
check(any(c.get("initial_occupant") for c in kept),
      "initial occupants are TAGGED, not silently dropped",
      "deleting the exit would undercount departures")

print()
print("=" * 74)
print("  it must not invent or destroy people")
print("=" * 74)
kept, d = E([])
check(kept == [] and d["in"] == 0, "empty input -> empty output")
odd = [{"t": 1, "track_id": None, "direction": "in"},
       {"t": 2, "track_id": "A", "direction": None}]
kept, d = E(odd)
check(len(kept) == 2,
      "rows it cannot reason about are PASSED THROUGH, never dropped",
      "a missing id or direction is not evidence the crossing did not happen")
two = [C(1, "A", "in"), C(1, "B", "in")]
kept, d = E(two)
check(d["in"] == 2,
      "two DIFFERENT people entering at the same instant both count",
      "state is per-person; a group arriving together is not one arrival")

print()
print("=" * 74)
print("  ordering")
print("=" * 74)
shuffled = [C(9, "A", "in"), C(1, "A", "in"), C(5, "A", "out")]
kept, d = E(shuffled)
check(d["in"] == 2 and d["out"] == 1,
      "input order does not change the answer — it sorts by time first",
      "crossings arrive grouped by line, not by clock")

print()
print("=" * 74)


def test_script_level_checks_passed():
    assert not FAILED, f"crossing-state checks failed: {FAILED}"


if __name__ == "__main__":
    if FAILED:
        print(f"  {len(FAILED)} FAILED:")
        for x in FAILED:
            print(f"    - {x}")
        sys.exit(1)
    print("  ALL PASS")
    print("=" * 74)
