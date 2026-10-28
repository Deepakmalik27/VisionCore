"""test_answers.py — a value over a denominator, both carrying their validity.

Two shipped failures drive this file:
  guests_tonight     came from a region fallback and printed as EXACT*
  desk_covered_pct   appeared as 56.4 in one section and 68.9 in another

Run: python tests/test_answers.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kevacv.answers import (ESTIMATE, EXACT, PROXY, UNKNOWN, WEAK,  # noqa: E402
                            answer_set, desk_coverage, desk_gaps, greet_latency,
                            guest_count, to_report_rows)

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


def ev(tid, zone, t_in, dur, role=None):
    return {"track_id": tid, "zone": zone, "t_in": float(t_in),
            "t_out": float(t_in + dur), "duration": float(dur), "role": role}


STAFF = ("reception",)
WAIT = ("waiting_area",)

print("=" * 74)
print("  THE DENOMINATOR IS OBSERVED TIME, NEVER ELAPSED TIME")
print("=" * 74)
# staff present 0-1800 of a 3600s hour
E = [ev("s1", "reception", 0, 1800, "staff")]
full = desk_coverage(E, STAFF, [(0.0, 3600.0)], roles={"s1": "staff"})
check(full.value == 50.0, "half an observed hour covered -> 50%", str(full.value))

# the camera was blind for the second half. That time is NOT uncovered.
blind = desk_coverage(E, STAFF, [(0.0, 1800.0)], roles={"s1": "staff"})
check(blind.value == 100.0,
      "the same footage, with the blind half excluded -> 100%",
      "a dead camera must not read as an uncovered desk")
check(blind.denominator == "observed footage", "and the denominator says so")

none = desk_coverage(E, STAFF, [], roles={"s1": "staff"})
check(none.tier == UNKNOWN and none.value is None,
      "no observed footage -> UNKNOWN, never 0%",
      "0% would claim the desk was abandoned all night")
check("denominator would be zero" in none.caveats[-1], "and says why")

print()
print("=" * 74)
print("  Q1 is EXACT because it needs no identity")
print("=" * 74)
two = [ev("s1", "reception", 0, 900, "staff"), ev("s2", "reception", 900, 900, "staff")]
a = desk_coverage(two, STAFF, [(0.0, 1800.0)], roles={"s1": "staff", "s2": "staff"})
check(a.value == 100.0 and a.tier == EXACT,
      "two different staff covering back to back is still 100% covered",
      "the STATION is the question, not who was standing at it")
check(not a.needs_identity, "and it is marked as not needing identity")
check(any("STATION" in c for c in a.caveats), "the caveat states that explicitly")
check(a.detail["meets_target"] is True, "and it is scored against the 90% bar")
check(any("below the ratified" in c for c in
          desk_coverage([], STAFF, [(0.0, 3600.0)]).caveats),
      "an empty desk is scored as FAILING the bar, not omitted")

print()
print("=" * 74)
print("  gaps say WHEN, and whether anyone was waiting through them")
print("=" * 74)
E2 = [ev("s1", "reception", 0, 600, "staff"),
      ev("s1", "reception", 1800, 1800, "staff"),
      ev("g1", "waiting_area", 700, 500)]
g = desk_gaps(E2, STAFF, [(0.0, 3600.0)], roles={"s1": "staff"},
              waiting_zones=WAIT)
check(g.value == 1, "one gap found", str(g.value))
check(g.detail["gaps"][0]["minutes"] == 20.0, "of the right length",
      str(g.detail["gaps"][0]["minutes"]))
check(g.detail["gaps"][0]["guests_waiting"] == 1,
      "and it knows a guest waited through it")
check(any("guests waiting" in c for c in g.caveats), "which is called out")
check(desk_gaps(E2, STAFF, [], roles={"s1": "staff"}).tier == UNKNOWN,
      "no observed footage -> UNKNOWN")

print()
print("=" * 74)
print("  Q2 is PROXY permanently, and refuses to invent a number")
print("=" * 74)
gl = greet_latency({"g1": 100.0, "g2": 200.0}, {"g1": [130.0], "g2": [260.0]})
check(gl.tier == PROXY, "greet latency is PROXY", gl.tier)
check(gl.value in (30.0, 45.0, 60.0), "median latency computed", str(gl.value))
check(any("not conversation" in c for c in gl.caveats),
      "and says proximity is not conversation")
check(gl.detail["n_ungreeted"] == 0, "everyone was reached")

nogreet = greet_latency({"g1": 100.0}, {})
check(nogreet.tier == UNKNOWN, "nobody near anyone -> UNKNOWN, not 0 s")
check(any("check the staff zone" in c for c in nogreet.caveats),
      "and points at the likely cause before blaming service")

noarr = greet_latency({}, {"g1": [130.0]})
check(noarr.tier == UNKNOWN, "no arrival times -> UNKNOWN")
check(any("broken entry line" in c for c in noarr.caveats),
      "and names the shipped failure that causes it")

print()
print("=" * 74)
print("  Q3 carries a RANGE, because identity had to hold")
print("=" * 74)
gc = guest_count(["a", "b", "c", "d"], {"a": 90, "b": 80, "c": 30, "d": 20})
check(gc.tier == ESTIMATE, "guest count is ESTIMATE, never EXACT", gc.tier)
check(gc.needs_identity, "and is marked as identity-dependent")
check((gc.low, gc.high) == (2, 4), "the range splits on per-person confidence",
      f"{gc.low}-{gc.high}")
check("range 2-4" in gc.display, "and the display shows it", gc.display)

noconf = guest_count(["a", "b"])
check(noconf.low == noconf.high == 2, "no confidence data -> a point range")
check(any("not as zero" in c for c in noconf.caveats),
      "which is flagged as unknown width, not as certainty")

empty = guest_count([])
check(empty.tier == UNKNOWN and empty.value is None,
      "zero arrivals -> UNKNOWN, never the integer 0")
check(any("cannot tell you which" in c for c in empty.caveats),
      "because a broken sensor and an empty venue look identical",
      "this is exactly what shipped as '2 people came through the door'")

print()
print("=" * 74)
print("  a run-level ERROR taints the identity answers, not the EXACT ones")
print("=" * 74)
S = answer_set(events=two, staff_zones=STAFF, waiting_zones=WAIT,
               observed_windows=[(0.0, 1800.0)],
               roles={"s1": "staff", "s2": "staff"},
               arrivals={"g1": 10.0}, contacts={"g1": [40.0]},
               unique_ids=["g1"], confidence={"g1": 90},
               findings=[("ERROR", "ENTRY ZONE MISPLACED")])
by = {a.key: a for a in S}
check(any("ENTRY ZONE MISPLACED" in c for c in by["guests"].caveats),
      "the guest count inherits the run-level blocker")
check(not any("ENTRY ZONE MISPLACED" in c for c in by["desk_covered_pct"].caveats),
      "but desk coverage does NOT — it never depended on the entry zone",
      "tainting everything equally would hide which numbers survive")
check([a.key for a in S][0] == "desk_covered_pct",
      "answers are ordered by what the evidence supports, Q1 first")

print()
print("=" * 74)
print("  the report rows carry the tier with the value")
print("=" * 74)
rows = to_report_rows(S)
check(all("tier" in r and "value" in r for r in rows), "every row has both")
check(any(r["tier"] == EXACT for r in rows), "EXACT appears")
check(any(r["tier"] in (ESTIMATE, UNKNOWN) for r in rows),
      "and so does the uncertain tier — no row is silently promoted")
check(len(rows) == len(S), "one row per answer")


print()
print("=" * 74)
print("  DERIVE: an empty room is OBSERVED, it is just empty")
print("=" * 74)
from kevacv.derive import (arrivals_by_id, enrich, guest_ids,  # noqa: E402
                           id_confidence, observed_windows, staff_contacts)

# frame_log holds only frames WITH detections. Deriving observed time from it
# removed every empty stretch from the denominator and turned a true 66.7%
# desk coverage into 79.7% — inflating the one metric that must be EXACT.
sparse = [(i, float(i), [("s1", 600, 300, 660, 480)])
          for i in list(range(0, 1200, 2)) + list(range(2400, 3600, 2))]
run = {"frame_log": sparse, "duration_s": 3600.0, "t_end": 3600.0}
w = observed_windows(run)
check(w == [(0.0, 3600.0)],
      "the ANALYSED SPAN is the denominator, not the frames that had people",
      str(w))
check(observed_windows({"observed_windows": [(0.0, 100.0)], "t_end": 3600.0})
      == [(0.0, 100.0)],
      "a real validity ledger still wins — that IS observation evidence")
check(observed_windows({}) == [],
      "no duration at all -> empty, so percentages report UNKNOWN not a guess")

E3 = [ev("s1", "reception", 0, 1200, "staff"),
      ev("s1", "reception", 2400, 1200, "staff")]
cov = desk_coverage(E3, STAFF, w, roles={"s1": "staff"})
check(cov.value == 66.7, "and desk coverage is the TRUE 66.7%, not 79.7%",
      str(cov.value))

print()
print("=" * 74)
print("  DERIVE: a broken door propagates as UNKNOWN, never as zero")
print("=" * 74)
noline = {"crossings": [], "roles": {}, "events": [ev("g1", "waiting_area", 10, 60)]}
check(arrivals_by_id(noline) == {},
      "no inward crossings -> no arrival times, not invented ones")
check(guest_ids(noline) == ["g1"],
      "the guest list falls back to interior sightings")
check(greet_latency(arrivals_by_id(noline), {}).tier == UNKNOWN,
      "so greet latency reports UNKNOWN — a broken line is not instant service")

withline = {"crossings": [{"track_id": "g1", "direction": "in", "t": 100.0},
                          {"track_id": "g1", "direction": "in", "t": 300.0},
                          {"track_id": "s1", "direction": "in", "t": 50.0}],
            "roles": {"s1": "staff"}, "events": []}
a = arrivals_by_id(withline)
check(a == {"g1": 100.0}, "the FIRST inward crossing wins, and staff excluded",
      str(a))

print()
print("=" * 74)
print("  DERIVE: contacts are proximity in BODY HEIGHTS, not pixels")
print("=" * 74)
fl = [(i, float(i), [("s1", 300, 320, 355, 470), ("g1", 320, 320, 375, 470)])
      for i in range(0, 20)]
c = staff_contacts({"frame_log": fl, "roles": {"s1": "staff"}})
check("g1" in c, "a staff member standing beside a guest is a contact")
far = [(i, float(i), [("s1", 20, 320, 75, 470), ("g1", 900, 320, 955, 470)])
       for i in range(0, 20)]
check(staff_contacts({"frame_log": far, "roles": {"s1": "staff"}}) == {},
      "across the room is not a contact")
brief = [(i, float(i), [("s1", 300, 320, 355, 470), ("g1", 320, 320, 375, 470)])
         for i in range(0, 2)]
check(staff_contacts({"frame_log": brief, "roles": {"s1": "staff"}}) == {},
      "a 1-second pass-by is not a greeting", "min_contact_s guards it")

print()
print("=" * 74)
print("  DERIVE: confidence orders identities, it is not a probability")
print("=" * 74)
run2 = {"events": [ev("solid", "waiting_area", 0, 300),
                   ev("flimsy", "waiting_area", 0, 5)],
        "crossings": [{"track_id": "solid", "direction": "in", "t": 0.0}],
        "canon_map": {"a": "flimsy", "b": "flimsy", "c": "flimsy", "d": "flimsy"},
        "roles": {}}
conf = id_confidence(run2, ["solid", "flimsy"])
check(conf["solid"] > conf["flimsy"],
      "a long-lived identity that crossed the door beats a stitched fragment",
      f"solid={conf['solid']} flimsy={conf['flimsy']}")
check(all(0 <= v <= 100 for v in conf.values()), "and stays in range")

full = enrich(dict(run))
for k in ("observed_windows", "arrivals_by_id", "guest_ids", "contacts",
          "id_confidence"):
    check(k in full, f"enrich() supplies {k}")

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
