"""Tests for kevacv/arrivals.py — an arrival count that survives a bad line.

Built from the real CAM.112 failure: the entry line fired ZERO times over a
full hour while 95 people moved through the zones. Every test below asks the
question that failure raised — would this have caught it, and would it have
given a usable number instead of a silent 0?

Run: python test_arrivals.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kevacv.arrivals import (arrivals_from_regions, cross_check, describe,
                             entry_zone_coverage)

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


ROLES = {"main_entrance": ["entry"], "dining_entrance": ["entry"],
         "waiting_area": ["wait"], "reception": ["staff"]}


def ev(tid, zone, t_in, dur=20.0):
    return {"track_id": tid, "zone": zone, "t_in": t_in, "t_out": t_in + dur,
            "duration": dur, "role": "customer"}


print("=" * 74)
print("  the basic claim: an arrival is entry-zone THEN interior")
print("=" * 74)
events = [ev(1, "main_entrance", 100), ev(1, "waiting_area", 108),
          ev(2, "main_entrance", 300), ev(2, "reception", 315),
          ev(3, "waiting_area", 500)]                      # never in an entry zone
n, arr, why = arrivals_from_regions(events, ROLES)
check(n == 2, "two people entered and moved inside -> 2", f"got {n}")
check(why == "", "no complaint when the zones support the question")
check({a["track_id"] for a in arr} == {1, 2}, "the right two")
check(3 not in {a["track_id"] for a in arr},
      "someone who only ever appears inside is NOT an arrival",
      "they were already there when the chunk started")

# stood in the doorway, turned round, left
n, _, _ = arrivals_from_regions([ev(9, "main_entrance", 50)], ROLES)
check(n == 0, "loitering in the doorway without coming in is not an arrival")

# staff are excluded
n, _, _ = arrivals_from_regions(
    [ev("sarah", "main_entrance", 10), ev("sarah", "reception", 20)],
    ROLES, roles={"sarah": "staff"})
check(n == 0, "staff walking in are not guests")

# both doors work
n, _, _ = arrivals_from_regions(
    [ev(4, "dining_entrance", 10), ev(4, "waiting_area", 20),
     ev(5, "main_entrance", 30), ev(5, "waiting_area", 40)], ROLES)
check(n == 2, "arrivals through EITHER entrance are counted", f"{n}")

print()
print("=" * 74)
print("  it must say 'cannot tell' rather than 0")
print("=" * 74)
n, _, why = arrivals_from_regions(events, {"waiting_area": ["wait"]})
check(n is None, "no entry zone -> None, NOT zero")
check("ENTRY role" in why, "and it says exactly what is missing", why[:52])
n, _, why = arrivals_from_regions(events, {"main_entrance": ["entry"]})
check(n is None, "no interior zone -> None, NOT zero")
check("arrive INTO" in why, "and says why", why[:52])
print("    -> 0 and 'cannot tell' must never look the same in a report")

print()
print("=" * 74)
print("  the CAM.112 failure: line says 0, people are everywhere")
print("=" * 74)
real = []
for i in range(23):                       # 23 guests walk in over the hour
    real += [ev(i, "main_entrance", i * 120), ev(i, "waiting_area", i * 120 + 9)]
n, _, _ = arrivals_from_regions(real, ROLES)
cc = cross_check(line_count=0, region_count=n, movers=95)
print("   " + describe(0, n, cc).replace("\n", "\n   "))
check(n == 23, "the region method still produces a real number", f"{n}")
check(cc["verdict"] == "LINE IS BROKEN", "the disagreement is named correctly")
check(cc["trust"] == "region", "and it says which number to use meanwhile")
check("redraw it wall to wall" in cc["detail"], "and how to fix the line")
print("    -> this is the number the last run SHOULD have reported instead of 0")

print()
print("=" * 74)
print("  every other disagreement shape")
print("=" * 74)
for line, region, movers, want in [
        (23, 22, 95, "the two methods AGREE"),
        (0, 0, 95, "BOTH ZERO but people were present"),
        (20, 0, 95, "entry zone is misplaced"),
        (5, 40, 95, "the two methods DISAGREE"),
        (0, 0, 0, "the two methods AGREE")]:
    cc = cross_check(line, region, movers)
    print(f"    line {line:>3}  region {region:>3}  movers {movers:>3}  ->  {cc['verdict']}")
    check(cc["verdict"] == want, f"line={line} region={region} -> {want}")
cc = cross_check(23, 22, 95)
check(cc["agree"] is True, "agreement within tolerance is reported as agreement")
check("strongest evidence" in cc["detail"],
      "and agreement is stated as EVIDENCE, not just a tick",
      "two independent signals agreeing is the point")
cc = cross_check(10, None, 95)
check(cc["trust"] == "line" and cc["agree"] is None,
      "no region count -> falls back to the line, honestly")

print()
print("=" * 74)
print("  the SECOND CAM.112 failure: the entry ZONE is misplaced too")
print("=" * 74)
# 31 people moved. Only 2 were ever seen in main_entrance. The region method
# returns 2 and the old cross_check called that a usable number.
mixed = []
for i in range(2):                        # two people really did use the door
    mixed += [ev(f"in{i}", "main_entrance", i * 60),
              ev(f"in{i}", "waiting_area", i * 60 + 9)]
for i in range(20):                       # twenty appear already inside
    mixed += [ev(f"mid{i}", "waiting_area", 300 + i * 30)]
n, _, _ = arrivals_from_regions(mixed, ROLES)
cov = entry_zone_coverage(mixed, ROLES)
check(n == 2, "region method still returns its small number", f"{n}")
check(cov["non_staff"] == 22, "coverage counts every non-staff person", str(cov["non_staff"]))
check(cov["with_entry"] == 2, "only two were ever in an entry zone")
check(round(cov["share_with_entry"], 2) == 0.09, "share is ~9%",
      f"{cov['share_with_entry']:.2f}")

cc_old = cross_check(line_count=0, region_count=n, movers=22)
check(cc_old["trust"] == "region",
      "WITHOUT coverage the old behaviour is unchanged (backwards compatible)")

cc = cross_check(line_count=0, region_count=n, movers=22, coverage=cov)
print("   " + describe(0, n, cc).replace("\n", "\n   "))
check(cc["trust"] == "neither", "WITH coverage it refuses to trust the region count")
check("ENTRY ZONE IS MISPLACED" in cc["verdict"], "and names the second fault")
check(cc["agree"] is False, "two broken sensors are not agreement")
print("    -> the last run published '2 people came through the door' from this")

# a genuinely healthy venue must NOT trip the new guard
healthy = []
for i in range(23):
    healthy += [ev(i, "main_entrance", i * 120), ev(i, "waiting_area", i * 120 + 9)]
cov_ok = entry_zone_coverage(healthy, ROLES)
cc_ok = cross_check(0, 23, movers=95, coverage=cov_ok)
check(cov_ok["share_with_entry"] == 1.0, "everyone came through the door -> 100%")
check(cc_ok["verdict"] == "LINE IS BROKEN",
      "a good entry zone still reports the line honestly", cc_ok["verdict"])
check(cc_ok["trust"] == "region", "and still hands over the usable number")

# staff are excluded from coverage, same as from the count
cov_staff = entry_zone_coverage(
    [ev("sarah", "reception", 10), ev(1, "main_entrance", 20),
     ev(1, "waiting_area", 30)], ROLES, roles={"sarah": "staff"})
check(cov_staff["non_staff"] == 1, "staff never count against entry coverage",
      str(cov_staff["non_staff"]))
check(entry_zone_coverage(mixed, {"waiting_area": ["wait"]}) is None,
      "no entry zone -> None, not a fake 0%")
# too few people to judge: don't cry misplaced-zone on a quiet chunk
cov_tiny = entry_zone_coverage([ev(1, "waiting_area", 10)], ROLES)
check(cross_check(0, 1, movers=2, coverage=cov_tiny)["trust"] != "neither",
      "under 5 non-staff people the guard stays quiet (too little evidence)")

print()
print("=" * 74)
print("  de-duplication, same rule as Tier A")
print("=" * 74)
frag = []
for tid in (81, 82, 83):                  # one person, three fragmented ids
    frag += [ev(tid, "main_entrance", 500), ev(tid, "waiting_area", 505)]
pos = {81: (100, 700), 82: (104, 702), 83: (98, 699)}
n, _, _ = arrivals_from_regions(frag, ROLES, positions=pos)
check(n == 1, "one person with three ids at the same place counts ONCE", f"{n}")
far = [ev(91, "main_entrance", 500), ev(91, "waiting_area", 505),
       ev(92, "main_entrance", 501), ev(92, "waiting_area", 506)]
n, _, _ = arrivals_from_regions(far, ROLES, positions={91: (100, 700), 92: (900, 700)})
check(n == 2, "two people far apart at the same moment count TWICE", f"{n}")
n, _, _ = arrivals_from_regions(frag, ROLES)      # no positions at all
check(n == 3, "with no positions it degrades to per-id (never silently merges)", f"{n}")

print()
print("=" * 74)
print("  edge cases")
print("=" * 74)
check(arrivals_from_regions([], ROLES)[0] == 0, "no events -> 0 arrivals, no crash")
check(arrivals_from_regions([], {})[0] is None, "no zones -> None")
n, _, _ = arrivals_from_regions(
    [ev(1, "waiting_area", 10), ev(1, "main_entrance", 200), ev(1, "reception", 210)],
    ROLES)
check(n == 1, "someone already inside who later re-enters is counted once", f"{n}")

print()
print("=" * 74)
if FAILED:
    print(f"  {len(FAILED)} FAILED:")
    for f in FAILED:
        print(f"    - {f}")
    sys.exit(1)
print("  ALL PASS")
print("=" * 74)
