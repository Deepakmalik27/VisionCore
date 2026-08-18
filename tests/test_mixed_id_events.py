"""test_mixed_id_events.py — a matched staff face must not kill the run.

track_id is an INT for a tracker fragment and a STRING for a gallery-named
staff member ("staff2"). OccupancyRecorder.events() sorted on a bare tuple
containing it, so the first run where a face actually matched died with

    TypeError: '<' not supported between instances of 'int' and 'str'

1h58m into a 2-hour analysis, at the last step before the report. Every 20s
smoke test passed, because the gallery never matched in 20 seconds -- the
failure needed SUCCESS elsewhere to trigger.

track_sort_key() already existed in the same file for exactly this, and is used
2000 lines further down. derive.py carries the lesson in a comment. This was
the third occurrence of one bug, so it gets a test.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kevacv.analytics import OccupancyRecorder, track_sort_key

_fail = False


def check(ok, what, detail=""):
    global _fail
    print(f"  {'PASS' if ok else 'FAIL'}  {what}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        _fail = True


print("=" * 74)
print("  int and str track ids coexist in one run")
print("=" * 74)

r = OccupancyRecorder(frame_step_s=1.0, min_event_s=0.5)
for t in (0.0, 1.0, 2.0, 3.0):
    r.add(t, "reception", 7)          # tracker fragment
    r.add(t, "reception", "staff2")   # gallery-named staff — the trigger
    r.add(t, "waiting area", 12)
try:
    ev = r.events()
    ok = True
except TypeError as e:
    ev, ok = [], False
    print("   ", e)
check(ok, "events() does not raise on mixed id types",
      "this is the exact crash that lost a 2-hour run")
check(len(ev) == 3, "all three events survive", f"{len(ev)} events")
check(any(e["track_id"] == "staff2" for e in ev),
      "the named staff member is present, not silently dropped")

# Deterministic order matters: events feed the report and the ledger, and an
# unstable sort makes two identical runs diff against each other.
o1 = [(e["t_in"], e["zone"], str(e["track_id"])) for e in r.events()]
o2 = [(e["t_in"], e["zone"], str(e["track_id"])) for e in r.events()]
check(o1 == o2, "the order is deterministic across calls")

print()
print("=" * 74)
print("  track_sort_key handles every id shape the pipeline produces")
print("=" * 74)
ids = [7, "staff2", 12, "receptionist_sarah", "31", 2]
try:
    s = sorted(ids, key=track_sort_key)
    check(True, "mixed ints, digit-strings and names all sort", str(s))
except TypeError as e:
    check(False, "mixed ids sort", str(e))
check(sorted([3, 1, 2], key=track_sort_key) == [1, 2, 3],
      "plain ints keep numeric order — not lexicographic",
      "'10' < '9' as strings would scramble a 45-person night")

print()
print("=" * 74)
print("  ALL PASS" if not _fail else "  FAILURES ABOVE")
print("=" * 74)
sys.exit(1 if _fail else 0)
