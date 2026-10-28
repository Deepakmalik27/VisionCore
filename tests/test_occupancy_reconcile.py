"""test_occupancy_reconcile.py — a check that measures nothing must say so.

WHY THIS EXISTS
    reconcile_occupancy compares two independent estimates of one quantity:

        expected(t) = initial + IN(<=t) - OUT(<=t)      the doors
        observed(t) = distinct people on screen         the tracker

    The caller hardcoded line_name="entry line". That line was deleted on
    2026-08-27 (it had been drawn across a wall), after which the filter
    matched NOTHING. IN/OUT stayed 0/0, `expected` froze at `initial` for all
    40 steps, and the published "max drift 17" was the room's own variation
    compared against a constant -- not a disagreement between two estimates.

    A check that silently measures nothing is worse than no check, because it
    publishes a number that reads as a finding.

Run: python tests/test_occupancy_reconcile.py
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


from kevacv.analytics import (reconcile_occupancy as R,  # noqa: E402
                              describe_reconciliation as D)


def EV(tid, t0, t1, zone="reception", role="customer"):
    # occupancy_timeline groups by role, so an event without one raises.
    return {"track_id": tid, "zone": zone, "t_in": t0, "t_out": t1,
            "role": role}


def CR(t, tid, d, line):
    return {"t": t, "track_id": tid, "direction": d, "line": line}


EVENTS = [EV("A", 0, 100), EV("B", 10, 90)]

print()
print("=" * 76)
print("  the bug: asking for a door that produced no traffic")
print("=" * 76)
cr = [CR(5, "A", "in", "dining entry"), CR(40, "B", "out", "staff entry")]
rec = R(EVENTS, cr, 100.0, step_s=25.0, line_name="entry line")
check(rec.get("n_crossings", 0) > 0,
      "it FALLS BACK to the doors that exist instead of matching nothing",
      f"used {rec.get('lines_used')}, {rec.get('n_crossings')} crossing(s)")
check(rec.get("measurable") is True,
      "and reports itself measurable once it has doors")

print()
print("=" * 76)
print("  and when there genuinely are NO crossings at all")
print("=" * 76)
rec0 = R(EVENTS, [], 100.0, step_s=25.0, line_name=None)
check(rec0.get("measurable") is False,
      "measurable=False when nothing crossed any door")
txt = D(rec0)
check("NOT MEASURABLE" in txt,
      "the human-readable output REFUSES to publish a drift number",
      "a frozen `expected` is not a disagreement")
check("max drift" not in txt.lower() or "NOT MEASURABLE" in txt,
      "it does not present the room's own variation as a finding")

print()
print("=" * 76)
print("  it still does its real job when doors ARE working")
print("=" * 76)
# spread across steps -- two crossings inside ONE step cannot make `expected`
# move between steps, which is correct behaviour and a fixture mistake I made
cr2 = [CR(5, "A", "in", "dining entry"), CR(60, "B", "in", "dining entry")]
rec2 = R(EVENTS, cr2, 100.0, step_s=25.0, line_name=None)
check(rec2["measurable"] and rec2["n_crossings"] == 2,
      "counts every door when asked for none in particular",
      f"lines_used={rec2['lines_used']}")
exp = [s["expected"] for s in rec2["steps"]]
check(len(set(exp)) > 1,
      "expected MOVES as people cross — the whole point of the check",
      f"expected over time: {exp}")
check("NOT MEASURABLE" not in D(rec2),
      "and the readable output prints the real comparison")

print()
print("=" * 76)
print("  the caller no longer hardcodes a door name")
print("=" * 76)
PIPE = (ROOT / "kevacv" / "pipeline.py").read_text(encoding="utf-8")
import re as _re
_calls = _re.findall(r'_rec\([^)]*line_name\s*=\s*([^,)]+)', PIPE)
check(_calls and all("None" in c for c in _calls),
      "pipeline.py passes line_name=None, not a hardcoded door",
      f"call sites: {_calls} — checking the CALL, not the comment that "
      f"records why the hardcoding was removed")

print()
print("=" * 76)


def test_script_level_checks_passed():
    assert not FAILED, f"reconciliation checks failed: {FAILED}"


if __name__ == "__main__":
    if FAILED:
        print(f"  {len(FAILED)} FAILED:")
        for x in FAILED:
            print(f"    - {x}")
        sys.exit(1)
    print("  ALL PASS")
    print("=" * 76)
