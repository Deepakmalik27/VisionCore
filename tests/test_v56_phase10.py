"""Tests for patch_v56_phase10.py — phantom filter + region arrivals, wired.

The question every check here asks is the same one: would this have changed
what the first real CAM.112 hour reported? That run printed "0 people came
through the door" as a fact while 46 people waited, and drew a box on a potted
plant labelled "staff" in every frame.

Run: python test_v56_phase10.py
"""
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kevacv.arrivals import arrivals_from_regions, cross_check
from kevacv.phantoms import in_phantom, phantom_regions

HERE = Path(__file__).resolve().parent.parent
NB = HERE / "notebooks" / "pipeline.ipynb"
MARKER = "PHASE10_PHANTOM_AND_REGION_ARRIVALS"

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


nb = json.loads(NB.read_text(encoding="utf-8"))
cells = nb["cells"]
code = [("".join(c["source"]), i) for i, c in enumerate(cells)
        if c["cell_type"] == "code"]
whole = "\n".join(s for s, _ in code)
eng = next(s for s, _ in code if "def process_video" in s)
met = next(s for s, _ in code if "def reception_report" in s)
boot = next(s for s, _ in code if "KEVACV_BOOTSTRAP" in s)

print("=" * 74)
print("  the patch landed, and in the right places")
print("=" * 74)
check(MARKER in whole, "phase-10 marker present")
check("D3 dropped" in eng, "D3 phantom filter is in the ENGINE cell")
check("phantom_regions(" in eng, "and it actually calls phantom_regions")
check(eng.index("D2 dropped") < eng.index("D3 dropped"),
      "D3 runs AFTER D2", "D2 is per-id, D3 is per-location; both are wanted")
check(eng.index("D3 dropped") < eng.index("ghost line-crossings"),
      "and BEFORE the ghost-crossing filter",
      "dropping a phantom's track must remove its crossings too")
for label, name in (("kevacv.phantoms", "from kevacv.phantoms import"),
                    ("describe_arrivals", "describe as describe_arrivals")):
    check(name in boot, f"bootstrap imports {label}")
for knob in ("ENABLE_PHANTOM_FILTER", "PHANTOM_MIN_SPAN_S",
             "PHANTOM_CENTRE_JITTER", "PHANTOM_SIZE_CV"):
    check(knob in whole, f"config knob {knob} exists")

print()
print("=" * 74)
print("  every touched cell still parses")
print("=" * 74)
for src, idx in code:
    if src.strip().startswith(("!", "%")):
        continue
    try:
        ast.parse(src)
        ok = True
    except SyntaxError as e:
        ok = False
        print(f"      cell {idx}: {e}")
    if not ok:
        check(False, f"cell {idx} parses")
check(True, f"all {len(code)} code cells parse")

print()
print("=" * 74)
print("  D1 can no longer report a number without its denominator")
print("=" * 74)
check('stats["size_seen"]' in eng, "D1 counts the detections it SAW")
check('"size_seen"' in eng and "_d1pct" in eng, "and prints a percentage")
check("more than a quarter of every" in eng,
      "and says so loudly past 25%",
      "29,124 dropped meant nothing without the total")
check('"size_seen": _supp_stats.get("size_seen", 0),' in eng,
      "the denominator travels with the run")

print()
print("=" * 74)
print("  THE HEADLINE FIX: a broken line must not publish 0 as a fact")
print("=" * 74)
check("arrivals_from_regions(" in met, "reception_report computes region arrivals")
check("cross_check(" in met, "and cross-checks them against the line")
check('_xc["trust"] == "region"' in met,
      "and prefers the region count when the line is judged broken")
check("run.get(\"zone_roles\")" in met, "using the zone roles carried on the run")
check('"zone_roles": dict(zone_roles),' in eng, "which the engine now provides")

# ── the actual CAM.112 shape, through the real functions ───────────────────
ROLES = {"main_entrance": ["entry"], "dining_entrance": ["entry"],
         "waiting_area": ["wait"], "reception": ["staff"]}
events = []
for i in range(23):
    events += [{"track_id": i, "zone": "main_entrance", "t_in": i * 120,
                "t_out": i * 120 + 8, "duration": 8, "role": "customer"},
               {"track_id": i, "zone": "waiting_area", "t_in": i * 120 + 9,
                "t_out": i * 120 + 60, "duration": 51, "role": "customer"}]
n, arr, why = arrivals_from_regions(events, ROLES)
xc = cross_check(0, n, movers=len({e["track_id"] for e in events}))
check(n == 23, "region arrivals still counts 23 with a dead line", f"{n}")
check(xc["trust"] == "region", "cross_check says to trust the region count")
per_person_ins = {}
if xc["trust"] == "region" and n:                 # the exact patched branch
    for a in arr:
        per_person_ins.setdefault(a["track_id"], []).append(a["t"])
check(len(per_person_ins) == 23,
      "so guests_tonight becomes 23, not 0",
      "this is the number the first real run should have printed")

print()
print("=" * 74)
print("  and the phantoms that made the video unwatchable")
print("=" * 74)
fl = []
for k in range(2000):                              # ~4.4 min at 7.5 fps
    t = k / 7.5
    fl.append((k, t, [
        (f"plant_{int(t // 4)}", 690, 350, 880, 800),      # id churns
        ("guest", 200 + 9 * t, 300, 280 + 9 * t, 700),     # walks past
    ]))
regs = phantom_regions(fl, frame_wh=(1280, 808))
check(len(regs) == 1, "the plant is found despite a fresh id every 4s", f"{len(regs)}")
check(in_phantom((690, 350, 880, 800), regs), "its box would be dropped")
check(not in_phantom((200, 300, 280, 700), regs), "the walking guest is kept")
# Anchor on the block's own comment and read to its end. Splitting on the bare
# string "D3" broke the moment phase 11 mentioned D3 in the renderer, which
# sits EARLIER in the same cell — a locator that silently points somewhere else
# is worse than no test, because it keeps passing.
_d3 = eng[eng.index("D3: static phantoms"):]
_d3 = _d3[:_d3.index("event_ids = {e")]
for _needle, _what in (
        ("protected_ids(crossings=crossings", "D3 computes the protected ids"),
        ("protected=_pprot", "and passes them into phantom_regions"),
        ("if _cid in _pprot", "and never drops a protected id's track")):
    check(_needle in _d3, _what, "a verified human is never furniture")

print()
print("=" * 74)
print("  idempotency — patches get re-run, that must be safe")
print("=" * 74)
check(eng.count("D3 dropped") == 1, "D3 block appears exactly once")
check(whole.count("ENABLE_PHANTOM_FILTER = True") == 1,
      "the config knob is defined exactly once")
check(met.count("arrivals_from_regions(") == 2,
      "region arrivals wired in the two intended spots, not duplicated",
      f"{met.count('arrivals_from_regions(')}")

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
