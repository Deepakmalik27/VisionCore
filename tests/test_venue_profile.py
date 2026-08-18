"""Tests for venue_profile.py — config as data, DST, and inferred entry direction.

Success criteria, stated before the code was written:
  1. every value currently hardcoded in Cell 2 has a default WITH a unit
  2. a profile file overrides defaults; a malformed one degrades, never crashes
  3. a typo that would produce a plausible-but-wrong report is caught loudly
  4. DST: a night crossing the US autumn change keeps correct wall clocks
     (this is the case an abbreviation like "CDT" cannot express at all)
  5. entry direction is inferred correctly from BOTH orientations, and returns
     None (not False) when the evidence is too thin to say

Run: python test_venue_profile.py
"""
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kevacv.venue_profile import (DEFAULTS, describe, infer_entry_direction,
                           load_profile, local_clock, validate, write_template)

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


print("=" * 74)
print("  1. every hardcoded value is now data, with a unit")
print("=" * 74)
# The list that was measured out of the notebook in the universality audit.
MUST_COVER = ["timezone", "hfov_deg", "person_height_m", "entry_line_flip",
              "min_seated_s", "wait_threshold_s", "party_gap_s", "min_party_s",
              "visit_min_s", "staff_override_min_s", "staff_min_video_share",
              "staff_dominance_ratio", "greet_min_contact_s", "greet_proximity_m",
              "turnaway_max_s", "long_wait_s", "micro_absence_s",
              "break_absence_s", "group_window_s", "group_radius_m",
              "max_walk_speed_mps"]
have = set(DEFAULTS["camera"]) | set(DEFAULTS["venue"])
missing = [k for k in MUST_COVER if k not in have]
check(not missing, f"all {len(MUST_COVER)} audited values have defaults",
      f"missing {missing}" if missing else "")
src = Path(__file__).resolve().parent.parent.joinpath("kevacv","venue_profile.py").read_text()
undocumented = []
for k in have:
    line = next((l for l in src.splitlines() if l.strip().startswith(f'"{k}"')), "")
    if "#" not in line:
        undocumented.append(k)
check(not undocumented, "every default carries a unit/reason comment",
      f"undocumented: {undocumented}" if undocumented else "")

print()
print("=" * 74)
print("  2. overrides, precedence and graceful degradation")
print("=" * 74)
p = load_profile()
check(p["venue"]["min_seated_s"] == 60, "defaults load", p["_source"])

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    vid = td / "some_other_cafe.mp4"
    vid.write_bytes(b"x")
    (td / "profile_some_other_cafe.json").write_text(json.dumps(
        {"camera": {"timezone": "Asia/Kolkata"}, "venue": {"min_seated_s": 300}}))
    p = load_profile(video_path=vid)
    check(p["venue"]["min_seated_s"] == 300, "profile file overrides a default")
    check(p["camera"]["timezone"] == "Asia/Kolkata", "camera override applied")
    check(p["venue"]["wait_threshold_s"] == 600, "untouched keys keep their default")
    check(p["camera"]["id"] == "some_other_cafe", "camera id falls back to the stem")
    check("profile_some_other_cafe.json" in p["_source"], "source is reported")

    zp = td / "zones_x.json"
    zp.write_text(json.dumps({"polygons": {}, "profile": {"venue": {"visit_min_s": 22}}}))
    p2 = load_profile(video_path=td / "x.mp4", zones_path=zp)
    check(p2["venue"]["visit_min_s"] == 22, "profile embedded in the zones file works")

    bad = td / "b.mp4"
    bad.write_bytes(b"x")
    (td / "profile_b.json").write_text("{ this is not json ")
    p3 = load_profile(video_path=bad)
    check(p3["venue"]["min_seated_s"] == 60, "malformed profile -> defaults, no crash")
    check("MALFORMED" in p3["_source"], "and says so loudly", p3["_source"][-40:])

    t = write_template(td / "tpl.json", "my_cam")
    tpl = json.loads(t.read_text())
    check(tpl["camera"]["id"] == "my_cam" and "_README" in tpl,
          "template is a file to edit, not code to edit")
    check(load_profile(explicit=tpl)["venue"]["group_radius_m"] == 3.0,
          "the emitted template round-trips back in")

print()
print("=" * 74)
print("  3. a typo must be caught, not silently reported")
print("=" * 74)
check(validate(load_profile()) == [], "defaults are valid")
for bad_prof, why in [
    ({"venue": {"greet_proximity_m": 150}}, "metres where pixels were meant"),
    ({"venue": {"min_seated_s": 0}}, "zero dwell threshold"),
    ({"camera": {"hfov_deg": 0.9}}, "field of view of 0.9 degrees"),
    ({"camera": {"person_height_m": 170}}, "cm typed as metres"),
    ({"venue": {"staff_min_video_share": 35}}, "percent typed as a fraction"),
]:
    probs = validate(load_profile(explicit=bad_prof))
    check(len(probs) >= 1, f"caught: {why}", probs[0][:56] if probs else "NOT CAUGHT")
probs = validate(load_profile(explicit={"camera": {"timezone": "CDT"}}))
check(any("IANA" in p for p in probs),
      "an abbreviation like 'CDT' is rejected with the reason why",
      probs[0][:60] if probs else "")

print()
print("=" * 74)
print("  4. DST — the case 'CDT' cannot express")
print("=" * 74)
# US autumn change 2026: 02:00 CDT -> 01:00 CST on Sunday 1 November.
start = datetime(2026, 11, 1, 0, 30, 0)
clk = local_clock(load_profile(explicit={"camera": {"timezone": "America/Chicago"}}), start)
before = clk(0)
across = clk(2 * 3600)          # 2 real hours later
print(f"    start {start:%Y-%m-%d %H:%M:%S} local, +2h of real time -> {across}")
check(before == "00:30:00", "start renders correctly", before)
check(across == "01:30:00",
      "two real hours across the fall-back lands at 01:30, not 02:30",
      f"got {across}")
naive_wrong = (start.hour + 2, start.minute)
check(across != f"{naive_wrong[0]:02d}:{naive_wrong[1]:02d}:00",
      "naive addition would have been an hour out — this is the bug DST causes")
clk_utc = local_clock(load_profile(), start)
check(clk_utc(3600) == "01:30:00", "UTC default still works")
check(local_clock(load_profile(), None)(5) == "", "no start clock -> empty, no crash")

print()
print("=" * 74)
print("  5. entry direction inferred from the footage (U5)")
print("=" * 74)
INTERIOR = {"waiting", "dining"}


def scenario(entering, exiting, flipped=False):
    """entering: people who really came IN. exiting: people who really left."""
    cr, ev, tid = [], [], 0
    for _ in range(entering):
        tid += 1
        cr.append({"t": 100.0, "track_id": tid,
                   "direction": "out" if flipped else "in"})
        ev.append({"track_id": tid, "zone": "waiting", "t_in": 102.0, "t_out": 400.0})
    for _ in range(exiting):
        tid += 1
        cr.append({"t": 500.0, "track_id": tid,
                   "direction": "in" if flipped else "out"})
        ev.append({"track_id": tid, "zone": "waiting", "t_in": 200.0, "t_out": 498.0})
    return cr, ev


cr, ev = scenario(10, 10, flipped=False)
flip, conf, e = infer_entry_direction(cr, ev, INTERIOR)
check(flip is False, "correctly-labelled footage -> no flip needed", str(e)[:70])

cr, ev = scenario(10, 10, flipped=True)
flip, conf, e = infer_entry_direction(cr, ev, INTERIOR)
check(flip is True, "backwards entry line -> flip DETECTED", str(e.get("verdict")))
print("    -> today this is a hand-set boolean; getting it wrong inverts the"
      " headline number")

cr, ev = scenario(2, 1)
flip, conf, e = infer_entry_direction(cr, ev, INTERIOR)
check(flip is None, "too little evidence -> None, NOT a guess", e.get("why", "")[:56])
check(conf == 0.0, "and confidence is zero")
check(infer_entry_direction([], [], INTERIOR)[0] is None, "empty input -> None")
check(infer_entry_direction(cr, [], INTERIOR)[0] is None,
      "crossings but no interior dwell -> None")

# One-directional nights. These are the cases a one-sided rule goes blind on,
# which is why both directions are scored with opposite sign.
cr, ev = scenario(20, 0)
check(infer_entry_direction(cr, ev, INTERIOR)[0] is False,
      "an opening shift (everyone arrives), labels correct -> no flip")
cr, ev = scenario(20, 0, flipped=True)
check(infer_entry_direction(cr, ev, INTERIOR)[0] is True,
      "an opening shift with a BACKWARDS line -> flip detected")
cr, ev = scenario(0, 20)
flip, conf, e = infer_entry_direction(cr, ev, INTERIOR)
check(flip is False,
      "a closing shift (everyone leaves), labels correct -> no flip",
      "a rule that only read inward crossings would have had ZERO evidence here")
cr, ev = scenario(0, 20, flipped=True)
check(infer_entry_direction(cr, ev, INTERIOR)[0] is True,
      "a closing shift with a BACKWARDS line -> flip detected")

print()
print("=" * 74)
print("  describe()")
print("=" * 74)
print("   " + describe(load_profile(explicit={"venue": {"min_seated_s": 300}})
                        ).replace("\n", "\n   "))
check("min_seated_s=300" in describe(load_profile(explicit={"venue": {"min_seated_s": 300}})),
      "describe shows only what DIFFERS from default")

print()
print("=" * 74)
if FAILED:
    print(f"  {len(FAILED)} FAILED:")
    for f in FAILED:
        print(f"    - {f}")
    sys.exit(1)
print("  ALL PASS")
print("=" * 74)
