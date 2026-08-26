"""test_coverage_p3.py — the coverage map and the canary must be WIRED.

WHY THIS FILE EXISTS
    This repo's recurring failure mode is not bad code, it is UNREACHABLE code.
    tools/wiring_audit.py currently reports six modules that were built, tested
    and never called; validity.DetectorCanary was one of them, sitting complete
    and unused while every run's accuracy claim was self-reported.

    So these checks are about CONNECTION, not arithmetic. coverage.py has its
    own demo() for the maths. What this file asserts is that the grid is fed,
    the canary is fed, and their findings actually arrive somewhere a human
    reads -- because each of those links has silently broken before.

Run: python tests/test_coverage_p3.py
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


ENG = (ROOT / "kevacv" / "engine.py").read_text(encoding="utf-8")
PIPE = (ROOT / "kevacv" / "pipeline.py").read_text(encoding="utf-8")

print()
print("=" * 74)
print("  the coverage grid is CONSTRUCTED and FED")
print("=" * 74)
check("_coverage = _CoverageGrid(an_w, an_h)" in ENG,
      "grid is built at the ANALYSED frame size",
      "building it at source size would put every anchor in the wrong cell")
check("_coverage.add_frame()" in ENG, "frames are counted")
check("_coverage.add_detection(bc," in ENG,
      "detections are recorded at bc — the anchor the counter uses")
check("_coverage.add_kill(" in ENG, "filter-chain kills are recorded too")

# the kill anchor must match the detection anchor, or the two halves of the
# map are measured in different places and kill_ratio is meaningless.
kill_line = next(l for l in ENG.splitlines() if "_coverage.add_kill(" in l)
check("_gy2" in kill_line and "+ _gx2) / 2.0" in kill_line,
      "kills use the SAME bottom-centre anchor as detections", kill_line.strip())

print()
print("=" * 74)
print("  the detector canary is CONSTRUCTED and FED")
print("=" * 74)
check("_canary = _DetectorCanary()" in ENG, "canary is built")
check("_canary.observe(t," in ENG, "canary observes every analysed frame")
check("_canary.close(" in ENG, "canary is closed, so a final open run is counted")
check("_motion_by_fi" in ENG, "motion is measured per frame for the canary")

# measuring motion must not have CHANGED the gate. The gate's own condition
# must still be the original one.
check("(_t - _last_det_t) > MOTION_IDLE_S" in ENG,
      "the motion GATE condition is unchanged",
      "measuring motion must not alter which frames are skipped")

print()
print("=" * 74)
print("  the findings REACH a human")
print("=" * 74)
check('"coverage": _coverage.as_dict()' in ENG,
      "the grid is returned as data, so two runs can be diffed mechanically")
check('"coverage_findings"' in ENG and '"canary_findings"' in ENG,
      "both findings lists are returned from process_video")
check('for _k in ("coverage_findings", "canary_findings")' in PIPE,
      "the pipeline merges them into result['findings']",
      "unmerged findings never reach the scorecard, SUMMARY or night report")

print()
print("=" * 74)
print("  behaviour: the canary fires only on SUSTAINED disagreement")
print("=" * 74)
from kevacv.validity import DetectorCanary  # noqa: E402

c = DetectorCanary(run_length=30)
for i in range(10):                       # a short blip must NOT fire
    c.observe(float(i), True, 0)
c.observe(10.0, True, 3)
c.close(11.0)
check(not c.findings(), "10 quiet frames is a blip, not a finding",
      "firing on blips would train everyone to ignore it")

c2 = DetectorCanary(run_length=30)
for i in range(60):                       # a sustained stretch MUST fire
    c2.observe(float(i), True, 0)
c2.close(60.0)
f = c2.findings()
check(f and f[0][0] == "ERROR", "60 moving frames with zero detections is an ERROR",
      f[0][1][:60] if f else "no finding")

c3 = DetectorCanary(run_length=30)
for i in range(60):                       # no motion -> silence is correct
    c3.observe(float(i), False, 0)
c3.close(60.0)
check(not c3.findings(), "a still, empty scene is NOT a canary event",
      "an empty venue must not look like a broken detector")

print()
print("=" * 74)
print("  behaviour: blind spots are only reported when classifiable")
print("=" * 74)
from kevacv.coverage import CoverageGrid  # noqa: E402

g = CoverageGrid(1920, 1080, cols=8, rows=4)
for _ in range(500):
    g.add_frame()
g.add_detection((300, 900), conf=0.9, height_px=200)
zone = g.cells_for_polygon([[0, 0], [900, 0], [900, 1079], [0, 1079]])
check(any(l == "ERROR" for l, _ in g.findings(zone_cells=zone)),
      "blind cells inside a counted zone raise an ERROR")
check(not any(l == "ERROR" for l, _ in g.findings()),
      "with no zone map it refuses to raise",
      "'12% of the frame is blind' with no idea WHICH 12% is a rumour")

print()
print("=" * 74)


def test_script_level_checks_passed():
    assert not FAILED, f"coverage/canary wiring checks failed: {FAILED}"


if __name__ == "__main__":
    if FAILED:
        print(f"  {len(FAILED)} FAILED:")
        for x in FAILED:
            print(f"    - {x}")
        sys.exit(1)
    print("  ALL PASS")
    print("=" * 74)
