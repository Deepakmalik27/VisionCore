"""Tests for patch_v56_phase5a.py — TRIAGE CASCADE.

The test verifies:
  T1  ENABLE_TRIAGE config exists in notebook
  T2  Triage parameters (SCAN_EVERY_S, PAD_S, etc.) are present
  T3  kevacv.triage functions are imported/available
  T4  triage planner produces valid segments, stats, miss_risk, and coverage_report
  T5  Syntax and notebook integrity check
"""
import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
NB = HERE / "notebooks" / "pipeline.ipynb"
sys.path.insert(0, str(HERE))

_pass = _fail = 0


def check(cond, label, detail=None):
    global _pass, _fail
    if cond:
        _pass += 1
        print(f"  ✅ {label}")
    else:
        _fail += 1
        d = f"  ({detail})" if detail else ""
        print(f"  ❌ {label}{d}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Load the notebook
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
nb = json.loads(NB.read_text(encoding="utf-8"))
cells = nb["cells"]
CODE = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
CELLS = [(s, i) for i, s in enumerate(CODE)]
SRC = "\n".join(CODE)

print("=" * 74)
print("  T1-T3 — Triage config and imports in notebook")
print("=" * 74)
check("ENABLE_TRIAGE" in SRC, "T1 ENABLE_TRIAGE config is present")
check("TRIAGE_SCAN_EVERY_S" in SRC, "T2 TRIAGE_SCAN_EVERY_S is present")
check("TRIAGE_PAD_S" in SRC, "T2 TRIAGE_PAD_S is present")
check("plan_segments" in SRC, "T3 plan_segments is present in notebook namespace/code")
check("PATCH_V56_PHASE5A" in SRC, "T1 phase 5a marker is present")

print()
print("=" * 74)
print("  T4 — Functional verification of kevacv.triage")
print("=" * 74)
from kevacv.triage import plan_segments, miss_risk, coverage_report, describe

# Generate synthetic scan data: 1 hour (3600s), sampled every 6s
# Active windows at 300-480s, 1500-1800s, 3000-3100s
scan_data = []
t = 0.0
while t < 3600.0:
    n = 3 if (300 <= t < 480) else (5 if (1500 <= t < 1800) else (2 if (3000 <= t < 3100) else 0))
    scan_data.append((t, n))
    t += 6.0

segs, stats = plan_segments(scan_data, pad_s=20.0)
check(len(segs) == 3, "planner finds exactly 3 active segments", f"found {len(segs)}")
check(stats["saving_pct"] > 70.0, "triage achieves >70% compute saving", f"{stats['saving_pct']:.1f}%")

cov = coverage_report(segs, (0, 3600), (0, 3600))
check(cov["accounted"], "coverage report accounts for 100% of time")
check(cov["unseen_pct"] == 0.0, "unseen time is 0% for fully scanned range")

risk = miss_risk(stats["scan_step_s"], typical_visit_s=25.0)
check(risk["miss_prob"] == 0.0, "scan step 6s with 25s visit has 0% miss risk")

desc_str = describe(stats, cov, risk)
check("TRIAGE" in desc_str and "ANALYSED" in desc_str, "describe formatting produces clean output")

print()
print("=" * 74)
print("  T5 — Notebook integrity & syntax check")
print("=" * 74)
syn = []
for i, s in enumerate(CODE):
    try:
        ast.parse(s)
    except SyntaxError as e:
        syn.append((i, e))
check(not syn, "every code cell parses", f"{len(syn)} error(s)")
for i, e in syn:
    print(f"      cell {i}: {e}")

check(len(nb["cells"]) >= 30, "notebook intact", f"{len(nb['cells'])} cells")

print()
print("=" * 74)
summary = f"  TOTAL: {_pass + _fail} checks — {_pass} pass, {_fail} fail"
if _fail:
    print(f"  ❌ {summary}")
else:
    print(f"  ✅ {summary}")
print("=" * 74)
# PYTEST VISIBILITY. This file's checks run at IMPORT and reported only via
# an exit code. A module-level sys.exit aborts pytest COLLECTION, which hid
# 59 of 74 test files -- and simply guarding the exit would have hidden the
# FAILURES instead. So the same condition is also asserted as a real test.
def test_script_level_checks_passed():
    assert not (1 if _fail else 0), "module-level checks in this file failed"


if __name__ == "__main__":
    sys.exit(1 if _fail else 0)
