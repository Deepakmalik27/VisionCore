"""test_profiling.py — the profiler, and the gate it exists to decide.

The bottleneck here was misdiagnosed three times (detector, then tiling, then
finally the proxy JPEG writer — found only by noticing the GPU idle at 14%).
Each wrong guess cost a run. Selective ReID is a 3-5x win if ReID dominates and
a wasted week if it does not, and nothing in the logs could tell those apart.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kevacv.profiling import Profile

_fail = False


def check(ok, what, detail=""):
    global _fail
    print(f"  {'PASS' if ok else 'FAIL'}  {what}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        _fail = True


print("=" * 74)
print("  it measures what it says it measures")
print("=" * 74)
p = Profile(); p.start_wall()
for _ in range(3):
    with p.stage("detect"):
        time.sleep(0.01)
with p.stage("track+reid"):
    time.sleep(0.03)
check(p.calls["detect"] == 3, "counts calls per stage")
check(p.total["detect"] >= 0.028, "accumulates time across calls",
      f"{p.total['detect']:.3f}s")
check(p.total["track+reid"] >= 0.028, "times a single long stage")
check("t_detect_ms" in p.counters(), "exports ledger counters in ms",
      sorted(p.counters()))

print()
print("=" * 74)
print("  the PHASE C gate is decided by data, not argument")
print("=" * 74)
hot = Profile(); hot.start_wall()
with hot.stage("track+reid"):
    time.sleep(0.06)
with hot.stage("detect"):
    time.sleep(0.01)
v = hot.verdict()
check("WORTH BUILDING" in v, "ReID dominant -> build selective ReID", v[:60])

cold = Profile(); cold.start_wall()
with cold.stage("detect"):
    time.sleep(0.06)
with cold.stage("track+reid"):
    time.sleep(0.005)
v2 = cold.verdict()
check("BELOW" in v2, "ReID minor -> do NOT build it", v2[:60])
check(Profile().verdict() is None, "no data -> no verdict, never a guess")

print()
print("=" * 74)
print("  disabled costs nothing and breaks nothing")
print("=" * 74)
off = Profile(enabled=False)
with off.stage("detect"):
    pass
check(off.total == {}, "disabled records nothing")
check("disabled" in off.describe(), "and says so rather than printing a fake table")

print()
print("=" * 74)
print("  ALL PASS" if not _fail else "  FAILURES ABOVE")
print("=" * 74)
sys.exit(1 if _fail else 0)
