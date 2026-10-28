"""test_graph_fusion_plane_trust.py — the hard veto must not run on a guess.

compute_pairwise_cost returns inf when two tracklets imply an impossible walking
speed. That is a HARD refusal: it forbids a merge and nothing downstream can
appeal it. It is computed from ground_plane.dist_m().

On CAM.112 the AUTOMATIC perspective fit produced eight camera heights in a
single hour (1.12m to 3.26m). A veto driven by a distance that may be 2.9x wrong
forbids real merges and permits impossible ones, roughly at random -- worse than
no veto, because it is confident. So the veto now requires an EXACT plane.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kevacv.graph_fusion import FusionWeights, compute_pairwise_cost  # noqa

_fail = False


def check(ok, what, detail=""):
    global _fail
    print(f"  {'PASS' if ok else 'FAIL'}  {what}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        _fail = True


class Plane:
    """Stand-in for GroundPlane. `mode` is the only thing under test."""
    def __init__(self, mode, metres):
        self.mode, self.ok, self._m = mode, True, metres

    def dist_m(self, a, b):
        return self._m


# Two tracklets 1 second apart. At 50 m that is 50 m/s -- impossible.
A = {"id": 1, "t_span": (0.0, 10.0), "exit_pos": (100, 100), "entry_pos": (100, 100)}
B = {"id": 2, "t_span": (11.0, 20.0), "exit_pos": (900, 700), "entry_pos": (900, 700)}
W = FusionWeights()

print("=" * 74)
print("  an EXACT plane is trusted — the veto fires")
print("=" * 74)
c = compute_pairwise_cost(A, B, W, Plane("exact", 50.0))
check(c == float("inf"), "50 m in 1 s is refused outright when the plane is MEASURED",
      f"cost={c}")
c = compute_pairwise_cost(A, B, W, Plane("exact", 1.0))
check(c != float("inf"), "and a plausible 1 m walk is allowed", f"cost={c:.3f}")

print()
print("=" * 74)
print("  an AUTO fit is NOT trusted — no hard veto")
print("=" * 74)
c_auto = compute_pairwise_cost(A, B, W, Plane("auto", 50.0))
check(c_auto != float("inf"),
      "the same impossible distance does NOT veto on an auto fit",
      f"cost={c_auto:.3f} — falls through to pixel distance, which can only "
      f"nudge the cost, never forbid the merge")
c_none = compute_pairwise_cost(A, B, W, None)
check(abs(c_auto - c_none) < 1e-9,
      "an auto fit behaves exactly like NO plane at all",
      "the distinction is measured-vs-guessed, not present-vs-absent")

print()
print("=" * 74)
print("  temporal feasibility is unaffected — it needs no geometry")
print("=" * 74)
OVER = {"id": 3, "t_span": (0.0, 10.0), "exit_pos": (1, 1), "entry_pos": (1, 1)}
check(compute_pairwise_cost(A, OVER, W, None) == float("inf"),
      "two tracklets alive at the same time are still refused",
      "one person cannot be in two places; that needs no plane")

print()
print("=" * 74)
print("  ALL PASS" if not _fail else "  FAILURES ABOVE")
print("=" * 74)
# PYTEST VISIBILITY. This file's checks run at IMPORT and reported only via
# an exit code. A module-level sys.exit aborts pytest COLLECTION, which hid
# 59 of 74 test files -- and simply guarding the exit would have hidden the
# FAILURES instead. So the same condition is also asserted as a real test.
def test_script_level_checks_passed():
    assert not (1 if _fail else 0), "module-level checks in this file failed"


if __name__ == "__main__":
    sys.exit(1 if _fail else 0)
