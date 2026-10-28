"""The ledger must make today's actual failures visible.

Each test replays a real one. If the ledger cannot surface it, the ledger is
not doing its job.
"""
import json, tempfile, os
from kevacv.decision_log import Ledger, CONFIG, DEFAULT, FALLBACK, DERIVED, HARDCODED


def test_names_the_real_mechanism_not_a_guess():
    """The log said "id churn at the line". It was a door filter deleting
    every interior crossing. The ledger records authority + recovery."""
    L = Ledger("t")
    with L.stage("tier-A dedupe", module="analytics.tier_a_crossings",
                 owns="engine.py:4659"):
        L.flow("crossings", 52, 45)
        L.drop(7, "crossings at interior doors",
               why="tier_a_crossings filters internally to venue entry lines",
               authority="venue_entry_lines({'entry line'})",
               recoverable="carry non-venue crossings through separately")
    txt = L.render()
    assert "interior doors" in txt and "venue_entry_lines" in txt
    assert "recover" in txt


def test_surfaces_a_hardcoded_value():
    """Drift hides in values with no knob."""
    L = Ledger("t")
    with L.stage("reid crops"):
        L.param("crop_conf_floor", 0.35, HARDCODED,
                why="literal in engine.py:3434; ignores the yaml detector floor")
    txt = L.render()
    assert "HARDCODED values with no knob: 1" in txt
    assert "cannot be A/B'd" in txt


def test_surfaces_a_fallback_as_a_degraded_answer():
    L = Ledger("t")
    with L.stage("ground plane"):
        L.param("mode", "auto-perspective-fit", FALLBACK,
                why="zones file has _ground_points_TEMPLATE with points: [] "
                    "so the exact homography could never run")
    txt = L.render()
    assert "FALLBACKS in force: 1" in txt
    assert "degraded answer" in txt


def test_derived_value_shows_its_formula():
    """The live re-id gate is 560/fps -- renamed from a fixed 140px, which is
    why the halving was invisible in any value-vs-value table."""
    L = Ledger("t")
    with L.stage("live reid"):
        L.param("max_dist_px", 70.0, DERIVED,
                formula="LIVE_REID_MAX_SPEED_PX_S 560 / eff_fps 8",
                why="notebook used a FIXED 140px; this is half")
    assert "560 / eff_fps 8" in L.render()


def test_warning_carries_why_it_matters():
    L = Ledger("t")
    with L.stage("gt"):
        L.warn("reference has 14 unique boxes across 600 rows",
               why_it_matters="scores grade a frozen scene, not tracking",
               likely_cause="labelling tool propagated boxes forward")
    txt = L.render()
    assert "matters because" in txt and "frozen scene" in txt


def test_stage_failure_is_recorded_not_swallowed():
    L = Ledger("t")
    try:
        with L.stage("static filter"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    txt = L.render()
    assert "FAILED" in txt and "as if it had passed" in txt


def test_writes_both_artifacts():
    L = Ledger("t")
    with L.stage("s"):
        L.flow("x", 1, 1)
    d = tempfile.mkdtemp()
    paths = L.write(d)
    assert all(os.path.exists(p) for p in paths)
    doc = json.load(open(os.path.join(d, "LEDGER.json")))
    assert doc["stages"][0]["stage"] == "s"


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
