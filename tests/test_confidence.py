"""Tiers must never launder a disagreement into a confident number.

Measured on CAM.112, three runs that all reported a single guest count:
    p0classfix2   line=1  region=4  -> reported 4     (disagree 75%)
    p0v4          line=5  region=5  -> reported 5     (agree)
    h_v1          line=3  region=7  -> reported 7     (disagree 57%)
Only the middle one deserves to be called confirmed.
"""
from kevacv.confidence import (arrival_tier, event_tier, summarise,
                               CONFIRMED, PROBABLE, UNCERTAIN, REJECTED)


def test_agreeing_estimators_are_confirmed():
    assert arrival_tier(5, 5)["tier"] == CONFIRMED
    assert arrival_tier(4, 5)["tier"] == CONFIRMED          # 20% apart


def test_the_real_disagreements_are_uncertain_not_a_number():
    assert arrival_tier(1, 4)["tier"] == UNCERTAIN          # p0classfix2
    assert arrival_tier(3, 7)["tier"] == UNCERTAIN          # h_v1


def test_a_single_estimator_is_never_confirmed():
    t = arrival_tier(5, None)
    assert t["tier"] == PROBABLE
    assert "nothing independent corroborates" in t["why"]


def test_agreement_on_a_broken_plane_is_downgraded():
    assert arrival_tier(5, 5, plane_ok=False)["tier"] == PROBABLE


def test_everyone_agreeing_on_zero_is_confirmed_not_uncertain():
    assert arrival_tier(0, 0)["tier"] == CONFIRMED


def test_no_estimator_at_all_is_uncertain_not_zero():
    t = arrival_tier(None, None)
    assert t["tier"] == UNCERTAIN


def test_a_veto_is_absolute_and_not_outvoted():
    t = event_tier(crossed_line=True, direction_known=True, track_seconds=99.0,
                   confirmed_not_uturn=True,
                   vetoes=["impossible topology: reception -> dining in 0.2s"])
    assert t["tier"] == REJECTED, \
        "a physical impossibility must not be averaged against good evidence"


def test_a_uturn_is_rejected():
    assert event_tier(crossed_line=True, direction_known=True,
                      track_seconds=4.0,
                      confirmed_not_uturn=False)["tier"] == REJECTED


def test_presence_without_a_crossing_is_not_an_event():
    t = event_tier(crossed_line=False, direction_known=True, track_seconds=9.0,
                   confirmed_not_uturn=True)
    assert t["tier"] == UNCERTAIN
    assert "presence is evidence, not a transition" in t["why"]


def test_a_clean_crossing_is_confirmed():
    assert event_tier(crossed_line=True, direction_known=True,
                      track_seconds=6.0,
                      confirmed_not_uturn=True)["tier"] == CONFIRMED


def test_one_weakness_is_probable_two_is_uncertain():
    one = event_tier(crossed_line=True, direction_known=False,
                     track_seconds=6.0, confirmed_not_uturn=True)
    two = event_tier(crossed_line=True, direction_known=False,
                     track_seconds=0.3, confirmed_not_uturn=True)
    assert one["tier"] == PROBABLE and two["tier"] == UNCERTAIN


def test_summary_separates_official_from_review():
    s = summarise([CONFIRMED, CONFIRMED, UNCERTAIN, PROBABLE, REJECTED])
    assert s["official"] == 2 and s["needs_review"] == 2
    assert s["headline"] == "2 confirmed, 2 need review, 1 rejected"


def test_the_scorecard_fails_a_run_whose_estimators_disagree():
    """p0classfix2 reported guests=4 from line=1/region=4. That must not read
    as a clean run just because a single number came out of it."""
    from kevacv.scorecard import verdicts, FAIL, PASS
    from kevacv.confidence import arrival_tier
    bad = {"arrival_confidence": arrival_tier(1, 4)}
    good = {"arrival_confidence": arrival_tier(5, 5)}
    assert {r["check"]: r["state"] for r in verdicts(bad)}["arrival confidence"] == FAIL
    assert {r["check"]: r["state"] for r in verdicts(good)}["arrival confidence"] == PASS
