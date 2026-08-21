"""A guest count whose estimators disagree must not read as a clean answer."""
from kevacv.answers import guest_count, WEAK, ESTIMATE
from kevacv.confidence import arrival_tier


def test_agreement_is_recorded_as_a_caveat():
    a = guest_count([1, 2, 3, 4, 5], agreement=arrival_tier(5, 5))
    assert a.tier == ESTIMATE
    assert any("cross-check CONFIRMED" in c for c in a.caveats)


def test_disagreement_downgrades_the_tier():
    # the real p0classfix2 case: line=1, region=4, published "4"
    a = guest_count([1, 2, 3, 4], agreement=arrival_tier(1, 4))
    assert a.tier == WEAK, "a 75% gap between sources is not an ESTIMATE"
    assert any("UNCERTAIN" in c for c in a.caveats)


def test_no_agreement_supplied_changes_nothing():
    before = guest_count([1, 2, 3])
    after = guest_count([1, 2, 3], agreement=None)
    assert before.tier == after.tier == ESTIMATE
    assert before.caveats == after.caveats
