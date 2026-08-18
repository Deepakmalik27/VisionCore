"""Co-visibility stops a GROUP collapsing into one arrival.

Ground truth eval/gt_entries_305_318.json: six guests entered CAM.112 in 13s.
tier_a_crossings folded them together because "close in time and space" is
what one person crossing once looks like -- and also what a party looks like.

WHAT THIS SIGNAL CAN AND CANNOT DO, stated honestly because the first version
of this test asserted a number I wanted rather than one the rule earns:

  CAN   prove two tracks are DIFFERENT people -- they were visible at the
        same instant, and nobody is in two places at once.
  CANNOT prove two tracks are the SAME person. Sequential tracks that never
        coexist stay ambiguous: one person whose id broke, or two people
        arriving one after the other, look identical here.

So it is a one-sided guarantee. It recovers the abreast case (measured: the
door counter went 3 -> 5 of 6 on the labelled window) and leaves the
single-file case to other evidence.
"""
from kevacv.analytics import tier_a_crossings


def _c(tid, t, x):
    return {"track_id": tid, "t": t, "direction": "in", "pos": (x, 1000.0),
            "line": "entry line"}


def test_covisible_pair_is_never_merged():
    """The core guarantee: overlapping lifetimes => two people."""
    pair = [_c(1, 100.0, 1700), _c(2, 100.9, 1740)]      # 0.9s, 40px apart
    spans = {1: (98.0, 130.0), 2: (99.5, 128.0)}         # visible together
    assert tier_a_crossings(pair)[0] == 1, "without spans they collapse"
    assert tier_a_crossings(pair, spans=spans)[0] == 2, "co-visible => two"


def test_sequential_fragments_of_one_person_still_merge():
    """The other side: no overlap => still treated as one person."""
    frags = [_c(7, 100.0, 1700), _c(8, 101.2, 1720)]
    spans = {7: (98.0, 100.4), 8: (100.9, 104.0)}        # never coexist
    assert tier_a_crossings(frags, spans=spans)[0] == 1


def test_absent_spans_keeps_old_behaviour_exactly():
    pair = [_c(1, 100.0, 1700), _c(2, 100.9, 1740)]
    spans = {1: (98.0, 130.0), 2: (99.5, 128.0)}
    assert tier_a_crossings(pair)[0] == 1
    assert tier_a_crossings(pair, spans=spans)[0] == 2


def test_far_apart_never_merged_either_way():
    far = [_c(1, 100.0, 200), _c(2, 101.0, 1700)]
    assert tier_a_crossings(far)[0] == 2
    assert tier_a_crossings(far, spans={1: (99, 105), 2: (100, 106)})[0] == 2


def test_real_group_from_ground_truth_improves():
    """The four abreast ids measured on CAM.112, with their real lifetimes.

    Both fixes are required and neither works alone: at the old dedupe_s=6.0
    the party collapses even WITH spans, and without spans it collapses at
    every window setting.
    """
    g = [_c(135, 308.9, 1739), _c(138, 309.8, 1785),
         _c(144, 314.6, 1696), _c(150, 315.6, 1766)]
    spans = {135: (306.0, 357.0), 138: (308.0, 311.3),
             144: (313.0, 318.2), 150: (315.0, 419.2)}
    assert tier_a_crossings(g, dedupe_s=6.0)[0] == 2, "the old bug"
    assert tier_a_crossings(g, dedupe_s=6.0, spans=spans)[0] == 2, \
        "co-visibility alone is not enough at a 6s window"
    assert tier_a_crossings(g, spans=spans)[0] == 4, \
        "new default + co-visibility counts all four"


def test_window_has_a_lower_bound_too():
    """Too tight and one person's re-minted id splits into two."""
    frags = [_c(7, 100.0, 1700), _c(8, 101.2, 1720)]
    spans = {7: (98.0, 100.4), 8: (100.9, 104.0)}
    assert tier_a_crossings(frags, spans=spans)[0] == 1, "2.5s holds"
    assert tier_a_crossings(frags, dedupe_s=1.0, spans=spans)[0] == 2, \
        "1.0s is too tight - documents the lower bound"


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
