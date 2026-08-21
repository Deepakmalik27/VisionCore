"""P3 needs a number that moves when tracking changes — without ground truth.

Traced by hand on CAM.112's held-out busy window: three real guests, detected,
whose tracks were born LEFT of the entry line (x~1160-1195) and died RIGHT of
it (x~1598). Nothing traversed, and the line scored 0 of 3. That is
fragmentation, and until now nothing in the pipeline counted it.
"""
from kevacv.scorecard import track_quality


def _walk(tid_pts):
    return {tid: [(t, x - 30, 400, x + 30, 900) for t, x in pts]
            for tid, pts in tid_pts.items()}


def test_one_clean_track_is_one_chain():
    q = track_quality(_walk({1: [(0.0, 100), (1.0, 200), (2.0, 300)]}))
    assert q["chains"] == 1 and q["fragments_per_chain"] == 1.0
    assert q["worst_chain"] == 1


def test_a_person_dropped_and_reacquired_is_ONE_chain_of_two():
    # id 1 dies at x=300, id 2 starts 0.5s later 40px away — same person
    q = track_quality(_walk({1: [(0.0, 100), (2.0, 300)],
                             2: [(2.5, 340), (4.0, 500)]}))
    assert q["track_ids"] == 2
    assert q["chains"] == 1, "two ids, one person"
    assert q["fragments_per_chain"] == 2.0
    assert q["links_made"] == 1


def test_two_genuinely_different_people_stay_separate():
    # far apart in space; must not be glued together
    q = track_quality(_walk({1: [(0.0, 100), (2.0, 200)],
                             2: [(2.5, 1700), (4.0, 1800)]}))
    assert q["chains"] == 2 and q["fragments_per_chain"] == 1.0


def test_a_long_time_gap_is_not_linked():
    q = track_quality(_walk({1: [(0.0, 100), (2.0, 300)],
                             2: [(60.0, 320), (61.0, 400)]}))
    assert q["chains"] == 2, "60s apart is not a hand-off"


def test_an_internal_gap_means_the_lost_buffer_RECOVERED_the_id():
    # same id, 1.5s hole in the middle: predict -> search -> recover worked
    q = track_quality(_walk({1: [(0.0, 100), (0.125, 110),
                                 (1.6, 300), (1.725, 310)]}), fps=8.0)
    assert q["ids_with_recovery"] == 1
    assert q["recovery_gaps"] == 1
    assert q["recovery_pct"] == 100.0


def test_a_continuous_track_reports_no_recovery():
    q = track_quality(_walk({1: [(0.0, 100), (0.125, 110), (0.25, 120)]}),
                      fps=8.0)
    assert q["ids_with_recovery"] == 0


def test_co_alive_and_close_is_swap_pressure():
    q = track_quality(_walk({1: [(0.0, 500), (0.125, 505)],
                             2: [(0.0, 540), (0.125, 545)]}), fps=8.0)
    assert q["swap_pressure_pairs"] == 1


def test_co_alive_but_far_apart_is_not():
    q = track_quality(_walk({1: [(0.0, 100), (0.125, 105)],
                             2: [(0.0, 1700), (0.125, 1705)]}), fps=8.0)
    assert q["swap_pressure_pairs"] == 0


def test_empty_input_does_not_divide_by_zero():
    q = track_quality({})
    assert q["track_ids"] == 0 and q["fragments_per_chain"] == 0.0
