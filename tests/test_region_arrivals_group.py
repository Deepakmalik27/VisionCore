"""arrivals_from_regions must not collapse a GROUP into one arrival.

This is the same bug tier_a_crossings was fixed for. It is the PREFERRED
arrival source (derive.arrivals_by_id prefers "region"), so it feeds the
headline guest number.
"""
from kevacv.arrivals import arrivals_from_regions

ZR = {"main_entrance": ["entry"], "waiting area": ["wait"]}


def _party(tid, t):
    return [{"track_id": tid, "zone": "main_entrance", "t_in": t,
             "t_out": t + 0.5, "duration": 0.5, "role": "customer"},
            {"track_id": tid, "zone": "waiting area", "t_in": t + 1.0,
             "t_out": t + 9.0, "duration": 8.0, "role": "customer"}]


# four guests abreast, ~1s apart, at nearly the same spot
EV = _party(1, 100.0) + _party(2, 101.0) + _party(3, 102.0) + _party(4, 103.0)
POS = {1: (1700, 1000), 2: (1720, 1005), 3: (1740, 1002), 4: (1760, 1008)}
SPANS = {1: (99.0, 140.0), 2: (100.0, 112.0), 3: (101.0, 115.0),
         4: (102.0, 160.0)}


def test_group_collapses_without_covisibility():
    n, _, _ = arrivals_from_regions(EV, ZR, positions=POS)
    assert n < 4, f"expected the bug to collapse the party, got {n}"


def test_group_survives_with_covisibility():
    n, _, _ = arrivals_from_regions(EV, ZR, positions=POS, spans=SPANS)
    assert n == 4, f"a party of four must count four, got {n}"


def test_one_person_two_ids_still_merges():
    """Sequential fragments that never coexist must still fold into one."""
    ev = _party(7, 100.0) + _party(8, 101.0)
    pos = {7: (1700, 1000), 8: (1710, 1002)}
    spans = {7: (99.0, 100.8), 8: (101.0, 110.0)}      # no overlap
    n, _, _ = arrivals_from_regions(ev, ZR, positions=pos, spans=spans)
    assert n == 1, f"one person wearing two ids must count once, got {n}"


def test_absent_spans_is_old_behaviour():
    assert arrivals_from_regions(EV, ZR, positions=POS)[0] < 4


if __name__ == "__main__":
    for n_, f in sorted(globals().items()):
        if n_.startswith("test_"):
            f(); print("ok", n_)
