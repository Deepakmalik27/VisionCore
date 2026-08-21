"""audit.txt RED FLAG #3: the overlay showed "people in frame = 5, entered = 8,
exited = 6" and never noticed 1 + 8 - 6 = 3. Both numbers were in hand; nothing
compared them."""
from kevacv.analytics import reconcile_occupancy, describe_reconciliation


def _ev(tid, t0, t1):
    return {"track_id": tid, "zone": "reception", "t_in": t0, "t_out": t1,
            "duration": t1 - t0, "role": "customer"}


def _cr(t, d, line="entry line"):
    return {"t": t, "direction": d, "line": line}


def test_agreeing_doors_and_room_show_no_drift():
    # 2 present from the start, 2 more walk in, nobody leaves -> 4 and 4
    events = [_ev("a", 0, 300), _ev("b", 0, 300),
              _ev("c", 100, 300), _ev("d", 100, 300)]
    cr = [_cr(100, "in"), _cr(100, "in")]
    r = reconcile_occupancy(events, cr, 300, step_s=100)
    assert r["max_abs_drift"] == 0, r["steps"]


def test_a_missed_exit_shows_up_as_drift():
    # doors say two left; the room still shows everyone
    events = [_ev(x, 0, 300) for x in "abcd"]
    cr = [_cr(50, "out"), _cr(60, "out")]
    r = reconcile_occupancy(events, cr, 300, step_s=100)
    assert r["max_abs_drift"] == 2
    assert r["final_drift"] == 2          # observed exceeds expected


def test_the_audits_own_example_is_flagged():
    # 5 observed, 8 in, 6 out, initial 1 -> expected 3, drift +2
    events = [_ev(x, 0, 100) for x in "abcde"]
    cr = [_cr(10 + i, "in") for i in range(8)] + [_cr(20 + i, "out")
                                                  for i in range(6)]
    r = reconcile_occupancy(events, cr, 100, step_s=100, initial=1)
    assert r["steps"][0]["expected"] == 3
    assert r["steps"][0]["observed"] == 5
    assert r["steps"][0]["drift"] == 2
    assert ">=2 people" in describe_reconciliation(r)


def test_initial_defaults_to_what_was_already_in_the_room():
    # assuming an empty room is guaranteed wrong at a staffed reception desk
    events = [_ev("staff", 0, 300)]
    r = reconcile_occupancy(events, [], 300, step_s=100)
    assert r["initial"] == 1
    assert r["max_abs_drift"] == 0
