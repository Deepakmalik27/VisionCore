"""One person who leaves and returns is ONE person and TWO visits.

The operator's rule, verbatim: "we just have to see that person has entered
right count it if that person go out cout out as 1 right ?? not like u will
minus it ?? i want count that person if that person comes agani back then that
person is not counted". Until now nothing in the pipeline could express it.
"""
from kevacv.visits import build_visits, visits_for_person


def X(t, d, line="entry line", tid="P1"):
    return {"t": t, "direction": d, "line": line, "track_id": tid}


def test_the_operators_rule_one_person_two_visits():
    r = build_visits([X(10, "in"), X(20, "out"), X(30, "in"), X(40, "out")])
    assert r["unique_people"] == 1, "re-entry must NOT create a second person"
    assert r["n_visits"] == 2
    assert r["entries"] == 2 and r["exits"] == 2
    assert r["repeat_visitors"] == 1


def test_a_count_is_never_decremented_by_an_exit():
    r = build_visits([X(10, "in"), X(20, "out")])
    assert r["unique_people"] == 1 and r["entries"] == 1
    # the exit is recorded, not subtracted
    assert r["exits"] == 1


def test_still_inside_at_the_end_is_an_open_visit_not_an_error():
    r = build_visits([X(10, "in")])
    assert r["open"] == 1 and r["closed"] == 0
    assert "still inside when observation ended" in r["visits"][0]["notes"][0]


def test_already_inside_before_we_started_is_named_not_dropped():
    r = build_visits([X(10, "out")])
    assert r["entry_missed"] == 1
    assert r["entries"] == 0 and r["exits"] == 1
    assert "already inside before observation began" in r["visits"][0]["notes"][0]


def test_a_missed_exit_is_visible_not_silently_merged():
    r = build_visits([X(10, "in"), X(30, "in"), X(40, "out")])
    assert r["n_visits"] == 2, "two INs are two visits even with an OUT missing"
    assert r["exit_missed"] == 1
    assert r["entries"] == 2


def test_two_people_are_not_pooled():
    r = build_visits([X(10, "in", tid="A"), X(20, "out", tid="A"),
                      X(11, "in", tid="B")])
    assert r["unique_people"] == 2 and r["n_visits"] == 2
    assert r["repeat_visitors"] == 0


def test_interior_doors_are_excluded_when_a_line_is_named():
    r = build_visits([X(10, "in", line="dining entry"),
                      X(12, "in", line="entry line")],
                     line_name="entry line")
    assert r["entries"] == 1, "a dining threshold is movement, not an arrival"


def test_out_of_order_input_is_sorted_not_trusted():
    r = build_visits([X(40, "out"), X(10, "in"), X(30, "in"), X(20, "out")])
    assert r["n_visits"] == 2 and r["closed"] == 2


def test_duration_and_median():
    r = build_visits([X(0, "in"), X(10, "out"), X(20, "in"), X(50, "out")])
    assert r["median_visit_s"] == 20.0
    assert [v["duration_s"] for v in r["visits"]] == [10.0, 30.0]


def test_empty_input_is_zero_not_a_crash():
    r = build_visits([])
    assert r["unique_people"] == 0 and r["n_visits"] == 0
    assert r["median_visit_s"] is None


def test_unique_people_and_visits_are_different_numbers():
    """The whole point: they were the same field before."""
    r = build_visits([X(0, "in"), X(5, "out"), X(9, "in"), X(14, "out"),
                      X(20, "in"), X(25, "out")])
    assert r["unique_people"] == 1
    assert r["n_visits"] == 3
    assert r["unique_people"] != r["n_visits"]
