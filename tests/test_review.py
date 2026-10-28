"""A tier that changes no behaviour is decoration. This is what consumes it."""
from kevacv.review import build_queue, render
from kevacv.confidence import arrival_tier
from kevacv.visits import build_visits


def X(t, d, tid="P1"):
    return {"t": t, "direction": d, "line": "entry line", "track_id": tid}


def test_a_confirmed_run_queues_nothing():
    q = build_queue(arrival_confidence=arrival_tier(5, 5),
                    visits=build_visits([X(0, "in"), X(9, "out")]))
    assert q["n_items"] == 0
    assert "empty" in render(q)


def test_the_real_disagreement_becomes_a_review_item():
    # p0classfix2: line=1, region=4, published "4"
    q = build_queue(arrival_confidence=arrival_tier(1, 4))
    assert q["n_items"] == 1
    assert q["items"][0]["kind"] == "arrival_disagreement"
    assert "count by eye" in q["items"][0]["action"]


def test_someone_who_left_without_arriving_is_flagged_with_a_clip():
    q = build_queue(visits=build_visits([X(30, "out")]))
    assert q["by_kind"]["entry_missed"] == 1
    at = q["items"][0]["look_at"]
    assert at["from_s"] == 25.0 and at["to_s"] == 35.0, \
        "a reviewer must not have to search a ten-hour video"


def test_a_missed_exit_is_flagged():
    q = build_queue(visits=build_visits([X(10, "in"), X(40, "in")]))
    assert q["by_kind"]["exit_missed"] == 1


def test_rejected_events_are_kept_for_audit_not_for_counting():
    q = build_queue(rejected_events=[{"t": 12.0, "track_id": "P3",
                                      "why": "U-turn inside 5s"}])
    assert q["by_kind"]["rejected_event"] == 1
    assert "indistinguishable from a bug" in q["items"][0]["action"]


def test_worst_first_ordering():
    q = build_queue(arrival_confidence=arrival_tier(1, 4),
                    visits=build_visits([X(30, "out")]),
                    rejected_events=[{"t": 5.0, "why": "x"}])
    kinds = [i["kind"] for i in q["items"]]
    assert kinds[0] == "arrival_disagreement"
    assert kinds.index("entry_missed") < kinds.index("rejected_event")


def test_an_open_visit_is_NOT_a_review_item():
    """Still being inside when the chunk ends is the normal state of anyone in
    the room at close — flagging it would bury the real items."""
    q = build_queue(visits=build_visits([X(10, "in")]))
    assert q["n_items"] == 0


def test_the_queue_is_capped_and_says_so():
    many = [{"t": float(i), "why": "x"} for i in range(50)]
    q = build_queue(rejected_events=many, limit=10)
    assert len(q["items"]) == 10 and q["truncated"] == 40
    assert "and 40 more" in render(q)
