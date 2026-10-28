"""The CV loop must never wait for a sink, and must never lose an event quietly."""
import json
import threading
import time
from kevacv.event_queue import EventQueue, jsonl_sink


def test_events_reach_the_sink():
    got = []
    q = EventQueue(sink=got.extend, batch=2, flush_s=0.05).start()
    for i in range(5):
        q.put({"i": i})
    s = q.close()
    assert s["written"] == 5 and s["lost"] == 0
    assert [g["i"] for g in got] == [0, 1, 2, 3, 4]


def test_a_full_queue_drops_rather_than_blocking_the_producer():
    """A dropped event that is COUNTED is recoverable. A stalled decoder is not."""
    started = threading.Event()
    release = threading.Event()

    def slow(rows):
        started.set()
        release.wait(5.0)

    q = EventQueue(sink=slow, maxsize=4, batch=1, flush_s=0.01).start()
    t0 = time.time()
    for i in range(200):
        q.put({"i": i})
    elapsed = time.time() - t0
    release.set()
    s = q.close()
    assert elapsed < 1.0, f"producer blocked for {elapsed:.2f}s"
    assert s["dropped"] > 0
    assert s["submitted"] == 200


def test_a_failing_sink_neither_kills_the_run_nor_claims_success():
    def boom(rows):
        raise RuntimeError("postgres is down")

    q = EventQueue(sink=boom, batch=1, flush_s=0.05).start()
    q.put({"a": 1})
    s = q.close()
    assert s["sink_errors"] >= 1
    assert s["written"] == 0, "a failed write must not be counted as written"
    assert s["lost"] == 1


def test_jsonl_sink_appends_and_survives_reopen(tmp_path):
    p = tmp_path / "events.jsonl"
    q = EventQueue(sink=jsonl_sink(str(p)), batch=2, flush_s=0.05).start()
    for i in range(3):
        q.put({"i": i})
    q.close()
    q2 = EventQueue(sink=jsonl_sink(str(p)), batch=1, flush_s=0.05).start()
    q2.put({"i": 99})
    q2.close()
    rows = [json.loads(l) for l in open(p)]
    assert [r["i"] for r in rows] == [0, 1, 2, 99], \
        "append-only: a crash at hour 23 must keep hours 1-22"


def test_close_is_idempotent():
    q = EventQueue(sink=lambda rows: None).start()
    q.close()
    assert q.close()["submitted"] == 0
