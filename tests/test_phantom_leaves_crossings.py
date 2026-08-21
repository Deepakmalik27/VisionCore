"""A phantom removed from the count must also leave `crossings`.

pipeline's phantom stage filtered events, guests, roles, arrivals and
contacts with a hand-rolled loop -- but not crossings. line_n is computed
from crossings a few lines later, so a plant or mirror flagged as a phantom
was deleted from every other structure and still counted as an ARRIVAL.
"""
from kevacv.detect_filters import drop_tracks


def _ev(tid):
    return {"track_id": tid, "zone": "waiting area", "t_in": 1.0,
            "t_out": 9.0, "duration": 8.0, "role": "customer"}


def _cr(tid):
    return {"track_id": tid, "t": 5.0, "direction": "in", "line": "entry line"}


def test_phantom_is_removed_from_crossings_too():
    events = [_ev(1), _ev(99)]
    crossings = [_cr(1), _cr(99)]
    frame_log = [(0, 0.0, [(1, 0, 0, 5, 5), (99, 9, 9, 14, 14)])]
    e, c, f = drop_tracks(events, crossings, frame_log, {99})
    assert [x["track_id"] for x in e] == [1]
    assert [x["track_id"] for x in c] == [1], \
        "the phantom survived in crossings and would still count as an arrival"
    assert all(b[0] != 99 for _fi, _t, bx in f for b in bx), \
        "the phantom is still DRAWN in the frame log"


def test_real_track_is_untouched():
    e, c, f = drop_tracks([_ev(1)], [_cr(1)], [(0, 0.0, [(1, 0, 0, 5, 5)])], set())
    assert len(e) == 1 and len(c) == 1 and len(f[0][2]) == 1


if __name__ == "__main__":
    for n, fn in sorted(globals().items()):
        if n.startswith("test_"):
            fn(); print("ok", n)
