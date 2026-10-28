"""The entry-label harness must refuse label sets that cannot detect the
failures we have already had.

Two real ones:
  * no quiet window  -> a counter that over-fires scores perfectly
  * no GROUP window HELD OUT -> exactly how a regression that halved recall
    on a 6-person party stayed invisible while held-out read 100%/0FP
"""
import json, sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from entry_label_kit import check


def _plan(windows):
    fh = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump({"windows": windows}, fh)
    fh.close()
    return fh.name


def _w(t0, truth, held=False, sealed=False):
    return {"t0": t0, "t1": t0 + 13, "truth": truth, "held_out": held,
            "sealed": sealed, "kind": "quiet" if truth == 0 else "busy",
            "entries": []}


def test_refuses_when_nothing_labelled():
    code, fail, _w2, _d = check(_plan([_w(0, None)]))
    assert code == 2 and any("nothing is labelled" in f for f in fail), fail


def test_refuses_without_a_quiet_window():
    code, fail, _w2, _d = check(_plan([_w(0, 4, held=True), _w(20, 3)]))
    assert code == 2 and any("NO QUIET" in f for f in fail), fail


def test_refuses_when_nothing_held_out():
    code, fail, _w2, _d = check(_plan([_w(0, 0), _w(20, 4)]))
    assert code == 2 and any("NOTHING HELD OUT" in f for f in fail), fail


def test_refuses_when_no_group_is_held_out():
    """A held-out set of only small windows cannot see the group regression."""
    ws = [_w(0, 0, held=True), _w(20, 1, held=True), _w(40, 6)]
    code, fail, _w2, _d = check(_plan(ws))
    assert code == 2 and any("NO GROUP" in f for f in fail), fail


def test_accepts_a_sound_label_set():
    ws = [_w(0, 0, held=True), _w(20, 5, held=True), _w(40, 3), _w(60, 0)]
    code, fail, _w2, _d = check(_plan(ws))
    assert code < 2, fail


def test_warns_while_still_sealed():
    ws = [_w(0, 0, held=True), _w(20, 5, held=True, sealed=True), _w(40, 3)]
    code, _f, warn, _d = check(_plan(ws))
    assert any("SEALED" in m for m in warn), warn


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
