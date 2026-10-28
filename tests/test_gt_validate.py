"""The GT validator must refuse the exact file that fooled us twice.

gt.txt: 600 rows, 100 frames, 14 unique boxes, 3 of 6 tracks frozen. It
produced "HOTA 0.4762 / recall 0.4883 / +107% / clears the target floor",
all of which was withdrawn. The point of these tests is that the refusal
cannot quietly regress.
"""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from gt_validate import validate


def _write(rows):
    fh = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    for r in rows:
        fh.write(",".join(str(x) for x in r) + "\n")
    fh.close()
    return fh.name


def _moving(n_frames=60, n_tracks=4):
    """A believable sequence: everyone drifts, shapes are person-like."""
    rows = []
    for f in range(1, n_frames + 1):
        for t in range(1, n_tracks + 1):
            rows.append((f, t, 100 + 7 * f + 40 * t, 200 + 3 * f,
                         90, 230, 1, -1, -1, -1))
    return rows


def test_refuses_copy_forward():
    rows = [(f, t, 100 * t, 200, 90, 230, 1, -1, -1, -1)
            for f in range(1, 101) for t in range(1, 7)]
    code, fail, _w, _i = validate(_write(rows))
    assert code == 2, "a fully propagated file must be REFUSED"
    assert any("COPY-FORWARD" in f for f in fail), fail


def test_refuses_frozen_tracks():
    rows = _moving()
    rows += [(f, 99, 500, 500, 90, 230, 1, -1, -1, -1) for f in range(1, 61)]
    rows += [(f, 98, 700, 500, 90, 230, 1, -1, -1, -1) for f in range(1, 61)]
    code, fail, warn, _i = validate(_write(rows))
    assert any("FROZEN" in m for m in fail + warn), (fail, warn)


def test_accepts_a_believable_sequence():
    code, fail, _w, _i = validate(_write(_moving()))
    assert code < 2, f"a moving sequence must not be refused: {fail}"


def test_flags_no_entry_events_and_fails_when_required():
    rows = _moving()                      # every track spans the whole clip
    code, fail, warn, _i = validate(_write(rows))
    assert any("NO ENTRY/EXIT" in m for m in warn), warn
    code2, fail2, _w2, _i2 = validate(_write(rows), need_entries=True)
    assert code2 == 2 and any("NO ENTRY/EXIT" in f for f in fail2), fail2


def test_entry_events_are_recognised():
    rows = _moving(n_frames=60, n_tracks=3)
    # a fourth person arrives halfway and leaves before the end
    rows += [(f, 4, 300 + 6 * f, 400, 90, 230, 1, -1, -1, -1)
             for f in range(25, 50)]
    _c, _f, warn, _i = validate(_write(rows))
    assert not any("NO ENTRY/EXIT" in m for m in warn), warn


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
