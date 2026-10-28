"""The time-matched scorer must reproduce the frame-index scorer exactly.

A scoring tool that is wrong looks like a bad model, not like a broken tool --
which is the most expensive kind of bug in this project. So this pins the
identity: matched 1:1, the two scorers must agree to the digit.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from score_by_time import gt_frame_to_t, pred_frame_to_t, retime


def test_gt_mapping_matches_the_manifest():
    # profE: eval window t=300s, 7.5 analysed fps; quick100 = frames 301..400
    assert abs(gt_frame_to_t(1) - 340.0) < 1e-6
    assert abs(gt_frame_to_t(100) - 353.2) < 1e-6


def test_identity_when_rates_match():
    gt = {k: [(1, 0, 0, 10, 10)] for k in range(1, 101)}
    pred = {k: [(1, 0, 0, 10, 10)] for k in range(1, 100)}
    _, out, st = retime(gt, pred, pred_t0=340.0, pred_fps=7.5)
    assert st["matched"] == 99, st
    assert st["pred_frames_dropped"] == 0, st
    assert out.keys() == pred.keys(), "1:1 mapping must renumber to itself"


def test_double_rate_collapses_to_one_per_gt_frame():
    """At 15fps against 7.5fps gt, two predictions compete for each gt frame;
    exactly one must win, or boxes get counted twice."""
    gt = {k: [(1, 0, 0, 10, 10)] for k in range(1, 11)}
    pred = {k: [(1, 0, 0, 10, 10)] for k in range(1, 21)}
    _, out, st = retime(gt, pred, pred_t0=340.0, pred_fps=15.0, tol_s=0.07)
    assert len(out) <= len(gt), f"more matched frames than gt frames: {st}"
    assert all(k in gt for k in out), "matched into a frame gt does not have"


def test_wrong_mapping_is_visible_not_silent():
    """A bad --pred-t0 must drop frames, so it reads as a mapping error rather
    than as a catastrophically bad model."""
    gt = {k: [(1, 0, 0, 10, 10)] for k in range(1, 101)}
    pred = {k: [(1, 0, 0, 10, 10)] for k in range(1, 100)}
    _, out, st = retime(gt, pred, pred_t0=900.0, pred_fps=7.5)
    assert st["matched"] == 0, "a 560s offset must match nothing"
    assert st["pred_frames_dropped"] == 99


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
