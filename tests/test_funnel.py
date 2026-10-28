"""test_funnel.py — detections are conserved, and losses are attributable.

WHY THIS EXISTS
    Eight filters sit between YOLO and a number on the report, and until now
    two of them printed a total and six printed nothing. "We counted 12 guests
    and the truth is 40" was unanswerable, because no record existed of which
    stage removed the other 28.

    These tests pin the accounting itself: in minus out equals dropped, at
    every stage, in run order — plus the two things a bare total hides, namely
    a stage that empties an occupied frame, and a stage that does nothing at
    all.

Run: python tests/test_funnel.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kevacv.funnel import DetectionFunnel

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


def _two_frames():
    """Two frames through a 3-stage chain: 5 raw -> 4 -> 3, then 1 -> 1 -> 0."""
    f = DetectionFunnel(label="test")
    f.record_first("yolo raw", 5)
    f.record("mask", 5, 4)
    f.record("size", 4, 3)
    f.record_first("yolo raw", 1)
    f.record("mask", 1, 1)
    f.record("size", 1, 0)
    return f


def test_conservation():
    f = _two_frames()
    d = {s["stage"]: s for s in f.as_dict()["stages"]}
    check(d["yolo raw"]["in"] == 6, "raw total accumulates across frames",
          str(d["yolo raw"]["in"]))
    check(d["mask"]["dropped"] == 1, "mask dropped 1", str(d["mask"]["dropped"]))
    check(d["size"]["dropped"] == 2, "size dropped 2", str(d["size"]["dropped"]))
    for s in f.as_dict()["stages"]:
        check(s["in"] - s["out"] == s["dropped"],
              f"in - out == dropped for {s['stage']}")


def test_stage_order_is_run_order():
    """The table must read top-to-bottom the way the frame does, or the stage
    that lost the people is impossible to find by eye."""
    f = _two_frames()
    names = [s["stage"] for s in f.as_dict()["stages"]]
    check(names == ["yolo raw", "mask", "size"], "first-seen order preserved",
          str(names))


def test_emptied_frames_counted_separately():
    """One detection lost from a frame of one is 100% of that frame and a
    rounding error in the total — but it is exactly how a person blinks out
    and returns with a new id."""
    f = _two_frames()
    d = {s["stage"]: s for s in f.as_dict()["stages"]}
    check(d["size"]["frames_emptied"] == 1, "the 1 -> 0 frame is flagged",
          str(d["size"]["frames_emptied"]))
    check(d["mask"]["frames_emptied"] == 0,
          "a stage that never emptied a frame reports 0")
    check(any("emptied" in m for _l, m in f.findings()),
          "and it is surfaced as a finding")


def test_share_is_of_raw_not_of_stage_input():
    """A late stage receiving 3 detections and dropping 3 has dropped 100% of
    its input but 50% of the run. The second number is the one that matters."""
    f = _two_frames()
    d = {s["stage"]: s for s in f.as_dict()["stages"]}
    check(abs(d["size"]["share_of_raw"] - 2 / 6) < 1e-9,
          "share is measured against the raw total",
          str(d["size"]["share_of_raw"]))


def test_greedy_stage_warns():
    f = DetectionFunnel()
    f.record_first("yolo raw", 100)
    f.record("greedy", 100, 50)
    levels = {lvl for lvl, _m in f.findings()}
    check("WARN" in levels, "a stage eating 50% of the run warns", str(levels))


def test_dead_stage_is_reported():
    f = DetectionFunnel()
    f.record_first("yolo raw", 100)
    f.record("does_nothing", 100, 100)
    msgs = [m for _l, m in f.findings()]
    check(any("removed nothing" in m for m in msgs),
          "a filter that never fires is reported, not silently trusted",
          str(msgs))


def test_raw_stage_never_warns_about_itself():
    """record_first is in == out by construction; it must not be reported as a
    'does nothing' filter — it is the denominator, not a filter."""
    f = DetectionFunnel()
    f.record_first("yolo raw", 10)
    f.record("real_filter", 10, 9)
    msgs = [m for _l, m in f.findings()]
    check(not any("yolo raw" in m for m in msgs),
          "the raw stage is excluded from findings", str(msgs))


def test_empty_funnel_describes_itself():
    f = DetectionFunnel()
    out = f.describe()
    check("never ran" in out, "an empty funnel says the loop never ran, not '0%'")


def test_describe_includes_every_stage():
    f = _two_frames()
    out = f.describe()
    for name in ("yolo raw", "mask", "size", "SURVIVED"):
        check(name in out, f"'{name}' appears in the printed table")


def main():
    print(__doc__.strip().splitlines()[0])
    for fn in (test_conservation,
               test_stage_order_is_run_order,
               test_emptied_frames_counted_separately,
               test_share_is_of_raw_not_of_stage_input,
               test_greedy_stage_warns,
               test_dead_stage_is_reported,
               test_raw_stage_never_warns_about_itself,
               test_empty_funnel_describes_itself,
               test_describe_includes_every_stage):
        print(f"\n{fn.__name__}")
        fn()
    print()
    if FAILED:
        print(f"FAILED ({len(FAILED)}): " + ", ".join(FAILED))
        return 1
    print("all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
