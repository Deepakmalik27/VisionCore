"""test_log_counters.py — the run ledger, and the diff that makes it useful.

WHY THIS EXISTS
    Stage.count() was first-class and the numbers were still thrown away: they
    were formatted into one log line and dropped. So "did that change help?"
    could only be answered by scrolling two logs side by side, and the
    measured_baseline block in config/cam112.yaml is a human hand-copying
    numbers out of one -- which is why it still claims entry_line_crossings: 0.

    The ledger keeps them. These tests hold the two properties that make it
    worth keeping: a crashed run still records how far it got, and a counter
    that FALLS TO ZERO is reported as a failure rather than as a diff row.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kevacv.log import (close, compare, flatten, setup, stage,  # noqa: E402
                        write_ledger)

_fail = False


def check(ok, what, detail=""):
    global _fail
    print(f"  {'PASS' if ok else 'FAIL'}  {what}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        _fail = True


print("=" * 74)
print("  counters survive the run that produced them")
print("=" * 74)

with tempfile.TemporaryDirectory() as td:
    setup(log_dir=td, name="runA", stream=False)
    with stage("analyse") as st:
        st.count("frames", 27060)
        st.count("crossings", 4)
    close()
    led = list(Path(td).glob("*_counters.json"))
    check(len(led) == 1, "a ledger is written next to the .log")
    rows = json.loads(led[0].read_text())["stages"]
    flat = flatten(rows)
    check(flat.get("analyse.frames") == 27060, "counters survive verbatim", flat)
    check(flat.get("analyse.crossings") == 4, "every counter, not just the last")

# A crash is when you MOST need to know how far it got.
with tempfile.TemporaryDirectory() as td:
    setup(log_dir=td, name="boom", stream=False)
    try:
        with stage("analyse") as st:
            st.count("frames", 900)
            raise RuntimeError("cuda oom")
    except RuntimeError:
        pass
    close()
    rows = json.loads(next(Path(td).glob("*_counters.json")).read_text())["stages"]
    check(flatten(rows).get("analyse.frames") == 900,
          "a CRASHED stage still records its counters",
          "the counters of a failed run say how far it got")
    check("cuda oom" in (rows[0].get("failed") or ""),
          "and records why it died")

print()
print("=" * 74)
print("  the diff answers 'did that change help?'")
print("=" * 74)

check(compare({"a.x": 1}, {"a.x": 1}) == [],
      "an unchanged counter is not noise")
check(compare({"a.x": 1}, {"a.x": 5}) == [("a.x", 1, 5)],
      "a moved counter is reported with both values")
check(compare({"a.x": 1}, {}) == [("a.x", 1, None)],
      "a counter that STOPPED being recorded is reported",
      "a stage that silently stopped running looks identical to a stable one")
check(compare({}, {"b.y": 2}) == [("b.y", None, 2)],
      "and so is a brand new one")

# The failure this pipeline keeps hitting: the entry line fired 0x and eight
# GM-facing numbers collapsed with it, and nothing was louder than a log line.
check(("zones.crossings", 12, 0) in compare({"zones.crossings": 12},
                                            {"zones.crossings": 0}),
      "a counter falling to ZERO shows up in the diff")

with tempfile.TemporaryDirectory() as td:
    setup(log_dir=td, name="run1", stream=False)
    with stage("zones") as st:
        st.count("crossings", 12)
    close()
    setup(log_dir=td, name="run2", stream=False)
    with stage("zones") as st:
        st.count("crossings", 0)
    import logging
    _recs = []
    _h = logging.Handler()
    _h.emit = _recs.append
    logging.getLogger("kevacv").addHandler(_h)
    write_ledger()
    logging.getLogger("kevacv").removeHandler(_h)
    close()
    msgs = [r.getMessage() for r in _recs]
    check(any("FELL TO ZERO" in m for m in msgs),
          "and is escalated by name, not buried in the diff",
          [m for m in msgs if "FELL TO ZERO" in m] or msgs[:3])
    check(any(r.levelno >= logging.ERROR for r in _recs),
          "at ERROR — a collapsed counter invalidates the run")

print()
print("=" * 74)
print("  ALL PASS" if not _fail else "  FAILURES ABOVE")
print("=" * 74)
# PYTEST VISIBILITY. This file's checks run at IMPORT and reported only via
# an exit code. A module-level sys.exit aborts pytest COLLECTION, which hid
# 59 of 74 test files -- and simply guarding the exit would have hidden the
# FAILURES instead. So the same condition is also asserted as a real test.
def test_script_level_checks_passed():
    assert not (1 if _fail else 0), "module-level checks in this file failed"


if __name__ == "__main__":
    sys.exit(1 if _fail else 0)
