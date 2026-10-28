"""test_clock.py — the product is timestamps, so the clock gets its own suite.

Built from a shipped failure: CHUNK_FILTER selected the 7:30pm file, the
4:30pm file was on disk, and the run stamped 19:30 onto 16:30 footage. Every
time in that report was three hours wrong and nothing complained, because each
step was individually correct.

Run: python tests/test_clock.py
"""
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kevacv.clock import (check_dst_span, check_frame_clock, describe,  # noqa: E402
                          localize, parse_start, verify_provenance)

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


REAL = ("CAM.112 (PP.09_12) 7-28-2026, 4.30.00pm CDT - "
        "7-28-2026, 5.30.00pm CDT.mp4")

print("=" * 74)
print("  the START time is read, not whichever stamp appears first anywhere")
print("=" * 74)
dt = parse_start(REAL)
check(dt == datetime(2026, 7, 28, 16, 30, 0), "the real filename parses to 16:30",
      str(dt))
check(parse_start(REAL).hour == 16,
      "and NOT to the range's end time",
      "a bare substring search once matched the previous hour's END")
check(parse_start("CAM.112 1-5-2026, 12.00.00am CDT.mp4").hour == 0,
      "12am is midnight, not noon")
check(parse_start("CAM.112 1-5-2026, 12.00.00pm CDT.mp4").hour == 12,
      "12pm is noon, not midnight")
check(parse_start("no timestamp here.mp4") is None, "an unparseable name -> None")
check(parse_start("CAM 13-45-2026, 4.30.00pm.mp4") is None,
      "an impossible date -> None, not a crash")

print()
print("=" * 74)
print("  naive local time is not a time")
print("=" * 74)
aware, f = localize(datetime(2026, 7, 28, 16, 30), "America/Chicago")
check(aware is not None and aware.tzinfo is not None, "an ordinary time localizes")
check(not f, "with no complaints", str(f))

# 2 Nov 2026, 01:30 America/Chicago happens twice — the clock goes back
_, f = localize(datetime(2026, 11, 1, 1, 30), "America/Chicago")
check(any("AMBIGUOUS" in m for _, m in f),
      "a repeated local hour is flagged AMBIGUOUS",
      "two different real moments would print the same time")
# 8 Mar 2026, 02:30 America/Chicago does not exist — the clock jumps forward
_, f = localize(datetime(2026, 3, 8, 2, 30), "America/Chicago")
check(any("does not exist" in m for _, m in f), "a skipped local hour is flagged")

_, f = localize(None, "America/Chicago")
check(any(l == "ERROR" for l, _ in f), "no timestamp at all is an ERROR")
_, f = localize(datetime(2026, 7, 28, 16, 30), "Mars/Olympus")
check(any("unknown timezone" in m for _, m in f), "a bogus zone warns, never crashes")

print()
print("=" * 74)
print("  a long run can START fine and END in a different offset")
print("=" * 74)
check(not check_dst_span(datetime(2026, 7, 28, 16, 30), 10, "America/Chicago"),
      "a normal 10-hour night is clean")
f = check_dst_span(datetime(2026, 11, 1, 0, 30), 4, "America/Chicago")
check(any("crosses a DST change" in m for _, m in f),
      "an overnight run through fall-back is caught",
      "checked on the SPAN, not just the start")

print()
print("=" * 74)
print("  frame_index/fps is only the clock if the container tells the truth")
print("=" * 74)
src, drift, f = check_frame_clock([(300, 10.0), (900, 30.0), (1800, 60.0)], 30.0)
check(src == "frame_index" and drift < 1.0, "a constant-rate file is trusted",
      f"drift {drift:.2f}%")
check(not f, "and produces no findings")

src, drift, f = check_frame_clock([(300, 10.0), (900, 33.0), (1800, 70.0)], 30.0)
check(src == "pos_msec", "a drifting file switches the time source", src)
check(any("VARIABLE FRAME RATE" in m for _, m in f), "and says why")
check(any("growing amount" in m for _, m in f),
      "and that the error GROWS — an early gap reads fine, a late one does not")

src, _, f = check_frame_clock([], 30.0)
check(src == "frame_index" and f, "no probes -> assume CFR but warn")

print()
print("=" * 74)
print("  the file we picked, decoded and clocked must be ONE file")
print("=" * 74)
SEVEN = "CAM.112 7-28-2026, 7.30.00pm CDT - 7-28-2026, 8.30.00pm CDT.mp4"
f = verify_provenance(SEVEN, REAL, SEVEN)
check(any("PROVENANCE MISMATCH" in m for _, m in f),
      "the exact shipped bug is caught",
      "selected 7:30pm, decoded 4:30pm, clocked 7:30pm")
check(any("every timestamp in this report would be wrong" in m.lower()
          for _, m in f), "and the consequence is stated plainly")
check(not verify_provenance(REAL, REAL, REAL), "one consistent file -> silence")
check(not verify_provenance(REAL, "/data/" + REAL, "./" + REAL),
      "different paths to the same file are fine")
f = verify_provenance("a_" + REAL, REAL, REAL)
check(f and f[0][0] == "WARN",
      "same start time but different names -> WARN, not ERROR")
check(not verify_provenance(None, None, None), "nothing supplied -> no false alarm")

print()
print("=" * 74)
print("  describe() is readable")
print("=" * 74)
txt = describe("pos_msec", 6.7, [("ERROR", "VARIABLE FRAME RATE")])
check("pos_msec" in txt and "6.70" in txt, "source and drift are shown", txt[:44])
check("no clock problems" in describe("frame_index", 0.0, []),
      "a clean clock says so explicitly")

print()
print("=" * 74)
# PYTEST VISIBILITY. This file's checks run at IMPORT and reported only via
# an exit code. A module-level sys.exit aborts pytest COLLECTION, which hid
# 59 of 74 test files -- and simply guarding the exit would have hidden the
# FAILURES instead. So the same condition is also asserted as a real test.
def test_script_level_checks_passed():
    assert not (FAILED), "module-level checks in this file failed"


if __name__ == "__main__":
    if FAILED:
        print(f"  {len(FAILED)} FAILED:")
        for f in FAILED:
            print(f"    - {f}")
        sys.exit(1)
print("  ALL PASS")
print("=" * 74)
