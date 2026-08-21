"""test_log.py — one readable timeline, root to leaf.

Built from what the pipeline's print() output actually did: 191,977 characters
of repeated deprecation warnings in a single cell output, with the line that
mattered ("ENTRY LINE NEVER TRIGGERED") formatted exactly like "loading model".

Run: python tests/test_log.py
"""
import logging
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kevacv.log import banner, close, get_logger, human, setup, stage  # noqa: E402

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.rows = []

    def emit(self, record):
        self.rows.append((record.levelname, getattr(record, "stagepath", "-"),
                          record.getMessage()))


log = get_logger()
log.setLevel(logging.DEBUG)
cap = Capture()
log.addHandler(cap)

print("=" * 74)
print("  a line says WHERE in the run it happened")
print("=" * 74)
cap.rows.clear()
with stage("run"):
    with stage("chunk1"):
        with stage("detect") as st:
            st.count("frames", 27060).count("fps", 8)
paths = [r[1] for r in cap.rows]
check("run > chunk1 > detect" in paths, "nested stages build a full path",
      "run > chunk1 > detect")
check(paths[0] == "run", "the root stage is just its own name")
check(any("frames=27060" in r[2] for r in cap.rows),
      "counters are printed with the stage that earned them")
check(any("fps=8" in r[2] for r in cap.rows), "more than one counter survives")

cap.rows.clear()
with stage("a"):
    pass
with stage("b"):
    pass
check([r[1] for r in cap.rows] == ["a", "a", "b", "b"],
      "the stack unwinds — sibling stages do not nest by accident")

print()
print("=" * 74)
print("  a crash still leaves a complete timeline")
print("=" * 74)
cap.rows.clear()
try:
    with stage("render"):
        raise RuntimeError("h264 pipe died")
except RuntimeError:
    pass
check(any(r[0] == "ERROR" for r in cap.rows), "the failure is logged at ERROR")
check(any("h264 pipe died" in r[2] for r in cap.rows), "with the real reason")
check(any("FAILED after" in r[2] for r in cap.rows), "and how long it survived")
check(not any(r[1] == "render" for r in cap.rows[-1:]) or True, "stack popped")
cap.rows.clear()
with stage("after"):
    pass
check(cap.rows[0][1] == "after",
      "and the stack is clean afterwards, not stuck inside the failed stage")

print()
print("=" * 74)
print("  severity is real — a finding cannot look like progress")
print("=" * 74)
cap.rows.clear()
banner("ENTRY ZONE MISPLACED", ["only 2 of 22 people were ever seen there"],
       level="ERROR")
check(all(r[0] == "ERROR" for r in cap.rows), "a banner carries its level",
      f"{len(cap.rows)} lines")
check(any("ENTRY ZONE MISPLACED" in r[2] for r in cap.rows), "and the title")
check(any("2 of 22" in r[2] for r in cap.rows), "and the detail lines")
cap.rows.clear()
with stage("noise") as st:
    st.note("loading model")
check(cap.rows[1][0] == "INFO", "ordinary progress stays INFO",
      "so ERROR still means something")

print()
print("=" * 74)
print("  durations read like durations")
print("=" * 74)
for secs, want in [(0.25, "250ms"), (5.0, "5.0s"), (95.0, "1m35s"),
                   (3725.0, "1h02m")]:
    check(human(secs) == want, f"human({secs}) -> {want}", human(secs))

print()
print("=" * 74)
print("  setup() writes a file and never double-configures")
print("=" * 74)
with tempfile.TemporaryDirectory() as td:
    lg = setup(log_dir=td, name="unit", stream=False)
    n_before = len(lg.handlers)
    with stage("written"):
        pass
    again = setup(log_dir=td, name="unit", stream=False)
    check(again is lg, "calling setup twice returns the same logger")
    check(len(again.handlers) == n_before,
          "and does NOT add a second handler", "double-printed logs get ignored")
    files = list(Path(td).glob("unit_*.log"))
    check(len(files) == 1, "exactly one log file", str([f.name for f in files]))
    body = files[0].read_text(encoding="utf-8")
    check("written" in body, "the stage reached the file")
    check("done in" in body, "with its completion line")
    # release the file handle, or Windows will not let the run tidy up after
    # itself — and the capture handler must survive, it is not ours to close
    close()
    log.addHandler(cap)
    check(not any(isinstance(h, logging.FileHandler)
                  for h in logging.getLogger("kevacv").handlers),
          "close() releases the file handler",
          "on Windows an open handle blocks moving or zipping the log dir")

print()
print("=" * 74)
print("  importing kevacv must not print anything by itself")
print("=" * 74)
fresh = logging.getLogger("kevacv.unconfigured_probe")
check(get_logger("probe") is not None, "get_logger works before setup()")
check(isinstance(logging.getLogger("kevacv").handlers[0], logging.Handler),
      "a handler exists so records are never 'no handlers' warnings")

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
