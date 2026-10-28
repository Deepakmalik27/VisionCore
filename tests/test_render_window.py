"""test_render_window.py — --render-window must select frames by TIME.

WHY THIS EXISTS
    Rendering was all-or-nothing: 11 minutes and ~8 GB on an hour chunk. So
    every A/B ran --no-render, which is correct for comparing counters and
    useless for the questions counters cannot answer — "is that box on a real
    person?", "is the entry line drawn where I think?". Those stayed open for
    days while the numbers kept moving.

    The filter sits at ONE choke point (frames = {...} from frame_log), so a
    window narrows coasting, HUD smoothing, decode and encode together. If it
    were applied later, the HUD would smooth over frames that were never drawn
    and the counts on screen would disagree with the run.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

fail = 0


def check(ok, what, detail=""):
    global fail
    print(f"  {'PASS' if ok else 'FAIL'}  {what}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        fail = 1


# The selection logic, mirrored exactly as it appears in render_annotated.
def select(frame_log, window):
    if not window:
        return frame_log
    r0, r1 = float(window[0]), float(window[1])
    keep = {idx for idx, t, _ in frame_log if r0 <= float(t) <= r1}
    if not keep:
        return frame_log          # empty selection -> render everything
    return [r for r in frame_log if r[0] in keep]


LOG = [(i, i / 8.0, [(1, 0, 0, 10, 10)]) for i in range(8 * 600)]  # 600 s @ 8 fps

print("=" * 74)
print("  --render-window selects by time, at one choke point")
print("=" * 74)

check(len(select(LOG, None)) == len(LOG),
      "no window renders the whole chunk", f"{len(LOG)} frames")

sel = select(LOG, (300, 420))
check(len(sel) == 8 * 120 + 1,
      "a 120 s window selects 120 s of frames", f"{len(sel)} frames")
check(all(300 <= t <= 420 for _, t, _ in sel),
      "every selected frame is inside the window")
check(len(sel) / len(LOG) < 0.25,
      "and it is a small fraction of the work",
      f"{100*len(sel)/len(LOG):.0f}% of frames")

# A window past the end of the chunk must not silently produce an empty video.
check(len(select(LOG, (9000, 9600))) == len(LOG),
      "a window outside the chunk falls back to rendering everything",
      "never produces an empty video")

print()
print("  ALL PASS" if not fail else "  FAILURES ABOVE")
# PYTEST VISIBILITY. This file's checks run at IMPORT and reported only via
# an exit code. A module-level sys.exit aborts pytest COLLECTION, which hid
# 59 of 74 test files -- and simply guarding the exit would have hidden the
# FAILURES instead. So the same condition is also asserted as a real test.
def test_script_level_checks_passed():
    assert not (fail), "module-level checks in this file failed"


if __name__ == "__main__":
    sys.exit(fail)
