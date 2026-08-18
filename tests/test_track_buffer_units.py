"""test_track_buffer_units.py — the lost-track memory must be the number of
SECONDS the config says.

WHY THIS EXISTS
    LOST_TRACK_BUFFER_S = 60 meant 16 seconds for months.

    boxmot's BotSort stores track_buffer in FRAMES AT 30 FPS and rescales:

        self.buffer_size   = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size

    Three call sites passed int(fps * LOST_TRACK_BUFFER_S), so the fps scaling
    landed twice:

        passed    8 * 60 = 480
        boxmot    int(8/30 * 480) = 128 frames = 16 s at 8 fps

    Nothing failed. Nothing warned. The tracker simply forgot people four times
    faster than intended, and a guest who stepped behind the plant for 20
    seconds returned as a new identity — 408 fragments resolving to 31 "people"
    in one hour at a desk that sees a handful.

    A units error inside a third-party rescaling is invisible in review and
    invisible at runtime. It is only visible in arithmetic, so the arithmetic
    gets a test.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kevacv import config  # noqa: E402

fail = 0


def check(ok, what, detail=""):
    global fail
    print(f"  {'PASS' if ok else 'FAIL'}  {what}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        fail = 1


def boxmot_seconds(track_buffer, fps):
    """Exactly what boxmot does with the number we hand it."""
    return int(fps / 30.0 * track_buffer) / fps


print("=" * 74)
print("  lost-track memory: config seconds vs what boxmot actually does")
print("=" * 74)

want = config.LOST_TRACK_BUFFER_S
for fps in (4, 8, 15, 30):
    got = boxmot_seconds(int(30 * want), fps)
    check(abs(got - want) <= 1.0,
          f"at {fps:>2} fps the tracker remembers {want}s",
          f"got {got:.1f}s")

# The old formula, kept as a guard so nobody reintroduces it.
bad = boxmot_seconds(int(8 * want), 8)
check(abs(bad - want) > 5.0,
      "the OLD formula int(fps * LOST_TRACK_BUFFER_S) is still wrong",
      f"it gives {bad:.1f}s instead of {want}s — do not go back to it")

print()
print("  ALL PASS" if not fail else "  FAILURES ABOVE")
sys.exit(fail)
