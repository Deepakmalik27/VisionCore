"""test_phantom_gates_measured.py — the phantom gates must admit the real statue.

WHY THIS EXISTS
    The statue on the reception desk was tracked as a person for 44 minutes and
    only removed by the end-of-chunk pass, after holding a canonical id the
    whole time. The discriminator that should have caught it — centre jitter —
    was never wrong. It was simply UNREACHABLE, because two gates in front of
    it were set from a synthetic test rather than from footage.

    Measured on real tracks (run10, 10 min, L40S):

        REAL static objects (145, 49)   size cv 0.034-0.045
                                        centre move 1.2 px
                                        span 129-146 s
        REAL moving people              size cv 0.116-0.299
                                        centre move 50-310 px

    Against those numbers:
        PHANTOM_SIZE_CV    was 0.015  -> BELOW the real statue -> never matched
        PHANTOM_MIN_SPAN_S was 240 s  -> ABOVE its span        -> never matched

    A synthetic fixture claimed a statue holds size cv ~0.004. It does not. Any
    threshold derived from that fixture is 8-10x too strict, which is also why
    PHANTOM_FAST_CV_RATIO measured exactly zero effect on real footage.

WHAT IS ENFORCED
    the real statue's measurements PASS every gate
    the real people's measurements FAIL at least one
    the thresholds sit inside the measured gap, not on top of either population
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kevacv import config as C  # noqa: E402

fail = 0


def check(ok, what, detail=""):
    global fail
    print(f"  {'PASS' if ok else 'FAIL'}  {what}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        fail = 1


# (label, size_cv, centre_move_px, span_s, body_px)  — from run10
STATIC = [("track 145", 0.034, 1.2, 129, 500),
          ("track 49",  0.045, 1.2, 146, 500)]
MOVING = [("track 143", 0.116, 63.9, 20, 500),
          ("track 153", 0.128, 69.7, 51, 500),
          ("track 99",  0.144, 79.1, 51, 500),
          ("track 96",  0.191, 310.4, 15, 500)]

print("=" * 74)
print("  phantom gates vs REAL measured tracks")
print("=" * 74)
print(f"  size_cv bar {C.PHANTOM_SIZE_CV}   span bar {C.PHANTOM_MIN_SPAN_S}s   "
      f"jitter bar {C.PHANTOM_CENTRE_JITTER}")
print()


def passes(size_cv, move_px, span_s, body_px):
    """All three gates, as phantom_regions applies them."""
    return (size_cv <= C.PHANTOM_SIZE_CV
            and (move_px / body_px) <= C.PHANTOM_CENTRE_JITTER
            and span_s >= C.PHANTOM_MIN_SPAN_S)


for name, cv, mv, sp, body in STATIC:
    check(passes(cv, mv, sp, body), f"{name} (real statue) IS caught",
          f"cv {cv} · jitter {mv/body:.4f} · span {sp}s")

for name, cv, mv, sp, body in MOVING:
    check(not passes(cv, mv, sp, body), f"{name} (real person) is NOT caught",
          f"cv {cv} · jitter {mv/body:.4f}")

# the thresholds must sit in the GAP, not on top of a population
check(0.045 < C.PHANTOM_SIZE_CV < 0.116,
      "size_cv bar sits inside the measured gap (0.045 .. 0.116)",
      str(C.PHANTOM_SIZE_CV))
check(C.PHANTOM_MIN_SPAN_S <= 129,
      "span bar is reachable by the real statue's track span (129 s)",
      f"{C.PHANTOM_MIN_SPAN_S}s")
check(0.0024 < C.PHANTOM_CENTRE_JITTER < 0.10,
      "jitter bar sits between statue (0.0024) and people (~0.10)",
      str(C.PHANTOM_CENTRE_JITTER))

# and the synthetic value that started this must NOT be able to set the bar
check(C.PHANTOM_SIZE_CV > 0.004 * 3,
      "the bar is NOT derived from the synthetic 0.004 fixture",
      "real statues measure 0.034-0.045, not 0.004")

print()
print("  ALL PASS" if not fail else "  FAILURES ABOVE")
sys.exit(fail)
