"""test_mirror_travel.py — a reflection MOVES. Two statues are not a mirror.

WHY THIS EXISTS
    mirrored_pair_ids promises, in its own docstring, to find tracks that
    "move in lockstep at a fixed offset". The implementation measured only the
    coefficient of variation of the SEPARATION — and never checked that either
    track moved at all.

    Two people standing at a reception counter are a fixed distance apart. CV
    near zero. Flagged as a person and their reflection.

    At a reception that is not a rare case, it is the common one. The first
    real run reported 105 lockstep pairs touching 92 of 231 identities, and
    one track was "mirrored" with two different partners at two different
    distances — which no actual reflection can be, since a reflection is a
    rigid transform of exactly one body.

    The evidence for a reflection is that two bodies TRAVELLED and stayed
    rigidly separated the whole way. Constant separation alone proves nothing.

Run: python tests/test_mirror_travel.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kevacv.detect_filters import mirrored_pair_ids

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


def log(tracks, n=200, fps=8.0):
    """tracks: {id: fn(i) -> (cx, foot_y)} -> a frame_log."""
    out = []
    for i in range(n):
        boxes = [(tid, x - 20, y - 100, x + 20, y)
                 for tid, fn in tracks.items() for x, y in [fn(i)]]
        out.append((i, i / fps, boxes))
    return out


def test_two_stationary_people_are_not_a_mirror():
    """THE regression. A queue at a counter is not a hall of mirrors."""
    pairs = mirrored_pair_ids(log({"A": lambda i: (300, 500),
                                   "B": lambda i: (600, 500)}))
    check(pairs == {}, "two people standing still are not flagged", str(pairs))


def test_a_real_reflection_is_still_caught():
    """The whole point of the filter must survive the fix."""
    pairs = mirrored_pair_ids(log({"A": lambda i: (200 + i * 3, 500),
                                   "B": lambda i: (500 + i * 3, 500)}))
    check(len(pairs) == 1, "a moving lockstep pair is still flagged", str(len(pairs)))
    ev = list(pairs.values())[0]
    check("travel_px" in ev, "and the evidence records how far each travelled",
          str(ev.get("travel_px")))


def test_independent_walkers_are_not_a_mirror():
    pairs = mirrored_pair_ids(log({"A": lambda i: (200 + i * 3, 500),
                                   "B": lambda i: (900 - i * 2, 520)}))
    check(pairs == {}, "two people walking apart are not flagged", str(pairs))


def test_one_moving_one_still_is_not_a_mirror():
    """A reflection cannot stand still while its owner walks."""
    pairs = mirrored_pair_ids(log({"A": lambda i: (200 + i * 3, 500),
                                   "B": lambda i: (900, 500)}))
    check(pairs == {}, "only one of the pair moving is not a mirror", str(pairs))


def test_travel_threshold_is_tunable():
    """A slow drift over a long co-visibility should be reachable by lowering
    the bar, not permanently excluded."""
    slow = log({"A": lambda i: (300 + i * 0.2, 500),
                "B": lambda i: (600 + i * 0.2, 500)})
    check(mirrored_pair_ids(slow, min_travel_px=200.0) == {},
          "a high travel bar rejects a slow drift")
    check(len(mirrored_pair_ids(slow, min_travel_px=10.0)) == 1,
          "a low travel bar accepts it")


def test_short_covisibility_still_rejected():
    """The pre-existing guards must keep working."""
    pairs = mirrored_pair_ids(log({"A": lambda i: (200 + i * 3, 500),
                                   "B": lambda i: (500 + i * 3, 500)}, n=20))
    check(pairs == {}, "too few samples is still rejected", str(pairs))


def main():
    print(__doc__.strip().splitlines()[0])
    for fn in (test_two_stationary_people_are_not_a_mirror,
               test_a_real_reflection_is_still_caught,
               test_independent_walkers_are_not_a_mirror,
               test_one_moving_one_still_is_not_a_mirror,
               test_travel_threshold_is_tunable,
               test_short_covisibility_still_rejected):
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
