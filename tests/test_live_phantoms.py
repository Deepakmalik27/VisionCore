"""test_live_phantoms.py — kill the plant before it takes an id, not after.

WHY THIS EXISTS
    phantom_regions() already finds static locations, and drop_tracks() then
    removes them from events, crossings and the video together. The final
    numbers are clean. So why suppress live at all?

    Because a phantom is not inert while it waits to be deleted. For its whole
    life it holds a canonical id — and co-visibility then BLOCKS a real person
    from resolving to that id, pushing them onto a fresh one. Deleting the
    plant at the end of the chunk cannot undo an id break it caused at 19:14.
    That is symptom 7 causing symptoms 4 and 11.

    The danger is the mirror image: suppress too eagerly and the receptionist
    standing still at the desk is deleted, which is far worse than the plant.
    Hence per-zone patience, and hence the tests below.

Run: python tests/test_live_phantoms.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kevacv.phantoms import OnlineStaticSuppressor

FAILED = []

DOORWAY_X, DESK_X = 100, 800
PLANT = (100, 300, 160, 500)
PERSON_AT_DESK = (800, 300, 860, 500)


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


def life_at(x, y):
    """Doorway 30 s, desk 240 s — the policy from STATIC_MIN_LIFE_BY_ROLE."""
    return 30.0 if x < 500 else 240.0


def sup(**kw):
    kw.setdefault("min_life_for", life_at)
    kw.setdefault("min_hits", 15)
    return OnlineStaticSuppressor((1280, 720), **kw)


def run(s, boxes, seconds, fps=8.0):
    """-> {box_index: first time it was dropped}"""
    dropped = {}
    for i in range(int(seconds * fps)):
        t = i / fps
        keep = s.filter_boxes(t, boxes)
        for j, k in enumerate(keep):
            if not k and j not in dropped:
                dropped[j] = t
    return dropped


def test_doorway_phantom_dies_at_its_patience():
    s = sup()
    d = run(s, [PLANT], 60)
    check(0 in d, "a rigid box in the doorway is suppressed")
    check(28 <= d[0] <= 32, "at ~30s, the doorway patience", f"t={d.get(0)}")


def test_person_at_the_desk_survives():
    """The regression that matters most. A receptionist holding position for a
    minute must not be deleted."""
    s = sup()
    d = run(s, [PERSON_AT_DESK], 60)
    check(0 not in d, "a rigid box at the desk is NOT suppressed in 60s",
          str(d))


def test_both_at_once_are_judged_separately():
    s = sup()
    d = run(s, [PLANT, PERSON_AT_DESK], 60)
    check(0 in d and 1 not in d,
          "the plant goes, the person stays, in the same frames", str(d))


def test_a_moving_box_is_never_suppressed():
    """The whole predicate is rigidity. Anything that walks must pass."""
    s = sup()
    dropped = False
    for i in range(int(60 * 8)):
        t = i / 8.0
        x = 100 + i * 2          # walking across the doorway
        if not s.filter_boxes(t, [(x, 300, x + 60, 500)])[0]:
            dropped = True
    check(not dropped, "a box that moves is never suppressed")


def test_quiet_cell_is_forgotten():
    """A person stands still, leaves, and hours later someone else stands in
    the same spot. That is two people, not one continuous rigid object."""
    s = sup(forget_after_s=10.0)
    for i in range(int(20 * 8)):          # 20s of stillness, under patience
        s.filter_boxes(i / 8.0, [PLANT])
    # long silence, then the location is occupied again
    d = run(s, [PLANT], 20)                # only 20s more
    check(0 not in d,
          "history resets after a gap, so 20s + 20s is not 40s of evidence",
          str(d))


def test_min_hits_guards_a_sparse_location():
    """Long span but few sightings is not evidence — it is a flicker."""
    s = sup(min_hits=100)
    d = run(s, [PLANT], 60)               # 480 frames, but require 100 hits
    check(0 in d, "enough hits still suppresses", str(d))
    s2 = sup(min_hits=100000)
    d2 = run(s2, [PLANT], 60)
    check(0 not in d2, "too few hits does not", str(d2))


def test_default_patience_without_a_policy():
    s = OnlineStaticSuppressor((1280, 720), default_life_s=10.0, min_hits=15)
    d = run(s, [PLANT], 30)
    check(0 in d and d[0] >= 10, "falls back to default_life_s", str(d))


def test_counts_and_description():
    s = sup()
    run(s, [PLANT], 60)
    check(s.n_suppressed > 0, "dropped detections are counted",
          str(s.n_suppressed))
    check("furniture" in s.describe(),
          "and the log explains what was suppressed and why")
    check("nothing suppressed" in sup().describe(),
          "a clean chunk says so plainly")


def main():
    print(__doc__.strip().splitlines()[0])
    for fn in (test_doorway_phantom_dies_at_its_patience,
               test_person_at_the_desk_survives,
               test_both_at_once_are_judged_separately,
               test_a_moving_box_is_never_suppressed,
               test_quiet_cell_is_forgotten,
               test_min_hits_guards_a_sparse_location,
               test_default_patience_without_a_policy,
               test_counts_and_description):
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
