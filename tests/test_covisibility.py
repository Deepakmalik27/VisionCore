"""test_covisibility.py — one canonical id may never belong to two raw tracks.

WHY THIS EXISTS
    A person cannot be two boxes. ENABLE_COVISIBILITY_BLOCK enforces that when
    a raw track is BORN, by refusing to merge a birth into any canonical id
    already visible under a different raw id that frame.

    Two holes made it leak anyway, and both showed up in review as the same
    symptom: one id label drawn on two different people at once.

      1. IN-FRAME. process_video seeded the blocked set once, before resolving
         any detection. Two raw ids born on the SAME frame are both absent from
         raw_to_canon, so neither seeded the set — the first could take canon C
         and the second was still handed a C-free blocked set.

      2. ACROSS FRAMES. Raw A binds to C while B is off-screen; later B binds
         to C while A is off-screen. Both bindings were legal when made. When A
         and B are finally visible together, resolve() short-circuits on
         `raw_tid in raw_to_canon` for both and the duplicate is permanent.

    Hole 1 is fixed in engine.process_video (the set is kept live). Hole 2 is
    fixed by _IdentityMemory.split_duplicate_raws, which this file pins.

Run: python tests/test_covisibility.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kevacv.analytics import _IdentityMemory

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


def _mem():
    """Memory with a null embedder — these cases are about the raw->canonical
    bookkeeping, which runs identically whether or not appearance is available.
    """
    return _IdentityMemory(embed_fn=lambda crop: None, memory_ttl_s=1e9)


def test_no_duplicate_survives():
    m = _mem()
    # A binds to canon "C" at t=0, B binds to the same canon at t=10 — the
    # across-frames hole, constructed directly.
    m.raw_to_canon = {"A": "C", "B": "C"}
    m.raw_bound_t = {"A": 0.0, "B": 10.0}

    evicted = m.split_duplicate_raws(["A", "B"], now_t=20.0)

    check(m.raw_to_canon["A"] == "C", "incumbent (earliest binding) keeps the id",
          f"A -> {m.raw_to_canon['A']}")
    check(m.raw_to_canon["B"] == "B", "later claimant is re-minted onto its own id",
          f"B -> {m.raw_to_canon['B']}")
    check(evicted == [("B", "C")], "the eviction is reported to the caller",
          str(evicted))
    check(m.duplicate_splits == 1, "the split is counted for the run log",
          str(m.duplicate_splits))
    check(len(set(m.raw_to_canon.values())) == 2,
          "no canonical id is claimed twice afterwards")


def test_incumbent_is_earliest_not_first_seen():
    """Ordering must come from raw_bound_t, not from the order the ids happen
    to arrive in the detection array — that order is confidence-ranked and
    changes frame to frame."""
    m = _mem()
    m.raw_to_canon = {"late": "C", "early": "C"}
    m.raw_bound_t = {"late": 99.0, "early": 1.0}

    m.split_duplicate_raws(["late", "early"], now_t=100.0)   # worst-case order

    check(m.raw_to_canon["early"] == "C", "earliest binding wins regardless of arg order")
    check(m.raw_to_canon["late"] == "late", "later binding is evicted")


def test_three_way_leaves_one_holder():
    m = _mem()
    m.raw_to_canon = {"A": "C", "B": "C", "D": "C"}
    m.raw_bound_t = {"A": 0.0, "B": 5.0, "D": 9.0}

    evicted = m.split_duplicate_raws(["A", "B", "D"], now_t=20.0)

    check(len(evicted) == 2, "two of three are evicted", str(evicted))
    check(m.raw_to_canon["A"] == "C", "only the incumbent still holds C")
    check(sorted(m.raw_to_canon.values()) == ["B", "C", "D"],
          "every survivor has a distinct id", str(m.raw_to_canon))


def test_noop_when_already_distinct():
    """The common case is every id already distinct — it must cost nothing and
    change nothing, or this runs on every frame and rewrites healthy state."""
    m = _mem()
    m.raw_to_canon = {"A": "C", "B": "D"}
    m.raw_bound_t = {"A": 0.0, "B": 1.0}
    before = dict(m.raw_to_canon)

    evicted = m.split_duplicate_raws(["A", "B"], now_t=5.0)

    check(evicted == [], "nothing evicted when ids are already distinct")
    check(m.raw_to_canon == before, "bindings untouched")
    check(m.duplicate_splits == 0, "counter not incremented")


def test_ignores_raw_ids_not_in_frame():
    """A duplicate that is NOT co-visible is not a contradiction — one of the
    two may simply be a stale binding for someone who left. Only ids present
    this frame are passed in, and only those may be split."""
    m = _mem()
    m.raw_to_canon = {"A": "C", "gone": "C"}
    m.raw_bound_t = {"A": 0.0, "gone": 10.0}

    evicted = m.split_duplicate_raws(["A"], now_t=20.0)   # 'gone' not visible

    check(evicted == [], "an absent duplicate holder is left alone")
    check(m.raw_to_canon["gone"] == "C", "its binding survives for a later re-appearance")


def test_unbound_raw_ids_are_skipped():
    m = _mem()
    m.raw_to_canon = {"A": "C"}
    m.raw_bound_t = {"A": 0.0}

    evicted = m.split_duplicate_raws(["A", "brand_new"], now_t=3.0)

    check(evicted == [], "a raw id with no binding yet is not a duplicate")
    check("brand_new" not in m.raw_to_canon, "and is not given one here")


def main():
    print(__doc__.strip().splitlines()[0])
    for fn in (test_no_duplicate_survives,
               test_incumbent_is_earliest_not_first_seen,
               test_three_way_leaves_one_holder,
               test_noop_when_already_distinct,
               test_ignores_raw_ids_not_in_frame,
               test_unbound_raw_ids_are_skipped):
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
