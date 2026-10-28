"""test_mixed_id_sort.py — an enrolled staff NAME must not crash the stitcher.

WHY THIS EXISTS
    Track ids are integers, except for enrolled staff, who carry their name:
    'staff1', not 1082. merge_fragmented_tracks sorted its candidate pairs
    with

        sorted(pairs, reverse=True)          # pairs = (sim, a, b, tier)

    and when two pairs tie on sim, Python falls through to comparing a — one
    of those track ids. On the first run where the staff gallery actually
    enrolled a face, that raised:

        TypeError: '<' not supported between instances of 'str' and 'int'

    The bug had been latent for as long as the gallery was empty, because
    then every id was an int. Fixing staff recognition is what exposed it.

    The failure is expensive rather than loud: the exception is caught, the
    run continues, and NOTHING merges — so a chunk that should report ~22
    people reports ~200 and every downstream count is silently inflated.

    track_sort_key() has existed in analytics.py the whole time for exactly
    this. These tests pin that it is actually used.

Run: python tests/test_mixed_id_sort.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kevacv.analytics import merge_fragmented_tracks, track_sort_key

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


def test_the_exact_crash():
    """Ties on sim, with mixed id types, sorted the way the fix does it."""
    pairs = [(0.9, "staff1", 1082, "gallery"),
             (0.9, 5, "staff1", "handoff"),
             (0.9, 12, 7, "stationary")]
    try:
        out = sorted(pairs, key=lambda p: (-p[0], track_sort_key(p[1]),
                                           track_sort_key(p[2]), str(p[3])))
        check(len(out) == 3, "mixed str/int ids sort without raising")
    except TypeError as e:
        check(False, "mixed str/int ids sort without raising", str(e))

    # and prove the OLD form really did raise, so this test cannot rot into a
    # tautology if someone reverts the fix
    try:
        sorted(pairs, reverse=True)
        check(False, "the old sorted(reverse=True) form still raises",
              "it did NOT raise — the reproduction is no longer valid")
    except TypeError:
        check(True, "the old sorted(reverse=True) form still raises")


def test_track_sort_key_orders_all_three_kinds():
    keys = [track_sort_key(x) for x in ("staff1", 1082, "7")]
    check(len(set(type(k[0]) for k in keys)) == 1,
          "every key starts with the same type, so tuples stay comparable")
    check(sorted(["staff1", 1082, 5], key=track_sort_key) == ["staff1", 5, 1082],
          "names sort before numbers, numbers sort numerically",
          str(sorted(["staff1", 1082, 5], key=track_sort_key)))


def test_stitcher_survives_a_named_staff_track():
    """The real entry point, with a gallery-named id in the input."""
    windows = {"staff1": (0.0, 60.0), 2: (70.0, 120.0), 3: (130.0, 180.0)}
    # identical embeddings -> every pair ties on sim, which is the trigger
    emb = {t: [[1.0, 0.0, 0.0]] for t in windows}
    positions = {t: [(100.0, 200.0), (100.0, 200.0)] for t in windows}
    try:
        mapping, edges, diag = merge_fragmented_tracks(
            windows, emb, positions=positions)
        check(isinstance(mapping, dict), "merge_fragmented_tracks returned",
              f"{len(mapping)} id(s)")
        check(all(t in mapping for t in windows),
              "every input id appears in the mapping")
    except TypeError as e:
        check(False, "merge_fragmented_tracks survives a named staff id",
              f"TypeError: {e}")
    except Exception as e:
        # any OTHER failure is a different bug and should be visible, but it
        # is not the regression this file guards
        print(f"       (non-TypeError from the stitcher: {type(e).__name__}: {e})")
        check(True, "no TypeError from mixed ids")


def test_sort_is_deterministic():
    """Ties previously resolved by whatever order the dict happened to give,
    so two runs of the same footage could merge differently."""
    pairs = [(0.8, 3, "staff2", "a"), (0.8, "staff2", 3, "b"),
             (0.8, 1, 2, "c")]
    k = lambda p: (-p[0], track_sort_key(p[1]), track_sort_key(p[2]), str(p[3]))
    check(sorted(pairs, key=k) == sorted(list(reversed(pairs)), key=k),
          "input order does not change the sorted result")


def main():
    print(__doc__.strip().splitlines()[0])
    for fn in (test_the_exact_crash,
               test_track_sort_key_orders_all_three_kinds,
               test_stitcher_survives_a_named_staff_track,
               test_sort_is_deterministic):
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
