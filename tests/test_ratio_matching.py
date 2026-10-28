"""test_ratio_matching.py — decide by MARGIN, not by an absolute bar.

WHY THIS EXISTS
    The live matcher accepted the first identity whose cosine cleared 0.62.
    The run measured, on this camera:

        same-person      p10 0.332   p50 0.461
        different-person p50 0.352   p90 0.538

    Those distributions OVERLAP. No single number separates them — 0.60
    rejects most true matches, 0.37 admits strangers. Picking a threshold only
    picks which error to make.

    And 45% of the footage is infrared, where every cosine is globally lower.
    An absolute bar is therefore too strict at night and too loose by day AT
    THE SAME TIME, which is exactly the reported "ids change too easily".

    Lowe's ratio test asks a relative question: is the best candidate
    decisively better than the runner-up? A constant factor applied to every
    similarity — which is what a modality change does — leaves that comparison
    unchanged. That invariance is the property these tests pin.

Run: python tests/test_ratio_matching.py
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


def unit(*v):
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / n for x in v]


def mem(**kw):
    kw.setdefault("memory_ttl_s", 1e9)
    kw.setdefault("max_dist_px", 1e9)
    m = _IdentityMemory(embed_fn=lambda c: None, **kw)
    return m


def seed(m, entries, t=0.0):
    """entries: {canon: vector} placed at the same spot, so only appearance
    distinguishes them and the physical gates never interfere."""
    for cid, vec in entries.items():
        m.bank[cid] = {"anchor": vec, "pos": (100.0, 100.0), "t": t,
                       "best_score": 1.0}
    return m


A = unit(1.0, 0.0, 0.0)
B = unit(0.0, 1.0, 0.0)


def test_one_clear_candidate_is_accepted():
    m = seed(mem(), {"alice": A})
    got = m.resolve("raw1", None, 0.9, (90, 90, 110, 110), 1.0, vec=A)
    check(got == "alice", "an unambiguous match is taken", str(got))


def test_two_similar_candidates_are_refused():
    """The case a threshold gets wrong: BOTH clear any bar, so it picks one
    arbitrarily. Ambiguity should mint a new id, not guess."""
    near = unit(0.99, 0.14, 0.0)      # very close to A
    m = seed(mem(), {"alice": A, "alice_twin": near})
    got = m.resolve("raw1", None, 0.9, (90, 90, 110, 110), 1.0, vec=A)
    check(got == "raw1", "two near-identical candidates -> a NEW id, not a guess",
          str(got))
    check(m.ratio_rejects == 1, "and it is counted as an ambiguity rejection",
          str(m.ratio_rejects))


def test_a_clear_winner_among_several_is_taken():
    m = seed(mem(), {"alice": A, "bob": B})
    got = m.resolve("raw1", None, 0.9, (90, 90, 110, 110), 1.0, vec=A)
    check(got == "alice", "a decisive winner beats an unrelated rival", str(got))


def test_infrared_shift_does_not_change_the_verdict():
    """THE point. Scale every similarity down, as switching to infrared does.
    A fixed bar flips; a ratio does not."""
    day = seed(mem(), {"alice": A, "bob": B})
    v_day = A
    got_day = day.resolve("raw1", None, 0.9, (90, 90, 110, 110), 1.0, vec=v_day)

    # a vector 60 degrees off A: cosine to alice drops from 1.0 to ~0.5, well
    # under the old 0.62 bar, while still being the clear best candidate
    v_night = unit(0.5, 0.05, 0.86)
    night = seed(mem(), {"alice": A, "bob": B})
    got_night = night.resolve("raw1", None, 0.9, (90, 90, 110, 110), 1.0,
                              vec=v_night)
    check(got_day == "alice", "daylight: matched", str(got_day))
    check(got_night == "alice",
          "infrared-like drop: STILL matched, where 0.62 would have refused",
          str(got_night))


def test_the_floor_still_rejects_nonsense():
    """Relative does not mean credulous: an orthogonal vector with one
    candidate has no runner-up, so the floor has to hold the line."""
    m = seed(mem(), {"alice": A})
    got = m.resolve("raw1", None, 0.9, (90, 90, 110, 110), 1.0, vec=B)
    check(got == "raw1", "an unrelated vector does not match", str(got))


def test_ratio_is_tunable():
    near = unit(0.99, 0.14, 0.0)
    strict = seed(mem(ratio=0.5), {"alice": A, "twin": near})
    loose = seed(mem(ratio=0.999), {"alice": A, "twin": near})
    check(strict.resolve("r", None, 0.9, (90, 90, 110, 110), 1.0, vec=A) == "r",
          "a strict ratio refuses the ambiguous pair")
    check(loose.resolve("r", None, 0.9, (90, 90, 110, 110), 1.0, vec=A) == "alice",
          "a loose ratio accepts it")


def test_covisibility_still_blocks():
    """The relative test must not weaken the hard constraint that one identity
    cannot be two bodies at once."""
    m = seed(mem(), {"alice": A, "bob": B})
    got = m.resolve("raw1", None, 0.9, (90, 90, 110, 110), 1.0, vec=A,
                    blocked_canons={"alice"})
    check(got != "alice", "a blocked canon is never returned", str(got))


def test_known_raw_id_short_circuits():
    m = seed(mem(), {"alice": A})
    m.raw_to_canon["raw1"] = "alice"
    got = m.resolve("raw1", None, 0.9, (90, 90, 110, 110), 5.0, vec=B)
    check(got == "alice", "an already-bound raw id keeps its identity", str(got))


def main():
    print(__doc__.strip().splitlines()[0])
    for fn in (test_one_clear_candidate_is_accepted,
               test_two_similar_candidates_are_refused,
               test_a_clear_winner_among_several_is_taken,
               test_infrared_shift_does_not_change_the_verdict,
               test_the_floor_still_rejects_nonsense,
               test_ratio_is_tunable,
               test_covisibility_still_blocks,
               test_known_raw_id_short_circuits):
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
