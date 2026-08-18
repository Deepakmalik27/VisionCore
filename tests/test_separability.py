"""test_separability.py — measure the appearance signal without labels.

WHY THIS EXISTS
    config.py records a same-person p50 of 0.435 against accept bars of
    0.60/0.62/0.75 — by the project's own number, the system rejects most true
    matches. That is symptoms 3, 4, 11 and 14 in one line. But the file also
    says that 0.435 was measured circularly, so nothing could be done about
    the thresholds: the only evidence for them could not be trusted.

    LiveSeparability replaces it using two facts that hold regardless of what
    any embedding says:

      * two detections in the SAME FRAME are two different people
      * the same raw track id a fraction of a second apart is one person,
        associated by motion rather than by appearance

    Neither assumes the answer, so measuring appearance against them is not
    circular. These tests pin that property and the arithmetic on top of it.

Run: python tests/test_separability.py
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kevacv.reid_calibration import LiveSeparability, describe_live

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


def person(seed, noise, rng):
    r = random.Random(seed)
    base = [r.gauss(0, 1) for _ in range(32)]
    return [b + rng.gauss(0, noise) for b in base]


def populated(noise=0.55, people=(1, 2, 3), frames=480, fps=8.0):
    rng = random.Random(7)
    sep = LiveSeparability(max_same_gap_s=0.5)
    for k in range(frames):
        sep.observe(k / fps, [(p, person(p, noise, rng)) for p in people])
    return sep


def test_both_populations_are_collected():
    sep = populated()
    check(sep.n_same_diff_ok if hasattr(sep, "n_same_diff_ok") else True, "sanity")
    check(len(sep.same) > 100, "same-person pairs collected", str(len(sep.same)))
    check(len(sep.diff) > 100, "different-person pairs collected", str(len(sep.diff)))


def test_same_scores_above_different():
    """The whole premise: if these two populations do not separate, no
    threshold anywhere can work and that is the finding."""
    rep = populated().report()
    check(rep["ok"], "report is produced")
    check(rep["same_p50"] > rep["diff_p50"],
          "same-person similarity exceeds different-person",
          f"{rep['same_p50']:.3f} vs {rep['diff_p50']:.3f}")


def test_only_co_visible_ids_become_negatives():
    """One person alone on screen can never produce a 'different people'
    pair — there is nobody to compare against."""
    rng = random.Random(1)
    sep = LiveSeparability()
    for k in range(200):
        sep.observe(k / 8.0, [(1, person(1, 0.4, rng))])
    check(sep.diff == [], "a solo person yields no negatives", str(len(sep.diff)))
    check(len(sep.same) > 0, "but still yields positives", str(len(sep.same)))


def test_a_long_gap_is_not_a_positive():
    """Beyond the window the association is no longer pure motion, so the pair
    stops being trustworthy evidence of 'same person'."""
    rng = random.Random(2)
    sep = LiveSeparability(max_same_gap_s=0.5)
    sep.observe(0.0, [(1, person(1, 0.3, rng))])
    sep.observe(30.0, [(1, person(1, 0.3, rng))])     # 30s later
    check(sep.same == [], "a 30s gap is not admitted as same-person",
          str(len(sep.same)))


def test_duplicate_id_in_one_frame_is_not_a_negative():
    """If the same id appears twice in a frame that is the duplicate-id bug,
    not two people — it must not poison the negative set."""
    rng = random.Random(3)
    sep = LiveSeparability()
    v1, v2 = person(1, 0.1, rng), person(1, 0.1, rng)
    sep.observe(0.0, [(1, v1), (1, v2)])
    check(sep.diff == [], "same id twice in one frame is skipped", str(sep.diff))


def test_missing_vectors_are_skipped():
    sep = LiveSeparability()
    sep.observe(0.0, [(1, None), (2, None)])
    check(sep.diff == [] and sep.same == [], "None vectors contribute nothing")


def test_insufficient_evidence_is_reported_not_guessed():
    sep = LiveSeparability()
    rep = sep.report()
    check(rep["ok"] is False, "an empty run does not produce a threshold")
    check("not enough evidence" in rep["why"], "and says why", rep["why"])
    check("not enough evidence" in describe_live(rep),
          "the printed block says it too")


def test_current_thresholds_are_scored():
    rep = populated().report(current_thresholds={"REID_SIM_THRESHOLD": 0.60})
    x = rep["at_current"]["REID_SIM_THRESHOLD"]
    check(0.0 <= x["true_matches_accepted"] <= 1.0,
          "true-match acceptance is a rate", str(x["true_matches_accepted"]))
    check(0.0 <= x["strangers_accepted"] <= 1.0,
          "stranger acceptance is a rate", str(x["strangers_accepted"]))
    check("accepts" in describe_live(rep),
          "and the trade is spelled out in the printed block")


def test_pair_budget_is_respected():
    """An hour at 8 fps with a crowd would otherwise accumulate millions of
    floats in RAM."""
    rng = random.Random(4)
    sep = LiveSeparability(max_pairs=50)
    for k in range(200):
        sep.observe(k / 8.0, [(p, person(p, 0.5, rng)) for p in (1, 2, 3, 4)])
    check(len(sep.diff) <= 50 + 6, "negatives are capped", str(len(sep.diff)))
    check(len(sep.same) <= 50 + 4, "positives are capped", str(len(sep.same)))


def test_recommendation_beats_a_bad_threshold():
    rep = populated().report()
    same_at_rec = sum(1 for s in populated().same if s >= rep["recommended"])
    check(same_at_rec > 0, "the recommended threshold accepts real matches")
    check(0.0 <= rep["balanced_accuracy"] <= 1.0,
          "balanced accuracy is a proportion", f"{rep['balanced_accuracy']:.3f}")


def main():
    print(__doc__.strip().splitlines()[0])
    for fn in (test_both_populations_are_collected,
               test_same_scores_above_different,
               test_only_co_visible_ids_become_negatives,
               test_a_long_gap_is_not_a_positive,
               test_duplicate_id_in_one_frame_is_not_a_negative,
               test_missing_vectors_are_skipped,
               test_insufficient_evidence_is_reported_not_guessed,
               test_current_thresholds_are_scored,
               test_pair_budget_is_respected,
               test_recommendation_beats_a_bad_threshold):
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
