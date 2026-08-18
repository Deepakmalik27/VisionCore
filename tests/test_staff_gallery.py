"""test_staff_gallery.py — two photos of one person are ONE staff member.

WHY THIS EXISTS
    The loader used the whole filename as the identity. The real gallery
    contains Staff2.png AND Staff2-same.png — plainly two shots of one person
    — and that enrolled them as TWO staff members. Every later "one body, one
    name" rule then had to arbitrate between identities that were never
    distinct, and the same person could be matched under either name from
    frame to frame. Symptom 2, and part of symptom 3.

    Fixing it also makes the gallery stronger rather than merely correct: a
    person may now have several reference shots, and matching takes the best
    one. That matters here because faces appear on only ~2% of tracks, so
    every extra reference is real recall.

Run: python tests/test_staff_gallery.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import kevacv.engine as E
from kevacv.pipeline import staff_gallery_findings

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


def test_the_real_gallery_groups_correctly():
    """The actual files that shipped — this is the bug, not a fixture."""
    check(E.staff_name_from_filename("staff_gallery/Staff2.png") == "staff2",
          "Staff2.png -> staff2")
    check(E.staff_name_from_filename("staff_gallery/Staff2-same.png") == "staff2",
          "Staff2-same.png -> staff2 (SAME person, was a second identity)")


def test_shot_suffixes():
    for name, want in (("priya.jpg", "priya"),
                       ("priya-2.jpg", "priya"),
                       ("priya_3.png", "priya"),
                       ("priya_same.png", "priya"),
                       ("priya-alt.jpg", "priya"),
                       ("priya-2nd.jpg", "priya")):
        got = E.staff_name_from_filename(f"staff_gallery/{name}")
        check(got == want, f"{name} -> {want}", got)


def test_a_real_hyphenated_name_is_not_split():
    """The suffix rule must not eat half of Anna-Maria."""
    got = E.staff_name_from_filename("staff_gallery/Anna-Maria.jpg")
    check(got == "anna-maria", "Anna-Maria.jpg stays whole", got)


def test_folder_per_person_wins():
    """The unambiguous form, for names the suffix rule could misread."""
    got = E.staff_name_from_filename("staff_gallery/anna-maria/shot1.jpg")
    check(got == "anna-maria", "a folder names the person", got)


def test_case_is_normalised():
    check(E.staff_name_from_filename("g/PRIYA.jpg")
          == E.staff_name_from_filename("g/priya.jpg"),
          "case does not create a second identity")


def test_multi_shot_match_takes_the_best():
    entry = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    check(abs(E._staff_gallery_sim([1.0, 0.0, 0.0], entry) - 1.0) < 1e-9,
          "the matching shot decides, not the average")


def test_legacy_flat_vector_still_works():
    """An older cached gallery must not crash deep inside the frame loop."""
    got = E._staff_gallery_sim([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    check(abs(got - 1.0) < 1e-9, "a single flat vector is accepted", str(got))


def test_empty_entry_is_not_a_match():
    check(E._staff_gallery_sim([1.0, 0.0, 0.0], []) < 0,
          "no shots enrolled is never a match")


def test_preflight_sees_the_real_gallery():
    out = staff_gallery_findings(ROOT / "staff_gallery")
    check(out and out[0][0] == "INFO",
          "the shipped gallery passes preflight", str(out[0][0]))


def test_preflight_flags_an_empty_gallery():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        out = staff_gallery_findings(d)
        check(out and out[0][0] == "ERROR",
              "an empty gallery is an ERROR, not silence", str(out))
        check("entirely disabled" in out[0][1],
              "and says the whole face path is off")


def main():
    print(__doc__.strip().splitlines()[0])
    for fn in (test_the_real_gallery_groups_correctly,
               test_shot_suffixes,
               test_a_real_hyphenated_name_is_not_split,
               test_folder_per_person_wins,
               test_case_is_normalised,
               test_multi_shot_match_takes_the_best,
               test_legacy_flat_vector_still_works,
               test_empty_entry_is_not_a_match,
               test_preflight_sees_the_real_gallery,
               test_preflight_flags_an_empty_gallery):
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
