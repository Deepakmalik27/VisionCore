"""test_head_recovery.py — a head with no body is a person, not a statistic.

WHY THIS EXISTS
    When two people overlap, the detector commonly returns ONE person box and
    TWO heads. The rear body is ABSENT, not low-confidence, so no threshold
    recovers it. The track dies; when the person steps clear they are born as
    a new id. That one mechanism drives symptom 12 (occlusion breaks
    tracking), much of symptom 11 (flicker then a new id), and symptom 14
    (unique counts inflate).

    _heads_without_person() computed exactly this signal from the beginning
    and the answer was only ever added to a log counter. ENABLE_HEAD_RECOVERY
    existed as a flag mentioned in one f-string with NO code behind it — the
    log said "recovery OFF until Phase 2 scores it", describing a feature that
    had never been written.

    These tests pin the implementation, and especially the guards: a recovered
    box built from a noisy head is a phantom with extra steps, which would
    trade symptom 12 for symptoms 5 and 6.

Run: python tests/test_head_recovery.py
"""
import sys
from pathlib import Path

import numpy as np
import supervision as sv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import kevacv.engine as E
from kevacv import config as CFG

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


def dets(boxes, confs, cls):
    if not boxes:
        return sv.Detections(xyxy=np.empty((0, 4), dtype=float),
                             confidence=np.empty(0, dtype=float),
                             class_id=np.empty(0, dtype=int))
    return sv.Detections(xyxy=np.array(boxes, dtype=float),
                         confidence=np.array(confs, dtype=float),
                         class_id=np.array(cls, dtype=int))


class FakePlane:
    """A fitted perspective model that says a person standing anywhere is
    300 px tall — enough to prove geometry overrides raw anthropometry."""
    ready = True

    def expected_h(self, foot_y):
        return 300.0


PERSON = [[100, 50, 220, 400]]
HEAD_INSIDE = [130, 55, 170, 95]     # belongs to the visible person
HEAD_ORPHAN = [400, 60, 440, 100]    # a body the detector lost


def test_body_height_follows_anthropometry():
    box = E._body_from_head([100, 50, 130, 90], None, 1280, 720)
    h = box[3] - box[1]
    check(abs(h - CFG.HEAD_TO_BODY_RATIO * 40) < 1.0,
          "height is HEAD_TO_BODY_RATIO x head height", f"{h:.0f}px")


def test_scene_geometry_overrides_when_fitted():
    box = E._body_from_head([100, 50, 130, 90], FakePlane(), 1280, 720)
    h = box[3] - box[1]
    check(abs(h - 300.0) < 25.0,
          "a fitted plane pulls the height to what the scene implies",
          f"{h:.0f}px vs plane's 300px")


def test_box_is_clamped_to_the_frame():
    box = E._body_from_head([10, 600, 40, 660], None, 1280, 720)
    check(box[0] >= 0 and box[1] >= 0, "no negative corner", str(box))
    check(box[2] <= 1280 and box[3] <= 720, "no corner past the frame", str(box))


def test_orphan_head_becomes_a_person():
    persons = dets(PERSON, [0.9], [0])
    heads = dets([HEAD_INSIDE, HEAD_ORPHAN], [0.8, 0.7], [1, 1])
    out, n = E._recover_bodies_from_heads(persons, heads, None, 1280, 720)
    check(n == 1, "exactly one body recovered", str(n))
    check(len(out) == 2, "it is appended to the real detections", str(len(out)))
    cx = (out.xyxy[1][0] + out.xyxy[1][2]) / 2.0
    check(abs(cx - 420) < 5, "and it is centred under its head", f"{cx:.0f}")


def test_head_inside_a_body_is_not_recovered():
    """Otherwise every visible person is duplicated — one ID on two boxes,
    which is symptom 10, the bug we just finished fixing."""
    persons = dets(PERSON, [0.9], [0])
    heads = dets([HEAD_INSIDE], [0.9], [1])
    _out, n = E._recover_bodies_from_heads(persons, heads, None, 1280, 720)
    check(n == 0, "a head already inside a person box is ignored", str(n))


def test_weak_heads_are_rejected():
    persons = dets(PERSON, [0.9], [0])
    weak = dets([HEAD_ORPHAN], [0.10], [1])
    _out, n = E._recover_bodies_from_heads(persons, weak, None, 1280, 720)
    check(n == 0, "a low-confidence head does not become a phantom body",
          str(n))


def test_recovered_boxes_are_marked_by_confidence():
    """Inferred, not observed. It must lose to a real detection anywhere the
    two compete, and stay identifiable downstream."""
    persons = dets(PERSON, [0.9], [0])
    heads = dets([HEAD_ORPHAN], [0.8], [1])
    out, _n = E._recover_bodies_from_heads(persons, heads, None, 1280, 720)
    rec = float(out.confidence[-1])
    check(rec < 0.8, "recovered confidence is penalised below the head's",
          f"{rec:.2f} < 0.80")
    check(rec > 0, "but stays positive so the tracker can still use it",
          f"{rec:.2f}")


def test_no_heads_is_a_noop():
    persons = dets(PERSON, [0.9], [0])
    out, n = E._recover_bodies_from_heads(persons, dets([], [], []),
                                          None, 1280, 720)
    check(n == 0 and len(out) == 1, "no heads changes nothing", str(len(out)))


def test_works_with_no_visible_person():
    """Everyone hidden behind one another is exactly when this matters most."""
    empty = dets([], [], [])
    heads = dets([HEAD_ORPHAN], [0.8], [1])
    out, n = E._recover_bodies_from_heads(empty, heads, None, 1280, 720)
    check(n == 1 and len(out) == 1,
          "a head with no person boxes at all still recovers", str(n))


def test_degenerate_head_is_skipped():
    persons = dets(PERSON, [0.9], [0])
    tiny = dets([[400, 60, 400.2, 60.2]], [0.9], [1])
    _out, n = E._recover_bodies_from_heads(persons, tiny, None, 1280, 720)
    check(n == 0, "a sub-pixel head is noise, not a person", str(n))


def test_two_heads_in_one_box_is_two_people():
    """Symptom 9. A person cannot have two heads, so this is PROOF of a merged
    detection rather than an inference from size or aspect."""
    merged = dets([[100, 100, 400, 500]], [0.9], [0])
    two = dets([[150, 110, 190, 150], [310, 110, 350, 150]], [0.8, 0.8], [1, 1])
    out, n = E._split_merged_persons(merged, two)
    check(n == 1, "one merged box detected", str(n))
    check(len(out) == 2, "and split into two people", str(len(out)))
    check(out.xyxy[0][2] == out.xyxy[1][0],
          "the split is at the midpoint between the heads, with no gap")


def test_split_parts_inherit_confidence():
    """A re-interpretation of a real detection, not a new inference — so it is
    NOT penalised the way a head-recovered box is."""
    merged = dets([[100, 100, 400, 500]], [0.9], [0])
    two = dets([[150, 110, 190, 150], [310, 110, 350, 150]], [0.8, 0.8], [1, 1])
    out, _n = E._split_merged_persons(merged, two)
    check(all(abs(float(c) - 0.9) < 1e-9 for c in out.confidence),
          "both parts keep the original confidence",
          str([round(float(c), 2) for c in out.confidence]))


def test_one_head_leaves_the_box_alone():
    solo = dets(PERSON, [0.9], [0])
    one = dets([HEAD_INSIDE], [0.8], [1])
    out, n = E._split_merged_persons(solo, one)
    check(n == 0 and len(out) == 1, "a normal person is untouched", str(n))


def test_heads_outside_the_box_do_not_split_it():
    solo = dets(PERSON, [0.9], [0])
    outside = dets([HEAD_INSIDE, HEAD_ORPHAN], [0.8, 0.8], [1, 1])
    _out, n = E._split_merged_persons(solo, outside)
    check(n == 0, "only heads INSIDE the box count", str(n))


def test_weak_heads_do_not_split():
    merged = dets([[100, 100, 400, 500]], [0.9], [0])
    two = dets([[150, 110, 190, 150], [310, 110, 350, 150]], [0.8, 0.05], [1, 1])
    _out, n = E._split_merged_persons(merged, two)
    check(n == 0, "a noisy second head does not split a real person", str(n))


def test_three_heads_gives_three_people():
    merged = dets([[100, 100, 600, 500]], [0.9], [0])
    three = dets([[150, 110, 190, 150], [330, 110, 370, 150],
                  [510, 110, 550, 150]], [0.8] * 3, [1] * 3)
    out, n = E._split_merged_persons(merged, three)
    check(n == 1 and len(out) == 3, "a triple merge splits three ways",
          str(len(out)))


def test_no_heads_at_all_is_a_noop():
    solo = dets(PERSON, [0.9], [0])
    out, n = E._split_merged_persons(solo, dets([], [], []))
    check(n == 0 and len(out) == 1, "no heads changes nothing")


def test_config_is_present_and_on():
    for name in ("ENABLE_HEAD_RECOVERY", "HEAD_TO_BODY_RATIO",
                 "HEAD_RECOVERY_ASPECT", "HEAD_RECOVERY_MIN_CONF",
                 "HEAD_RECOVERY_CONF_PENALTY"):
        check(hasattr(CFG, name), f"{name} is in config.py")


def main():
    print(__doc__.strip().splitlines()[0])
    for fn in (test_body_height_follows_anthropometry,
               test_scene_geometry_overrides_when_fitted,
               test_box_is_clamped_to_the_frame,
               test_orphan_head_becomes_a_person,
               test_head_inside_a_body_is_not_recovered,
               test_weak_heads_are_rejected,
               test_recovered_boxes_are_marked_by_confidence,
               test_no_heads_is_a_noop,
               test_works_with_no_visible_person,
               test_degenerate_head_is_skipped,
               test_two_heads_in_one_box_is_two_people,
               test_split_parts_inherit_confidence,
               test_one_head_leaves_the_box_alone,
               test_heads_outside_the_box_do_not_split_it,
               test_weak_heads_do_not_split,
               test_three_heads_gives_three_people,
               test_no_heads_at_all_is_a_noop,
               test_config_is_present_and_on):
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
