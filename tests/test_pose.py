"""Pose is a layer on top of identity, and must never decide identity."""
import math
from kevacv.pose import (classify, select_tracks, timeline,
                         STANDING, WALKING, BENDING, SITTING, UNKNOWN)


def _kps(sh=(100, 200), hip=(100, 400), knee=(100, 600), ank=(100, 800)):
    k = [None] * 17
    k[5] = (sh[0] - 20, sh[1]); k[6] = (sh[0] + 20, sh[1])
    k[11] = (hip[0] - 15, hip[1]); k[12] = (hip[0] + 15, hip[1])
    if knee: k[13] = (knee[0] - 15, knee[1]); k[14] = (knee[0] + 15, knee[1])
    if ank: k[15] = (ank[0] - 15, ank[1]); k[16] = (ank[0] + 15, ank[1])
    return k


def test_upright_and_still_is_standing():
    assert classify(_kps())["activity"] == STANDING


def test_upright_and_moving_is_walking():
    assert classify(_kps(), foot_move_px=40)["activity"] == WALKING


def test_a_bent_torso_is_bending():
    # hips well to the side of the shoulders = leaning over
    assert classify(_kps(sh=(100, 200), hip=(400, 300)))["activity"] == BENDING


def test_knees_level_with_hips_is_sitting():
    assert classify(_kps(hip=(100, 400), knee=(100, 420)))["activity"] == SITTING


def test_missing_keypoints_report_unknown_not_a_guess():
    k = [None] * 17
    r = classify(k)
    assert r["activity"] == UNKNOWN and "not visible" in r["why"]


def test_every_answer_carries_its_reason():
    for r in (classify(_kps()), classify(_kps(), foot_move_px=40),
              classify(_kps(sh=(100, 200), hip=(400, 300)))):
        assert r["why"]


def test_only_long_lived_tracks_are_selected():
    tracks = {1: [(0.0,), (5.0,)], 2: [(0.0,), (0.5,)], 3: [(0.0,), (30.0,)]}
    sel = select_tracks(tracks, max_tracks=8, min_seconds=2.0)
    assert sel == [3, 1], "shortest-lived cannot support an activity claim"


def test_the_compute_budget_is_respected():
    tracks = {i: [(0.0,), (float(i) + 3,)] for i in range(20)}
    assert len(select_tracks(tracks, max_tracks=5)) == 5


def test_one_bad_frame_does_not_become_an_activity_change():
    samples = [{"t": 0, "activity": STANDING}, {"t": 1, "activity": STANDING},
               {"t": 2, "activity": WALKING},                 # single blip
               {"t": 3, "activity": STANDING}, {"t": 4, "activity": STANDING}]
    spans = timeline(samples, min_run=2)
    assert [s["activity"] for s in spans] == [STANDING], \
        "never make an important decision from one frame"


def test_a_sustained_change_IS_reported():
    samples = ([{"t": i, "activity": STANDING} for i in range(3)]
               + [{"t": 3 + i, "activity": WALKING} for i in range(4)])
    spans = timeline(samples, min_run=2)
    assert [s["activity"] for s in spans] == [STANDING, WALKING]
