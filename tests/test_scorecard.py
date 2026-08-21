"""The scorecard must never turn 'cannot tell' into 'PASS'."""
from kevacv.scorecard import (track_stats, pixel_height, score_windows,
                              verdicts, render, PASS, FAIL, NOTRUTH)


def _tracks():
    # id 1: born and dies mid-scene (fragmentation). id 2: edge to edge (a person).
    return {1: [(0.0, 900, 400, 1000, 600), (5.0, 1100, 400, 1200, 600)],
            2: [(0.0, 0, 400, 80, 600), (5.0, 1840, 400, 1920, 600)]}


def test_fragmentation_is_counted_and_edge_traffic_is_not():
    s = track_stats(_tracks(), 1920, 1080, zone_x=1500.0)
    assert s["born_mid_scene"] == 1 and s["died_mid_scene"] == 1
    assert s["traversed_zone_x"] == 1


def test_a_track_clipped_by_the_window_is_not_blamed():
    s = track_stats(_tracks(), 1920, 1080, t_first=0.0, t_last=5.0)
    assert s["born_mid_scene"] == 0 and s["died_mid_scene"] == 0


def test_network_height_reflects_both_downscales():
    p = pixel_height({1: [(0.0, 0, 0, 100, 600)]}, analysis_w=1920,
                     source_w=3840, imgsz=1280)
    assert abs(p["scale_total"] - 1 / 3) < 1e-6
    assert abs(p["network_px_median"] - 400.0) < 1e-6   # 600 analysis px * 2/3


def test_no_ground_truth_is_NOT_a_pass():
    card = {"entry_score": score_windows([], eval_dir="does_not_exist")}
    v = {r["check"]: r["state"] for r in verdicts(card)}
    assert v["entry line vs truth"] == NOTRUTH, \
        "an unmeasurable check must never report PASS"


def test_impossible_ground_plane_fails():
    card = {"ground_plane": {"ok": True, "camera_h_m": 5.51, "horizon_row": -429}}
    v = {r["check"]: r["state"] for r in verdicts(card)}
    assert v["ground plane"] == FAIL


def test_plausible_ground_plane_passes():
    card = {"ground_plane": {"ok": True, "camera_h_m": 2.9, "horizon_row": 240}}
    v = {r["check"]: r["state"] for r in verdicts(card)}
    assert v["ground plane"] == PASS


def test_tuned_window_is_never_summed_into_heldout():
    import json, os, tempfile
    d = tempfile.mkdtemp()
    json.dump({"window_s": [305.0, 318.0], "truth_count": 6},
              open(os.path.join(d, "gt_entries_305_318.json"), "w"))
    json.dump({"window_s": [26.0, 39.0], "kind": "quiet", "truth_count": 0},
              open(os.path.join(d, "gt_entries_026_039.json"), "w"))
    cross = [{"t": 310.0, "line": "entry line", "direction": "in"}]
    s = score_windows(cross, eval_dir=d)
    assert s["tuned_truth"] == 6 and s["tuned_got"] == 1
    assert s["heldout_truth"] == 0 and s["heldout_got"] == 0
    v = {r["check"]: r["state"] for r in verdicts({"entry_score": s})}
    assert v["entry line vs truth"] == NOTRUTH, \
        "a tuned-only score must not be reported as evidence"


def test_render_does_not_crash_on_a_sparse_card():
    assert "RUN SCORECARD" in render({"build": {}})


def test_dominant_funnel_stage_fails():
    card = {"funnel": {"person/head split": 65.5, "dedup NMS": 0.0}}
    v = {r["check"]: r["state"] for r in verdicts(card)}
    assert v["detector funnel"] == FAIL


def test_labels_from_another_chunk_are_never_scored():
    """The scorecard once scored a 7.30pm run against 6.30pm labels and
    reported held-out 1/3. Truth must be bound to the footage it came from."""
    import json, os, tempfile
    from kevacv.scorecard import score_windows, verdicts, NOTRUTH
    d = tempfile.mkdtemp()
    json.dump({"window_s": [221.0, 234.0], "kind": "busy", "truth_count": 3,
               "source_chunk": "6.30.00pm"},
              open(os.path.join(d, "gt_entries_221_234.json"), "w"))
    cross = [{"t": 225.0, "line": "entry line", "direction": "in"}]

    right = score_windows(cross, eval_dir=d, video="CAM.112 ... 6.30.00pm ....mp4")
    assert right["heldout_truth"] == 3 and right["heldout_got"] == 1

    wrong = score_windows(cross, eval_dir=d, video="CAM.112 ... 7.30.00pm ....mp4")
    assert wrong["heldout_truth"] == 0 and wrong["heldout_got"] == 0
    assert len(wrong["skipped"]) == 1
    assert verdicts({"entry_score": wrong})[0]["state"] == NOTRUTH


def _door_scene():
    """A doorway polygon mid-frame, plus two tracks that end inside it."""
    door = [(800, 800), (1200, 800), (1200, 1080), (800, 1080)]
    tracks = {
        1: [(0.0, 0, 400, 100, 600), (5.0, 900, 800, 1000, 900)],   # edge -> doorway
        2: [(0.0, 0, 400, 100, 600), (5.0, 500, 400, 600, 600)],    # edge -> open floor
    }
    return tracks, {"dining": door}


def test_a_track_ending_in_a_doorway_is_not_a_tracker_failure():
    from kevacv.scorecard import track_stats
    tracks, zones = _door_scene()
    s = track_stats(tracks, 1920, 1080, zones=zones)
    assert s["died_mid_scene"] == 2, "both end mid-frame"
    assert s["died_by_zone"].get("dining") == 1
    assert s["died_open_floor"] == 1, "only the open-floor one is unexplained"


def test_verdict_judges_open_floor_not_the_aggregate():
    from kevacv.scorecard import track_stats, verdicts, PASS
    # 10 tracks, all dying in the doorway: aggregate 100%, unexplained 0%.
    door = [(800, 800), (1200, 800), (1200, 1080), (800, 1080)]
    # born at the FRAME EDGE (a person walking in), dying in the doorway
    tracks = {i: [(0.0, 0, 400, 100, 600), (5.0, 900, 800, 1000, 900)]
              for i in range(10)}
    s = track_stats(tracks, 1920, 1080, zones={"dining": door})
    assert s["died_mid_pct"] == 100.0
    v = {r["check"]: r for r in verdicts({"tracks": s})}
    assert v["track fragmentation"]["state"] == PASS, \
        "100% mid-scene through a DOORWAY must not read as a tracker failure"


def test_without_zones_it_refuses_to_call_it_a_failure_rate():
    from kevacv.scorecard import track_stats, verdicts, INFO
    tracks, _ = _door_scene()
    s = track_stats(tracks, 1920, 1080)
    v = {r["check"]: r for r in verdicts({"tracks": s})}
    assert v["track fragmentation"]["state"] == INFO
    assert "NOT a failure rate" in v["track fragmentation"]["detail"]
