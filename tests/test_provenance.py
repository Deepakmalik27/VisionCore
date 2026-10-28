"""'Why did yesterday give 2,104 and today 2,083?' must be answerable."""
from kevacv.provenance import build_stamp, fingerprint, diff


def S(**kw):
    base = dict(build_id="abc123", config_path="config/a.yaml",
                zones_path="zones/z.json", video="chunk_6.30pm.mp4",
                detector="yolo11x.pt", tracker="botsort-reid",
                reid_weights="clip_market1501.pt", fps=8.0,
                analysis_w=1920.0, imgsz=1280.0,
                ground_plane={"camera_h_m": 3.25, "horizon_row": -240},
                changed={"MASK_REQUIRE_MOTION": "0.0 -> 12.0"})
    base.update(kw)
    return build_stamp(**base)


def test_identical_runs_share_a_fingerprint():
    assert S()["fingerprint"] == S()["fingerprint"]


def test_a_changed_knob_changes_the_fingerprint():
    a = S()
    b = S(changed={"MASK_REQUIRE_MOTION": "0.0 -> 12.0",
                   "ENABLE_GMC": "True -> False"})
    assert a["fingerprint"] != b["fingerprint"]


def test_the_duplicate_key_failure_is_now_visible():
    """The GMC A/B reported 'identical to baseline' because a duplicated yaml
    key meant the flag never applied. The FILE said false; the RUN used true.
    Only `changed` records what the run actually did."""
    intended = S(changed={"ENABLE_GMC": "True -> False"})
    actual = S(changed={})           # flag never applied
    assert intended["fingerprint"] != actual["fingerprint"]
    d = diff(intended, actual)
    assert "changed.ENABLE_GMC" in d


def test_different_footage_is_never_the_same_run():
    """A held-out score once got computed against ground truth from a
    different HOUR of footage."""
    assert S()["fingerprint"] != S(video="chunk_7.30pm.mp4")["fingerprint"]


def test_the_plane_is_recorded_because_it_is_the_ruler():
    a = S()
    b = S(ground_plane={"camera_h_m": 4.04, "horizon_row": -261})
    assert a["calibration"]["camera_h_m"] == 3.25
    assert "calibration.camera_h_m" in diff(a, b)


def test_diff_names_exactly_what_moved():
    d = diff(S(), S(detector="best.pt", fps=15.0))
    assert d["detector"] == ["yolo11x.pt", "best.pt"]
    assert d["fps"] == [8.0, 15.0]
    assert "tracker" not in d


def test_identical_stamps_diff_to_nothing():
    assert diff(S(), S()) == {}


def test_a_missing_field_does_not_crash():
    s = build_stamp()
    assert s["build_id"] == "?" and len(s["fingerprint"]) == 12
