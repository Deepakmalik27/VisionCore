"""test_observe.py — the per-frame observation row.

Run: python3 tests/test_observe.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from kevacv import observe

FAILED = []

def check(cond, msg):
    print(("  ✅ " if cond else "  ❌ ") + msg)
    if not cond:
        FAILED.append(msg)

SQUARE = {"dining": np.array([[100, 100], [300, 100], [300, 300], [100, 300]], dtype=float)}

def _box_at(x, y, w=40, h=100):
    """A box whose FOOT anchor lands on (x, y)."""
    return (x - w / 2.0, y - h, x + w / 2.0, y)

def test_deep_inside_beats_near_edge():
    mid, _c1 = observe.zone_and_confidence(_box_at(200, 200), SQUARE, set(), True)
    edge, _c2 = observe.zone_and_confidence(_box_at(200, 295), SQUARE, set(), True)
    check(mid == "dining" and edge == "dining", "both anchors land in dining")
    check(_c1 > _c2, f"centre {_c1} is more confident than edge {_c2}")

def test_outside_is_no_zone():
    z, c = observe.zone_and_confidence(_box_at(50, 50), SQUARE, set(), True)
    check(z is None and c == 0.0, "anchor outside every polygon -> zone None, conf 0.0")

def test_hidden_feet_halve_confidence():
    _z, seen = observe.zone_and_confidence(_box_at(200, 200), SQUARE, set(), True)
    _z, hid = observe.zone_and_confidence(_box_at(200, 200), SQUARE, set(), False)
    check(abs(hid - seen / 2.0) < 1e-6, f"hidden feet halve conf: {seen} -> {hid}")

def test_confidence_is_bounded():
    _z, c = observe.zone_and_confidence(_box_at(200, 200, w=2), SQUARE, set(), True)
    check(0.0 <= c <= 1.0, f"a tiny box cannot exceed 1.0 (got {c})")

def test_stationary_track_has_no_heading():
    s = [(0.0, (100.0, 200.0)), (0.125, (100.0, 200.0)), (0.25, (100.0, 200.0))]
    m = observe.motion(s)
    check(m["speed_px_s"] == 0.0, "identical anchors -> 0 px/s")
    check(m["stationary"] is True, "and stationary True")
    check(m["heading_deg"] is None, "and no heading (a still body has no direction)")

def test_single_sample_is_unknown_not_zero():
    m = observe.motion([(0.0, (10.0, 10.0))])
    check(m["speed_px_s"] is None, "one sample cannot measure speed -> None")
    check(m["stationary"] is None, "and cannot claim stationary either")

def test_moving_right_reads_east():
    s = [(0.0, (100.0, 200.0)), (0.125, (110.0, 200.0)), (0.25, (120.0, 200.0))]
    m = observe.motion(s)
    check(abs(m["speed_px_s"] - 80.0) < 1e-6, f"10px per 0.125s = 80 px/s (got {m['speed_px_s']})")
    check(m["stationary"] is False, "80 px/s is not stationary")
    check(abs(m["heading_deg"] - 90.0) < 1e-6, f"straight right = 90 deg (got {m['heading_deg']})")

def test_one_jittery_interval_does_not_dominate():
    """The oldest anchor is the bad one: intervals are [784, 8, 8] px/s, so the
    MEDIAN reads 8 and the mean would read 267. One outlying interval is what a
    median can absorb -- a spike in the MIDDLE corrupts two intervals and is
    deliberately not claimed to be fixed here."""
    steady = [(0.0, (0.0, 0.0)), (0.125, (1.0, 0.0)), (0.25, (2.0, 0.0)), (0.375, (3.0, 0.0))]
    spike = [(0.0, (99.0, 0.0)), (0.125, (1.0, 0.0)), (0.25, (2.0, 0.0)), (0.375, (3.0, 0.0))]
    check(observe.motion(spike)["speed_px_s"] == observe.motion(steady)["speed_px_s"],
          "median over the window absorbs one outlying interval")

def test_speed_mps_is_none_without_a_ground_plane():
    from kevacv.ground_plane import GroundPlane
    s = [(0.0, (100.0, 200.0)), (0.125, (110.0, 200.0))]
    m = observe.motion(s, ground=GroundPlane.none("test"))
    check(m["speed_mps"] is None, "no calibration -> metres NOT invented from pixels")

def test_pose_never_sampled_is_null_not_unknown():
    p = observe.pose_field(None, 12.0)
    check(p["available"] is False, "no sample -> available False")
    check(p["activity"] is None and p["age_s"] is None,
          "and NULL activity/age, not a guessed 'unknown'")

def test_pose_fresh_sample_has_zero_age():
    p = observe.pose_field((12.0, "standing"), 12.0)
    check(p == {"available": True, "activity": "standing", "age_s": 0.0}, str(p))

def test_pose_carry_forward_states_its_age():
    p = observe.pose_field((12.0, "sitting"), 12.875)
    check(p["activity"] == "sitting", "activity carries forward between strides")
    check(abs(p["age_s"] - 0.875) < 1e-6, f"age is reported, not hidden (got {p['age_s']})")

def _row(**kw):
    base = dict(run_id="r1", camera_id="cam01", frame_idx=10, t_s=1.25,
                ts="2026-08-20T20:15:32.400000+00:00", box=(180.0, 100.0, 220.0, 200.0),
                det_conf=0.94, raw_track_id=103, canon_id=None, emb_id=None,
                polygons=SQUARE, staff_zones=set(), feet_visible=True, is_ir=False,
                samples=[(1.125, (200.0, 200.0)), (1.25, (200.0, 200.0))],
                ground=None, pose_last=None, stationary_px_s=8.0)
    base.update(kw)
    return observe.build_row(**base)

def test_row_has_exactly_the_declared_columns():
    r = _row()
    check(set(r) == set(observe.OBS_COLUMNS) | {"kind"},
          f"row keys match OBS_COLUMNS; diff={set(r) ^ (set(observe.OBS_COLUMNS) | {'kind'})}")
    check(r["kind"] == "obs", "tagged obs so one JSONL can carry both kinds")

def test_row_keeps_both_ids():
    r = _row(canon_id="guest_7")
    check(r["raw_track_id"] == "103" and r["canon_id"] == "guest_7",
          "raw AND canonical id both survive; raw is stringified for a text PK")

def test_row_is_json_serialisable():
    import json
    json.dumps(_row(det_conf=np.float32(0.94)))   # numpy scalars must not leak
    check(True, "numpy scalars are cast before they reach the JSONL")

def test_emb_row_carries_a_real_vector():
    e = observe.emb_row(run_id="r1", camera_id="cam01", raw_track_id=103,
                        frame_idx=10, emb_id="r1_103_10", vec=np.zeros(512, dtype=np.float32),
                        blur_score=41.2)
    check(len(e["vec"]) == 512 and isinstance(e["vec"][0], float),
          "vector is stored as plain floats, not an opaque id")

def main():
    for fn in (test_deep_inside_beats_near_edge, test_outside_is_no_zone,
               test_hidden_feet_halve_confidence, test_confidence_is_bounded,
               test_stationary_track_has_no_heading, test_single_sample_is_unknown_not_zero,
               test_moving_right_reads_east, test_one_jittery_interval_does_not_dominate,
               test_speed_mps_is_none_without_a_ground_plane,
               test_pose_never_sampled_is_null_not_unknown, test_pose_fresh_sample_has_zero_age,
               test_pose_carry_forward_states_its_age,
               test_row_has_exactly_the_declared_columns, test_row_keeps_both_ids,
               test_row_is_json_serialisable, test_emb_row_carries_a_real_vector):
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
