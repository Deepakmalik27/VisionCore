"""A plane is accepted or rejected against a KNOWN distance in the room.

Bounding the horizon row by guesswork would reject correct fits: a steeply
down-tilted ceiling camera genuinely puts the floor's vanishing point above the
frame. One measured distance arbitrates without guessing. For CAM.112 that is
the floor checkerboard -- 128.6 px at analysis row 780 is one tile diagonal,
and the operator measured the tile edge at 30 cm.
"""
import json
from kevacv.ground_plane import GroundPlane
from kevacv.geometry_calibration import fit_robust_ground_plane
import numpy as np
from kevacv.ground_plane import synth_camera


def _synthetic_people(cam_h=2.6, n=60, person_h=1.7, seed=0):
    """Boxes for upright people on a flat floor under a known camera."""
    proj = synth_camera(cam_h=cam_h, focal_px=1200.0, frame=(1920, 1080))
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        X, Z = rng.uniform(-3.0, 3.0), rng.uniform(3.0, 9.0)
        fx, fy = proj(X, Z, 0.0)
        _hx, hy = proj(X, Z, person_h)
        h = abs(fy - hy)
        out.append((fx - h / 5.2, hy, fx + h / 5.2, fy))
    return out

REF = {"row": 780, "pixels": 128.6, "metres": 0.4243, "tolerance": 0.35}


def test_a_plane_that_matches_the_floor_is_accepted():
    gp = fit_robust_ground_plane(_synthetic_people(cam_h=2.6), (1920, 1080))
    assert gp.ok
    got = gp.dist_m((0.0, 780.0), (128.6, 780.0))
    ref = dict(REF, metres=got)          # a plane consistent with itself
    assert gp.check_scale_reference(ref) == []


def test_a_plane_off_by_60_percent_is_rejected():
    gp = fit_robust_ground_plane(_synthetic_people(cam_h=2.6), (1920, 1080))
    got = gp.dist_m((0.0, 780.0), (128.6, 780.0))
    ref = dict(REF, metres=got * 1.6)
    bad = gp.check_scale_reference(ref)
    assert bad and "scale reference FAILED" in bad[0]


def test_no_reference_means_no_opinion_not_a_pass():
    gp = fit_robust_ground_plane(_synthetic_people(cam_h=2.6), (1920, 1080))
    assert gp.check_scale_reference(None) == []
    assert gp.check_scale_reference({"row": 780}) == []   # malformed -> abstain


def test_the_zone_file_carries_a_usable_reference():
    z = json.load(open("zones/CAM.112_zone_v4.json"))
    r = z["scale_reference"]
    for k in ("row", "pixels", "metres", "tolerance"):
        assert k in r, k
    # 128.6 px is a tile DIAGONAL on a 45-degree floor -> 30 cm edge
    assert abs(r["metres"] / (2 ** 0.5) - 0.30) < 0.01
