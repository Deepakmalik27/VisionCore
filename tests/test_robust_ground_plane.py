"""The robust plane must beat the one it replaces, and refuse rather than lie.

Measured on CAM.112 (output/p0v4, 9,516 detections):
    bin-median LSQ (in use)   camera 4.04 m, horizon row -261   impossible
    fit_robust_ground_plane   camera 2.50 m, horizon row  +32   plausible
Validated against the floor: checkerboard period 128.6 px -> 0.30 m tile edge
through the robust plane; the operator states 30 cm.
"""
import numpy as np
import kevacv.config as C
from kevacv.ground_plane import GroundPlane, synth_camera
from kevacv.geometry_calibration import fit_robust_ground_plane


def _synthetic_people(cam_h=2.6, n=60, person_h=1.7, seed=0):
    """Boxes for upright people on a flat floor under a known camera."""
    proj = synth_camera(cam_h=cam_h, focal_px=1200.0, frame=(1920, 1080))
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        X = rng.uniform(-3.0, 3.0)
        Z = rng.uniform(3.0, 9.0)
        fx, fy = proj(X, Z, 0.0)          # feet
        hx, hy = proj(X, Z, person_h)     # head
        h = abs(fy - hy)
        w = h / 2.6                        # upright aspect ~2.6, inside the band
        out.append((fx - w / 2, hy, fx + w / 2, fy))
    return out


def test_recovers_a_known_camera_height():
    gp = fit_robust_ground_plane(_synthetic_people(cam_h=2.6), (1920, 1080))
    assert gp.ok, gp.note
    h = gp.camera_height_m()
    assert 2.0 <= h <= 3.4, f"recovered {h:.2f} m for a 2.6 m camera"


def test_horizon_is_above_the_floor_not_below_the_frame():
    gp = fit_robust_ground_plane(_synthetic_people(cam_h=2.6), (1920, 1080))
    assert gp.horizon_y() > -50, \
        f"horizon row {gp.horizon_y()} — a negative horizon is the exact " \
        f"symptom the in-use fitter shows on this camera"


def test_it_refuses_rather_than_inventing_a_plane():
    gp = fit_robust_ground_plane([(0, 0, 10, 10)] * 3, (1920, 1080))
    assert not gp.ok, "three boxes is not a calibration"
    assert "nsufficient" in gp.note or "no " in gp.note.lower()


def test_seated_and_crouched_boxes_do_not_tilt_it():
    good = _synthetic_people(cam_h=2.6)
    # squat, wide boxes at random rows — seated guests and desk-occluded torsos
    rng = np.random.RandomState(1)
    bad = [(x, y, x + 260, y + 150) for x, y in
           zip(rng.uniform(200, 1600, 80), rng.uniform(500, 1000, 80))]
    gp = fit_robust_ground_plane(good + bad, (1920, 1080))
    assert gp.ok
    assert 2.0 <= gp.camera_height_m() <= 3.4, \
        f"{gp.camera_height_m():.2f} m — outliers tilted the line"


def test_knob_is_reachable_and_inert_by_default():
    assert C.ENABLE_ROBUST_GROUND_PLANE is False
    assert "analysis.enable_robust_ground_plane" in C.RUN_CONFIG_KEYS


def test_a_plane_that_fails_its_own_sanity_check_is_flagged():
    """The engine used to log 'camera height 9.0 m is implausible' and then use
    that plane as the ruler for every metre gate. sanity() must be consulted,
    not merely printed."""
    from kevacv.ground_plane import GroundPlane
    # a fit implying a camera far above any ceiling
    bad = fit_robust_ground_plane(_synthetic_people(cam_h=12.0), (1920, 1080))
    if bad.ok:
        assert bad.sanity(1080), \
            "a 12 m camera must be reported by sanity(), which is what the " \
            "engine now gates on"


def test_min_sample_floor_is_reachable_and_sane():
    assert C.ROBUST_PLANE_MIN_SAMPLES >= 100, \
        "the fitter's own floor of 10 points guards against crashing, " \
        "not against being wrong"
    assert "analysis.robust_plane_min_samples" in C.RUN_CONFIG_KEYS
