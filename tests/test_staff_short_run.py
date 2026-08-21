"""The 10-minute delivery run reported "0 staff" over a permanently-manned
reception desk. Cause was arithmetic, not detection: min_spread is a FRACTION
of the observation window while sole_dwell_s was an ABSOLUTE 600 s, so on a
600 s clip the sole-occupancy rule demanded 100% of the window with zero
missed frames. And the spread rule cannot catch someone who never leaves the
desk, because that is ONE contiguous visit -- the exact case sole occupancy
exists to cover. Neither rule could fire, for any input.
"""
from kevacv.analytics import apply_staff_zone_override

STAFF = {"reception"}
W = 600.0


def _ev(tid, t0, t1, zone="reception"):
    return {"track_id": tid, "zone": zone, "t_in": t0, "t_out": t1,
            "duration": t1 - t0, "role": "customer"}


def _role(events, **kw):
    return apply_staff_zone_override(events, STAFF, observation_s=W, **kw)


def test_desk_bound_receptionist_is_staff_on_a_short_run():
    # one visit, never leaves -> spread cannot help, sole occupancy must
    assert _role([_ev("r", 10, 560)])[0]["role"] == "staff"


def test_a_long_customer_interaction_is_still_a_customer():
    # ~5 minutes at the counter is a big check-in, not a shift
    assert _role([_ev("g", 10, 300)])[0]["role"] == "customer"
    assert _role([_ev("g", 10, 120)])[0]["role"] == "customer"


def test_threshold_never_falls_below_min_staff_dwell():
    # a tiny window must not make everyone staff
    out = apply_staff_zone_override([_ev("x", 0, 20)], STAFF,
                                    observation_s=40.0)
    assert out[0]["role"] == "customer"


def test_long_run_keeps_the_original_absolute_bar():
    # 2 hours: 0.5*window is 3600 s, so the 600 s absolute stays in force
    out = apply_staff_zone_override([_ev("s", 0, 700)], STAFF,
                                    observation_s=7200.0)
    assert out[0]["role"] == "staff"
