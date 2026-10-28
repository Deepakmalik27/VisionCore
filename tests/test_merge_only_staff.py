"""A wrong Re-ID merge must not be able to manufacture a staff member.

Replays the measured CAM.112 nostaff run: pre-merge the spread rule called
1 of 32 people staff; post-merge it called 7 of 13, and the guest arrival
count collapsed to 1. Five of the six new "staff" had exactly ONE visit of
their own — their returns were the merge, not behaviour.
"""
from kevacv.analytics import apply_staff_zone_override

OBS = 600.0
DESK = {"reception"}


def _ev(tid, spans):
    return [{"track_id": tid, "zone": "reception", "t_in": a, "t_out": b,
             "duration": b - a, "role": "customer"} for a, b in spans]


def _staff_of(events, premerge=None):
    out = apply_staff_zone_override(
        events, DESK, 60.0, observation_s=OBS, min_visits=2, min_spread=0.25,
        sole_dwell_s=600.0, premerge_visits=premerge)
    return {e["track_id"] for e in out if e["role"] == "staff"}


# id 1: five guests glued together — one visit each, scattered across the hour.
FAKE = _ev(1, [(5, 20), (140, 160), (300, 330), (450, 470), (580, 594)])
# id 9: the real receptionist — earned 2 visits under a single pre-merge id.
REAL = _ev(9, [(30, 60), (520, 548)])


def test_merge_manufactures_staff_without_the_guard():
    assert _staff_of(FAKE) == {1}, "reproduces the bug: spread fires on a merge"


def test_guard_reverts_the_manufactured_one():
    assert _staff_of(FAKE, premerge={1: 1}) == set()


def test_guard_keeps_the_real_receptionist():
    assert _staff_of(REAL, premerge={9: 2}) == {9}


def test_sole_occupancy_is_untouched_by_the_guard():
    # dwell cannot be faked by relabelling, so rule (b) must still fire
    hog = _ev(4, [(0, 600)])
    assert _staff_of(hog, premerge={4: 1}) == {4}


def test_replays_the_measured_run():
    """1-of-32 pre-merge -> 7-of-13 post-merge -> 1 real staff with the guard."""
    events = FAKE + REAL
    assert _staff_of(events) == {1, 9}            # both, as the run reported
    assert _staff_of(events, premerge={1: 1, 9: 2}) == {9}


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
