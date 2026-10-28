"""test_staff_role.py — a guest at the counter is not staff.

WHY THIS EXISTS
    The staff detector was "dwell in a staff zone >= 60 s". At a
    one-receptionist desk that labelled 31 of 45 identities staff, because a
    GUEST STANDING AT THE COUNTER also spends ~100% of a short track there.
    kevacv/config.py carries the post-mortem and names the fix: SPREAD.

    Dwell and share both describe ONE visit. One visit is exactly what a
    customer makes. Staff RETURN — across a shift their desk time is scattered
    over the whole window, while a guest's is a single blob however long.

    The first test below is the exact case that shipped wrong. The rest pin the
    boundaries so a future tuning pass cannot quietly re-open it.

Run: python tests/test_staff_role.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kevacv.analytics import (apply_staff_zone_override,
                              describe_staff_decision, staff_evidence)

FAILED = []
STAFF = {"reception"}
HOUR = 3600.0


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


def ev(tid, zone, t_in, dur, role="customer"):
    return {"track_id": tid, "zone": zone, "t_in": float(t_in),
            "t_out": float(t_in + dur), "duration": float(dur), "role": role}


def roles_of(events):
    return {e["track_id"]: e["role"] for e in events}


def _apply(events, **kw):
    kw.setdefault("observation_s", HOUR)
    return apply_staff_zone_override(events, STAFF, **kw)


def test_the_bug_that_shipped():
    """A guest checks in: one visit to the counter, 90 seconds. The old rule
    (dwell >= 60 s) called this staff."""
    out = _apply([ev("guest", "reception", 600, 90)])
    check(roles_of(out)["guest"] == "customer",
          "a 90 s single counter visit is a CUSTOMER, not staff",
          roles_of(out)["guest"])


def test_receptionist_is_staff():
    """Same total time as a patient guest, but spread over the hour in four
    separate returns."""
    out = _apply([ev("rec", "reception", 0, 300),
                  ev("rec", "reception", 900, 300),
                  ev("rec", "reception", 2000, 300),
                  ev("rec", "reception", 3200, 300)])
    check(roles_of(out)["rec"] == "staff",
          "four returns spanning the hour is STAFF")


def test_long_single_visit_still_caught_by_sole_occupancy():
    """The backstop. Nobody stands at a reception counter for 20 minutes
    straight as a customer, so dwell alone still decides that case."""
    out = _apply([ev("anchored", "reception", 100, 1200)])
    check(roles_of(out)["anchored"] == "staff",
          "a 20-minute continuous occupancy is STAFF via sole occupancy")


def test_a_patient_guest_is_not_staff():
    """The case sole-occupancy must NOT swallow: a guest waiting a genuinely
    long time, but under the bar and in one visit."""
    out = _apply([ev("waiting", "reception", 100, 480)])   # 8 min, one visit
    check(roles_of(out)["waiting"] == "customer",
          "8 minutes in ONE visit is still a customer",
          roles_of(out)["waiting"])


def test_two_quick_returns_are_not_enough():
    """Visits alone must not be sufficient — a guest who steps to the counter
    twice in three minutes is still a guest. Spread is what separates them."""
    out = _apply([ev("g", "reception", 100, 30),
                  ev("g", "reception", 240, 30)])
    check(roles_of(out)["g"] == "customer",
          "two visits 2 minutes apart in an hour is a customer")


def test_evidence_explains_every_verdict():
    events = [ev("rec", "reception", 0, 300), ev("rec", "reception", 3200, 300),
              ev("guest", "reception", 600, 90)]
    out, evd = apply_staff_zone_override(events, STAFF, observation_s=HOUR,
                                         return_evidence=True)
    check(evd["rec"]["visits"] == 2, "visits counted", str(evd["rec"]["visits"]))
    check(evd["rec"]["reason"] == "spread", "staff verdict names its rule",
          str(evd["rec"]["reason"]))
    check(evd["guest"]["reason"] is None, "a customer has no staff reason")
    text = describe_staff_decision(evd)
    check("rec" in text and "guest" in text, "both appear in the printed table")
    check("1 of 2" in text, "the table states how many were called staff")


def test_non_staff_zone_time_is_ignored():
    """Time in the waiting area must not create staff evidence, or every guest
    who lingers becomes a candidate."""
    e = staff_evidence([ev("g", "waiting_area", 0, 3000)], STAFF,
                       observation_s=HOUR)
    check(e == {}, "a track that never entered a staff zone has no evidence",
          str(e))


def test_share_uses_lifetime_not_summed_dwell():
    """Overlapping polygons (a host stand inside a lobby) must not dilute a
    desk-anchored person's share below the bar."""
    e = staff_evidence([ev("s", "reception", 0, 100),
                        ev("s", "waiting_area", 0, 100)], STAFF,
                       observation_s=HOUR)
    check(abs(e["s"]["share"] - 1.0) < 1e-6,
          "share is dwell / lifetime, so overlap does not dilute it",
          str(e["s"]["share"]))


def test_legacy_path_without_observation_window():
    """A caller that cannot supply a window must degrade to the OLD answer,
    not to 'nobody is staff' — silent zero staff is worse than a weak rule."""
    out = apply_staff_zone_override([ev("x", "reception", 0, 90)], STAFF,
                                    observation_s=None)
    check(roles_of(out)["x"] == "staff",
          "no observation window falls back to the dwell rule")


def test_input_is_never_mutated():
    src = [ev("g", "reception", 0, 5000)]
    before = dict(src[0])
    _apply(src)
    check(src[0] == before, "the caller's events are untouched")


def test_no_staff_zones_is_a_noop():
    src = [ev("g", "reception", 0, 5000)]
    out = apply_staff_zone_override(src, set(), observation_s=HOUR)
    check(roles_of(out)["g"] == "customer", "no staff zones -> nobody is staff")


def main():
    print(__doc__.strip().splitlines()[0])
    for fn in (test_the_bug_that_shipped,
               test_receptionist_is_staff,
               test_long_single_visit_still_caught_by_sole_occupancy,
               test_a_patient_guest_is_not_staff,
               test_two_quick_returns_are_not_enough,
               test_evidence_explains_every_verdict,
               test_non_staff_zone_time_is_ignored,
               test_share_uses_lifetime_not_summed_dwell,
               test_legacy_path_without_observation_window,
               test_input_is_never_mutated,
               test_no_staff_zones_is_a_noop):
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
