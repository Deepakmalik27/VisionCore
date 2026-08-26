"""test_role_lock.py — a face match must outrank a zone inference.

THE BUG THIS PINS (measured, 19:30-19:50 Delilah chunk, 2026-08-25)

    person   role      role_from       minutes  waited_s  greeted
    staff3   customer  zone-inferred      19.8       136       no
    staff2   customer  zone-inferred      17.0        29       no
    staff1   customer  zone-inferred       8.5        10       no

    staff3 was on camera 19.8 of 20 minutes -- plainly the receptionist -- and
    was reported as a GUEST who waited 2m16s and was never greeted.

WHY IT HAPPENED
    role_votes are cast inside the frame loop, where every track id is still
    numeric, so every vote a future staff member casts is "customer". The face
    gallery sweep runs AFTER the loop, renames the chain to "staff3", and the
    customer majority came along with the name.

WHY ONE ROW MATTERS SO MUCH
    That single inversion is enough to produce, from the same data:
      * desk_covered = 30.5% (the actual receptionist not counted as staff)
      * an inflated guest count (staff counted as guests)
      * a FABRICATED service anomaly ("guest waited 2m16s ungreeted")
    Three wrong headline numbers, one wrong role.

THE CONVENTION BEING RELIED ON
    A non-numeric track id can only have come from the face gallery -- nothing
    else in this pipeline mints a named id. derive.py already encodes this
    ("source": "face" if non-numeric else "staff-zone") and so does
    render_annotated. This test pins that convention so a future change that
    starts minting named ids elsewhere fails loudly here.

Run: python tests/test_role_lock.py
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


# The lock as the engine implements it, extracted so it can be exercised
# without a video, a GPU or a model. Kept in lockstep with engine.py by the
# source assertions at the bottom of this file.
def apply_role_lock(roles, events):
    named = [t for t in roles if isinstance(t, str) and not str(t).isdigit()]
    n_ev = 0
    for t in named:
        roles[t] = "staff"
    for e in events:
        if e.get("track_id") in set(named) and e.get("role") != "staff":
            e["role"] = "staff"
            n_ev += 1
    return named, n_ev


print()
print("=" * 74)
print("  a face-matched identity cannot be downgraded by a zone vote")
print("=" * 74)

# exactly the observed failure: named ids carrying a customer majority
roles = {1: "customer", 7: "customer",
         "staff1": "customer", "staff2": "customer", "staff3": "customer"}
events = [
    {"track_id": "staff3", "role": "customer", "zone": "reception",
     "t_in": 10.0, "t_out": 1199.0},
    {"track_id": "staff2", "role": "customer", "zone": "reception",
     "t_in": 23.0, "t_out": 1043.0},
    {"track_id": 7, "role": "customer", "zone": "waiting area",
     "t_in": 100.0, "t_out": 160.0},
]
named, n_ev = apply_role_lock(roles, events)

check(sorted(named) == ["staff1", "staff2", "staff3"],
      "every non-numeric id is recognised as face-matched", str(sorted(named)))
check(all(roles[t] == "staff" for t in named),
      "all three are forced to staff")
check(roles[1] == "customer" and roles[7] == "customer",
      "NUMERIC ids are left alone",
      "the lock must not promote everybody to staff")
check(n_ev == 2, "the zone events carrying a stale role are corrected too",
      f"{n_ev} corrected")
check(all(e["role"] == "staff" for e in events if e["track_id"] in named),
      "reception events now read staff")
check(events[2]["role"] == "customer",
      "a real guest's event is untouched")

print()
print("=" * 74)
print("  the downstream numbers the inversion was corrupting")
print("=" * 74)

# desk coverage counts STAFF time in a staff zone. Before the lock, the
# receptionist's 19.8 minutes counted for nothing.
def staff_seconds(evs, zone="reception"):
    return sum(e["t_out"] - e["t_in"] for e in evs
               if e["zone"] == zone and e["role"] == "staff")

before = [
    {"track_id": "staff3", "role": "customer", "zone": "reception",
     "t_in": 10.0, "t_out": 1199.0},
]
after = [dict(before[0], role="staff")]
check(staff_seconds(before) == 0.0,
      "BEFORE: the receptionist contributed 0 s of desk coverage",
      "which is how 19.8 minutes of cover became 30.5%")
check(abs(staff_seconds(after) - 1189.0) < 1e-6,
      "AFTER: the same person contributes 1189 s", f"{staff_seconds(after)}")

# and the fabricated anomaly: a "guest" who waited and was never greeted
def ungreeted_guests(rows):
    return [r for r in rows if r["role"] == "customer"
            and r.get("waited_s", 0) >= 120 and r.get("greeted") == "no"]

rows_before = [{"person": "staff3", "role": "customer", "waited_s": 136,
                "greeted": "no"}]
rows_after = [dict(rows_before[0], role="staff")]
check(len(ungreeted_guests(rows_before)) == 1,
      "BEFORE: a staff member appears as an ungreeted guest",
      "a fabricated service failure in the GM's report")
check(len(ungreeted_guests(rows_after)) == 0,
      "AFTER: the fabricated anomaly disappears")

print()
print("=" * 74)
print("  the engine really does this, at the right place")
print("=" * 74)
ENG = (ROOT / "kevacv" / "engine.py").read_text(encoding="utf-8")
check("ROLE LOCK" in ENG, "the lock exists in engine.py")
check("_face_named" in ENG, "it selects on the non-numeric-id convention")

# It MUST run after `roles` is composed from final_roles + events, or it locks
# a dict that is about to be overwritten.
i_roles = ENG.index("roles = {**rec.final_roles(),")
i_lock = ENG.index("_face_named = [t for t in roles")
check(i_lock > i_roles, "the lock runs AFTER roles is composed",
      "locking before composition would be silently undone")

# events must be corrected too: desk coverage reads events, not roles.
seg = ENG[i_lock:i_lock + 1400]
check('_e["role"] = "staff"' in seg,
      "it corrects the events list as well as the roles map",
      "fixing one and not the other is how this bug hid in the first place")
check("_log.warning" in seg, "it says out loud what it changed",
      "a silent role rewrite is as bad as the wrong role")

print()
print("=" * 74)


def test_script_level_checks_passed():
    assert not FAILED, f"role-lock checks failed: {FAILED}"


if __name__ == "__main__":
    if FAILED:
        print(f"  {len(FAILED)} FAILED:")
        for x in FAILED:
            print(f"    - {x}")
        sys.exit(1)
    print("  ALL PASS")
    print("=" * 74)
