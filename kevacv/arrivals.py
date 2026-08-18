"""arrivals.py — PHASE 9. Count arrivals without depending on the line.

WHY THIS EXISTS
    On the first full hour of CAM.112 the entry line triggered ZERO times while
    95 people moved through the zones and reception was visited 74 times. Every
    arrival-based number in the report was 0, and "0 people entered" reads
    exactly like a fact.

    The line was too short — people walked around both ends. That is a drawing
    mistake, and it will happen again on the next camera, and the one after.

THE PRINCIPLE
    A single sensor with no cross-check is a single point of silent failure.
    Industry practice treats a tripwire and a region as COMPLEMENTARY — the
    tripwire answers "who crossed where", the region answers "who was here" —
    so we compute BOTH and make them check each other.

    A region-based arrival needs no line at all: someone who appears in an
    entry zone and then appears somewhere interior has arrived. It is coarser
    than a line (it cannot tell you the exact instant, and it cannot separate
    in from out), but it is nearly impossible to break by drawing badly, which
    is exactly the failure a line has.

    When the two disagree by a lot, that is itself the finding.
"""
from __future__ import annotations

import math

ENTRY_ROLES = {"entry"}
INTERIOR_ROLES = {"wait", "staff", "seating", "service"}


def _zones_with(zone_roles, roles):
    return {z for z, rs in (zone_roles or {}).items() if set(rs) & roles}


def arrivals_from_regions(events, zone_roles, roles=None, dedupe_s=6.0,
                          dedupe_m=1.2, plane=None, positions=None):
    """Arrivals inferred from zone transitions instead of a line crossing.

    An arrival is a track that is seen in an ENTRY zone and then, later, in an
    INTERIOR zone. Walking in is exactly that transition, and it does not care
    where a line was drawn.

    -> (count, arrivals, why). `why` is never None: when the zones cannot
    support the question at all it says so rather than returning 0, because 0
    and "cannot tell" must never look the same in a report.
    """
    ez = _zones_with(zone_roles, ENTRY_ROLES)
    iz = _zones_with(zone_roles, INTERIOR_ROLES)
    if not ez:
        return None, [], ("no zone has the ENTRY role — region arrivals cannot "
                          "be computed. Name a zone entrance/door/gate/entry.")
    if not iz:
        return None, [], ("no INTERIOR zone (wait/staff/seating/service) — "
                          "there is nowhere to arrive INTO.")

    per = {}
    for e in events:
        per.setdefault(e["track_id"], []).append(e)

    out = []
    for tid, evs in per.items():
        if roles and roles.get(tid) == "staff":
            continue
        evs = sorted(evs, key=lambda e: e["t_in"])
        first_entry = next((e for e in evs if e["zone"] in ez), None)
        if first_entry is None:
            continue
        moved_in = next((e for e in evs if e["zone"] in iz
                         and e["t_in"] >= first_entry["t_in"]), None)
        if moved_in is None:
            continue                      # stood in the doorway and left again
        out.append({"track_id": tid, "t": moved_in["t_in"],
                    "from_zone": first_entry["zone"], "to_zone": moved_in["zone"],
                    "pos": (positions or {}).get(tid)})

    out.sort(key=lambda a: a["t"])
    kept = []
    for a in out:
        dup = False
        for k in reversed(kept):
            if a["t"] - k["t"] > dedupe_s:
                break
            p, q = a.get("pos"), k.get("pos")
            if p and q:
                d = plane.dist_m(p, q) if (plane is not None and plane.ok) else None
                if (d is not None and d <= dedupe_m) or (
                        d is None and math.hypot(p[0] - q[0], p[1] - q[1]) <= 140):
                    dup = True
                    break
            elif k["track_id"] == a["track_id"]:
                dup = True
                break
        if not dup:
            kept.append(a)
    return len(kept), kept, ""


def entry_zone_coverage(events, zone_roles, roles=None):
    """What share of non-staff people were EVER seen in an entry zone?

    WHY THIS EXISTS
        A region arrival needs an entry-zone sighting before an interior one.
        If hardly anyone produces that first sighting, the entry polygon is not
        over the door — and the region count is then just as broken as a badly
        drawn line, only quietly, because it still returns a small positive
        number instead of 0.

        On CAM.112 the line fired 0 times, the region method returned 2, and 31
        people moved through the zones. cross_check called that "LINE IS BROKEN,
        trust the region" and the report published "2 people came through the
        door". Two independent sensors were both wrong and nothing said so.

    -> None when no zone carries the ENTRY role (the question cannot be asked).
    """
    ez = _zones_with(zone_roles, ENTRY_ROLES)
    if not ez:
        return None
    per = {}
    for e in events:
        per.setdefault(e["track_id"], []).append(e)
    seen = never = 0
    for tid, evs in per.items():
        if roles and roles.get(tid) == "staff":
            continue
        if any(e["zone"] in ez for e in evs):
            seen += 1
        else:
            never += 1
    total = seen + never
    return {"with_entry": seen, "without_entry": never, "non_staff": total,
            "share_with_entry": (seen / total) if total else 0.0}


def cross_check(line_count, region_count, movers=0, tolerance=0.25,
                coverage=None, min_entry_share=0.5):
    """Compare the two independent arrival counts and say what to believe.

    The point is not to pick a winner. It is that two numbers which should
    agree and do not are evidence of a specific, nameable fault — and that a
    silent 0 is the one outcome nobody should ever be handed.

    `coverage` is entry_zone_coverage(). Passing it adds the check that the
    region count is worth trusting at all: a region method whose entry zone
    most people never enter is not a cross-check, it is a second failure
    wearing the costume of a measurement.
    """
    if region_count is None:
        return {"verdict": "no cross-check available",
                "detail": "zones cannot support a region arrival count",
                "trust": "line", "agree": None}
    # Before believing the region number, ask whether the entry zone is even in
    # the right place. This must run BEFORE "LINE IS BROKEN", because that
    # branch's whole purpose is to hand the region count to the report.
    if (coverage and coverage["non_staff"] >= 5
            and coverage["share_with_entry"] < min_entry_share):
        return {"verdict": "ENTRY ZONE IS MISPLACED TOO — trust neither",
                "detail": (f"only {coverage['with_entry']} of "
                           f"{coverage['non_staff']} non-staff people were ever "
                           f"seen inside an entry zone "
                           f"({coverage['share_with_entry']*100:.0f}%). The "
                           f"region count of {region_count} is not a "
                           f"cross-check — the entry polygon is not over the "
                           f"door people actually use. Redraw the entry zone "
                           f"AND the line before believing either number."),
                "trust": "neither", "agree": False}
    if line_count == 0 and region_count == 0 and movers >= 5:
        return {"verdict": "BOTH ZERO but people were present",
                "detail": (f"{movers} people moved through the zones and neither "
                           f"method saw an arrival. The entry zone is probably not "
                           f"where people actually come in."),
                "trust": "neither", "agree": True}
    if line_count == 0 and region_count > 0:
        return {"verdict": "LINE IS BROKEN",
                "detail": (f"the line counted 0 while {region_count} people were "
                           f"seen entering by zone transition. The line is almost "
                           f"certainly too short or in the wrong place — redraw it "
                           f"wall to wall across the threshold. Use the region "
                           f"count until then."),
                "trust": "region", "agree": False}
    if region_count == 0 and line_count > 0:
        return {"verdict": "entry zone is misplaced",
                "detail": (f"the line counted {line_count} but nobody was seen in "
                           f"an entry zone first. The entry polygon is probably not "
                           f"over the doorway."),
                "trust": "line", "agree": False}
    hi = max(line_count, region_count)
    delta = abs(line_count - region_count) / hi if hi else 0.0
    if delta <= tolerance:
        return {"verdict": "the two methods AGREE",
                "detail": (f"line {line_count} vs region {region_count} "
                           f"({delta*100:.0f}% apart) — two independent signals "
                           f"agreeing is the strongest evidence this number is real."),
                "trust": "line", "agree": True}
    return {"verdict": "the two methods DISAGREE",
            "detail": (f"line {line_count} vs region {region_count} "
                       f"({delta*100:.0f}% apart). Something is clipping one of "
                       f"them — check the line spans the full doorway and the "
                       f"entry polygon covers where people actually walk in."),
            "trust": "region" if region_count > line_count else "line",
            "agree": False}


def describe(line_count, region_count, cc):
    L = ["ARRIVALS — two independent methods"]
    L.append(f"  line crossing   {line_count if line_count is not None else 'n/a'}")
    L.append(f"  zone transition {region_count if region_count is not None else 'n/a'}")
    L.append(f"  -> {cc['verdict']}")
    L.append(f"     {cc['detail']}")
    return "\n".join(L)
