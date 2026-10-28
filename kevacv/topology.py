"""topology.py — a re-appearance gate that stays sharp when appearance doesn't.

WHY THIS EXISTS
    The live identity gate asks "could a person have walked that far in the
    time available":

        _md > max_speed_mps * max(gap, 0.35) + 0.35

    At 2.2 m/s a 900-second gap allows ~1,980 metres. Every point in the frame
    passes. So beyond roughly ten seconds the spatial constraint contributes
    nothing and the merge is decided by appearance alone — exactly the regime
    where appearance is weakest. On CAM.112 the measured separability is:

        same person      p50 = 0.435
        different person p50 = 0.370, p90 = 0.573
        best possible balanced accuracy = 0.658

    No threshold rescues that. A stronger backbone might move it a few points.
    A different KIND of constraint moves it more.

THE PRINCIPLE
    A fixed camera has a topology. People do not materialise in the middle of
    the room — they come through a door. So a re-appearance is only physically
    possible in three shapes:

        1. the track died and was reborn in the SAME PLACE
           -> the tracker lost them behind something. Plausible.
        2. the track died AT A DOOR and was reborn AT A DOOR
           -> they left and came back. Plausible, at any gap.
        3. anything else
           -> they would have had to cross the room unseen. Not plausible.

    This is a HARD constraint, like co-visibility, not a score. It does not
    degrade as the gap grows, because a door stays a door. It is the natural
    partner to the co-visibility rule already in _IdentityMemory: that one says
    "one person cannot be in two places at once", this one says "one person
    cannot get from A to B without crossing the space between".

WHAT IT DOES NOT DO
    It never merges anything. It only ever says "this pair is impossible" —
    a veto, applied before appearance is consulted. Vetoes are safe in a way
    that merges are not: a wrong veto costs one fragment, a wrong merge fuses
    two people and corrupts every downstream count.

    Doors come from kevacv.learn_zones.learn_entry_zones (learned from where
    tracks are actually born and die), so this needs no hand-drawn zone.
"""
from __future__ import annotations

import math

# A door is a place, not a point: allow this fraction of the frame diagonal
# around a learned door centre before calling a position "not at the door".
DOOR_RADIUS_FRAC = 0.10
# Below this gap the existing positional/velocity gates are trustworthy and
# this module stays out of the way.
SHORT_GAP_S = 10.0
# How far a body may drift while the tracker has lost it and still count as
# "the same place" (fraction of frame diagonal).
STILL_DRIFT_FRAC = 0.06


def _diag(frame_wh):
    w, h = float(frame_wh[0]), float(frame_wh[1])
    return math.hypot(w, h)


def _nearest(pt, doors):
    """-> (distance_px, index) to the nearest door centre, or (inf, None)."""
    best, idx = float("inf"), None
    for i, d in enumerate(doors or []):
        dist = math.hypot(pt[0] - d[0], pt[1] - d[1])
        if dist < best:
            best, idx = dist, i
    return best, idx


def doors_from_zones(zone_polygons, zone_roles, roles=("entry",)):
    """Door centres from named entry polygons, when zones ARE hand-drawn.

    Falls back to nothing rather than guessing: no entry zone means this gate
    abstains, which is the correct behaviour for a module that only vetoes.
    """
    want = {z for z, rs in (zone_roles or {}).items() if set(rs) & set(roles)}
    out = []
    for name, poly in (zone_polygons or {}).items():
        # `not poly` RAISES on a numpy array, and engine.py stores polygons as
        # numpy arrays -- so this line, not the verdict logic, is what threw
        # "truth value of an array is ambiguous" on every real run. The veto
        # therefore died while BUILDING the doors: it never evaluated a single
        # pair, the engine's handler failed open, and every merge went through
        # unfiltered. Length, not truthiness.
        if name not in want or poly is None or len(poly) == 0:
            continue
        xs = [float(p[0]) for p in poly]
        ys = [float(p[1]) for p in poly]
        out.append((sum(xs) / len(xs), sum(ys) / len(ys)))
    return out


def doors_from_endpoints(entry_zones):
    """Door centres from kevacv.learn_zones.learn_entry_zones() output.

    Accepts either dicts with a 'centre'/'center' key or bare (x, y) pairs, so
    it does not care which shape that module returns.
    """
    out = []
    for z in entry_zones or []:
        if isinstance(z, dict):
            c = z.get("centre") or z.get("center")
            if c is None and z.get("polygon"):
                xs = [float(p[0]) for p in z["polygon"]]
                ys = [float(p[1]) for p in z["polygon"]]
                c = (sum(xs) / len(xs), sum(ys) / len(ys))
            if c is not None:
                out.append((float(c[0]), float(c[1])))
        elif z is not None and len(z) >= 2:
            out.append((float(z[0]), float(z[1])))
    return out


def reappearance_verdict(death_pos, birth_pos, gap_s, doors, frame_wh,
                         door_radius_frac=DOOR_RADIUS_FRAC,
                         still_drift_frac=STILL_DRIFT_FRAC,
                         short_gap_s=SHORT_GAP_S):
    """Is it physically possible that these two tracks are one person?

    -> {"allow": bool, "shape": str, "why": str, ...}

    `allow=True` never means "these ARE the same person" — appearance and the
    other tiers still decide that. It means only that topology does not forbid
    it. `allow=False` is a hard veto.

    Abstains (allow=True, shape="abstain") when there are no doors to reason
    about, because a gate with no information must not block anything.
    """
    if not doors:
        return {"allow": True, "shape": "abstain",
                "why": "no doors known — topology cannot judge this pair"}
    if gap_s is None or gap_s <= short_gap_s:
        return {"allow": True, "shape": "continuous",
                "why": (f"gap {gap_s}s is within the short-gap window "
                        f"({short_gap_s}s); the positional gate governs here")}

    diag = _diag(frame_wh)
    door_r = diag * door_radius_frac
    still_r = diag * still_drift_frac

    d_exit, exit_i = _nearest(death_pos, doors)
    d_entry, entry_i = _nearest(birth_pos, doors)
    at_exit = d_exit <= door_r
    at_entry = d_entry <= door_r
    drift = math.hypot(birth_pos[0] - death_pos[0], birth_pos[1] - death_pos[1])

    base = {"gap_s": gap_s, "drift_px": round(drift, 1),
            "exit_door_px": round(d_exit, 1), "entry_door_px": round(d_entry, 1),
            "door_radius_px": round(door_r, 1), "at_exit": at_exit,
            "at_entry": at_entry}

    # 2. left through a door, came back through a door
    if at_exit and at_entry:
        return {**base, "allow": True, "shape": "door_to_door",
                "why": (f"last seen at door {exit_i}, reappeared at door "
                        f"{entry_i} — leaving and returning is exactly this")}
    # 1. never went near a door, reappeared where they vanished
    if not at_exit and not at_entry and drift <= still_r:
        return {**base, "allow": True, "shape": "occlusion_recovery",
                "why": (f"vanished and reappeared {drift:.0f}px apart, away "
                        f"from any door — the tracker lost a body that never "
                        f"left")}
    # 3. everything else requires crossing the room unseen
    if at_exit and not at_entry:
        why = ("last seen leaving through a door but reappeared mid-room — "
               "they would have had to walk back in unobserved")
    elif at_entry and not at_exit:
        why = ("vanished mid-room but reappeared at a door — they would have "
               "had to walk to the door unobserved")
    else:
        why = (f"vanished and reappeared {drift:.0f}px apart with no door at "
               f"either end — no path exists that the camera would not have "
               f"seen")
    return {**base, "allow": False, "shape": "impossible", "why": why}


def veto_pairs(pairs, doors, frame_wh, **kw):
    """Filter candidate merge pairs, returning (kept, vetoed).

    `pairs` are dicts with at least death_pos, birth_pos and gap_s. Everything
    else on the dict is carried through untouched, so this drops into an
    existing merge pipeline without reshaping its data.
    """
    # NUMPY AMBIGUITY (seen 2026-08-20: "truth value of an array with more
    # than one element is ambiguous"). Any caller handing in a numpy point or
    # gap makes `at_exit and at_entry` an array test, which raises -- and the
    # engine's handler fails OPEN, so impossible pairs get merged silently and
    # the unique-person count moves. Coerce at the boundary: this kills the
    # whole class rather than the one caller that happened to trip it.
    def _pt(v):
        return (float(v[0]), float(v[1]))

    kept, vetoed = [], []
    for p in pairs:
        _g = p.get("gap_s")
        v = reappearance_verdict(_pt(p["death_pos"]), _pt(p["birth_pos"]),
                                 None if _g is None else float(_g),
                                 doors, frame_wh, **kw)
        (kept if v["allow"] else vetoed).append({**p, "topology": v})
    return kept, vetoed


def describe(kept, vetoed):
    L = [f"TOPOLOGY GATE — {len(kept)} possible, {len(vetoed)} vetoed"]
    shapes = {}
    for p in kept + vetoed:
        s = p["topology"]["shape"]
        shapes[s] = shapes.get(s, 0) + 1
    for s, n in sorted(shapes.items(), key=lambda kv: -kv[1]):
        L.append(f"  {s:<20} {n}")
    for p in vetoed[:5]:
        L.append(f"  VETO {p.get('a', '?')} -> {p.get('b', '?')}: "
                 f"{p['topology']['why']}")
    return "\n".join(L)
