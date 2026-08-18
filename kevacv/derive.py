"""derive.py — turn what the engine returns into what the answers need.

WHY THIS EXISTS
    process_video returns tracks, events, crossings and a frame log. The
    answers need observed windows, per-guest arrival times, staff contacts,
    a guest list and a per-person confidence. In the notebook those gaps were
    closed by Cell 18 reaching into whatever globals happened to be lying
    around. As a codebase they have to be derived explicitly, from the run
    dict, with the derivation visible.

    Nothing here invents data. Every function returns None or an empty result
    when the inputs cannot support it, because a derived number that quietly
    substitutes for a missing measurement is the exact failure that produced
    "2 people came through the door".

THE ONE RULE
    A derivation must be WEAKER than the thing it stands in for, and must say
    so. observed_windows derived from a frame log is a lower bound on what we
    watched; id_confidence derived from fragment counts is not a probability.
    Both are useful; neither may be presented as measurement.
"""
from __future__ import annotations

import math
from collections import defaultdict

from .log import get_logger

_log = get_logger("derive")

# staff within this many BODY HEIGHTS of a guest counts as a service touch
GREET_RADIUS_BODIES = 1.2
GREET_MIN_CONTACT_S = 3.0
# A guest who stood in a wait zone this long with no staff contact is the
# anomaly the report exists to surface. Not a statistical outlier — a service
# failure a human should watch. 120s is a starting point, not a measured one.
LONG_WAIT_S = 120.0


def observed_windows(run):
    """What span did we actually analyse? -> [(t0, t1), ...]

    THE MISTAKE THIS FUNCTION USED TO MAKE
        The first version derived these from frame_log timestamps. But
        frame_log only contains frames that HAD DETECTIONS, so every empty
        stretch looked unobserved and was removed from the denominator. On a
        test fixture that turned a true 66.7% desk coverage into 79.7% —
        inflating the headline metric by excluding exactly the minutes when
        nobody was at the desk.

        An empty room is OBSERVED. It is just empty. Confusing "no detections"
        with "no observation" is the same error as confusing a broken entry
        line with an empty venue, and it lands on the one metric that is
        supposed to be EXACT.

    SO: the analysed span is the denominator, unless something that genuinely
    knows better says otherwise:

        run["observed_windows"]  a validity.ValidityLedger ran and excluded
                                 blind/undecodable frames — that IS observation
                                 evidence, so it wins
        otherwise                the whole analysed span, because the engine
                                 sampled every frame in it whether or not it
                                 found anybody
    """
    if run.get("observed_windows"):
        return [tuple(w) for w in run["observed_windows"]]
    end = float(run.get("t_end") or run.get("duration_s") or 0.0)
    start = float(run.get("start_seconds") or 0.0)
    if end <= start:
        _log.warning("no duration on the run — observed time is unknown, and "
                     "every percentage will report UNKNOWN rather than guess")
        return []
    return [(start, end)]


def line_arrivals_by_id(run, roles=None):
    """First inward door crossing per non-staff track. -> {track_id: t}

    Returns {} — not a fabricated set — when the line never fired. A broken
    entry line must propagate as "no arrival times", so greet latency comes
    back UNKNOWN instead of silently becoming zero.
    """
    roles = roles or run.get("roles") or {}
    # ONLY THE VENUE DOOR IS AN ARRIVAL. Every crossing carries {"line": name}
    # and this loop ignored it, so a staff member stepping out of the staff room
    # and a guest walking INTO the dining room both became "a guest arrived".
    # The 20s smoke run on 2026-08-13 reported arrivals=1 from the DINING line.
    from .analytics import venue_entry_lines
    _lines = {c.get("line") for c in (run.get("crossings") or [])
              if c.get("line") is not None}
    _venue = venue_entry_lines(_lines) if _lines else None
    out = {}
    for c in run.get("crossings") or []:
        if c.get("direction") != "in":
            continue
        if _venue is not None and c.get("line") is not None \
                and c["line"] not in _venue:
            continue
        tid = c.get("track_id")
        if roles.get(tid) == "staff":
            continue
        t = float(c.get("t", 0.0))
        if tid not in out or t < out[tid]:
            out[tid] = t
    return out


def region_arrivals_by_id(run, roles=None):
    """First entry-zone -> interior-zone transition per non-staff track.

    Walking in IS that transition, and unlike a line it does not depend on
    where somebody drew a 218-pixel segment. -> {track_id: t}
    """
    from .arrivals import arrivals_from_regions
    roles = roles or run.get("roles") or {}
    _n, arr, _why = arrivals_from_regions(
        run.get("events") or [], run.get("zone_roles") or {}, roles=roles)
    out = {}
    for a in arr or []:
        tid, t = a.get("track_id"), float(a.get("t", 0.0))
        if tid is not None and (tid not in out or t < out[tid]):
            out[tid] = t
    return out


def arrivals_by_id(run, roles=None, prefer="region"):
    """First arrival time per non-staff track. -> {track_id: t}

    TWO SENSORS, AND WHICH ONE LEADS
        The line is precise and directional but all-or-nothing: on CAM.112 it
        fired ZERO times, because the segment is 218 px at analysis width and
        both its endpoints sit outside the entrance polygon. Everything that
        consumed arrival times — greet latency, the guest list — therefore
        died or silently fell back, and the report said "no arrivals" when it
        meant "the sensor is broken".

        Region arrivals ask the same question of the zone events instead: was
        this person in an entry zone, and then later inside? That survives a
        badly drawn line, so it LEADS (operator decision, 2026-08-12), and the
        line becomes the cross-check that can contradict it.

    prefer: "region" (default) or "line". The non-preferred sensor is still
        computed — kevacv.arrivals.cross_check compares them, and two numbers
        that should agree and do not are the evidence that something is wrong.

    run["arrival_source"] records which sensor actually supplied the times, so
    no downstream reader has to guess.
    """
    roles = roles or run.get("roles") or {}
    line = line_arrivals_by_id(run, roles)
    region = region_arrivals_by_id(run, roles)
    order = ((region, "region"), (line, "line")) if prefer == "region" else (
        (line, "line"), (region, "region"))
    for arr, name in order:
        if arr:
            run["arrival_source"] = name
            return arr
    # Both empty. Do NOT invent a fallback here: "nobody arrived" and "both
    # sensors are broken" must stay distinguishable, and the caller reports it.
    run["arrival_source"] = "none"
    return {}


def guest_ids(run, roles=None, arrivals=None):
    """Who counts as a guest. -> [track_id]

    Prefers door crossings. Falls back to "non-staff identity seen in an
    interior zone", which is weaker and is labelled as such by the caller —
    guest_count() already reports its source.
    """
    roles = roles or run.get("roles") or {}
    arrivals = arrivals if arrivals is not None else arrivals_by_id(run, roles)
    if arrivals:
        return sorted(arrivals, key=str)
    seen = {e.get("track_id") for e in (run.get("events") or [])
            if roles.get(e.get("track_id")) != "staff"}
    return sorted((t for t in seen if t is not None), key=str)


def staff_contacts(run, roles=None, radius_bodies=GREET_RADIUS_BODIES,
                   min_contact_s=GREET_MIN_CONTACT_S):
    """When was a staff member near each guest? -> {guest_id: [t_start, ...]}

    THE PROXY, STATED HONESTLY
        This measures proximity, sustained for min_contact_s. It is not
        conversation and cannot become conversation with a better tracker —
        SUCCESS_CRITERIA says the honest upgrade is a VLM on a short clip.

        The radius is in BODY HEIGHTS, not pixels: a person near the camera and
        one across the room are the same distance apart in the world at very
        different pixel separations, and a fixed pixel radius would make greet
        detection depend on where in the frame someone stood.
    """
    roles = roles or run.get("roles") or {}
    fl = run.get("frame_log") or []
    if not fl:
        return {}
    near = defaultdict(list)               # guest -> [t where staff was close]
    for _i, t, boxes in fl:
        staff, guests = [], []
        for tid, x1, y1, x2, y2 in boxes:
            cx = (float(x1) + float(x2)) / 2.0
            foot = float(y2)
            h = max(1.0, float(y2) - float(y1))
            (staff if roles.get(tid) == "staff" else guests).append(
                (tid, cx, foot, h))
        if not staff or not guests:
            continue
        for gid, gx, gy, gh in guests:
            lim = gh * radius_bodies
            if any(math.hypot(gx - sx, gy - sy) <= lim
                   for _s, sx, sy, _sh in staff):
                near[gid].append(float(t))

    out = {}
    for gid, ts in near.items():
        ts.sort()
        runs, start, prev = [], ts[0], ts[0]
        step = 1.0
        for t in ts[1:]:
            if t - prev > max(step * 3, 2.0):
                if prev - start >= min_contact_s:
                    runs.append(start)
                start = t
            prev = t
        if prev - start >= min_contact_s:
            runs.append(start)
        if runs:
            out[gid] = runs
    return out


def id_confidence(run, guests=None):
    """How much to trust each identity. -> {track_id: 0-100}

    NOT A PROBABILITY. It combines signals the pipeline already has — how long
    the identity lived, how many fragments were stitched into it, whether a
    face confirmed it, whether it crossed the door — into a number whose only
    job is to split a guest count into a range. Treat it as an ordering, not a
    measurement.
    """
    roles = run.get("roles") or {}
    windows = defaultdict(lambda: [math.inf, -math.inf])
    for e in run.get("events") or []:
        tid = e.get("track_id")
        w = windows[tid]
        w[0] = min(w[0], float(e["t_in"]))
        w[1] = max(w[1], float(e.get("t_out", e["t_in"])))
    frags = defaultdict(int)
    for a, b in (run.get("canon_map") or {}).items():
        frags[b] += 1
    faced = set(run.get("staff_matched_names") or [])
    crossed = {c.get("track_id") for c in (run.get("crossings") or [])}

    out = {}
    for tid in (guests if guests is not None else list(windows)):
        w = windows.get(tid)
        score = 40
        if w and w[1] > w[0]:
            life = w[1] - w[0]
            score += 20 if life >= 60 else 10 if life >= 20 else 0
        n = frags.get(tid, 1)
        score += 15 if n <= 1 else 5 if n <= 3 else -10   # many fragments = doubt
        if tid in crossed:
            score += 15          # a door crossing is physical evidence
        if tid in faced or (isinstance(tid, str) and not str(tid).isdigit()):
            score += 20          # a recognised face is the strongest signal
        out[tid] = max(0, min(100, score))
    return out


def _span_by_track(run):
    """-> {track_id: (t_first, t_last)} across all zones."""
    w = defaultdict(lambda: [math.inf, -math.inf])
    for e in run.get("events") or []:
        s = w[e.get("track_id")]
        s[0] = min(s[0], float(e["t_in"]))
        s[1] = max(s[1], float(e.get("t_out", e["t_in"])))
    return {t: (a, b) for t, (a, b) in w.items() if b >= a}


def _wait_seconds(run):
    """-> {track_id: seconds spent in a WAIT-role zone}."""
    zr = run.get("zone_roles") or {}
    waits = {z for z, rs in zr.items() if "wait" in (rs or [])}
    out = defaultdict(float)
    for e in run.get("events") or []:
        if e.get("zone") in waits:
            out[e.get("track_id")] += float(
                e.get("duration", e.get("t_out", 0) - e.get("t_in", 0)))
    return dict(out)


def report_rows(run, clock=None):
    """Build the people / staff / anomaly rows the report writes.

    WHY THIS LIVES HERE AND NOT IN THE PIPELINE
        report_slim.write_slim_outputs takes `people`, `staff` and `anomalies`,
        and the pipeline passed `result.get(...) or []` for all three — keys it
        never set. So people.csv had a header and no rows, WHO WORKED THE DESK
        said "nobody was identified", and WHAT WENT WRONG said "nothing
        flagged", on a run with 45 tracked people and 6 desk gaps.

        None of those were findings. They were three unwired sections that
        render, when empty, as confident negative statements. That is the same
        failure the tier system exists to prevent: "not measured" printed as
        "measured zero".

    -> (people, staff, anomalies). Every field that was not measured is None,
    never 0, so report_slim writes "" rather than a number that reads as fact.
    """
    roles = run.get("roles") or {}
    spans = _span_by_track(run)
    waited = _wait_seconds(run)
    arrivals = run.get("arrivals_by_id") or {}
    contacts = run.get("contacts") or {}
    conf = run.get("id_confidence") or {}
    guests = set(run.get("guest_ids") or [])

    def _t(v):
        return clock(v) if (clock and v is not None) else (
            None if v is None else f"{int(v) // 60:02d}:{int(v) % 60:02d}")

    people = []
    # ids are int for tracker fragments and str for gallery-named staff, so a
    # bare sorted() raises TypeError comparing the two. Sort on (time, str).
    for tid, (t0, t1) in sorted(spans.items(),
                                key=lambda kv: (kv[1][0], str(kv[0]))):
        if roles.get(tid) == "staff":
            continue
        cs = contacts.get(tid) or []
        arr = arrivals.get(tid)
        flags = []
        if arr is None:
            flags.append("no-door-crossing")
        if (conf.get(tid, 0) or 0) < 55:
            flags.append("low-confidence")
        if tid not in guests:
            flags.append("not-counted-as-guest")
        people.append({
            "person": tid,
            # blank until something actually banks a per-person crop.
            # A path to a file that was never written is a claim the
            # report did not earn.
            "snap": None,
            "role": roles.get(tid) or "customer",
            "role_from": "zone-inferred" if roles.get(tid) else "default",
            "first_seen": _t(t0), "last_seen": _t(t1),
            "minutes": round((t1 - t0) / 60.0, 1),
            "waited_s": round(waited[tid]) if tid in waited else None,
            "greeted": "yes" if cs else "no",
            # greet latency needs BOTH an arrival and a contact. Without the
            # door it is unknowable, and None keeps it out of the CSV as blank
            # rather than as a zero anyone could average.
            "greet_s": (round(cs[0] - arr, 1)
                        if (cs and arr is not None) else None),
            "confidence": conf.get(tid),
            "flags": " ".join(flags) or None,
        })

    obs = run.get("observed_windows") or []
    observed_s = sum(b - a for a, b in obs) or None

    # TIME AT THE DESK, not time on camera.
    #
    # This reported (last_seen - first_seen): how long the TRACK EXISTED. Under
    # a heading that says "WHO WORKED THE DESK", next to a bar chart, that
    # reads as occupancy and is not. On the first real run track 524 showed
    # "42.7 min  71%" while the staff-decision table recorded its actual
    # reception dwell as 101 seconds — the 71% was its SPREAD, the span from
    # first to last sighting.
    #
    # It also contradicted the headline: desk coverage said 23.4% on the same
    # events, because desk_coverage measures presence and this measured
    # lifetime. Two numbers from one run that cannot both be true is worse
    # than either being wrong, because it makes the whole report unbelievable.
    _staff_zones = {z for z, rs in (run.get("zone_roles") or {}).items()
                    if "staff" in (rs or [])}
    _iv = defaultdict(list)
    for e in run.get("events") or []:
        if e.get("zone") in _staff_zones:
            _iv[e["track_id"]].append((float(e["t_in"]), float(e["t_out"])))

    def _union_s(ivs):
        """Total covered seconds. Zones can overlap and a track can re-enter,
        so summing durations double-counts."""
        if not ivs:
            return 0.0
        merged = []
        for a, b in sorted(ivs):
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        return sum(b - a for a, b in merged)

    staff = []
    for tid, (t0, t1) in sorted(spans.items(), key=lambda kv: str(kv[0])):
        if roles.get(tid) != "staff":
            continue
        desk_s = _union_s(_iv.get(tid, []))
        mins = desk_s / 60.0
        staff.append({
            "name": tid,
            "minutes": round(mins, 1),
            "pct": (round(100.0 * desk_s / observed_s)
                    if observed_s else None),
            # kept so the two are never confused again: how long they were on
            # camera at all, which is what this used to print as "minutes"
            "on_camera_min": round((t1 - t0) / 60.0, 1),
            # a non-numeric id came from the face gallery; a numeric one was
            # only ever inferred from standing in the staff zone
            "source": ("face" if isinstance(tid, str) and not str(tid).isdigit()
                       else "staff-zone"),
            "confidence": conf.get(tid),
        })

    anomalies = []
    for p in people:
        w = p["waited_s"]
        if w is not None and w >= LONG_WAIT_S and p["greeted"] == "no":
            anomalies.append({
                "time": p["first_seen"],
                "what": (f"{p['person']} waited {int(w // 60)}m{int(w % 60):02d}s "
                         f"with no staff within {GREET_RADIUS_BODIES} body "
                         f"heights"),
                "clip": p["snap"],
                "severity": "high" if w >= 2 * LONG_WAIT_S else "medium",
            })
    anomalies.sort(key=lambda a: a["severity"] != "high")
    return people, staff, anomalies


def enrich(run):
    """Add every derived field the answers need. Returns the same dict.

    Called once, in the pipeline, so the derivation happens in exactly one
    place and a later reader can see what was measured and what was inferred.
    """
    roles = run.get("roles") or {}
    run.setdefault("observed_windows", observed_windows(run))
    arr = arrivals_by_id(run, roles)
    run.setdefault("arrivals_by_id", arr)
    gids = guest_ids(run, roles, arr)
    run.setdefault("guest_ids", gids)
    run.setdefault("contacts", staff_contacts(run, roles))
    run.setdefault("id_confidence", id_confidence(run, gids))
    obs = run["observed_windows"]
    _src = run.get("arrival_source", "none")
    _log.info(f"derived: observed={len(obs)} window(s) "
              f"({sum(b - a for a, b in obs):.0f}s)  "
              f"arrivals={len(arr)} (source: {_src})  "
              f"guests={len(gids)}  contacts={len(run['contacts'])}")
    if _src == "line":
        _log.warning("arrival times came from the LINE, not the region method "
                     "— the region method produced nothing, which means the "
                     "entry/interior zones cannot support the question. Check "
                     "the entry polygon before trusting greet latency.")
    if not arr:
        _log.warning("BOTH arrival sensors are empty — the line never fired "
                     "AND no entry->interior transition was seen. Greet "
                     "latency reports UNKNOWN rather than 0, and the guest "
                     "list falls back to interior-zone sightings. This is a "
                     "zone problem, not an empty venue, unless the venue was "
                     "genuinely empty.")
    return run
