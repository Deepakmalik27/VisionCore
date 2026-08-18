"""analytics.py — tracks in, people out.

EXTRACTED FROM notebook Cell 5 ("analytics logic, fully unit-tested"). This is
the layer that turns raw tracker output into identities and then into the
numbers a manager reads: who merged with whom, how long the desk was covered,
who waited.

WHAT IT DOES, IN ORDER
    interval helpers          overlap / coverage arithmetic on time windows
    OccupancyRecorder         how many people were in a zone, second by second
    zone metrics              entered / seated / waited / reception_absence
    _IdentityMemory           LIVE re-id: one canonical id per body, per frame
    merge_fragmented_tracks   OFFLINE re-id: six evidence tiers, greedy union
    calibrate_appearance_*    measures this run's own same/different similarity
    cross_validate / veto     a second opinion may BLOCK a merge, never make one

THE ONE NUMBER THAT EXPLAINS THE REST
    On CAM.112 the merge tiers accepted 69 merges: stationary 54, hand-off 7,
    gallery 6, attire 1. Physics produced 90% of them; the whole appearance
    stack produced 9%. Meanwhile 356 candidates were blocked by window overlap
    — five times more than were accepted — because the greedy union widens a
    group's window each time it absorbs a fragment, so later merges collide
    with windows that earlier merges donated.

    kevacv.graph_fusion (min-cost flow) and kevacv.topology (a hard veto on
    physically impossible re-appearances) both exist to attack that. Neither
    is called from here yet.

CONFIG
    Tunables come from kevacv.config, not from notebook globals. Cell 2e used
    to override three of them after the audit had already printed the Cell 2
    values; config.py holds the values that actually run.
"""
from __future__ import annotations

from .log import get_logger, stage, banner

_log = get_logger("analytics")


import math
from collections import Counter, defaultdict

import cv2
import numpy as np

from .config import (ANCHOR_SIM_THRESHOLD, APPEARANCE_HSV_VETO_SIM,
                     DEFAULT_HFOV_DEG, ENABLE_APPEARANCE_HSV_VETO,
                     ENABLE_HANDOFF_APPEARANCE_VETO, ENABLE_HANDOFF_HSV_VETO,
                     FAR_GAP_PENALTY, FAR_GAP_S, HANDOFF_HSV_VETO_SIM,
                     HANDOFF_VETO_SIM, IR_TRACK_FRAC, MAX_BODY_GAP_S,
                     MAX_PLAUSIBLE_SPEED_PX, MAX_SPATIAL_PENALTY,
                     NEAR_GAP_BONUS, NEAR_GAP_S, OCCLUSION_CONTAIN,
                     LIVE_REID_ABS_FLOOR, OCCLUSION_IOU, REID_MAX_GAP_S,
                     REID_RATIO, REID_SIM_THRESHOLD,
                     SPATIAL_PENALTY_SCALE)
from .ground_plane import PERSON_H_M, GroundPlane

# Cell 5 — analytics logic (fully unit-tested)
"""Core zone-analytics logic for the restaurant video POC.

Pure Python (no numpy/pandas/cv2) so every metric is unit-testable
without a video. The notebook's video engine produces per-frame zone
occupancy; everything below turns that into events and answers.

Time is always seconds from video start (float).
Roles are "customer" / "staff" / "unknown".
"""

# v42: staff string names always win as canonical ID over numeric track IDs
def _canon_priority_key(track_windows):
    """Sort key: string names (staff) first, then by appearance time."""
    def key(t):
        if isinstance(t, str) and not str(t).isdigit():
            return (0, 0.0, str(t))        # staff names always first
        return (1, track_windows.get(t, (0,0))[0], str(t))
    return key

def find_optimal_threshold(same_sims, diff_sims):
    """Find the threshold that maximizes separation (EER approach).

    Sweeps thresholds to find where false-accept-rate == false-reject-rate.
    This is the 'best spot' where the signal is cleanest.

    Returns: (optimal_threshold, metrics_dict)
    """
    import numpy as _np
    if not same_sims or not diff_sims:
        return 0.55, {"note": "insufficient data for EER sweep"}
    same_arr = _np.array(same_sims)
    diff_arr = _np.array(diff_sims)
    thresholds = _np.arange(0.0, 1.001, 0.01)
    best_thresh, best_score = 0.55, -1.0
    for t in thresholds:
        fr = _np.mean(same_arr < t)     # false reject: same-person below threshold
        fa = _np.mean(diff_arr >= t)    # false accept: diff-person above threshold
        score = 1.0 - (fr + fa) / 2.0  # balanced accuracy
        if score > best_score:
            best_thresh, best_score = float(t), float(score)
    return round(best_thresh, 2), {
        "balanced_accuracy": round(best_score, 4),
        "same_person_median": round(float(_np.median(same_arr)), 4),
        "diff_person_median": round(float(_np.median(diff_arr)), 4),
        "separation_gap": round(float(_np.median(same_arr) - _np.median(diff_arr)), 4),
    }

import math
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Interval helpers
# ---------------------------------------------------------------------------

def merge_intervals(intervals, gap=0.0):
    """Merge [start, end] intervals, joining any separated by <= gap seconds."""
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda iv: iv[0])
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [tuple(iv) for iv in merged]


def total_duration(intervals):
    return sum(end - start for start, end in intervals)


def covered_windows(run):
    """The stretches of the night we actually have video for. A missing hour is
    not a quiet hour, and no metric may average over footage that doesn't
    exist."""
    cb = run.get("chunk_bounds")
    if cb:
        return merge_intervals([(float(s), float(e)) for _k, s, e in cb])
    return [(0.0, float(run.get("t_end", 0.0)))]


def clip_to(intervals, windows):
    """Intersection of two interval lists."""
    out = []
    for a0, a1 in intervals:
        for b0, b1 in windows:
            lo, hi = max(a0, b0), min(a1, b1)
            if hi > lo:
                out.append((lo, hi))
    return sorted(out)


def complement_intervals(intervals, t_start, t_end):
    """Gaps inside [t_start, t_end] not covered by intervals (assumed merged)."""
    gaps, cursor = [], t_start
    for start, end in sorted(intervals, key=lambda iv: iv[0]):
        if start > cursor:
            gaps.append((cursor, min(start, t_end)))
        cursor = max(cursor, end)
        if cursor >= t_end:
            break
    if cursor < t_end:
        gaps.append((cursor, t_end))
    return [(s, e) for s, e in gaps if e - s > 1e-9]


# ---------------------------------------------------------------------------
# Occupancy -> events
# ---------------------------------------------------------------------------

class OccupancyRecorder:
    """Collects per-frame zone occupancy and consolidates it into events.

    add(t, zone, track_id): call once per (processed frame, zone, visible track).
    Consolidation merges consecutive sightings into [t_in, t_out] events,
    bridging gaps <= gap_merge_s (occlusion tolerance) and dropping events
    shorter than min_event_s (detection flicker).
    """

    def __init__(self, frame_step_s, gap_merge_s=15.0, min_event_s=2.0):
        self.frame_step_s = frame_step_s
        self.gap_merge_s = gap_merge_s
        self.min_event_s = min_event_s
        self._sightings = defaultdict(list)   # (zone, track_id) -> [t, ...]
        self.role_votes = defaultdict(Counter)  # track_id -> Counter(role)

    def add(self, t, zone, track_id):
        self._sightings[(zone, track_id)].append(t)

    def vote_role(self, track_id, role):
        self.role_votes[track_id][role] += 1

    def final_roles(self):
        return {tid: votes.most_common(1)[0][0]
                for tid, votes in self.role_votes.items()}

    def events(self, min_event_by_zone=None):
        """-> list of dicts {track_id, role, zone, t_in, t_out, duration}.
        min_event_by_zone: optional {zone: seconds} override — entry zones need
        a lower bar because a doorway transit at walking pace is ~1s and the
        global 2.0s minimum silently deleted every arrival event."""
        roles = self.final_roles()
        out = []
        for (zone, tid), times in self._sightings.items():
            _min_s = (min_event_by_zone or {}).get(zone, self.min_event_s)
            # each sighting covers one frame step
            ivs = merge_intervals(
                [(t, t + self.frame_step_s) for t in sorted(times)],
                gap=self.gap_merge_s,
            )
            for t_in, t_out in ivs:
                if t_out - t_in >= _min_s:
                    out.append({
                        "track_id": tid,
                        "role": roles.get(tid, "unknown"),
                        "zone": zone,
                        "t_in": round(t_in, 2),
                        "t_out": round(t_out, 2),
                        "duration": round(t_out - t_in, 2),
                    })
        # MIXED ID TYPES. track_id is an int for a tracker fragment and a STRING
        # for a gallery-named staff member ("staff2"), so a bare tuple sort
        # raises TypeError the moment a face finally matches. It killed a run
        # 1h58m in, at the last step before the report, on 2026-08-14 -- the
        # first run where the gallery matched anybody, which is why 20-second
        # smoke tests could never reach it.
        #
        # track_sort_key exists in this very file for exactly this and is used
        # 2000 lines below; this call site simply did not use it. derive.py
        # carries the same lesson in a comment. Third occurrence, same bug.
        out.sort(key=lambda e: (e["t_in"], str(e["zone"]),
                                track_sort_key(e["track_id"])))
        return out


# ---------------------------------------------------------------------------
# Door-camera metrics
# ---------------------------------------------------------------------------

def venue_entry_lines(line_names):
    """Which door lines mean "arrived at the venue" -> set of names.

    A venue has more than one threshold and they do NOT mean the same thing:

        entry line     street -> venue        an ARRIVAL
        dining entry   venue  -> dining       a seating transit
        staff entry    staff room <-> floor   a staff movement

    Derived from the operator's own labels through the same classify_zones the
    engine uses, so nothing is hardcoded to one venue: a line is a venue
    entrance when it reads as ENTRY and does NOT also read as an interior
    destination (seating/staff/service). "dining entry" classifies as
    entry+seating, "staff entry" as entry+staff; only "entry line" is entry
    alone.

    Fallback: if that leaves nothing, every entry line counts. A single oddly
    named door should still be counted rather than silently producing zero.
    """
    # NOT classify_zones: it deliberately makes "entry" WIN over "seating" in a
    # compound name, so "dining entry" comes back as ['entry'] alone and the
    # interior test below would never fire. That precedence is right for a
    # ZONE (a dining entrance is an entrance) and wrong for a DOOR, where the
    # question is which side of it you end up on. So match the raw table and
    # keep every role the name earns.
    from .helpers import ZONE_ROLE_KEYWORDS
    interior = {"seating", "staff", "service"}

    def _roles(name):
        low = str(name).lower()
        return {r for r, kws in ZONE_ROLE_KEYWORDS.items()
                if any(k in low for k in kws)}

    all_roles = {n: _roles(n) for n in (line_names or [])}
    pure = {n for n, rs in all_roles.items()
            if "entry" in rs and not (rs & interior)}
    return (pure or {n for n, rs in all_roles.items() if "entry" in rs}
            or set(line_names or []))


def entered_count(crossings, roles=None, lines=None):
    """Unique track_ids that crossed INWARD at a venue entrance.

    `lines`: the door names that count as a venue entrance. Default None = every
    line, which is exactly what this did unconditionally before -- and it was
    wrong. Every crossing already carried {"line": name} from engine.py and this
    function simply never read it, so a staff member stepping out of the staff
    room and a guest walking into the dining room were both counted as GUEST
    ARRIVALS. That is the headline business question ("How many guests?")
    inflated by every interior movement in the building.

    Pass venue_entry_lines(<line names>) to get it right.
    """
    keep = set(lines) if lines else None
    seen = set()
    for c in crossings:
        if c["direction"] != "in":
            continue
        # A crossing with no "line" predates per-door recording; count it rather
        # than drop it, so an old run does not silently become zero.
        if keep is not None and c.get("line") is not None and c["line"] not in keep:
            continue
        tid = c["track_id"]
        if roles and roles.get(tid) == "staff":
            continue
        seen.add(tid)
    return len(seen)


def seated_count(events, table_zones, min_seated_s=60.0):
    """Unique customers who dwelled >= min_seated_s in any table/dining zone."""
    seated = set()
    for e in events:
        if (e["zone"] in table_zones and e["role"] == "customer"
                and e["duration"] >= min_seated_s):
            seated.add(e["track_id"])
    return len(seated)


def waited_over(events, waiting_zone, threshold_s=600.0, gap=30.0):
    """Customers whose merged dwell in the waiting zone >= threshold_s.
    Returns (count, {track_id: total_wait_s}) - dict includes ALL waiters
    so the demo can show the distribution, not just the threshold count.
    """
    per_track = defaultdict(list)
    for e in events:
        if e["zone"] == waiting_zone and e["role"] == "customer":
            per_track[e["track_id"]].append((e["t_in"], e["t_out"]))
    waits = {tid: round(total_duration(merge_intervals(ivs, gap=gap)), 1)
             for tid, ivs in per_track.items()}
    over = {tid: w for tid, w in waits.items() if w >= threshold_s}
    return len(over), waits


def reception_absence(events, reception_zone, t_start, t_end, min_gap_s=30.0):
    """How long was reception unstaffed?  Union of STAFF presence intervals in
    the reception zone, complemented over [t_start, t_end].
    Returns dict: total_away_s, longest_away_s, gaps (only gaps >= min_gap_s).
    """
    staffed = [(e["t_in"], e["t_out"]) for e in events
               if e["zone"] == reception_zone and e["role"] == "staff"]
    gaps = [(s, e) for s, e in
            complement_intervals(merge_intervals(staffed), t_start, t_end)
            if e - s >= 1.0]   # drop sub-second boundary slivers
    big_gaps = [(round(s, 1), round(e, 1)) for s, e in gaps if e - s >= min_gap_s]
    return {
        "total_away_s": round(sum(e - s for s, e in gaps), 1),
        "longest_away_s": round(max((e - s for s, e in gaps), default=0.0), 1),
        "gaps": big_gaps,
    }


# ---------------------------------------------------------------------------
# Table-camera metrics
# ---------------------------------------------------------------------------

def detect_parties(events, table_zone, party_gap_s=120.0, min_party_s=60.0):
    """A 'party' = a maximal window where the table has customers, with
    interior gaps < party_gap_s merged (people lean out of frame, etc.)."""
    cust = [(e["t_in"], e["t_out"]) for e in events
            if e["zone"] == table_zone and e["role"] == "customer"]
    return [iv for iv in merge_intervals(cust, gap=party_gap_s)
            if iv[1] - iv[0] >= min_party_s]


def staff_visits(events, table_zone, window, visit_min_s=8.0, merge_gap_s=30.0):
    """Distinct staff visits to a table overlapping [window]. Consecutive staff
    presence separated by < merge_gap_s counts as ONE visit."""
    w_start, w_end = window
    ivs = [(e["t_in"], e["t_out"]) for e in events
           if e["zone"] == table_zone and e["role"] == "staff"
           and e["t_out"] > w_start and e["t_in"] < w_end]
    return [iv for iv in merge_intervals(ivs, gap=merge_gap_s)
            if iv[1] - iv[0] >= visit_min_s]


def table_service_metrics(events, table_zones, party_gap_s=120.0,
                          min_party_s=60.0, visit_min_s=8.0):
    """Per party per table: seating->first-visit ('order' proxy),
    first->second visit ('food' proxy), total visit count.
    Returns (per_party_rows, averages_dict)."""
    rows = []
    for table in sorted(table_zones):
        for party in detect_parties(events, table, party_gap_s, min_party_s):
            visits = staff_visits(events, table, party, visit_min_s)
            seat_t = party[0]
            order_t = visits[0][0] if len(visits) >= 1 else None
            food_t = visits[1][0] if len(visits) >= 2 else None
            rows.append({
                "table": table,
                "party_start": round(seat_t, 1),
                "party_end": round(party[1], 1),
                "n_staff_visits": len(visits),
                "seating_to_order_s": (round(order_t - seat_t, 1)
                                       if order_t is not None else None),
                "order_to_food_s": (round(food_t - order_t, 1)
                                    if food_t is not None else None),
            })

    def avg(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return round(sum(vals) / len(vals), 1) if vals else None

    averages = {
        "avg_seating_to_order_s": avg("seating_to_order_s"),
        "avg_order_to_food_s": avg("order_to_food_s"),
        "avg_visits_per_party": (round(sum(r["n_staff_visits"] for r in rows)
                                       / len(rows), 2) if rows else None),
        "n_parties": len(rows),
    }
    return rows, averages



# v42 mix track_id sort key (some are integers, some are string names like 'jane')
def track_sort_key(tid):
    if isinstance(tid, str) and not tid.isdigit():
        return (0, tid)
    try:
        return (1, int(tid))
    except ValueError:
        return (2, str(tid))

def staff_evidence(events, staff_zones, observation_s=None, exclude_ids=()):
    """Per-track evidence for the staff/customer decision, as data.

    Returned separately from the decision so a run can PRINT why each identity
    was called staff. "31 of 45 identities are staff" is unactionable; "31 were
    called staff, 28 of them on a single 70 s visit" names the bug.

    Per track id:
        dwell_s      total time inside staff-only zones
        lifetime_s   last_seen - first_seen across all zones
        share        dwell_s / lifetime_s, capped at 1.0
        visits       separate staff-zone events (returns to the desk)
        span_s       last staff-zone exit - first staff-zone entry
        spread       span_s / observation_s, 0.0 when observation_s is unknown

    exclude_ids: tracks that are KNOWN not to be people — the static/phantom
        filter's verdict. A potted plant at the reception desk produces
        detections that gap out and resume all evening, and to a rule counting
        "returns to the desk" that is indistinguishable from a receptionist
        coming and going. The first real run labelled track 256 staff on 11
        visits and a spread of 0.84, and the static filter then removed it as
        furniture that had not moved in 3084 seconds. The filter ran AFTER the
        role was assigned, so the label was already downstream.
    """
    exclude_ids = set(exclude_ids or ())
    dwell = defaultdict(float)
    visits = defaultdict(int)
    first_seen, last_seen = {}, {}
    s_first, s_last = {}, {}
    for e in events:
        tid = e["track_id"]
        if tid in exclude_ids:
            continue
        first_seen[tid] = min(first_seen.get(tid, e["t_in"]), e["t_in"])
        last_seen[tid] = max(last_seen.get(tid, e["t_out"]), e["t_out"])
        if e["zone"] in staff_zones:
            dwell[tid] += e["duration"]
            visits[tid] += 1
            s_first[tid] = min(s_first.get(tid, e["t_in"]), e["t_in"])
            s_last[tid] = max(s_last.get(tid, e["t_out"]), e["t_out"])
    out = {}
    for tid, d in dwell.items():
        # share of LIFETIME, not of summed zone dwells — overlapping zones
        # (a host stand inside a larger lobby polygon) would otherwise dilute
        # a desk-anchored person's share below the threshold
        lifetime = max(last_seen[tid] - first_seen[tid], 1e-9)
        span = max(s_last[tid] - s_first[tid], 0.0)
        out[tid] = {
            "dwell_s": d,
            "lifetime_s": lifetime,
            "share": min(d / lifetime, 1.0),
            "visits": visits[tid],
            "span_s": span,
            "spread": (span / observation_s) if observation_s else 0.0,
        }
    return out


def apply_staff_zone_override(events, staff_zones, min_staff_dwell_s=60.0,
                              min_share=0.6, min_share_dwell_s=15.0,
                              observation_s=None, min_visits=2,
                              min_spread=0.25, sole_dwell_s=None,
                              return_evidence=False, exclude_ids=(),
                              premerge_visits=None):
    """Re-label tracks as 'staff' from where and how they spent their time.

    THE RULE THAT WAS HERE, AND WHY IT WAS WRONG
        "dwell in a staff zone >= 60 s" plus a share variant for fragments.
        At a one-receptionist desk that labelled 31 of 45 identities staff,
        because A GUEST STANDING AT THE COUNTER ALSO SPENDS ~100% OF A SHORT
        TRACK THERE. Dwell and share cannot separate them: both describe a
        single visit, and a single visit is exactly what a customer makes.
        kevacv/config.py records the diagnosis and names the fix — SPREAD.

    THE DISCRIMINATOR
        Staff RETURN. Across a shift a receptionist leaves the desk and comes
        back many times, so their staff-zone activity is scattered across the
        whole observation window. A guest arrives once, is served, and leaves;
        their activity is one contiguous blob however long it lasts.

        So a track is staff when EITHER

          (a) SPREAD — it visited staff zones >= min_visits separate times AND
              those visits span >= min_spread of the observation window. This
              is the rule that actually separates the two populations, and it
              gets stronger the longer the run, where dwell gets weaker.

          (b) SOLE OCCUPANCY — dwell alone reaches sole_dwell_s, a bar set far
              above any plausible customer interaction. Defaults to 10x
              min_staff_dwell_s. This is the backstop for a single-camera hour
              too short for (a) to have evidence, NOT the primary rule.

        The old dwell/share rules remain available but are now OFF unless the
        caller sets observation_s=None — i.e. unless it cannot compute spread.
        A caller with no observation window gets the old behaviour and a
        weaker answer, rather than silently getting no staff at all.

    observation_s: length of the analysed period. Without it spread cannot be
        computed and the function falls back to dwell/share.
    premerge_visits: optional {track_id: visits} measured BEFORE Re-ID
        stitching. When given, rule (a) additionally requires the identity to
        have earned min_visits on its own, un-merged evidence. Without this a
        wrong merge fabricates a staff member out of several guests: pooling
        N one-visit fragments yields N visits spread across the whole window,
        which is precisely the signature (a) looks for. Rule (b) is unaffected
        — real dwell cannot be manufactured by relabelling.
    return_evidence: also return the per-track evidence dict, for logging.

    Returns a new events list; input is never mutated.
    """
    if not staff_zones:
        return (events, {}) if return_evidence else events
    if sole_dwell_s is None:
        sole_dwell_s = min_staff_dwell_s * 10.0

    ev = staff_evidence(events, staff_zones, observation_s=observation_s,
                        exclude_ids=exclude_ids)
    staff_ids = set()
    for tid, x in ev.items():
        if observation_s:
            by_spread = (x["visits"] >= min_visits
                         and x["spread"] >= min_spread)
            # MERGE-MANUFACTURED SPREAD (see premerge_visits in the docstring):
            # when this identity only reaches min_visits because Re-ID glued
            # several one-visit fragments together, the "returns" are an
            # artefact of the merge, not behaviour. Measured on CAM.112: the
            # staff verdict went 1-of-32 before stitching to 7-of-13 after,
            # and five of the six new ones had exactly ONE visit pre-merge.
            if by_spread and premerge_visits is not None:
                if premerge_visits.get(tid, 0) < min_visits:
                    by_spread = False
                    x["merge_only"] = True
            by_sole = x["dwell_s"] >= sole_dwell_s
            x["reason"] = ("spread" if by_spread else
                           "sole_occupancy" if by_sole else None)
            if by_spread or by_sole:
                staff_ids.add(tid)
        else:
            # legacy path: no observation window, so no spread. Kept so an
            # older caller degrades to the previous answer instead of to none.
            by_dwell = x["dwell_s"] >= min_staff_dwell_s
            by_share = (x["dwell_s"] >= min_share_dwell_s
                        and x["share"] >= min_share)
            x["reason"] = ("dwell" if by_dwell else
                           "share" if by_share else None)
            if by_dwell or by_share:
                staff_ids.add(tid)

    out = [dict(e, role="staff") if e["track_id"] in staff_ids else dict(e)
           for e in events]
    return (out, ev) if return_evidence else out


def describe_staff_decision(evidence, width=78):
    """Every track that spent ANY time in a staff zone, and the call made on
    it. Printed in full because the failure mode is silent misclassification:
    a wrong staff label looks identical to a right one in every downstream
    number."""
    L = ["=" * width, "  STAFF DECISION — who was called staff, and on what", "=" * width]
    if not evidence:
        L.append("  (nobody entered a staff zone — check the zone is drawn)")
        L.append("=" * width)
        return "\n".join(L)
    L.append(f"  {'track':<14}{'dwell':>8}{'share':>8}{'visits':>8}"
             f"{'spread':>8}   verdict")
    L.append("  " + "-" * (width - 4))
    for tid, x in sorted(evidence.items(),
                         key=lambda kv: -kv[1]["dwell_s"]):
        verdict = x.get("reason") or "customer"
        L.append(f"  {str(tid):<14}{x['dwell_s']:>7.0f}s{x['share']:>8.2f}"
                 f"{x['visits']:>8}{x['spread']:>8.2f}   {verdict}")
    n_staff = sum(1 for x in evidence.values() if x.get("reason"))
    L.append("  " + "-" * (width - 4))
    L.append(f"  {n_staff} of {len(evidence)} staff-zone visitors called staff")
    L.append("=" * width)
    return "\n".join(L)


def occupancy_timeline(events, t_end, step_s=10.0):
    """Distinct people present per role at each time step (for occupancy charts).
    Returns (times, {role: [count per step]}). Same person in two zones in one
    step counts once."""
    n = max(1, int(math.ceil(t_end / step_s)))
    times = [round(i * step_s, 2) for i in range(n)]
    roles = sorted({e["role"] for e in events}) or ["customer"]
    counts = {r: [0] * n for r in roles}
    for i, t0 in enumerate(times):
        t1 = t0 + step_s
        seen = defaultdict(set)
        for e in events:
            if e["t_out"] > t0 and e["t_in"] < t1:
                seen[e["role"]].add(e["track_id"])
        for r in roles:
            counts[r][i] = len(seen[r])
    return times, counts


# ---------------------------------------------------------------------------
# Re-ID stitching: merge track fragments that are the same person
# ---------------------------------------------------------------------------

def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Cross-validation (v26): independent HSV-histogram check on OSNet merges.
# Deliberately a DIFFERENT feature space from the OSNet embeddings used by
# merge_fragmented_tracks — the point is corroboration, not a second vote
# with the same blind spots. Runs on the exact crops already banked for this
# run (track_crops), so track IDs match the OSNet mapping 1:1 with no manual
# reconciliation, unlike re-running tracking in a separate script.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# LIVE identity memory (v27): runs INSIDE the per-frame tracking loop, not
# as a post-hoc offline stitch. Purpose: when the online tracker (BotSORT)
# mints a brand-new raw track_id, decide BEFORE that id ever reaches
# events/zones/render whether it's actually a person we already know —
# someone whose id just got swapped mid-visibility, not someone who left
# and came back. That second case (real gap) is what merge_fragmented_tracks
# already handles offline; this handles the case it structurally can't
# (overlapping/near-continuous visibility windows).
# ---------------------------------------------------------------------------


def _dynamic_accept_thresh(gap, base_thresh):
    near_s, far_s = NEAR_GAP_S, FAR_GAP_S
    if gap <= near_s:
        return base_thresh - NEAR_GAP_BONUS
    if gap >= far_s:
        return base_thresh + FAR_GAP_PENALTY
    frac = (gap - near_s) / (far_s - near_s)
    return (base_thresh - NEAR_GAP_BONUS) + frac * (NEAR_GAP_BONUS + FAR_GAP_PENALTY)


def _boxes_occluding(a, b):
    """v44: True if two xyxy boxes mutually occlude — either their IoU is high
    or one is largely contained in the other. During such an overlap a body
    crop is contaminated by the neighbouring body, so appearance must NOT be
    trusted to update anchors or drive re-assignment."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return False
    aarea = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    barea = max(1.0, (bx2 - bx1) * (by2 - by1))
    iou = inter / (aarea + barea - inter)
    contain = inter / min(aarea, barea)
    return iou >= OCCLUSION_IOU or contain >= OCCLUSION_CONTAIN


_NOVEC = object()   # v55 sentinel: 'no vector supplied, compute it yourself'


class _IdentityMemory:

    """Rolling memory of recently-seen identities: one best snapshot
    (anchor embedding), last known position, and last-seen time per
    canonical id. `resolve()` is called once per detection per frame and
    returns the canonical id that detection should be recorded under.

    Design intentionally mirrors the anchor-snapshot logic already used
    offline (ANCHOR_SIM_THRESHOLD / merge_fragmented_tracks): single
    best-quality crop, not a pooled/averaged embedding, compared 1:1.
    A live false-positive here is worse than an offline one (no
    cross-validation pass reviews it after the fact), so the bar
    (LIVE_REID_SIM_THRESHOLD) is set stricter than the offline gallery
    threshold on purpose.
    """
    def __init__(self, embed_fn, sim_threshold=0.72, max_dist_px=140.0,
                memory_ttl_s=6.0, dist_scale=1.0, plane=None,
                max_speed_mps=2.2, ratio=None, abs_floor=None):
        self.embed_fn = embed_fn
        self.sim_threshold = sim_threshold
        # G2: with a ground plane the gate becomes "could a person physically
        # have walked that far in the time available" — one statement about the
        # world, correct in every part of the frame. max_dist_px survives only
        # as the fallback for footage where no plane could be fitted.
        self.plane = plane
        self.max_speed_mps = max_speed_mps
        self.max_dist_px = max_dist_px
        self.memory_ttl_s = memory_ttl_s
        self.dist_scale = dist_scale   # v45: resolution-relative scaling factor
        self.bank = {}          # canonical_id -> {"anchor","pos","t","best_score"}
        self.raw_to_canon = {}  # raw tracker track_id -> canonical_id
        self.raw_bound_t = {}   # raw track_id -> when that binding was made
        # RELATIVE decision (Lowe's ratio) with the absolute bar demoted to a
        # sanity floor. See resolve() for why a fixed threshold cannot work on
        # footage that is 45% infrared.
        self.ratio = REID_RATIO if ratio is None else ratio
        self.abs_floor = (LIVE_REID_ABS_FLOOR if abs_floor is None
                          else abs_floor)
        self.ratio_rejects = 0  # matches refused as ambiguous, not as too weak
        self.reassignments = 0  # count of births redirected to an existing id
        self.duplicate_splits = 0   # raw ids evicted from a shared canonical id

    def _prune(self, now_t):
        dead = [cid for cid, rec in self.bank.items()
                if now_t - rec["t"] > self.memory_ttl_s]
        for cid in dead:
            del self.bank[cid]

    def _remember(self, canon, vec, pos, now_t, score):
        rec = self.bank.get(canon)
        if vec is not None and (rec is None or score >= rec.get("best_score", -1)):
            self.bank[canon] = {"anchor": vec, "pos": pos, "t": now_t,
                                "best_score": score, "t_emb": now_t}
        elif rec is not None:
            rec["pos"] = pos
            rec["t"] = now_t
            if vec is not None:
                rec["t_emb"] = now_t
        elif canon not in self.bank:
            self.bank[canon] = {"anchor": vec, "pos": pos, "t": now_t,
                                "best_score": score}

    def resolve(self, raw_tid, crop, conf, bbox, now_t, frozen=False,
                blocked_canons=None, vec=_NOVEC):
        """crop may be None (detection too small/low-conf to embed) — still
        tracked positionally, just can't confirm/deny via appearance.

        v44 params:
          frozen         -- this detection is mutually occluded THIS frame; its
                            crop is contaminated by the overlapping body, so we
                            NEVER overwrite the appearance anchor and NEVER let
                            appearance drive a re-assignment. Position stays
                            fresh so the track survives the overlap.
          blocked_canons -- canonical ids already visible under a DIFFERENT raw
                            id this frame. A person can't be in two tracks at
                            once, so a birth may not merge into any of these
                            (co-visibility hard constraint)."""
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self._prune(now_t)
        score = conf * max(1.0, bbox[3] - bbox[1])

        if raw_tid in self.raw_to_canon:
            canon = self.raw_to_canon[raw_tid]
            if frozen:
                # occluded: keep last-known position fresh but do NOT touch the
                # appearance anchor with a contaminated crop.
                rec = self.bank.get(canon)
                if rec is not None:
                    rec["pos"] = (cx, cy)
                    rec["t"] = now_t
            else:
                if vec is _NOVEC:
                    vec = self.embed_fn(crop) if crop is not None else None
                self._remember(canon, vec, (cx, cy), now_t, score)
            return canon

        # Brand-new raw track_id -- before minting a new canonical identity,
        # check whether it plausibly belongs to someone already in memory.
        if frozen:
            # Don't trust an occluded/contaminated crop to merge a birth into an
            # existing identity. Mint a positional-only id now; a later clean
            # frame can still be matched normally.
            canon = raw_tid
            self.raw_to_canon[raw_tid] = canon
            self.raw_bound_t[raw_tid] = now_t
            rec = self.bank.get(canon)
            if rec is None:
                self.bank[canon] = {"anchor": None, "pos": (cx, cy),
                                    "t": now_t, "best_score": -1}
            else:
                rec["pos"] = (cx, cy)
                rec["t"] = now_t
            return canon

        if vec is _NOVEC:
            vec = self.embed_fn(crop) if crop is not None else None
        # ── CANDIDATE GATHERING, then a RELATIVE decision ───────────────────
        #
        # WHY NOT A FIXED THRESHOLD
        #     This used to accept the first identity whose cosine cleared an
        #     absolute bar (0.62). On this camera that bar is measurably wrong
        #     in BOTH directions at once: the run measured same-person p50 =
        #     0.461 and different-person p90 = 0.538. The distributions
        #     OVERLAP, so no single number separates them — 0.60 rejects most
        #     true matches, 0.37 admits strangers. Tuning it only chooses which
        #     error you prefer.
        #
        #     And 45% of this footage is infrared, where every cosine is
        #     globally lower. An absolute bar is therefore simultaneously too
        #     strict at night and too loose in daylight, which is exactly the
        #     "ids change too easily" complaint.
        #
        # WHAT REPLACES IT
        #     Lowe's ratio test, the standard answer in feature matching: do
        #     not ask "is this match good enough", ask "is the best candidate
        #     DECISIVELY better than the runner-up". A constant factor applied
        #     to every similarity — which is what changing modality does —
        #     leaves that comparison unchanged. The same code works day and
        #     night with no retuning.
        #
        #     The absolute floor survives only as a weak sanity bar for the
        #     case where there IS no runner-up to compare against.
        cands = []
        if vec is not None:
            for cid, rec in self.bank.items():
                if rec["anchor"] is None:
                    continue
                if blocked_canons and cid in blocked_canons:
                    continue  # v44 co-visibility: id already on-screen elsewhere
                dist = math.hypot(cx - rec["pos"][0], cy - rec["pos"][1])
                gap = now_t - rec["t"]
                # G2: metric first, pixels only if the plane is unavailable here
                # (e.g. the point is above the horizon). These PHYSICAL gates
                # are unchanged and still run first — a relative test only ever
                # chooses among candidates a body could actually have reached.
                _md = None
                if self.plane is not None and self.plane.ok:
                    _md = self.plane.dist_m((cx, cy), rec["pos"])
                if _md is not None:
                    # a standing-start allowance so a stationary person whose box
                    # jitters is never gated out
                    if _md > self.max_speed_mps * max(gap, 0.35) + 0.35:
                        continue
                elif dist > self.max_dist_px:
                    continue

                # the spatial penalty still applies, as a DEDUCTION from the
                # score rather than an addition to a bar — so implausible-but-
                # possible candidates rank lower instead of being compared
                # against a moving target
                penalty = 0.0
                if _md is not None and self.plane is not None:
                    _pl = self.max_speed_mps * max(gap, 0.35) + 0.35
                    excess = max(0.0, _md - _pl) / max(_pl, 1e-6)
                    penalty = min(MAX_SPATIAL_PENALTY,
                                  excess * SPATIAL_PENALTY_SCALE)
                else:
                    plausible_dist = MAX_PLAUSIBLE_SPEED_PX * self.dist_scale * gap
                    if plausible_dist > 0:
                        excess = max(0.0, dist - plausible_dist) / plausible_dist
                        penalty = min(MAX_SPATIAL_PENALTY,
                                      excess * SPATIAL_PENALTY_SCALE)
                # a long gap is weaker evidence; keep that as a score deduction
                penalty += (_dynamic_accept_thresh(gap, self.sim_threshold)
                            - self.sim_threshold)
                cands.append((_cosine(vec, rec["anchor"]) - penalty, cid))

        best_cid, best_sim = None, 0.0
        if cands:
            cands.sort(reverse=True)
            best_sim, best_cid = cands[0]
            if best_sim < self.abs_floor:
                # nothing here is even plausibly this person
                best_cid = None
            elif len(cands) > 1 and self.ratio is not None:
                runner_up = cands[1][0]
                # Lowe, expressed for similarities: the runner-up must be at
                # most `ratio` of the winner. Scale-free, so a modality shift
                # that lowers every score equally does not change the verdict.
                if runner_up > 0 and (runner_up / max(best_sim, 1e-6)) > self.ratio:
                    best_cid = None          # ambiguous — mint a new id instead
                    self.ratio_rejects += 1

        if best_cid is not None:
            canon = best_cid
            self.reassignments += 1
        else:
            canon = raw_tid

        self.raw_to_canon[raw_tid] = canon
        self.raw_bound_t[raw_tid] = now_t
        self._remember(canon, vec, (cx, cy), now_t, score)
        return canon

    def split_duplicate_raws(self, raw_ids, now_t):
        """A canonical identity may belong to exactly ONE raw track at a time —
        a person cannot be two boxes.

        ENABLE_COVISIBILITY_BLOCK enforces that at BIRTH, but only against the
        ids visible on that frame. Raw ids A and B can therefore each bind to
        canon C legally, on different frames, while the other was off-screen or
        occluded; when they later appear together, resolve() short-circuits on
        `raw_tid in self.raw_to_canon` for both and hands the same id to two
        boxes forever. Nothing downstream can recover from that: zone dwell,
        crossings and role votes all double-count under one id.

        Called once per frame with the raw ids present. The INCUMBENT — the
        earliest binding — keeps the canonical id; every later claimant is
        re-minted onto its own id, which resolve() may legitimately merge again
        later on appearance evidence. Returns the (raw_id, canon) pairs evicted.
        """
        by_canon = defaultdict(list)
        for r in raw_ids:
            c = self.raw_to_canon.get(r)
            if c is not None:
                by_canon[c].append(r)
        evicted = []
        for canon, raws in by_canon.items():
            if len(raws) < 2:
                continue
            # str() tie-break so the choice is deterministic across runs when
            # two bindings share a timestamp (same frame, same t).
            raws.sort(key=lambda r: (self.raw_bound_t.get(r, 0.0), str(r)))
            for r in raws[1:]:
                self.raw_to_canon[r] = r
                self.raw_bound_t[r] = now_t
                evicted.append((r, canon))
        self.duplicate_splits += len(evicted)
        return evicted

    def try_swap(self, a_raw, a_crop, b_raw, b_crop, margin=0.10):
        """v53: post-occlusion swap re-validation. When two tracks exit an
        overlap, the tracker may have TRADED their ids mid-occlusion, and
        resolve() would rubber-stamp the trade forever (known raw ids pass
        straight through). Here both fresh clean crops are checked against
        both stored anchors; only if EACH matches the OTHER identity better
        by `margin` is the trade undone. Symmetric requirement = extremely
        conservative, so a false un-swap is far less likely than the swap
        it corrects."""
        ca = self.raw_to_canon.get(a_raw)
        cb = self.raw_to_canon.get(b_raw)
        if ca is None or cb is None or ca == cb:
            return False
        ra, rb = self.bank.get(ca), self.bank.get(cb)
        if not ra or not rb or ra["anchor"] is None or rb["anchor"] is None:
            return False
        va = self.embed_fn(a_crop) if a_crop is not None else None
        vb = self.embed_fn(b_crop) if b_crop is not None else None
        if va is None or vb is None:
            return False
        a_own   = _cosine(va, ra["anchor"])
        a_other = _cosine(va, rb["anchor"])
        b_own   = _cosine(vb, rb["anchor"])
        b_other = _cosine(vb, ra["anchor"])
        if a_other - a_own > margin and b_other - b_own > margin:
            self.raw_to_canon[a_raw], self.raw_to_canon[b_raw] = cb, ca
            self.swap_corrections = getattr(self, "swap_corrections", 0) + 1
            return True
        return False


def _hsv_embed_single(crop):
    """One crop -> one HSV color-histogram vector (torso region only)."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    torso = hsv[64:192, 16:112]
    hist = [cv2.calcHist([torso], [ch], None, [16],
                         [0, 180 if ch == 0 else 256]) for ch in range(3)]
    v = np.concatenate([h.flatten() for h in hist])
    n = float(np.linalg.norm(v)) or 1e-9
    return (v / n).tolist()


def _hsv_gallery_sim(crops_a, crops_b):
    """Best pairwise HSV cosine across two crop lists — same gallery
    philosophy as _gallery_sim, applied to the HSV feature space instead
    of OSNet, so pose/viewpoint spread doesn't wash out a real match here
    either."""
    if not crops_a or not crops_b:
        return 0.0
    vecs_a = [_hsv_embed_single(c) for c in crops_a]
    vecs_b = [_hsv_embed_single(c) for c in crops_b]
    return max(_cosine(va, vb) for va in vecs_a for vb in vecs_b)



def _handoff_appearance_contradicts(a, b, anchor_embeddings):
    """v47: veto a PURE-SPATIAL hand-off/stationary merge when both tracks carry
    body embeddings whose similarity is clearly 'different person'. When either
    track has no embedding (small/dim fragment — the case the spatial rule exists
    for), returns False so the spatial merge still stands. Without this, in a
    crowd a person vanishing behind a fixture and a DIFFERENT person appearing
    near that spot within the gap/px window get merged into one identity — the
    exact failure the calibration 'worst same-person' audit reveals."""
    if not ENABLE_HANDOFF_APPEARANCE_VETO or not anchor_embeddings:
        return False
    va = anchor_embeddings.get(a)
    vb = anchor_embeddings.get(b)
    if va is None or vb is None:
        return False
    return _cosine(va, vb) < HANDOFF_VETO_SIM


def _appearance_hsv_contradicts(a, b, raw_crops):
    """v53: veto an APPEARANCE-tier merge when the independent HSV colour
    space actively disagrees. Abstains (returns False) when either track has
    no banked crop, so tracks without colour evidence are never penalised —
    this only removes merges that a second, different feature space says are
    wrong. Directly targets the hub-id over-merge seen in the run log."""
    if not ENABLE_APPEARANCE_HSV_VETO or not raw_crops:
        return False
    ca, cb = raw_crops.get(a), raw_crops.get(b)
    if not ca or not cb:
        return False
    try:
        return _hsv_gallery_sim(ca[:3], cb[:3]) < APPEARANCE_HSV_VETO_SIM
    except Exception:
        return False


def _handoff_hsv_contradicts(a, b, raw_crops):
    """v48: independent 2nd opinion on a pure-spatial merge. Within a
    hand-off-sized gap clothing color cannot change, so clearly-different
    torso histograms (< HANDOFF_HSV_VETO_SIM) mean different people. Uses a
    DIFFERENT feature space from the CLIP veto (colors vs learned Re-ID), so
    the two vetoes cover each other's blind spots. Crop-less -> no veto."""
    if not ENABLE_HANDOFF_HSV_VETO or not raw_crops:
        return False
    ca, cb = raw_crops.get(a), raw_crops.get(b)
    if not ca or not cb:
        return False
    try:
        return _hsv_gallery_sim(ca[:3], cb[:3]) < HANDOFF_HSV_VETO_SIM
    except Exception:
        return False


def calibrate_appearance_threshold(windows, positions, anchor_embeddings,
                                   handoff_gap_s=2.5, handoff_px=90.0,
                                   stationary_px=30.0, role_hint=None,
                                   duplicate_px=40.0):
    """Diagnostic (v30): measures THIS run's actual same-person vs
    different-person cosine-similarity distribution for whichever appearance
    backbone loaded (CLIP-ReID or OSNet), instead of reusing OSNet's
    2026-07-13 calibration (~0.72 same / ~0.41 different) for a possibly
    different embedding space.

    Two ground-truth sets, BOTH independent of appearance (no circularity):
      different-person pairs: any two tracks CO-VISIBLE in the same frame
        (their visibility windows overlap) must be different people --
        physically impossible for one person to occupy two tracks at once.
      same-person pairs: tracks linked by the hand-off/stationary rule --
        near-instant (<= handoff_gap_s), near-zero-distance (<= handoff_px,
        or <= stationary_px with no gap) reappearance right where the other
        track died. This is SPATIAL evidence, not appearance evidence, so
        checking appearance similarity on these pairs is a fair test.

    role_hint: (v38) optional {track_id: "staff"|"customer"} -- a fragment
        pair that has EARNED opposite roles from real zone dwell is excluded
        from the same-person ground truth even if spatially close. A busy
        counter has staff and customers constantly swapping the same few
        square feet; without this guard, "staff fragment ends, a DIFFERENT
        customer's fragment starts nearby a moment later" gets miscounted as
        a genuine same-person hand-off, which drags the measured same-person
        distribution down and makes the auto-calibrated threshold LOOSER
        than the footage actually supports -- the opposite of the intent.
        Same rule merge_fragmented_tracks already applies to real merges;
        this just applies it to the ground truth used to calibrate them.

    duplicate_px: (v40) guards the OTHER ground-truth set. "Co-visible in
        the same frame -> different person" only holds if the two tracks
        are actually two separate bodies. A tracker duplicate -- one
        physical person who briefly got two track IDs (e.g. an
        occlusion-ambiguous re-detect) -- is also technically "co-visible,"
        but both IDs stay glued to the same spot the whole time they
        coexist, start-to-start and end-to-end. Appearance correctly says
        these are the same person (high similarity); the ground truth was
        wrong to call it "different," and that mislabeled pair dragged
        diff_p90 UP, making the auto-calibrated bar stricter than the
        footage supports -- the mirror image of the role-conflict bug this
        same guard-pattern already fixed on the same-person side.

    Purely diagnostic -- returns numbers, never touches a threshold itself.
    """
    def _role_conflict(a, b):
        if not role_hint:
            return False
        ra, rb = role_hint.get(a), role_hint.get(b)
        return bool(ra) and bool(rb) and ra != rb

    def _looks_duplicate(pa, pb):
        # pa/pb are (first_pos, last_pos). Two IDs co-located at BOTH ends
        # of their shared lifetime aren't "two people who happened to be
        # close" -- that's one body wearing two IDs.
        d_start = math.hypot(pa[0][0] - pb[0][0], pa[0][1] - pb[0][1])
        d_end = math.hypot(pa[1][0] - pb[1][0], pa[1][1] - pb[1][1])
        return d_start <= duplicate_px and d_end <= duplicate_px

    ids = [t for t in windows if t in anchor_embeddings]
    diff_sims, diff_pairs = [], []
    duplicate_excluded = 0
    for i, a in enumerate(ids):
        wa, pa = windows[a], positions.get(a)
        for b in ids[i + 1:]:
            wb = windows[b]
            if wa[0] < wb[1] and wb[0] < wa[1]:   # overlap -> different person
                pb = positions.get(b)
                if pa and pb and _looks_duplicate(pa, pb):
                    duplicate_excluded += 1
                    continue
                s = _cosine(anchor_embeddings[a], anchor_embeddings[b])
                diff_sims.append(s)
                diff_pairs.append((a, b, s))

    same_sims, same_pairs = [], []
    role_conflicts_excluded = 0
    for i, a in enumerate(ids):
        wa, pa = windows[a], positions.get(a)
        if not pa:
            continue
        for b in ids[i + 1:]:
            wb, pb = windows[b], positions.get(b)
            if not pb:
                continue
            if _role_conflict(a, b):
                role_conflicts_excluded += 1
                continue
            if wa[0] <= wb[0]:
                first_end, first_pos = wa[1], pa[1]
                second_start, second_pos = wb[0], pb[0]
            else:
                first_end, first_pos = wb[1], pb[1]
                second_start, second_pos = wa[0], pa[0]
            gap = second_start - first_end
            if gap < 0:
                continue
            dist = math.hypot(first_pos[0] - second_pos[0],
                              first_pos[1] - second_pos[1])
            if (((gap <= handoff_gap_s and dist <= handoff_px) or dist <= stationary_px)
                    and not _handoff_appearance_contradicts(a, b, anchor_embeddings)):
                s = _cosine(anchor_embeddings[a], anchor_embeddings[b])
                same_sims.append(s)
                same_pairs.append((a, b, s))

    def _pct(xs, p):
        if not xs:
            return None
        xs = sorted(xs)
        k = (len(xs) - 1) * p
        f, c = int(k), min(int(k) + 1, len(xs) - 1)
        return xs[f] + (xs[c] - xs[f]) * (k - f)

    return {
        "same_n": len(same_sims), "diff_n": len(diff_sims),
        "same_p10": _pct(same_sims, 0.10), "same_p50": _pct(same_sims, 0.50),
        "diff_p50": _pct(diff_sims, 0.50), "diff_p90": _pct(diff_sims, 0.90),
        "same_sims": same_sims,
        "diff_sims": diff_sims,
        # (v36) the actual pairs behind the numbers, worst-first, so you can
        # LOOK at what's confusing the model instead of only reading a
        # percentile. "Worst same-person" = spatially-certain same person
        # but LOWEST appearance similarity (should be high, isn't).
        # "Worst different-person" = certainly different people but HIGHEST
        # appearance similarity (should be low, isn't).
        "same_pairs_worst": sorted(same_pairs, key=lambda p: p[2])[:8],
        "diff_pairs_worst": sorted(diff_pairs, key=lambda p: -p[2])[:8],
        # (v38) how many spatially-close pairs were kept OUT of the
        # same-person ground truth because they'd earned opposite roles --
        # a high number here on a counter-heavy venue confirms this guard
        # is doing real work, not just defensive code.
        "role_conflicts_excluded": role_conflicts_excluded,
        # (v40) how many co-visible pairs were kept OUT of the
        # different-person ground truth because they look like one body
        # wearing two track IDs rather than two people -- a high number
        # here on footage with occlusion-heavy tracking confirms this
        # guard is doing real work.
        "duplicate_excluded": duplicate_excluded,
    }


def suggest_appearance_thresholds(cal, current_thresh, current_anchor,
                                  floor=0.35, ceiling=0.85):
    """(v37) Turns calibrate_appearance_threshold()'s report into thresholds
    that are actually USED this run. Uses find_optimal_threshold (EER balanced accuracy)
    if sims lists are present, otherwise falls back to percentile heuristics.
    """
    same_sims, diff_sims = cal.get("same_sims"), cal.get("diff_sims")
    if same_sims and diff_sims:
        opt_thresh, metrics = find_optimal_threshold(same_sims, diff_sims)
        suggested = opt_thresh
        anchor_suggested = opt_thresh + 0.08
        _log.info(f"  [EER Sweep] Optimal threshold found at {opt_thresh:.3f} (balanced accuracy = {metrics['balanced_accuracy']:.4f})")
    else:
        same_p10, diff_p90 = cal.get("same_p10"), cal.get("diff_p90")
        if same_p10 is None or diff_p90 is None:
            return current_thresh, current_anchor
        if same_p10 > diff_p90:
            suggested = (same_p10 + diff_p90) / 2
            anchor_suggested = suggested + 0.08
        else:
            suggested = diff_p90 + 0.02
            anchor_suggested = suggested + 0.10
            
    suggested = max(floor, min(ceiling, suggested))
    anchor_suggested = max(floor, min(ceiling, anchor_suggested))
    return round(suggested, 3), round(anchor_suggested, 3)


def cross_validate_faces(mapping, face_embeddings, windows,
                         sim_threshold=0.35, max_gap_s=480.0):
    """Diagnostic (v30): where a face embedding exists for BOTH tracks in a
    pair, face similarity is the strongest identity signal this pipeline has
    -- robust to pose AND clothing change, unlike body-appearance backbones.
    Coverage will be sparse (many CCTV crops never resolve a usable face) --
    this only ever covers a fraction of merges, and that's fine: it's a
    corroborating signal, same "tag not gate" philosophy as the HSV check.
    Never touches `mapping`.
    """
    from collections import defaultdict as _dd
    groups = _dd(list)
    for tid, canon in mapping.items():
        groups[canon].append(tid)

    agree, disagree = [], []
    checked = set()
    for canon, members in groups.items():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                checked.add(frozenset((a, b)))
                if a not in face_embeddings or b not in face_embeddings:
                    continue
                sim = _cosine(face_embeddings[a], face_embeddings[b])
                (agree if sim >= sim_threshold else disagree).append((a, b, sim))

    face_only = []
    ids = list(face_embeddings)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if frozenset((a, b)) in checked:
                continue
            wa, wb = windows.get(a), windows.get(b)
            if not wa or not wb:
                continue
            gap = (wb[0] - wa[1]) if wa[0] <= wb[0] else (wa[0] - wb[1])
            if gap < 0 or gap > max_gap_s:
                continue
            sim = _cosine(face_embeddings[a], face_embeddings[b])
            if sim >= sim_threshold:
                face_only.append((a, b, sim))

    return {"agree": agree, "disagree": disagree, "face_only": face_only,
            "coverage": len(face_embeddings)}


def apply_face_veto(mapping, merge_edges, face_embeddings, windows,
                    sim_threshold=0.35, veto_margin=0.15,
                    max_edge_score_for_veto=0.80):
    """ACTIVE veto (v32, redesigned) — reverses a merge when face evidence
    confidently contradicts it, but ONLY the specific direct edge that
    caused the merge, and ONLY when that edge's own evidence was
    appearance-only.

    What changed from v31 and why: v31 checked every pairwise combination of
    members inside a transitively-merged group, including pairs that were
    NEVER directly compared to each other by merge_fragmented_tracks, and
    reversed any confident face mismatch regardless of how strong the
    original merge evidence was. That let one noisy face embedding undo a
    near-certain hand-off match (person physically can't teleport between a
    death and a birth 2px and 0.3s apart) just because some OTHER pair in
    the same transitive group had a low face score. In practice this showed
    up as visible ID flicker in the rendered video exactly where tracking
    was already correct -- the fix, not the bug.

    Two changes fix that:
      1. Only merge_edges (the literal accepted union pairs from
         merge_fragmented_tracks, each with its own evidence score) are
         checked -- never a pair that was merely co-transitive.
      2. An edge whose own score already reflects hand-off/stationary/
         anchor-tier evidence (score >= max_edge_score_for_veto, default
         0.80) is evidence independent of appearance/face entirely
         (position + timing, or a clean 1:1 crop match) -- a face signal
         alone doesn't outrank it. Those edges are FLAGGED for manual
         review instead of reversed. Only ordinary gallery/appearance-only
         edges (score < 0.80) can actually be vetoed.

    Reversing an edge is done by REPLAYING merge_edges into a fresh
    union-find in original (already-accepted, strongest-first) order,
    skipping vetoed edges -- not by patching the flat `mapping` in place.
    Patching in place (v31's approach) could leave a grandchild of a vetoed
    node still pointing at the old canonical id even after its direct
    parent was detached; replaying the actual edge list keeps the result
    consistent with what a rerun of merge_fragmented_tracks minus the
    vetoed edges would have produced.

    mapping:         {track_id: canonical_id} from merge_fragmented_tracks
                     (only used here to know every track_id that exists).
    merge_edges:     [(sim, a, b, tier), ...] from merge_fragmented_tracks --
                     the literal accepted union edges, in accepted order.
    face_embeddings: {track_id: [float, ...]} -- sparse by nature.
    windows:         {track_id: (t_first, t_last)} -- for the earliest-
                     member-becomes-canonical tie-break, same as
                     merge_fragmented_tracks.

    Returns (new_mapping, vetoed, flagged):
      vetoed  -- edges actually reversed: (a, b, face_sim, edge_score).
      flagged -- high-tier edges with a confident face mismatch that were
                 NOT reversed (independent physical/anchor evidence
                 outranks a single face signal here) -- surfaced for
                 manual review only, `mapping` is unaffected by these.
    Does not mutate the input `mapping`.
    """
    veto_bar = max(0.0, sim_threshold - veto_margin)
    parent = {t: t for t in mapping}

    def _find(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    vetoed, flagged = [], []
    for edge_score, a, b, _tier in merge_edges:
        do_veto = False
        if a in face_embeddings and b in face_embeddings:
            fsim = _cosine(face_embeddings[a], face_embeddings[b])
            if fsim < veto_bar:
                if edge_score < max_edge_score_for_veto:
                    do_veto = True
                    vetoed.append((a, b, fsim, edge_score))
                else:
                    flagged.append((a, b, fsim, edge_score))
        if do_veto:
            continue   # skip this union -- a and b stay in separate groups
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    new_mapping, canon = {}, {}
    _cpk = _canon_priority_key(windows)
    for t in sorted(mapping, key=_cpk):
        root = _find(t)
        canon.setdefault(root, t)      # staff names win; then earliest
        new_mapping[t] = canon[root]

    return new_mapping, vetoed, flagged


def cross_validate_reid(mapping, track_crops, track_windows,
                        gallery_sim_threshold=0.60, anchor_sim_threshold=0.75,
                        max_gap_s=480.0):
    """Independently re-check the OSNet-based merge mapping with HSV
    color-histogram embeddings on the SAME banked crops from this run.

    Returns a report dict:
      agree      -- merged pairs (same canonical id) where HSV also clears
                    either the gallery or anchor bar (corroborated).
      disagree   -- merged pairs where HSV does NOT corroborate — the OSNet
                    merge may still be correct (color histograms are weak
                    under lighting/color changes), but these are worth a
                    manual crop look.
      hsv_only   -- pairs OSNet did NOT merge, but HSV alone would have
                    (same non-overlap + gap-window rules as
                    merge_fragmented_tracks). Candidates OSNet may have
                    missed, not confirmed merges.

    Purely diagnostic — never changes `mapping` or downstream events.
    """
    from collections import defaultdict as _dd
    groups = _dd(list)
    for tid, canon in mapping.items():
        groups[canon].append(tid)

    agree, disagree = [], []
    already_checked = set()
    for canon, members in groups.items():
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda t: track_windows[t][0])
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                already_checked.add(frozenset((a, b)))
                crops_a = [c for _, c in track_crops.get(a, [])]
                crops_b = [c for _, c in track_crops.get(b, [])]
                if not crops_a or not crops_b:
                    continue
                gal_sim = _hsv_gallery_sim(crops_a, crops_b)
                anc_sim = _cosine(_hsv_embed_single(crops_a[0]),
                                  _hsv_embed_single(crops_b[0]))
                corroborated = (gal_sim >= gallery_sim_threshold
                                or anc_sim >= anchor_sim_threshold)
                rec = (a, b, gal_sim, anc_sim)
                (agree if corroborated else disagree).append(rec)

    hsv_only = []
    ids_with_crops = [t for t in track_windows if track_crops.get(t)]
    for i, a in enumerate(ids_with_crops):
        for b in ids_with_crops[i + 1:]:
            if frozenset((a, b)) in already_checked:
                continue
            wa, wb = track_windows[a], track_windows[b]
            gap = (wb[0] - wa[1]) if wa[0] <= wb[0] else (wa[0] - wb[1])
            if gap < 0 or gap > max_gap_s:
                continue
            crops_a = [c for _, c in track_crops[a]]
            crops_b = [c for _, c in track_crops[b]]
            gal_sim = _hsv_gallery_sim(crops_a, crops_b)
            anc_sim = _cosine(_hsv_embed_single(crops_a[0]),
                              _hsv_embed_single(crops_b[0]))
            if gal_sim >= gallery_sim_threshold or anc_sim >= anchor_sim_threshold:
                hsv_only.append((a, b, gal_sim, anc_sim))

    return {"agree": agree, "disagree": disagree, "hsv_only": hsv_only}


def _as_gallery(vecs):
    """Accept either a single flat vector [float,...] or a gallery of vectors
    [[float,...], ...] and always return a gallery (list of vectors)."""
    if not vecs:
        return []
    if isinstance(vecs[0], (list, tuple)):
        return vecs
    return [vecs]


def _gallery_sim(gal_a, gal_b):
    """Best-match similarity between two crop galleries (2026-07-13).

    A single averaged embedding is viewpoint-fragile: if a track's banked
    crops all happen to be one pose/angle (e.g. a person turned into the
    espresso machine the whole fragment), the mean vector doesn't resemble
    a different fragment of the SAME person seen frontally, even though one
    good crop-to-crop pair would clearly match. Comparing every stored crop
    in A against every stored crop in B and taking the best pairwise cosine
    means only ONE matching viewpoint needs to line up, not all of them —
    much more robust to a person moving/turning mid-track.
    """
    gal_a, gal_b = _as_gallery(gal_a), _as_gallery(gal_b)
    if not gal_a or not gal_b:
        return 0.0
    return max(_cosine(va, vb) for va in gal_a for vb in gal_b)


def _windows_overlap(wins_a, wins_b, tolerance_s=2):
    for a0, a1 in wins_a:
        for b0, b1 in wins_b:
            if a0 < b1 - tolerance_s and b0 < a1 - tolerance_s:
                return True
    return False

# ===========================================================================
# v46 GLOBAL TRACKLET ASSOCIATION  (unit-tested in scratchpad; 11/11)
# Treats identity assignment as one global optimisation over the whole video:
# link tracklet tails->heads with the Hungarian algorithm so competing long-
# range links are resolved jointly, not greedily. Pure Python; scipy if present
# else a built-in Hungarian. Enforces the same hard constraints as the live
# path (co-visibility, gap, spatial plausibility, time-decayed appearance bar).
# ===========================================================================
try:
    from scipy.optimize import linear_sum_assignment as _gta_lsa
    _GTA_SCIPY = True
except Exception:
    _GTA_SCIPY = False


def _gta_hungarian(cost):
    import copy
    n = len(cost)
    if n == 0:
        return [], []
    m = copy.deepcopy(cost)
    INF = float("inf")
    u = [0.0]*(n+1); v = [0.0]*(n+1); p = [0]*(n+1); way = [0]*(n+1)
    for i in range(1, n+1):
        p[0] = i; j0 = 0
        minv = [INF]*(n+1); used = [False]*(n+1)
        while True:
            used[j0] = True; i0 = p[j0]; delta = INF; j1 = -1
            for j in range(1, n+1):
                if not used[j]:
                    cur = m[i0-1][j-1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur; way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]; j1 = j
            for j in range(n+1):
                if used[j]:
                    u[p[j]] += delta; v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
            if j0 == 0:
                break
    rows = [0]*n
    for j in range(1, n+1):
        if p[j] != 0:
            rows[p[j]-1] = j-1
    return list(range(n)), rows


def _gta_solve(cost):
    if _GTA_SCIPY:
        r, c = _gta_lsa(cost); return list(r), list(c)
    return _gta_hungarian([list(row) for row in cost])


class _GtaUF:
    def __init__(self, items): self.p = {x: x for x in items}
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[rb] = ra


def _gta_sim_floor(gap, base, near_s, far_s, near_bonus, far_penalty):
    if gap <= near_s: return base - near_bonus
    if gap >= far_s:  return base + far_penalty
    frac = (gap - near_s) / (far_s - near_s)
    return (base - near_bonus) + frac * (near_bonus + far_penalty)


def link_tracklets_global(tracklets, sim_fn, accept_sim=0.60, max_gap_s=7200.0,
                          overlap_tol_s=0.5, max_speed_px=220.0, speed_slack_px=60.0,
                          near_s=15.0, far_s=180.0, near_bonus=0.04, far_penalty=0.05,
                          no_link_cost=1.0):
    import math as _m
    T = list(tracklets); n = len(T)
    if n <= 1:
        return {t["id"]: t["id"] for t in T}

    def eligible_cost(a, b):
        A, B = T[a], T[b]
        gap = B["t_start"] - A["t_end"]
        if gap < -overlap_tol_s: return None
        gap = max(0.0, gap)
        if gap > max_gap_s: return None
        if A["emb"] is None or B["emb"] is None: return None
        pe, psb = A.get("pos_end"), B.get("pos_start")
        if pe is not None and psb is not None:
            dist = _m.hypot(psb[0]-pe[0], psb[1]-pe[1])
            if dist > max_speed_px*gap + speed_slack_px: return None
        floor = _gta_sim_floor(gap, accept_sim, near_s, far_s, near_bonus, far_penalty)
        sim = sim_fn(A["emb"], B["emb"])
        if sim < floor: return None
        return (1.0 - sim) + 0.001*gap

    BIG = 1e6; size = 2*n
    cost = [[BIG]*size for _ in range(size)]
    for a in range(n):
        for b in range(n):
            if a == b: continue
            c = eligible_cost(a, b)
            if c is not None: cost[a][b] = c
    for i in range(n):
        cost[i][n+i] = no_link_cost
        cost[n+i][i] = no_link_cost
    for i in range(n):
        for j in range(n):
            cost[n+i][n+j] = 0.0
    rows, cols = _gta_solve(cost)
    uf = _GtaUF([t["id"] for t in T])
    for r, c in zip(rows, cols):
        if r < n and c < n and cost[r][c] < BIG:
            uf.union(T[r]["id"], T[c]["id"])
    groups = {}
    for t in T:
        groups.setdefault(uf.find(t["id"]), []).append(t)
    mapping = {}
    for root, members in groups.items():
        canon = min(members, key=lambda t: (t["t_start"], str(t["id"])))["id"]
        for t in members: mapping[t["id"]] = canon
    return mapping


def _apply_global_tracklet_pass(mapping, windows, positions, anchor_embeddings, base_sim):
    """Additive: run the global linker over the GREEDY groups. Builds one
    super-tracklet per current identity (union window, earliest/latest
    positions, representative anchor) and links those. Returns a composed
    fragment->identity mapping. Never splits an existing group."""
    groups = {}
    for frag, canon in mapping.items():
        groups.setdefault(canon, []).append(frag)
    tracklets = []
    for canon, frags in groups.items():
        fw = [f for f in frags if f in windows]
        if not fw:
            continue
        fw.sort(key=lambda f: windows[f][0])
        first, last = fw[0], fw[-1]
        ps = pe = None
        if positions:
            if first in positions and positions[first]:
                ps = positions[first][0]
            if last in positions and positions[last]:
                pe = positions[last][1]
        emb = None
        for f in [canon] + fw:
            if anchor_embeddings and anchor_embeddings.get(f) is not None:
                emb = anchor_embeddings[f]; break
        tracklets.append(dict(id=canon, t_start=windows[first][0],
                              t_end=windows[last][1], pos_start=ps,
                              pos_end=pe, emb=emb))
    if len(tracklets) <= 1:
        return mapping
    group_map = link_tracklets_global(
        tracklets, _cosine, accept_sim=base_sim, max_gap_s=REID_MAX_GAP_S,
        max_speed_px=MAX_PLAUSIBLE_SPEED_PX, near_s=NEAR_GAP_S, far_s=FAR_GAP_S,
        near_bonus=NEAR_GAP_BONUS, far_penalty=FAR_GAP_PENALTY)
    return {frag: group_map.get(old, old) for frag, old in mapping.items()}


def merge_fragmented_tracks(track_windows, embeddings,
                            sim_threshold=0.65, max_gap_s=480.0,
                            positions=None, handoff_gap_s=2.5,
                            handoff_px=90.0, role_hint=None,
                            stationary_px=30.0, anchor_embeddings=None,
                            anchor_sim_threshold=0.75,
                            raw_crops=None, hsv_sim_threshold=0.75,
                            face_embeddings=None, face_sim_threshold=0.35,
                            plane=None, handoff_m=1.6, stationary_m=0.6,
                            ir_hint=None):
    """Merge track fragments that are the same person reappearing.

    track_windows: {track_id: (t_first, t_last)} visibility window per track
    embeddings:    {track_id: [[float, ...], ...]} a GALLERY of appearance
                   vectors per track (one per banked crop). Matching uses the
                   best pairwise cosine across both galleries (_gallery_sim),
                   not a single averaged vector, so a track survives even if
                   its crops span multiple viewpoints/poses. A flat single
                   vector [float, ...] per track_id is still accepted for
                   backwards compatibility (auto-wrapped as a 1-item gallery).
    positions:     optional {track_id: (start_xy, end_xy)} — enables the
                   HAND-OFF rule: when one fragment dies and another is born
                   within handoff_gap_s seconds AND handoff_px pixels, they
                   merge even without appearance evidence (covers small/dim
                   fragments that never yielded a usable crop).
    role_hint:     optional {track_id: "staff"|"customer"} — a fragment that
                   has EARNED a confident role from real zone dwell (not just
                   a guess) can never merge with a fragment that earned the
                   opposite role. Appearance similarity alone cannot override
                   this: a staff member and a seated customer are never the
                   same person no matter how similar their embeddings look.
    raw_crops:     (v37) optional {track_id: [crop_bgr, ...]} — the SAME
                   banked crop images used for the appearance embeddings.
                   Enables an ATTIRE tier using HSV color-histogram gallery
                   similarity (_hsv_gallery_sim) as a genuine merge signal,
                   not just a post-hoc corroboration check. On footage where
                   the body-appearance backbone doesn't cleanly separate
                   same/different person (see calibrate_appearance_threshold),
                   HSV cross-validation has repeatedly found large numbers of
                   genuine same-person pairs the appearance backbone alone
                   never merged — this promotes that same evidence to where
                   it can actually fix the fragmentation, instead of only
                   flagging it for manual review afterward.
    face_embeddings: (v37) optional {track_id: [float, ...]} — ArcFace-style
                   embeddings, same ones used for face corroboration/veto
                   elsewhere. A clear face match is the strongest available
                   identity evidence (survives pose AND clothing change,
                   unlike body appearance or attire color), so it's ranked
                   with/above hand-off, and can close gaps neither the
                   appearance backbone nor HSV can (e.g. someone who stood up
                   and changed which way they're facing, or changed jackets).
    A pair merges when it passes any rule, their windows do NOT overlap
    (one body can't be in frame twice), and the gap <= max_gap_s. Greedy
    best-evidence-first with transitive groups.
    Returns (mapping, merge_edges):
      mapping     -- {track_id: canonical_id} (canonical = earliest-
                     seen member), same as before.
      merge_edges -- (v32) [(sim, a, b), ...] the actual direct pairs
                     whose evidence caused a union, in accepted order.
                     sim encodes the evidence TIER: >=0.92 face match,
                     >=0.90 hand-off, 0.85-0.94 stationary, 0.80-0.89
                     anchor, 0.76-0.82 attire/HSV (v37), below that an
                     ordinary gallery/appearance match. Lets a downstream
                     check (apply_face_veto) tell "person physically can't
                     teleport" evidence apart from a weaker appearance-only
                     match.
    """
    def _role_conflict(a, b):
        if not role_hint:
            return False
        ra, rb = role_hint.get(a), role_hint.get(b)
        return bool(ra) and bool(rb) and ra != rb

    def _ir_mismatch(a, b):
        # A4: colour evidence compared across a colour<->IR boundary is noise
        # in BOTH directions — a pre-dusk anchor vs a post-dusk greyscale+CLAHE
        # vector matches (or misses) by accident. ir_hint is {track_id: frac of
        # frames under IR}. Only pairs on OPPOSITE sides of the boundary are
        # blocked; face + hand-off/stationary tiers still bridge it (faces
        # survive near-IR, physics doesn't care about colour).
        if not ir_hint:
            return False
        fa, fb = ir_hint.get(a, 0.0), ir_hint.get(b, 0.0)
        _thr = float(globals().get("IR_TRACK_FRAC", 0.5))
        # F10b: same boundary as the attire-evidence exclusion, so a track
        # withheld from colour evidence is also 'IR' to this guard
        return (fa >= _thr) != (fb >= _thr)

    # (v38) diagnostics — WHY a candidate pair did or didn't end up merged.
    # Every entry in `pairs` now carries a tier label so we can report a
    # per-tier accepted-merge breakdown, and every rejection (role conflict
    # or window overlap) is counted + sampled so a slam-dunk match that got
    # silently dropped (e.g. by greedy processing order) is visible instead
    # of just missing from the output.
    role_conflict_count = 0
    # F11 instrumentation: how often the colour veto was waived because the two
    # tracks sit on opposite sides of the IR boundary, and how many of those it
    # would actually have blocked. The second number is the fix's real effect.
    _hsv_veto_skipped_ir = 0
    _hsv_veto_would_have_blocked = 0
    # F12: WHY a specific pair did not become one person.
    #
    # The existing diagnostics are aggregates — "356 blocked by overlap", "712
    # merges HSV disputed". Aggregates tell you a population is unhealthy, never
    # which gate killed the patient in front of you. Five separate fixes for
    # identity fragmentation were designed off those aggregates and every one
    # was inert or wrong (duplicate boxes, the ReID bar, window length, phantom
    # patience, the IR colour veto), each costing a 35-47 minute run.
    #
    # So record, per candidate pair, the FIRST gate that rejected it. The engine
    # already computes which pairs SHOULD obviously have merged (near in space,
    # inside the gap, different final identities) and prints them as "unmerged
    # near-pair". Joining the two turns
    #     "unmerged near-pair: 1066->1118 dist=2.7px gap=128.1s"
    # into
    #     "...rejected by: <gate>"
    # which is the sentence five sessions of inference failed to produce.
    _pair_reject = {}
    _overlap_transitive = [0]   # blocked by GROUP overlap, not their own

    def _rej(a, b, why):
        _pair_reject.setdefault((a, b), why)
        _pair_reject.setdefault((b, a), why)

    pairs = []
    # (v37) evaluate every track that has ANY identity signal at all — not
    # just the ones with an appearance embedding — so a track that only
    # yielded a face or attire signal (but no usable appearance embedding)
    # still gets a fair chance to match instead of being silently excluded
    # from every tier except hand-off/stationary.
    # v54 SCALE FIX: sort by WINDOW START (was: track id). With starts
    # ascending, wa[0] <= wb[0] always, so gap = wb[0]-wa[1] rises
    # monotonically as b advances — the moment it exceeds max_gap_s every
    # later b is further still, so `break` is exactly equivalent to
    # `continue` and costs O(n*k) instead of O(n^2). On a 10-hour run with
    # ~8k tracks this is the difference between minutes and hours.
    ids_signal = sorted((t for t in track_windows
                         if t in embeddings
                         or (raw_crops and t in raw_crops)
                         or (face_embeddings and t in face_embeddings)),
                        key=lambda t: track_windows[t][0])
    for i, a in enumerate(ids_signal):
        for b in ids_signal[i + 1:]:
            wa, wb = track_windows[a], track_windows[b]
            gap = wb[0] - wa[1]
            if gap > max_gap_s:
                _rej(a, b, f"gap {gap:.0f}s > max_gap_s {max_gap_s:.0f}s")
                break            # every later b is even further away
            if gap < 0:
                _rej(a, b, "windows overlap (co-visible, so two people)")
                continue         # windows overlap — not a reappearance
            if _role_conflict(a, b):
                role_conflict_count += 1
                _rej(a, b, "role conflict (one staff, one customer)")
                continue
            if a in embeddings and b in embeddings:
                sim = _gallery_sim(embeddings[a], embeddings[b])
                gap = (wb[0] - wa[1]) if wa[0] <= wb[0] else (wa[0] - wb[1])
                required = _dynamic_accept_thresh(gap, sim_threshold)
                
                # Soft spatial penalty
                spatial_penalty = 0.0
                # V77: positions is documented optional, and .get on None
                # raised AttributeError for every caller that took the default.
                pa, pb = ((positions or {}).get(a), (positions or {}).get(b))
                if pa and pb and gap > 0:
                    pos_last = pa[1] if wa[0] <= wb[0] else pb[1]
                    pos_first = pb[0] if wa[0] <= wb[0] else pa[0]
                    dist = math.hypot(pos_first[0] - pos_last[0], pos_first[1] - pos_last[1])
                    plausible_dist = MAX_PLAUSIBLE_SPEED_PX * gap
                    if plausible_dist > 0:
                        excess_ratio = max(0.0, dist - plausible_dist) / plausible_dist
                        spatial_penalty = min(MAX_SPATIAL_PENALTY, excess_ratio * SPATIAL_PENALTY_SCALE)
                required += spatial_penalty
                
                if (sim >= required and not _ir_mismatch(a, b)
                        and not _appearance_hsv_contradicts(a, b, raw_crops)):
                    pairs.append((sim, a, b, "gallery"))
                elif sim < required:
                    _rej(a, b, f"appearance sim {sim:.3f} < required "
                               f"{required:.3f} (gap {gap:.0f}s)")
                elif _ir_mismatch(a, b):
                    _rej(a, b, "appearance withheld: opposite sides of the "
                               "colour/IR boundary")
                else:
                    _rej(a, b, "appearance HSV veto (colour disagrees)")
            # ANCHOR TIER (v25): a clean 1:1 match between each track's single
            # best-quality crop (see ANCHOR_SIM_THRESHOLD) is stronger, cleaner
            # evidence than the best-of-gallery score above — that score can be
            # inflated by one lucky crop pairing even when the tracks are
            # different people. When two anchors clear the stricter anchor bar,
            # rank the match ABOVE ordinary gallery matches in the greedy
            # sorted-descending loop below, so it can't get starved out by a
            # pile of weaker matches that happen to sort higher (same failure
            # mode the hand-off/stationary fix below already addresses).
            if (anchor_embeddings and a in anchor_embeddings
                    and b in anchor_embeddings):
                asim = _cosine(anchor_embeddings[a], anchor_embeddings[b])
                if (asim >= anchor_sim_threshold
                        and not _ir_mismatch(a, b)
                        and not _appearance_hsv_contradicts(a, b, raw_crops)):
                    conf = min(1.0, (asim - anchor_sim_threshold)
                               / max(1e-6, 1.0 - anchor_sim_threshold))
                    pseudo = 0.80 + 0.09 * conf   # 0.80-0.89 — below hand-off/
                                                   # stationary (those are near-
                                                   # certain physical evidence),
                                                   # above generic gallery sims
                    pairs.append((pseudo, a, b, "anchor"))
            # ATTIRE TIER (v37): HSV color-histogram gallery match, computed
            # on the same banked crops. Ranked below anchor (clothing color
            # alone is fooled by identical uniforms) but treated as a real
            # merge trigger, not just a diagnostic — see raw_crops docstring
            # above for why.
            if raw_crops and a in raw_crops and b in raw_crops:
                hsim = _hsv_gallery_sim(raw_crops[a], raw_crops[b])
                if hsim >= hsv_sim_threshold and not _ir_mismatch(a, b):
                    conf = min(1.0, (hsim - hsv_sim_threshold)
                               / max(1e-6, 1.0 - hsv_sim_threshold))
                    pseudo = 0.76 + 0.06 * conf   # 0.76-0.82
                    pairs.append((pseudo, a, b, "attire"))
            # FACE TIER (v37): a confident face match outranks every other
            # signal here — see face_embeddings docstring above.
            if (face_embeddings and a in face_embeddings
                    and b in face_embeddings):
                fsim = _cosine(face_embeddings[a], face_embeddings[b])
                if fsim >= face_sim_threshold:
                    conf = min(1.0, (fsim - face_sim_threshold)
                               / max(1e-6, 1.0 - face_sim_threshold))
                    pseudo = 0.92 + 0.07 * conf   # 0.92-0.99
                    pairs.append((pseudo, a, b, "face"))

    if positions:
        all_ids = sorted(track_windows, key=lambda t: track_windows[t][0])
        for i, a in enumerate(all_ids):
            for b in all_ids[i + 1:]:
                wa, wb = track_windows[a], track_windows[b]
                if wb[0] < wa[0]:
                    a2, b2 = b, a
                else:
                    a2, b2 = a, b
                if _role_conflict(a2, b2):
                    role_conflict_count += 1
                    _rej(a2, b2, "role conflict (one staff, one customer)")
                    continue
                gap = track_windows[b2][0] - track_windows[a2][1]
                if gap > max_gap_s:
                    _rej(a2, b2, f"gap {gap:.0f}s > max_gap_s {max_gap_s:.0f}s")
                    break        # v54: all_ids is start-sorted, gap only grows
                if gap < 0:
                    _rej(a2, b2, "windows overlap (co-visible, so two people)")
                    continue
                pa, pb = positions.get(a2), positions.get(b2)
                if not pa or not pb:
                    _rej(a2, b2, "no position record for one of the tracks")
                    continue
                dx = pa[1][0] - pb[0][0]
                dy = pa[1][1] - pb[0][1]
                dist = math.hypot(dx, dy)
                # G2: "died and reappeared within 1.6 m" is the same physical
                # claim everywhere in the frame; "within 160 px" is not.
                _dm = plane.dist_m(pa[1], pb[0]) if (plane is not None and plane.ok) else None
                _near_handoff = (_dm <= handoff_m) if _dm is not None else (dist <= handoff_px)
                _near_static = (_dm <= stationary_m) if _dm is not None else (dist <= stationary_px)
                # F11: a COLOUR veto is meaningless across the IR boundary.
                # _handoff_hsv_contradicts compares torso hue histograms and
                # justifies itself with "within a hand-off-sized gap clothing
                # color cannot change" — true only if both crops came from the
                # same imaging mode. 66% of this camera's frames are infrared,
                # i.e. greyscale, so the SAME person photographed either side of
                # a switch has two unrelated histograms and the veto fires on
                # physics rather than on identity.
                #
                # That contradicts the design already stated 140 lines up, for
                # the cross-boundary guard: "face + hand-off/stationary tiers
                # still bridge it (faces survive near-IR, physics doesn't care
                # about colour)". The guard exempts these tiers; the veto inside
                # them did not. Measured cost on the 18:30 hour: pairs 2.7px
                # apart across a 128s gap left unmerged, 408 fragments becoming
                # 31 identities, and the staff rule then reading that scatter as
                # 18 people working a one-receptionist desk.
                #
                # Abstain, never invert: when the modes differ we simply have no
                # colour opinion, and the spatial evidence (a body that did not
                # move) stands on its own.
                #
                # Reuses the _ir_mismatch predicate defined above rather than
                # adding a second notion of "is this pair across the boundary" —
                # two such definitions would drift apart, and this is precisely
                # the guard whose own comment already exempts these tiers.
                _hsv_veto_ok = not _ir_mismatch(a2, b2)
                # Count what this actually changes. On the 10-minute chunk the
                # fix produced a bit-identical run (72->16 people, 57 merges,
                # 8 of 10 called staff), and "no effect" has two very different
                # causes: the veto never fired across the boundary, or it fired
                # and the pair failed some later gate anyway. Without a counter
                # both look the same and the fix gets believed on its story.
                if (not _hsv_veto_ok) and (_near_handoff or _near_static):
                    _hsv_veto_skipped_ir += 1
                    if _handoff_hsv_contradicts(a2, b2, raw_crops):
                        _hsv_veto_would_have_blocked += 1
                if (gap <= handoff_gap_s and _near_handoff
                        and not _handoff_appearance_contradicts(a2, b2, anchor_embeddings)
                        and not (_hsv_veto_ok
                                 and _handoff_hsv_contradicts(a2, b2, raw_crops))):
                    # A near-instant, near-zero-distance hand-off is *stronger*
                    # evidence than a typical appearance match (osnet_x0_25 on a
                    # small/dim crop is easy to fool; a person can't teleport).
                    # BUG FIX (was): pseudo was capped at `sim_threshold - 0.001`,
                    # which forced every spatial/hand-off match to rank BELOW
                    # every appearance match in the sorted-descending greedy loop.
                    # That let weaker (sometimes wrong) appearance merges claim
                    # union-find roots first, so by the time a rock-solid 0px
                    # hand-off pair was processed, its root's group window
                    # already overlapped something else and the merge was
                    # silently rejected. Scoring it ABOVE ordinary appearance
                    # matches means it gets first claim, as it should.
                    conf = 1.0 - dist / handoff_px
                    pseudo = 0.90 + 0.09 * conf   # 0.90-0.99
                    pairs.append((pseudo, a2, b2, "hand-off"))
                elif (_near_static
                        and not _handoff_appearance_contradicts(a2, b2, anchor_embeddings)
                        and not (_hsv_veto_ok
                                 and _handoff_hsv_contradicts(a2, b2, raw_crops))):
                    # near-zero displacement across a long gap = same seat,
                    # same person, even with a weak/inconclusive embedding.
                    # Same fix as above: rank above typical appearance sims
                    # (but slightly below a true hand-off, since the gap here
                    # can be much longer and the identity claim slightly less
                    # certain) instead of being capped below sim_threshold.
                    conf = 1.0 - dist / stationary_px
                    pseudo = 0.85 + 0.09 * conf   # 0.85-0.94
                    pairs.append((pseudo, a2, b2, "stationary"))
                elif not _near_static:
                    _rej(a2, b2, f"{dist:.0f}px apart — beyond stationary_px "
                                 f"{stationary_px:.0f} and hand-off_px "
                                 f"{handoff_px:.0f}")
                elif _handoff_appearance_contradicts(a2, b2, anchor_embeddings):
                    _rej(a2, b2, "spatial match VETOED by CLIP anchor "
                                 "(appearance says different people)")
                else:
                    _rej(a2, b2, "spatial match VETOED by HSV colour")

    parent = {t: t for t in track_windows}

    def find(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    group_windows = {t: [track_windows[t]] for t in track_windows}
    merge_edges = []   # (v32) the ACTUAL accepted direct union edges, in the
                       # order they fired -- (sim, a, b, tier) -- as scored
                       # above. Kept separate from the final transitive
                       # `mapping` because a downstream veto (apply_face_veto)
                       # needs to know WHICH specific pair caused a merge and
                       # HOW STRONG that evidence was (hand-off/stationary
                       # near-certain physical evidence vs. an ordinary
                       # appearance-only gallery match) -- checking every
                       # pairwise combination inside a transitively-merged
                       # group conflates edges that were never directly
                       # compared with edges that actually drove the merge.
    tier_counts = defaultdict(int)          # accepted merges, per tier
    overlap_blocked = []   # (v38) high-confidence pairs REJECTED only because
                           # their groups' time windows already overlapped —
                           # this is the specific failure mode where a strong
                           # match (e.g. HSV=0.999) gets starved out by
                           # processing order rather than by weak evidence.
    # V77: re-check the HARD constraints at GROUP level, not just pairwise.
    #     A(staff) x C(customer) is refused as a pair...
    #     ...but A-B merges, B-C merges, and union-find makes A-C transitive.
    # A probe confirmed exactly that: A and C landed in one identity while the
    # diagnostics reported "2 blocked by role conflict". A constraint that
    # transitivity walks around is not a hard constraint.
    members = {t: {t} for t in track_windows}
    role_blocked_transitive = 0

    def _roles_of(ids):
        return {role_hint.get(i) for i in ids
                if role_hint and role_hint.get(i)} if role_hint else set()

    def _named_staff(ids):
        # An enrolled staff id is a NAME, not a number. Two different enrolled
        # people are KNOWN to be two people; no similarity may fuse them, and
        # doing so silently deletes one of their names from the report.
        return {i for i in ids if isinstance(i, str) and not str(i).isdigit()}

    # SORT KEY, NOT reverse=True. `pairs` is (sim, a, b, tier), and when two
    # pairs tie on sim Python falls through to comparing a — which is a TRACK
    # ID. Enrolled staff carry a NAME ('staff1'); everyone else carries an
    # int. Comparing those raises, and the whole stitching pass dies:
    #
    #   TypeError: '<' not supported between instances of 'str' and 'int'
    #
    # It was invisible for as long as the staff gallery was empty, because
    # then every id was an int. The first run with faces actually enrolled hit
    # it immediately — and the failure is expensive rather than loud: nothing
    # merges, so a chunk that should report ~22 people reports ~200.
    #
    # track_sort_key already exists in this module for exactly this mixed-type
    # problem. -sim keeps similarity descending while the tie-breaks stay
    # ascending and, more importantly, TOTAL — so the merge order is
    # deterministic across runs rather than dependent on dict ordering.
    for sim, a, b, tier in sorted(
            pairs, key=lambda p: (-p[0], track_sort_key(p[1]),
                                  track_sort_key(p[2]), str(p[3]))):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if _windows_overlap(group_windows[ra], group_windows[rb]):
            overlap_blocked.append((sim, a, b, tier))
            # THE DISTINCTION THAT DECIDES THE FIX.
            #
            # 1,517 of 2,703 obviously-mergeable pairs die here. But "blocked
            # by overlap" covers two completely different situations:
            #
            #   DIRECT     the two tracks THEMSELVES are on screen together.
            #              They are two people. The rejection is CORRECT and
            #              there is nothing to fix in the merger.
            #
            #   TRANSITIVE the two tracks never co-exist, but the GROUPS they
            #              were greedily merged into do. The block is an
            #              artefact of processing ORDER — an earlier, weaker
            #              merge polluted a group and shut the door on a
            #              stronger one. THIS is what a global solver fixes.
            #
            # Without separating them, "56% blocked by overlap" argues equally
            # well for "the merger is broken" and "the merger is working". One
            # boolean decides whether to rewrite the association step at all.
            _direct = _windows_overlap([track_windows[a]], [track_windows[b]])
            if _direct:
                _rej(a, b, f"tier '{tier}' accepted it, but the two tracks are "
                           f"DIRECTLY co-visible — they are two people")
            else:
                _rej(a, b, f"tier '{tier}' ACCEPTED it (score {sim:.3f}), then "
                           f"blocked TRANSITIVELY: the tracks never co-exist, "
                           f"but their greedily-merged GROUPS do")
                _overlap_transitive[0] += 1
            continue   # same-time fragments cannot be one person
        ga, gb = members[ra], members[rb]
        if len(_roles_of(ga) | _roles_of(gb)) > 1:
            role_blocked_transitive += 1
            _rej(a, b, f"tier '{tier}' accepted it, then group roles "
                       f"conflicted (staff vs customer, transitively)")
            continue   # staff and customer cannot become one person
        if len(_named_staff(ga) | _named_staff(gb)) > 1:
            role_blocked_transitive += 1
            _rej(a, b, f"tier '{tier}' accepted it, then two DIFFERENT enrolled "
                       f"staff names would have fused")
            continue   # two DIFFERENT enrolled staff are two people
        parent[rb] = ra
        members[ra] = ga | gb
        group_windows[ra] = group_windows[ra] + group_windows[rb]
        merge_edges.append((sim, a, b, tier))
        tier_counts[tier] += 1

    mapping = {}
    canon = {}
    _cpk = _canon_priority_key(track_windows)
    for t in sorted(track_windows, key=_cpk):
        root = find(t)
        canon.setdefault(root, t)      # staff names win; then earliest
        mapping[t] = canon[root]

    diagnostics = {
        "tier_counts": dict(tier_counts),
        "role_conflicts_blocked": role_conflict_count,
        "hsv_veto_skipped_ir": _hsv_veto_skipped_ir,
        "hsv_veto_rescued": _hsv_veto_would_have_blocked,
        # F12: {(a, b): "first gate that rejected this pair"}. Consumed by the
        # engine's unmerged-near-pair report, which knows which pairs SHOULD
        # obviously have merged but until now could not say why they did not.
        "pair_reject": _pair_reject,
        "role_blocked_transitive": role_blocked_transitive,
        "overlap_blocked_count": len(overlap_blocked),
        # of those, how many were blocked only because their GROUPS overlap —
        # i.e. an artefact of greedy merge ORDER, not of the pair's own
        # evidence. This is the number that says whether replacing the greedy
        # association step is worth doing.
        "overlap_blocked_transitive": _overlap_transitive[0],
        # sorted highest-confidence-first: these are the "should have merged,
        # got starved by order" candidates worth a manual look.
        # Same mixed-type hazard as the merge loop above: these tuples end in
        # track ids, and a tie on sim would compare 'staff1' against 1082.
        "overlap_blocked_samples": sorted(
            overlap_blocked,
            key=lambda p: (-p[0], track_sort_key(p[1]),
                           track_sort_key(p[2]), str(p[3])))[:30],
    }
    return mapping, merge_edges, diagnostics


# ---------------------------------------------------------------------------
# Persistent identity memory (v38)
# ---------------------------------------------------------------------------
# The merge step above solves fragmentation WITHIN one processing run. These
# two functions are what let a person's ID stay FIXED going forward: build a
# durable "dossier" (memory snapshot) per confirmed person out of every
# evidence type collected on them, then cross-verify any new track fragment
# against that memory using the exact same evidence priority as the merge
# step (face > hand-off/stationary > anchor > attire > plain-gallery) before
# ever assigning it a brand-new id. A fragment only ever gets folded into an
# existing person when it clears one of these tiers' bars; anything weaker
# becomes its own new, equally trackable identity rather than a guess.

def build_identity_dossiers(mapping, merge_edges, track_windows,
                            embeddings=None, anchor_embeddings=None,
                            face_embeddings=None, raw_crops=None,
                            positions=None):
    """Build one persistent memory record per canonical person.

    Pools every fragment that was folded into that person plus every
    evidence snapshot collected across those fragments (appearance gallery,
    anchor crop embedding, face embedding, attire/HSV crops, last-known
    position) and records WHICH evidence tier(s) actually proved the
    fragments belong together. This dossier is the "memory" a later call to
    match_track_to_dossiers cross-verifies new sightings against.

    Returns {person_id: dossier_dict}. See match_track_to_dossiers for how
    the dossier is consumed.
    """
    fragments_by_person = defaultdict(list)
    for tid, cid in mapping.items():
        fragments_by_person[cid].append(tid)

    tiers_by_person = defaultdict(set)
    for sim, a, b, tier in merge_edges:
        tiers_by_person[mapping.get(a, a)].add(tier)

    dossiers = {}
    for cid, frags in fragments_by_person.items():
        frags = sorted(frags, key=lambda t: track_windows[t][0])
        gallery, crops, positions_track = [], [], []
        anchor_vec, face_vec = None, None
        for t in frags:
            if embeddings:
                gallery.extend(_as_gallery(embeddings.get(t, [])))
            if raw_crops:
                crops.extend(raw_crops.get(t, []))
            if anchor_embeddings and anchor_vec is None and t in anchor_embeddings:
                anchor_vec = anchor_embeddings[t]
            if face_embeddings and face_vec is None and t in face_embeddings:
                face_vec = face_embeddings[t]
            if positions and t in positions:
                positions_track.append((t, positions[t]))
        dossiers[cid] = {
            "person_id": cid,
            "fragment_ids": frags,
            "t_first": min(track_windows[t][0] for t in frags),
            "t_last": max(track_windows[t][1] for t in frags),
            "evidence_tiers_used": sorted(tiers_by_person.get(cid, [])),
            "appearance_gallery": gallery,      # plain-gallery snapshot
            "anchor_embedding": anchor_vec,      # anchor snapshot
            "face_embedding": face_vec,          # face snapshot
            "attire_crops": crops,               # attire/HSV snapshot
            "positions_by_fragment": positions_track,  # hand-off/stationary
        }
    return dossiers


def match_track_to_dossiers(track_id, track_window, dossiers,
                            embeddings=None, anchor_embeddings=None,
                            face_embeddings=None, raw_crops=None,
                            position=None, sim_threshold=0.65,
                            anchor_sim_threshold=0.75, hsv_sim_threshold=0.75,
                            face_sim_threshold=0.35, handoff_gap_s=2.5,
                            handoff_px=90.0, stationary_px=30.0,
                            max_gap_s=480.0):
    """Cross-verify one new track fragment against persistent identity
    memory before deciding whether it's a KNOWN person (reuse their id) or
    a genuinely NEW one (mint a new id).

    Checks the same evidence tiers merge_fragmented_tracks uses, in the same
    priority (face > hand-off/stationary > anchor > attire > gallery), takes
    the single strongest tier that clears its own threshold across every
    candidate dossier, and returns that dossier's person_id. If nothing
    clears any bar, returns (None, None, None) -- the caller should assign a
    brand-new id rather than force a match on weak/absent evidence, so a
    genuinely new person never inherits a stale identity.

    Returns (person_id_or_None, evidence_tier_or_None, score_or_None).
    """
    candidates = []  # (pseudo_score, tier, person_id)
    for cid, d in dossiers.items():
        gap = (track_window[0] - d["t_last"] if track_window[0] >= d["t_last"]
               else d["t_first"] - track_window[1])
        if gap < 0 or gap > max_gap_s:
            continue  # co-visible (can't be same body) or too far apart

        if (face_embeddings and track_id in face_embeddings
                and d["face_embedding"] is not None):
            fsim = _cosine(face_embeddings[track_id], d["face_embedding"])
            if fsim >= face_sim_threshold:
                conf = min(1.0, (fsim - face_sim_threshold)
                           / max(1e-6, 1.0 - face_sim_threshold))
                candidates.append((0.92 + 0.07 * conf, "face", cid))

        if position and d["positions_by_fragment"]:
            _, last_pos = d["positions_by_fragment"][-1]
            dist = math.hypot(last_pos[1][0] - position[0][0],
                              last_pos[1][1] - position[0][1])
            if gap <= handoff_gap_s and dist <= handoff_px:
                conf = 1.0 - dist / handoff_px
                candidates.append((0.90 + 0.09 * conf, "hand-off", cid))
            elif dist <= stationary_px:
                conf = 1.0 - dist / stationary_px
                candidates.append((0.85 + 0.09 * conf, "stationary", cid))

        if (anchor_embeddings and track_id in anchor_embeddings
                and d["anchor_embedding"] is not None):
            asim = _cosine(anchor_embeddings[track_id], d["anchor_embedding"])
            if asim >= anchor_sim_threshold:
                conf = min(1.0, (asim - anchor_sim_threshold)
                           / max(1e-6, 1.0 - anchor_sim_threshold))
                candidates.append((0.80 + 0.09 * conf, "anchor", cid))

        if raw_crops and track_id in raw_crops and d["attire_crops"]:
            hsim = _hsv_gallery_sim(raw_crops[track_id], d["attire_crops"])
            if hsim >= hsv_sim_threshold:
                conf = min(1.0, (hsim - hsv_sim_threshold)
                           / max(1e-6, 1.0 - hsv_sim_threshold))
                candidates.append((0.76 + 0.06 * conf, "attire", cid))

        if embeddings and track_id in embeddings and d["appearance_gallery"]:
            sim = _gallery_sim(embeddings[track_id], d["appearance_gallery"])
            if sim >= sim_threshold:
                candidates.append((sim, "gallery", cid))

    if not candidates:
        return None, None, None
    score, tier, cid = max(candidates)
    return cid, tier, score


def _rebuild_plane(run):
    """Rebuild the ground plane from a run dict.

    The GroundPlane object cannot survive the per-chunk JSON cache, but its two
    coefficients can, so it is reconstructed wherever it is needed downstream
    rather than silently falling back to pixels.

    Lives HERE, in the analytics cell, and not next to its first caller: the
    answers cell CALLS answer_for_run() at the bottom, so a helper defined in a
    later cell is a NameError 40 minutes into a paid GPU session. Caught by
    check_notebook_runtime.py, which is why that script exists.
    """
    c = run.get("perspective_coeffs")
    fs = run.get("frame_size_analysed")
    if not c or not fs:
        return GroundPlane.none("no perspective coefficients in this run")
    # P2: person_h is a DEFAULT ARGUMENT bound at def time to the module
    # constant, so setting a notebook global named PERSON_H_M does nothing. It
    # has to be passed. Without this a venue configuring person_height_m: 1.62
    # was silently measured with a 1.70 m ruler and every distance was ~5% out.
    _ph = ((globals().get("VENUE_PROFILE") or {}).get("camera", {})
           .get("person_height_m")) or PERSON_H_M
    _hf = ((globals().get("VENUE_PROFILE") or {}).get("camera", {})
           .get("hfov_deg")) or DEFAULT_HFOV_DEG
    return GroundPlane.from_perspective(c[0], c[1], fs[0], fs[1],
                                        person_h=_ph, hfov_deg=_hf)


def confirm_crossings(crossings, confirm_s=5.0, per_line=None,
                      track_first_seen=None, min_prior_s=0.0):
    """Drop crossings that a person did not COMMIT to. -> (kept, dropped)

    THE PROBLEM, in one sentence: your entry line sits where guests pause.

    A person who steps over the threshold, hesitates, and steps back produces
    IN then OUT. Both are real line crossings and both are wrong: nobody
    arrived. tier_a_crossings cannot see this — it collapses events close in
    time and space in the SAME direction, so an IN/OUT pair survives as one
    arrival and one departure.

    This is the industry's "confirmation wait" / "throttling" and its "U-turn
    filter", which are the same idea seen from two sides: a crossing counts
    only if the person was still on the far side after `confirm_s`. If they
    came back through the same door sooner than that, they never really left
    the side they started on, so BOTH events are cancelled.

    Deliberately symmetric. A guest hovering at the door (in->out) and a staff
    member stepping out for a moment (out->in) are the same physical event, and
    counting either one inflates a different number.

    per_line: {door: seconds} — a street entrance wants ~5s of commitment; an
    interior threshold people walk briskly through wants far less, or the wait
    itself starts deleting real transits.

    PRIOR STABILITY (min_prior_s), added 2026-08-18.

    The published spec for a valid crossing has six conditions; this function
    implemented four of them. The one missing here is the FIRST: a person must
    have been stably visible on one side BEFORE the crossing counts.

    Without it, a track that is BORN near the line counts as a transit even
    though nobody walked anywhere. That is not hypothetical on this camera —
    16.3% of detections cannot start a track at all, ids churn constantly, and
    the operator's frame-by-frame audit found exactly this signature: "dining
    entry IN 2" with a single staff member visible, and "staff entry OUT 6"
    while those staff were still in the room. Counts accumulating with no
    corresponding movement.

    So a crossing is discarded unless its track existed for at least
    min_prior_s BEFORE the crossing moment. A real transit has a history of
    approaching; a track invented on the line does not.

    track_first_seen: {track_id: t} — when each track was first observed.
    Without it this check cannot run and is skipped, which is the old
    behaviour: no silent half-enforcement.

    Pure and order-independent: takes a list, returns lists. No plane, no
    frames, no GPU — so the edge cases are tested rather than argued about.
    """
    per_line = per_line or {}
    track_first_seen = track_first_seen or {}
    by = {}
    for c in crossings or []:
        by.setdefault((c.get("track_id"), c.get("line") or "entry"), []).append(c)
    kept, dropped = [], []
    for (_tid, line), evs in by.items():
        evs.sort(key=lambda c: float(c.get("t", 0.0)))
        window = float(per_line.get(line, confirm_s))
        used = [False] * len(evs)
        for i in range(len(evs) - 1):
            if used[i]:
                continue
            a, b = evs[i], evs[i + 1]
            if used[i + 1] or a.get("direction") == b.get("direction"):
                continue
            if float(b.get("t", 0.0)) - float(a.get("t", 0.0)) <= window:
                # crossed and came straight back: neither event happened
                used[i] = used[i + 1] = True
                dropped += [a, b]
        for e, u in zip(evs, used):
            if u:
                continue
            # PRIOR STABILITY: was this track already around before it crossed?
            if min_prior_s > 0 and track_first_seen:
                born = track_first_seen.get(e.get("track_id"))
                if born is not None and float(e.get("t", 0.0)) - float(born) < min_prior_s:
                    e = dict(e, dropped_reason="born at the line "
                             f"({float(e.get('t',0.0)) - float(born):.1f}s of "
                             f"history, need {min_prior_s:.1f}s)")
                    dropped.append(e)
                    continue
            kept.append(e)
    kept.sort(key=lambda c: float(c.get("t", 0.0)))
    return kept, dropped


def tier_a_crossings(crossings, plane=None, dedupe_s=2.5, dedupe_m=1.2,
                     dedupe_px=140.0, direction="in", roles=None, spans=None,
                     overlap_s=0.4):
    """TIER A: how many people came through the door, WITHOUT depending on
    identity holding together for ten hours.

    entered_count() counts unique track_ids after the night-long Re-ID stitch,
    so it inherits every Re-ID error: a person whose id fragments at the door is
    counted twice, and two people wrongly merged are counted once. But crossing
    a line is a three-second event. It does not need a ten-hour identity, only a
    local one.

    So: count crossing EVENTS, then collapse events that are close in BOTH time
    and space — because that is what one person crossing once looks like, no
    matter how many track ids they wore while doing it.

    Needs each crossing to carry a position; crossings without one fall back to
    unique-id behaviour for that event, which is the old answer, never worse.

    CO-VISIBILITY (spans), added 2026-08-18 on measured evidence.

    "Close in time and space" is what one person crossing once looks like. It
    is ALSO what a GROUP walking in abreast looks like, and this function had
    no way to tell them apart -- so a party arriving together collapsed to a
    single arrival.

    Ground truth (eval/gt_entries_305_318.json): six guests entered CAM.112
    between 305s and 318s, ids 135/138/144/150 beginning within seconds of one
    another at x=1739/1785/1696/1766 -- well inside dedupe_s=6.0 and
    dedupe_px=140. The window counted 2 of 6.

    Two tracks that are visible AT THE SAME MOMENT cannot be one person, so
    they are never folded together however close they are. The pipeline
    already relies on this principle for Re-ID (ENABLE_COVISIBILITY_BLOCK);
    it simply was not applied to counting.

    spans: {track_id: (t_first, t_last)}. Without it the check is skipped and
    behaviour is exactly as before -- no silent half-enforcement.

    dedupe_s 6.0 -> 2.5, and it is NOT independent of the co-visibility guard:
    measured on the four abreast ids above, neither change works alone.

        dedupe_s   no spans   +co-visibility
           6.0         2            2     <- old default, fails either way
           4.0         2            4
           2.5         2            4     <- chosen
           1.5         2            4
           1.0         2            4

    Without spans the party collapses at EVERY window, because the ids really
    are close in time and space. With spans but a 6s window, a middle id stays
    in `kept` long enough for the later arrivals to match against it instead of
    against a co-visible one. 2.5s is bounded below too: at 1.0s a single
    person whose id is re-minted 1.2s later splits into two.
    """
    # Only crossings at a VENUE ENTRANCE. A dining-room threshold and a staff
    # gate are movements INSIDE the building, not arrivals, and counting them
    # inflates the headline guest number by every interior transit.
    _lines = {c.get("line") for c in crossings if c.get("line") is not None}
    _venue = venue_entry_lines(_lines) if _lines else None
    evs = [c for c in crossings if c.get("direction") == direction
           and not (roles and roles.get(c.get("track_id")) == "staff")
           and (_venue is None or c.get("line") is None
                or c["line"] in _venue)]
    evs.sort(key=lambda c: c["t"])
    kept = []
    for c in evs:
        p = c.get("pos")
        dup = False
        for k in reversed(kept):
            if c["t"] - k["t"] > dedupe_s:
                break
            if p is None or k.get("pos") is None:
                if k.get("track_id") == c.get("track_id"):
                    dup = True
                    break
                continue
            d = plane.dist_m(p, k["pos"]) if (plane is not None and plane.ok) else None
            if (d is not None and d <= dedupe_m) or (
                    d is None and math.hypot(p[0] - k["pos"][0],
                                             p[1] - k["pos"][1]) <= dedupe_px):
                if spans is not None:
                    _a = spans.get(c.get("track_id"))
                    _b = spans.get(k.get("track_id"))
                    if (_a and _b
                            and min(_a[1], _b[1]) - max(_a[0], _b[0]) > overlap_s):
                        continue      # co-visible -> two people, not one
                dup = True
                break
        if not dup:
            kept.append(c)
    return len(kept), kept


def detect_groups(arrivals, plane=None, window_s=25.0, radius_m=3.0,
                  radius_px=320.0):
    """G4: people who arrive together are one PARTY. A restaurant books parties,
    not individuals, so 12 arrivals in 4 groups is a different night from 12
    walk-ins. Chaining is deliberate — a family of five spreads wider than the
    radius but every member is close to the one before.

    Only sane in metres: a 3 m radius is ~470 px at the door and ~96 px at the
    back of the room, so in pixels this rule would silently mean different
    things in different parts of the same frame.
    """
    evs = sorted(arrivals, key=lambda c: c["t"])
    groups = []
    for c in evs:
        placed = False
        for g in groups:
            if c["t"] - g["t_last"] > window_s:
                continue
            for m in g["members"]:
                p, q = c.get("pos"), m.get("pos")
                if p is None or q is None:
                    continue
                d = plane.dist_m(p, q) if (plane is not None and plane.ok) else None
                near = (d <= radius_m) if d is not None else (
                    math.hypot(p[0] - q[0], p[1] - q[1]) <= radius_px)
                if near:
                    g["members"].append(c)
                    g["t_last"] = c["t"]
                    placed = True
                    break
            if placed:
                break
        if not placed:
            groups.append({"t_first": c["t"], "t_last": c["t"], "members": [c]})
    for g in groups:
        g["size"] = len(g["members"])
    return groups


def remap_events(events, crossings, mapping):
    """Apply a track merge mapping to events + crossings; role per canonical
    person = role of its longest-dwelling fragment."""
    dwell = defaultdict(lambda: defaultdict(float))
    for e in events:
        cid = mapping.get(e["track_id"], e["track_id"])
        dwell[cid][e["role"]] += e["duration"]
    role_of = {cid: max(roles.items(), key=lambda kv: kv[1])[0]
               for cid, roles in dwell.items()}
    new_events = [dict(e, track_id=mapping.get(e["track_id"], e["track_id"]),
                       role=role_of.get(mapping.get(e["track_id"],
                                                    e["track_id"]), e["role"]))
                  for e in events]
    new_crossings = [dict(c, track_id=mapping.get(c["track_id"], c["track_id"]))
                     for c in crossings]
    return new_events, new_crossings, role_of


# ---------------------------------------------------------------------------
# Minute-by-minute table (mentor's output contract)
# ---------------------------------------------------------------------------

def minute_summaries(events, crossings, camera_id, t_end, window_s=60.0,
                     roles=None):
    """One row per window: who was present, what happened. Deterministic
    (non-LLM) summaries grounded in events; matches the mentor's JSON keys.
    When roles is given, entered/exited counts use the SAME definition as the
    headline answer: unique non-staff people, not raw crossings."""
    def _count(dir_, w_start, w_end):
        hits = [c for c in crossings
                if c["direction"] == dir_ and w_start <= c["t"] < w_end]
        if roles is None:
            return len(hits)
        return len({c["track_id"] for c in hits
                    if roles.get(c["track_id"]) != "staff"})

    rows = []
    n_windows = int(t_end // window_s) + (1 if t_end % window_s > 1e-9 else 0)
    for w in range(n_windows):
        w_start, w_end = w * window_s, min((w + 1) * window_s, t_end)
        present = sorted({e["track_id"] for e in events
                          if e["t_out"] > w_start and e["t_in"] < w_end}, key=track_sort_key)
        ins = [None] * _count("in", w_start, w_end)
        outs = [None] * _count("out", w_start, w_end)
        zones_active = sorted({e["zone"] for e in events
                               if e["t_out"] > w_start and e["t_in"] < w_end})
        bits = []
        if ins:
            bits.append(f"{len(ins)} entered")
        if outs:
            bits.append(f"{len(outs)} exited")
        if zones_active:
            bits.append("activity in: " + ", ".join(zones_active))
        summary = "; ".join(bits) if bits else "no activity"
        mm = lambda s: f"{int(s // 60):02d}:{int(s % 60):02d}"
        rows.append({
            "time_window": f"{mm(w_start)}-{mm(w_end)}",
            "camera_id": camera_id,
            "people_count": len(present),
            "person_ids": [f"P{tid}" if (isinstance(tid, int) or str(tid).isdigit()) else str(tid) for tid in present],
            "action": summary,
            "possible_next_camera": None,   # single-camera run — the cross-
            "confidence": None,             # camera phase fills both fields
            "summary": f"{len(present)} people visible. {summary}.",
        })
    return rows
