"""venue_profile.py — PHASE 4 / U4-U8. Config is DATA, not edits to Cell 2.

THE COMPLAINT THIS ANSWERS
    "if I have to change the pipeline for each video, that is not engineering."
    Correct. Today ~20 values are hardcoded in the config cell — CAMERA_ID,
    DRIVE_TZ, ENTRY_LINE_FLIP, MIN_SEATED_S, STAFF_DOMINANCE_RATIO and the rest.
    Pointing the pipeline at a different camera means editing the pipeline.

    Everything camera-specific or venue-specific now lives in a profile that
    travels WITH the footage, next to the zones file. The code carries defaults;
    the profile overrides them; nothing about a new venue requires a code edit.

WHAT IS DELIBERATELY *NOT* HERE
    U6 was going to estimate a distribution of person heights. It is not worth
    building: a wrong PERSON_H_M scales every distance UNIFORMLY, so relative
    comparisons are unaffected and absolute thresholds shift by the same
    percentage. 1.70 vs 1.75 m is a 3% error, against 25% from a 20-degree
    camera tilt. It is a documented, configurable constant and that is enough.

LOOKUP ORDER (first hit wins)
    1. explicit argument                    (a notebook override, for testing)
    2. profile_<stem>.json beside the video
    3. a "profile" key inside zones_<stem>.json
    4. DEFAULTS below
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except ImportError:                                   # pragma: no cover
    ZoneInfo = None

# Every value carries its UNIT and the reason it exists. A config entry whose
# meaning has to be reverse-engineered from the code is not configuration.
DEFAULTS = {
    "camera": {
        "id": None,                    # str  — falls back to the video stem
        "timezone": "UTC",             # IANA name, e.g. "America/Chicago".
                                       # U8: NOT an abbreviation. "CDT" cannot
                                       # express a night that crosses a DST
                                       # change; "America/Chicago" can.
        "hfov_deg": 82.0,              # deg  — lens horizontal field of view.
                                       # Only scales the DEPTH axis of the auto
                                       # ground plane. Supply ground_points in
                                       # the zones file and this stops mattering.
        "person_height_m": 1.70,       # m    — the metric ruler. See the note
                                       # above: a small error here is uniform.
        "zone_tol_frac": 0.008,        # frac of frame diagonal — how far a zone
                                       # may shift before the run is INVALID.
                                       # Derived, see camera_health.py.
        "entry_line_flip": None,       # None = infer it from the footage (U5).
                                       # True/False pins it.
        "static": True,                # False for a PTZ: zone checks cannot
                                       # apply and must not silently pretend to.
    },
    "venue": {
        "type": "generic",             # free text, for the report only
        "min_seated_s": 60,            # s  — dwell that counts as "seated"
        "wait_threshold_s": 600,       # s  — "waited too long"
        "party_gap_s": 120,            # s  — gap inside one party's occupancy
        "min_party_s": 60,             # s
        "visit_min_s": 8,              # s  — staff presence that counts as a visit
        "staff_override_min_s": 60,    # s  — dwell in a staff zone => staff
        "staff_min_video_share": 0.35, # frac of the video inside a staff zone
        "staff_dominance_ratio": 3.0,  # x   — staff-zone time vs everywhere else
        "greet_min_contact_s": 3.0,    # s  — brief pass-by is not a greeting
        "greet_proximity_m": 1.5,      # m  — conversational distance (PROXY)
        "turnaway_max_s": 90.0,        # s  — in and out this fast, unserved
        "long_wait_s": 180.0,          # s  — the "waited" line in the report
        "micro_absence_s": 90.0,       # s  — desk gap that is doing the job
        "break_absence_s": 600.0,      # s  — desk gap that is a break
        "group_window_s": 25.0,        # s  — arrivals this close in time...
        "group_radius_m": 3.0,         # m  — ...and this close are one party
        "max_walk_speed_mps": 2.2,     # m/s — brisk walk, gates identity
    },
    "privacy": {
        "face_scope": "staff_only",    # "staff_only" | "all"
                                       # staff_only: post-processing face embeddings
                                       # (corroboration, merge tier, veto) are
                                       # restricted to tracks already identified as
                                       # staff. Customer faces are never embedded.
                                       # Staff gallery matching (per-frame) is always
                                       # on — staff are enrolled by name/photo.
                                       # all: every track gets face embeddings for
                                       # merge quality. Requires explicit biometric
                                       # consent from customers at the venue.
        "blur_exports": False,         # True = face-blur exported clip frames.
                                       # Not yet implemented; placeholder for Ph9.
    },
}

# Allowed values for enum-like config fields. Not bounds — discrete choices.
ENUMS = {
    "privacy.face_scope": ("staff_only", "all"),
}

# Bounds exist to catch a typo that would otherwise produce a plausible-looking
# wrong report — 600 vs 60, or metres typed where seconds were meant.
BOUNDS = {
    "camera.hfov_deg": (20.0, 180.0),
    "camera.person_height_m": (1.2, 2.2),
    "camera.zone_tol_frac": (0.001, 0.05),
    "venue.min_seated_s": (5, 3600),
    "venue.wait_threshold_s": (10, 7200),
    "venue.greet_proximity_m": (0.3, 6.0),
    "venue.group_radius_m": (0.5, 15.0),
    "venue.max_walk_speed_mps": (0.5, 6.0),
    "venue.staff_min_video_share": (0.0, 1.0),
    "venue.staff_dominance_ratio": (1.0, 100.0),
}


def _merge(base, over):
    out = {k: dict(v) for k, v in base.items()}
    for section, vals in (over or {}).items():
        if section in out and isinstance(vals, dict):
            out[section].update(vals)
        else:
            out[section] = vals
    return out


def load_profile(video_path=None, zones_path=None, explicit=None):
    """Merge the first profile found over DEFAULTS. Never raises: a missing or
    malformed profile falls back to defaults and reports it in `_source`."""
    src, over = "defaults", {}
    if explicit:
        src, over = "explicit argument", explicit
    else:
        cands = []
        if video_path:
            v = Path(video_path)
            cands.append(v.with_name(f"profile_{v.stem}.json"))
            cands.append(v.with_name("profile.json"))
        for p in cands:
            if p.exists():
                try:
                    over, src = json.loads(p.read_text()), str(p)
                    break
                except json.JSONDecodeError as e:
                    src = f"{p} (MALFORMED: {e}) — using defaults"
        else:
            if zones_path and Path(zones_path).exists():
                try:
                    z = json.loads(Path(zones_path).read_text())
                    if isinstance(z.get("profile"), dict):
                        over, src = z["profile"], f"{Path(zones_path).name}:profile"
                except json.JSONDecodeError:
                    pass
    prof = _merge(DEFAULTS, over)
    prof["_source"] = src
    if not prof["camera"].get("id") and video_path:
        prof["camera"]["id"] = Path(video_path).stem
    return prof


def validate(profile):
    """-> list of problems. Empty means usable. Loud beats plausible-but-wrong."""
    problems = []
    for path, (lo, hi) in BOUNDS.items():
        sec, key = path.split(".")
        v = profile.get(sec, {}).get(key)
        if v is None:
            continue
        if not isinstance(v, (int, float)) or not (lo <= v <= hi):
            problems.append(f"{path} = {v!r} is outside the sane range [{lo}, {hi}]")
    # Enum validation — discrete choices, not ranges.
    for path, allowed in ENUMS.items():
        sec, key = path.split(".")
        v = profile.get(sec, {}).get(key)
        if v is not None and v not in allowed:
            problems.append(
                f"{path} = {v!r} is not a valid choice. "
                f"Allowed: {', '.join(repr(a) for a in allowed)}")
    tz = profile.get("camera", {}).get("timezone")
    if tz and ZoneInfo is not None:
        try:
            ZoneInfo(tz)
        except Exception:
            problems.append(
                f"camera.timezone = {tz!r} is not an IANA zone name. Use e.g. "
                f"'America/Chicago', not 'CDT' — an abbreviation cannot express "
                f"a night that crosses a daylight-saving change.")
    return problems


def local_clock(profile, naive_start):
    """U8: naive wall time from the filename -> a DST-correct clock function.

    Returns f(seconds_into_video) -> 'HH:MM:SS'. Adding a timedelta to an
    aware datetime crosses a DST boundary correctly; adding it to a string
    called 'CDT' does not.
    """
    tz = profile.get("camera", {}).get("timezone") or "UTC"
    if naive_start is None:
        return lambda s: ""
    if ZoneInfo is None:
        return lambda s: (naive_start + timedelta(seconds=float(s))).strftime("%H:%M:%S")
    try:
        start = naive_start.replace(tzinfo=ZoneInfo(tz))
    except Exception:
        start = naive_start.replace(tzinfo=ZoneInfo("UTC"))

    def at(seconds):
        # arithmetic in UTC, render in local: the only order that survives a
        # DST jump without silently shifting or duplicating an hour
        return (start.astimezone(ZoneInfo("UTC")) + timedelta(seconds=float(seconds))
                ).astimezone(start.tzinfo).strftime("%H:%M:%S")

    return at


def infer_entry_direction(crossings, events, interior_zones, min_evidence=6):
    """U5: work out which way through the door is IN, from the footage itself.

    ENTRY_LINE_FLIP is currently a hand-set boolean. Get it wrong and the
    headline number is exactly backwards, and the only existing guard is a
    heuristic that fires when outward crossings outnumber inward ones by 1.6x.

    The evidence is already in the data: someone who ENTERS spends time in
    interior zones AFTER crossing; someone who LEAVES spent it BEFORE.

    BOTH directions are used, with opposite sign. A one-sided version (only
    looking at crossings labelled "in") throws away half the evidence and goes
    blind on any venue where that label happens to be rare — e.g. a closing
    shift, where a correctly-configured line produces almost no inward
    crossings at all. Signed symmetrically:
        labelled "in"  -> expect dwell AFTER  (+1 x (after - before))
        labelled "out" -> expect dwell BEFORE (-1 x (after - before))
    so a positive total means the labels are right and negative means flipped,
    regardless of which way the traffic happened to run that night.

    -> (flip_needed, confidence 0..1, evidence dict). flip_needed is None when
    there is not enough evidence to say, which must stay distinct from False.
    """
    by_track = {}
    for e in events:
        if e.get("zone") in interior_zones:
            by_track.setdefault(e["track_id"], []).append((e["t_in"], e["t_out"]))

    score, n = 0.0, 0
    for c in crossings:
        sign = {"in": 1.0, "out": -1.0}.get(c.get("direction"))
        if sign is None:
            continue
        ivs = by_track.get(c.get("track_id"))
        if not ivs:
            continue
        t = c["t"]
        after = sum(max(0.0, b - max(a, t)) for a, b in ivs)
        before = sum(max(0.0, min(b, t) - a) for a, b in ivs)
        if after == before == 0:
            continue
        score += sign * (after - before)
        n += 1

    ev = {"tracks_with_evidence": n, "net_dwell_after_minus_before_s": round(score, 1)}
    if n < min_evidence:
        ev["why"] = (f"only {n} crossing(s) had interior dwell either side "
                     f"(need {min_evidence}) — keeping the configured value")
        return None, 0.0, ev
    total = sum(abs(x) for x in (score,)) or 1.0
    conf = min(1.0, abs(score) / max(total, 1.0))
    ev["verdict"] = ("labels look correct" if score > 0
                     else "labels look BACKWARDS — in/out are swapped")
    return (score < 0), (1.0 if n >= min_evidence else conf), ev


def describe(profile):
    lines = [f"profile source: {profile.get('_source')}"]
    for sec in ("camera", "venue"):
        diffs = {k: v for k, v in profile.get(sec, {}).items()
                 if v != DEFAULTS.get(sec, {}).get(k)}
        lines.append(f"  {sec}: " + (", ".join(f"{k}={v}" for k, v in diffs.items())
                                     if diffs else "all defaults"))
    return "\n".join(lines)


def write_template(path, video_stem=""):
    """Emit a filled-in profile so a new venue is a file to edit, not code."""
    tpl = {"camera": dict(DEFAULTS["camera"]), "venue": dict(DEFAULTS["venue"])}
    tpl["camera"]["id"] = video_stem or "CAMERA_ID"
    tpl["_README"] = ("Everything the pipeline needs to know about THIS camera "
                      "and THIS venue. Put it next to the video as "
                      "profile_<video-stem>.json. timezone must be an IANA name "
                      "(America/Chicago), never an abbreviation. entry_line_flip "
                      "null = infer it from the footage.")
    Path(path).write_text(json.dumps(tpl, indent=2))
    return Path(path)
