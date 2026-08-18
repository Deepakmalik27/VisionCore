"""answers.py — the questions the project exists to answer.

NOT A PORT OF CELL 18
    Cell 18 computes the same quantities, and its versions shipped wrong twice:
    guests_tonight came from a region fallback and was labelled EXACT*, and
    desk_covered_pct appeared as 56.4 in one place and 68.9 in another in the
    same report. Reproducing that faithfully would reproduce the faults.

THE PRINCIPLE
    An answer is a VALUE over a DENOMINATOR, and both carry their own validity.

    Three consequences, and every design decision here follows from them:

    1. THE DENOMINATOR IS OBSERVED TIME, NEVER ELAPSED TIME.
       "The desk was covered 68.9% of the night" is meaningless unless you say
       68.9% of WHAT. If the camera was blind for twenty minutes, those minutes
       are not "uncovered" — they are unmeasured, and dividing by them turns a
       dead camera into a quiet venue.

    2. AN ANSWER THAT NEEDS IDENTITY MUST CARRY A RANGE.
       Measured re-id separability on this footage is ~0.66. Any count that
       requires one person to stay one person for minutes is uncertain by
       construction, and a single integer hides that. Desk coverage does NOT
       need identity — it only asks "is anyone in this polygon" — which is why
       it is the one metric that can honestly reach 90%.

    3. EVERY ANSWER STATES WHAT WOULD MAKE IT WRONG.
       Not a confidence score: a sentence. `caveats` is the difference between
       a number a manager can act on and a number they should not.

    An answer whose inputs cannot support it returns tier=UNKNOWN with a
    reason. It never returns 0. "None arrived" and "we could not tell" are
    different facts, and the pipeline has already published one as the other.

TIERS
    EXACT     measured directly; needs no identity to hold over time
    PROXY     a stand-in for the real thing (proximity is not conversation)
    ESTIMATE  needed identity to hold; carries a range
    WEAK      may over-count; never act on alone
    UNKNOWN   the inputs could not support the question
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .log import get_logger

_log = get_logger("answers")

EXACT, PROXY, ESTIMATE, WEAK, UNKNOWN = (
    "EXACT", "PROXY", "ESTIMATE", "WEAK", "UNKNOWN")


@dataclass
class Answer:
    """One question, one answer, and everything needed to judge it."""
    key: str
    label: str
    value: object = None
    tier: str = UNKNOWN
    unit: str = ""
    denominator: str = ""          # what the value is a fraction OF
    low: object = None             # ESTIMATE range
    high: object = None
    needs_identity: bool = False
    caveats: list = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    @property
    def display(self):
        if self.value is None:
            return "n/a"
        if self.tier == ESTIMATE and self.low is not None:
            return f"{self.value}{self.unit}   range {self.low}-{self.high}"
        return f"{self.value}{self.unit}"

    def unknown(self, why):
        self.tier, self.value = UNKNOWN, None
        self.caveats.append(why)
        return self


def _merge(intervals, gap=0.0):
    out = []
    for s, e in sorted(intervals):
        if out and s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


def _clip(intervals, windows):
    """Intersection — a metric may only count time we actually observed."""
    if not windows:
        return list(intervals)
    out = []
    for a, b in intervals:
        for w0, w1 in windows:
            lo, hi = max(a, w0), min(b, w1)
            if hi > lo:
                out.append((lo, hi))
    return _merge(out)


def _total(intervals):
    return sum(b - a for a, b in intervals)


def _presence(events, zones, roles=None, want_role=None):
    """Intervals during which ANYONE (optionally of a role) was in `zones`.

    Deliberately identity-free: it asks "was a body inside this polygon",
    which is the question, and never "which body", which is the hard part.
    """
    zs = set(zones or ())
    out = []
    for e in events or []:
        if e.get("zone") not in zs:
            continue
        if want_role and (roles or {}).get(e.get("track_id"),
                                           e.get("role")) != want_role:
            continue
        out.append((float(e["t_in"]), float(e.get("t_out", e["t_in"]))))
    return _merge(out)


# ── Q1 ─────────────────────────────────────────────────────────────────────
def desk_coverage(events, staff_zones, observed_windows, roles=None,
                  target=0.90):
    """Was the desk covered? EXACT — this is the metric that needs no identity.

    SUCCESS_CRITERIA calls this CRITICAL and >=90%, and says why it is
    achievable: "Zone dwell is the strongest signal we have — identity doesn't
    even need to be perfect." Only presence is required.
    """
    a = Answer("desk_covered_pct", "Was the desk covered?", unit="%",
               denominator="observed footage", tier=EXACT)
    obs = _total(observed_windows or [])
    if obs <= 0:
        return a.unknown("no observed footage — the denominator would be zero")
    covered = _clip(_presence(events, staff_zones, roles, "staff"),
                    observed_windows)
    pct = _total(covered) / obs * 100.0
    a.value = round(pct, 1)
    a.detail = {"covered_s": round(_total(covered), 1), "observed_s": round(obs, 1),
                "target_pct": target * 100, "meets_target": pct >= target * 100}
    a.caveats.append("counts the STATION, not any one person — a different "
                     "staff member at the desk is still covered")
    if not a.detail["meets_target"]:
        a.caveats.append(f"below the ratified {target*100:.0f}% bar")
    return a


def desk_gaps(events, staff_zones, observed_windows, roles=None,
              min_gap_s=60.0, waiting_zones=()):
    """When was it empty, for how long, and was anyone waiting?

    A percentage tells a manager how much; this tells them WHEN, which is the
    part they can act on.
    """
    a = Answer("desk_gaps", "When was the desk empty?", tier=EXACT,
               denominator="observed footage")
    if not observed_windows:
        return a.unknown("no observed footage")
    covered = _clip(_presence(events, staff_zones, roles, "staff"),
                    observed_windows)
    gaps = []
    for w0, w1 in observed_windows:
        cur = w0
        for c0, c1 in covered:
            if c1 <= w0 or c0 >= w1:
                continue
            if c0 - cur >= min_gap_s:
                gaps.append((cur, c0))
            cur = max(cur, c1)
        if w1 - cur >= min_gap_s:
            gaps.append((cur, w1))
    waiting = _presence(events, waiting_zones) if waiting_zones else []
    rows = []
    for g0, g1 in gaps:
        n_waiting = sum(1 for w0, w1 in waiting if w1 > g0 and w0 < g1)
        rows.append({"from_s": round(g0, 1), "to_s": round(g1, 1),
                     "minutes": round((g1 - g0) / 60.0, 1),
                     "guests_waiting": n_waiting})
    rows.sort(key=lambda r: -r["minutes"])
    a.value = len(rows)
    a.detail = {"gaps": rows,
                "longest_min": rows[0]["minutes"] if rows else 0.0}
    if any(r["guests_waiting"] for r in rows):
        a.caveats.append("at least one gap had guests waiting through it")
    return a


# ── Q2 ─────────────────────────────────────────────────────────────────────
def greet_latency(arrivals, contacts, tier=PROXY):
    """How long from arrival to a staff member being near them?

    PROXY, permanently, and SUCCESS_CRITERIA already says so: proximity is not
    conversation. The honest upgrade is a VLM on a 10-second clip, not a better
    tracker. Labelling this EXACT would be the same failure as guests_tonight.
    """
    a = Answer("greet_latency_s", "How fast were guests greeted?", unit=" s",
               tier=tier, denominator="guests with a known arrival time")
    if not arrivals:
        return a.unknown("no arrival times — greet latency needs a door event, "
                         "and a broken entry line produces none")
    lat = []
    ungreeted = []
    for tid, t_in in arrivals.items():
        cs = [c for c in (contacts or {}).get(tid, []) if c >= t_in]
        if cs:
            lat.append(min(cs) - t_in)
        else:
            ungreeted.append(tid)
    if not lat:
        a.value = None
        a.tier = UNKNOWN
        a.caveats.append("nobody was ever near a staff member — check the "
                         "staff zone before reading this as bad service")
    else:
        lat.sort()
        a.value = round(lat[len(lat) // 2], 1)
        a.detail = {"median_s": a.value, "slowest_s": round(lat[-1], 1),
                    "n_greeted": len(lat), "n_ungreeted": len(ungreeted),
                    "ungreeted_ids": ungreeted[:20]}
    a.caveats.append("proximity, not conversation — a service-touch signal, "
                     "never proof someone was spoken to")
    return a


# ── Q3 ─────────────────────────────────────────────────────────────────────
def guest_count(unique_ids, confidence=None, low_conf_bar=60, source="line"):
    """How many guests? ESTIMATE with a range, because identity had to hold.

    The range is not decoration. Measured separability is ~0.66, so a single
    integer asserts a precision the evidence cannot support. Splitting on the
    per-person confidence the pipeline already computes turns that into an
    honest interval instead of a hidden error bar.
    """
    a = Answer("guests", "How many guests?", tier=ESTIMATE,
               needs_identity=True, denominator="distinct identities")
    ids = list(unique_ids or [])
    if not ids:
        return a.unknown(f"no arrivals were detected by the {source} method — "
                         f"that is a broken sensor OR an empty venue, and this "
                         f"cannot tell you which")
    if confidence:
        high = [i for i in ids if confidence.get(i, 0) >= low_conf_bar]
        a.value = len(ids)
        a.low, a.high = len(high), len(ids)
        a.detail = {"high_confidence": len(high),
                    "low_confidence": len(ids) - len(high)}
    else:
        a.value = len(ids)
        a.low, a.high = len(ids), len(ids)
        a.caveats.append("no per-person confidence available, so the range is "
                         "a point — treat the width as unknown, not as zero")
    a.caveats.append(f"derived from the {source} method; identity must hold "
                     f"across the whole visit for this to be right")
    return a


def answer_set(*, events=None, staff_zones=(), waiting_zones=(),
               observed_windows=None, roles=None, arrivals=None, contacts=None,
               unique_ids=None, confidence=None, arrival_source="line",
               findings=()):
    """The full set, in the ratified priority order, ready for report_slim.

    Q1 first because it is CRITICAL and achievable; Q3 last because it is the
    one the evidence supports least. Ordering the report by confidence rather
    than by drama is itself a design decision.
    """
    obs = observed_windows or []
    out = [desk_coverage(events, staff_zones, obs, roles),
           desk_gaps(events, staff_zones, obs, roles, waiting_zones=waiting_zones),
           greet_latency(arrivals, contacts),
           guest_count(unique_ids, confidence, source=arrival_source)]
    # A run-level finding invalidates individual answers; say so on the answer
    # itself rather than only in a banner nobody re-reads.
    blockers = [m for lvl, m in findings if lvl == "ERROR"]
    for a in out:
        if blockers and a.needs_identity:
            a.caveats.extend(blockers)
    for a in out:
        if a.tier == UNKNOWN:
            _log.warning(f"{a.key}: {a.caveats[-1] if a.caveats else 'unknown'}")
    return out


def to_report_rows(answers):
    """-> the shape report_slim.summary_txt() expects."""
    rows = []
    for a in answers:
        extra = []
        if a.key == "desk_covered_pct" and a.detail:
            extra.append(("target", f"{a.detail.get('target_pct', 90):.0f}%",
                          "EXACT" if a.detail.get("meets_target") else "EXACT"))
        if a.key == "desk_gaps" and a.detail.get("gaps"):
            g = a.detail["gaps"][0]
            extra.append(("longest gap", f"{g['minutes']} min", "EXACT"))
            extra.append(("guests waiting in it", g["guests_waiting"], "EXACT"))
        if a.key == "greet_latency_s" and a.detail:
            extra.append(("never greeted", a.detail.get("n_ungreeted", 0), PROXY))
        if a.key == "guests" and a.detail:
            extra.append(("high confidence", a.detail.get("high_confidence"), WEAK))
        rows.append({"label": a.label, "value": a.display, "tier": a.tier,
                     "extra": extra})
    return rows
