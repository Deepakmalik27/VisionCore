"""Event confidence tiers — CONFIRMED / PROBABLE / UNCERTAIN / REJECTED.

WHY
---
This pipeline currently emits one number and asks you to believe it. When two
independent estimators disagree it picks one (`trust=line` or `trust=region`)
and the disagreement disappears from the answer. Measured on CAM.112:

    p0classfix2   line=1  region=4   -> reported 4
    p0v4          line=5  region=5   -> reported 5
    h_v1          line=3  region=7   -> reported 7

The first and third are not the same kind of answer as the second, and nothing
downstream can tell. A tier says so: agreement is CONFIRMED, a single source is
PROBABLE, disagreement is UNCERTAIN and belongs in a review queue rather than in
a headline.

DESIGN
------
HARD VETOES ARE NOT WEIGHTS. A physically impossible association is rejected
outright, never averaged against a high appearance score -- follow_up.txt makes
this point and it is right: "Re-ID = 0.96 BUT impossible topology -> reject."
So this is a small decision table, not a weighted sum. A weighted sum lets a
confident detector outvote physics.

ADDITIVE. Tiers are reported ALONGSIDE the existing count and change nothing
about it, so this cannot regress a number. What it can do is tell you which
numbers were worth believing.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

CONFIRMED = "CONFIRMED"
PROBABLE = "PROBABLE"
UNCERTAIN = "UNCERTAIN"
REJECTED = "REJECTED"

ORDER = [CONFIRMED, PROBABLE, UNCERTAIN, REJECTED]


def arrival_tier(line_n: Optional[int], region_n: Optional[int],
                 slit_n: Optional[int] = None,
                 plane_ok: bool = True,
                 agree_frac: float = 0.25) -> Dict[str, Any]:
    """Tier for a WINDOW's arrival count, from how far the estimators agree.

    agree_frac: relative gap two estimators may show and still count as
    agreeing. 0.25 means 4 vs 5 agrees, 3 vs 7 does not.
    """
    have = [(n, s) for n, s in ((line_n, "line"), (region_n, "region"),
                                (slit_n, "slit")) if n is not None]
    if not have:
        return {"tier": UNCERTAIN, "why": "no arrival estimator produced a number",
                "sources": {}}
    src = {s: n for n, s in have}
    if len(have) == 1:
        return {"tier": PROBABLE,
                "why": f"only one estimator available ({have[0][1]}); nothing "
                       f"independent corroborates it",
                "sources": src}
    lo = min(n for n, _ in have)
    hi = max(n for n, _ in have)
    spread = (hi - lo) / float(hi) if hi else 0.0
    if hi == 0 and lo == 0:
        return {"tier": CONFIRMED, "why": "every estimator says nobody arrived",
                "sources": src}
    if spread <= agree_frac:
        tier = CONFIRMED if plane_ok else PROBABLE
        why = (f"{len(have)} independent estimators agree within "
               f"{spread*100:.0f}%")
        if not plane_ok:
            why += " but the ground plane failed its own check, so any "\
                   "metre-based gate that fed them is unreliable"
        return {"tier": tier, "why": why, "sources": src}
    return {"tier": UNCERTAIN,
            "why": f"estimators disagree by {spread*100:.0f}% "
                   f"({', '.join(f'{s}={n}' for n, s in have)}) — this belongs "
                   f"in a review queue, not in a headline count",
            "sources": src}


def event_tier(*, crossed_line: bool, direction_known: bool,
               track_seconds: float, confirmed_not_uturn: bool,
               plane_ok: bool = True, vetoes: Optional[List[str]] = None,
               min_track_s: float = 1.0) -> Dict[str, Any]:
    """Tier for ONE crossing event. Vetoes are absolute."""
    if vetoes:
        return {"tier": REJECTED, "why": "; ".join(vetoes)}
    if not crossed_line:
        return {"tier": UNCERTAIN,
                "why": "no line crossing — presence is evidence, not a transition"}
    if not confirmed_not_uturn:
        return {"tier": REJECTED,
                "why": "crossed and came back inside the confirmation window — "
                       "nobody arrived or left"}
    weak = []
    if not direction_known:
        weak.append("direction could not be established")
    if track_seconds < min_track_s:
        weak.append(f"track lived only {track_seconds:.1f}s "
                    f"(under {min_track_s:.1f}s)")
    if not plane_ok:
        weak.append("ground plane failed its own scale check")
    if not weak:
        return {"tier": CONFIRMED,
                "why": f"line crossed, direction established, track stable "
                       f"{track_seconds:.1f}s, survived U-turn confirmation"}
    if len(weak) == 1:
        return {"tier": PROBABLE, "why": weak[0]}
    return {"tier": UNCERTAIN, "why": "; ".join(weak)}


def summarise(tiers: List[str]) -> Dict[str, Any]:
    """Counts per tier plus the headline sentence a report should print."""
    counts = {t: sum(1 for x in tiers if x == t) for t in ORDER}
    n_official = counts[CONFIRMED]
    n_review = counts[UNCERTAIN] + counts[PROBABLE]
    return {"counts": counts,
            "official": n_official,
            "needs_review": n_review,
            "headline": (f"{n_official} confirmed"
                         + (f", {n_review} need review" if n_review else "")
                         + (f", {counts[REJECTED]} rejected"
                            if counts[REJECTED] else ""))}
