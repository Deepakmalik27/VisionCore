"""The review queue — what a human should look at, and why.

WHY
---
kevacv/confidence.py grades events and arrival counts CONFIRMED / PROBABLE /
UNCERTAIN / REJECTED, and the scorecard prints the grade. Nothing consumed it.
A tier that changes no behaviour is decoration: the UNCERTAIN cases still went
into the same total as the confirmed ones, so "5 guests" and "5 guests, and two
of those are guesses" printed the same number.

This turns the tier into an action. Ambiguous things leave the count and enter a
queue, each with the timestamp, the person, the reason and where to look. The
report can then say

    5 confirmed, 2 need review

which is a claim that survives being checked -- the shape both accuracy.txt and
follow_up.txt argued for, and the honest answer to "is it 99%?": 99% of what we
confirm, with the rest isolated rather than averaged in.

WHAT LANDS HERE
---------------
    arrival disagreement   independent estimators differ beyond tolerance
    entry_missed           left without ever being seen to arrive
    exit_missed            a second IN while already inside; an OUT was lost
    rejected event         kept for audit, NOT for counting -- a rejection you
                           cannot inspect is indistinguishable from a bug

Sorted worst-first, capped, and every item carries `look_at` so the reviewer
does not have to search a ten-hour video for the moment in question.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

SEVERITY = {"arrival_disagreement": 0, "entry_missed": 1, "exit_missed": 1,
            "rejected_event": 2, "low_confidence_event": 3}


def _clip(t: Optional[float], pad: float = 5.0) -> Optional[Dict[str, float]]:
    if t is None:
        return None
    return {"from_s": max(0.0, float(t) - pad), "to_s": float(t) + pad}


def build_queue(*, arrival_confidence: Optional[Dict[str, Any]] = None,
                visits: Optional[Dict[str, Any]] = None,
                rejected_events: Optional[List[Dict[str, Any]]] = None,
                camera: str = "CAM", limit: int = 200) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []

    a = arrival_confidence or {}
    if a.get("tier") == "UNCERTAIN":
        items.append({
            "kind": "arrival_disagreement",
            "camera": camera,
            "why": a.get("why", ""),
            "sources": a.get("sources", {}),
            "look_at": None,
            "action": "watch the entrance for the whole window and count by "
                      "eye; the estimators cannot be reconciled automatically",
        })

    for v in (visits or {}).get("visits", []):
        st = v.get("status")
        if st not in ("entry_missed", "exit_missed"):
            continue
        t = v.get("t_out") if st == "entry_missed" else v.get("t_in")
        items.append({
            "kind": st,
            "camera": camera,
            "person": v.get("person"),
            "why": (v.get("notes") or [""])[0],
            "look_at": _clip(t),
            "action": ("check whether this person was already inside when the "
                       "chunk began — if so it is not a missed entry"
                       if st == "entry_missed" else
                       "check for a missed exit between these two arrivals"),
        })

    for e in (rejected_events or []):
        items.append({
            "kind": "rejected_event",
            "camera": camera,
            "person": e.get("track_id"),
            "why": e.get("why", "rejected"),
            "look_at": _clip(e.get("t")),
            "action": "confirm the rejection was right — a rejection nobody "
                      "can inspect is indistinguishable from a bug",
        })

    items.sort(key=lambda i: (SEVERITY.get(i["kind"], 9),
                             (i.get("look_at") or {}).get("from_s", 0.0)))
    truncated = max(0, len(items) - limit)
    return {"items": items[:limit],
            "n_items": len(items),
            "truncated": truncated,
            "by_kind": {k: sum(1 for i in items if i["kind"] == k)
                        for k in {i["kind"] for i in items}}}


def render(queue: Dict[str, Any], width: int = 78) -> str:
    if not queue or not queue.get("n_items"):
        return "  REVIEW QUEUE  empty — nothing was ambiguous enough to flag"
    L = ["-" * width,
         f"  REVIEW QUEUE  {queue['n_items']} item(s)  "
         + "  ".join(f"{k}={v}" for k, v in sorted(queue["by_kind"].items()))]
    for i in queue["items"][:12]:
        at = i.get("look_at")
        when = (f"{at['from_s']:.0f}-{at['to_s']:.0f}s" if at else "whole window")
        who = f" {i['person']}" if i.get("person") is not None else ""
        L.append(f"      [{i['kind']:<18s}]{who:<6s} {when:>14s}  {i['why'][:70]}")
    if queue["truncated"]:
        L.append(f"      ... and {queue['truncated']} more")
    return "\n".join(L)
