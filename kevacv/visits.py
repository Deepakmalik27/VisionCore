"""Person -> Visit -> Event. The IN/OUT state machine, and the shape of a count.

WHY
---
The operator's own counting rule, stated verbatim:

    "we just have to see that person has entered right count it if that person
     go out cout out as 1 right ?? not like u will minus it ?? i want count
     that person if that person comes agani back then that person is not
     counted"

i.e. one person who leaves and returns is ONE person and TWO visits. The
pipeline has never been able to express that: `guests` is a count of distinct
identities and there is no visit anywhere, so "unique people" and "visits" are
the same field and re-entry is invisible.

THE STATE MACHINE
-----------------
    OUTSIDE --IN--> INSIDE --OUT--> OUTSIDE

Every real sequence violates it somewhere, and each violation means something
different. Naming them is the point -- a machine that only accepts clean input
just moves the problem into whoever reads the number.

    IN with no OUT      still inside when observation ended. NOT an error;
                        it is the normal state of anyone in the room at close.
    OUT with no IN      already inside before observation began. Also normal
                        at the START of a chunk, and the reason a region
                        estimator over-counts: it sees presence, not arrival.
    IN, IN              an OUT was missed. Close the first visit as
                        `exit_missed` rather than dropping it, so the loss is
                        visible instead of silently halving the visit count.
    OUT, OUT            an IN was missed. Same treatment.

Counting rules that follow, and they are NOT the same number:

    unique_people  distinct identities seen                    (never subtract)
    visits         completed or open IN->OUT episodes
    entries        IN events                                   (>= visits)
    exits          OUT events

Pure and dependency-free on purpose: this is the piece a database schema and a
report both need, and it must be testable without a video.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

OUTSIDE, INSIDE = "OUTSIDE", "INSIDE"


@dataclass
class Visit:
    person: Any
    t_in: Optional[float] = None
    t_out: Optional[float] = None
    status: str = "open"          # open | closed | entry_missed | exit_missed
    notes: List[str] = field(default_factory=list)

    @property
    def duration_s(self) -> Optional[float]:
        if self.t_in is None or self.t_out is None:
            return None
        return self.t_out - self.t_in

    def as_dict(self) -> Dict[str, Any]:
        return {"person": self.person, "t_in": self.t_in, "t_out": self.t_out,
                "status": self.status, "duration_s": self.duration_s,
                "notes": list(self.notes)}


def visits_for_person(person: Any, events: Iterable[Dict[str, Any]]) -> List[Visit]:
    """events: [{"t": float, "direction": "in"|"out"}, ...] for ONE person."""
    evs = sorted(events, key=lambda e: float(e["t"]))
    out: List[Visit] = []
    state, cur = OUTSIDE, None
    for e in evs:
        t = float(e["t"])
        d = str(e.get("direction", "")).lower()
        if d == "in":
            if state == INSIDE and cur is not None:
                cur.status = "exit_missed"
                cur.notes.append("a second IN arrived while already inside — "
                                 "an OUT was missed, not a person removed")
                out.append(cur)
            cur = Visit(person=person, t_in=t)
            state = INSIDE
        elif d == "out":
            if state == OUTSIDE or cur is None:
                v = Visit(person=person, t_out=t, status="entry_missed")
                v.notes.append("left without having been seen to arrive — "
                               "already inside before observation began, or "
                               "the entry was missed")
                out.append(v)
                continue
            cur.t_out = t
            cur.status = "closed"
            out.append(cur)
            cur, state = None, OUTSIDE
    if cur is not None:
        cur.notes.append("still inside when observation ended")
        out.append(cur)
    return out


def build_visits(crossings: Iterable[Dict[str, Any]],
                 line_name: Optional[str] = None,
                 person_key: str = "track_id") -> Dict[str, Any]:
    """All visits from a crossings list, plus the counts that follow.

    line_name: restrict to one door (the venue entrance). A dining threshold is
    an interior movement, not an arrival, and mixing them is how a reception
    ends up reporting corridor traffic as guests.
    """
    per: Dict[Any, List[Dict[str, Any]]] = {}
    for c in crossings:
        if line_name is not None and c.get("line") != line_name:
            continue
        per.setdefault(c.get(person_key), []).append(c)

    all_visits: List[Visit] = []
    for pid, evs in per.items():
        all_visits.extend(visits_for_person(pid, evs))

    entries = sum(1 for v in all_visits if v.t_in is not None)
    exits = sum(1 for v in all_visits if v.t_out is not None)
    closed = [v for v in all_visits if v.status == "closed"]
    return {
        "visits": [v.as_dict() for v in all_visits],
        "unique_people": len(per),
        "n_visits": len(all_visits),
        "entries": entries,
        "exits": exits,
        "closed": len(closed),
        "open": sum(1 for v in all_visits if v.status == "open"),
        "entry_missed": sum(1 for v in all_visits if v.status == "entry_missed"),
        "exit_missed": sum(1 for v in all_visits if v.status == "exit_missed"),
        "repeat_visitors": sum(
            1 for pid in per
            if len([v for v in all_visits if v.person == pid]) > 1),
        "median_visit_s": _median([v.duration_s for v in closed
                                   if v.duration_s is not None]),
    }


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0
