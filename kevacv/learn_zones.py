"""learn_zones.py — PHASE 12. Find the door in the data instead of drawing it.

WHY THIS EXISTS
    On the first real CAM.112 hour the entry line was drawn ~200px too short.
    People walked around both ends, the line fired ZERO times, and eight
    GM-facing numbers silently collapsed to 0 while 46 people waited. Nothing
    in the pipeline noticed, because hand-drawn geometry fails silently and
    totally.

    That is the structural problem with manual zone mapping, and it does not
    get better with care — it gets worse with scale. Nobody re-draws 400
    polygons across a fleet when a camera is nudged during cleaning.

THE PRINCIPLE
    An entrance is not a place someone drew a line. It is THE PLACE WHERE
    TRACKS ARE BORN AND WHERE THEY DIE. People appear at doors and disappear
    at doors; in the middle of a room they neither materialise nor vanish.

    So cluster the first and last position of every track. The door falls out
    of the data. This is old, well-proven work — Makris & Ellis, "Learning
    semantic scene models from observing activity" (2002) — and it is what
    makes a fleet deployment scale: propose zones automatically, have a human
    confirm once, instead of drawing every polygon by hand.

THE HONEST CEILING — read this before trusting a proposal
    A tracker that fragments creates FALSE births and deaths in the middle of
    the room. On the real run identity lifetime had a median of 38s, so this
    footage fragments a lot, and naive endpoint clustering would happily
    propose a "door" in the centre of the lobby.

    Three guards, none of them free:
      1. a track must LIVE long enough (min_life_s) to be evidence
      2. it must TRAVEL far enough (min_travel_frac of the frame diagonal) —
         a fragment that appears and dies on the spot is not a journey
      3. a real door has BOTH births and deaths; a fragmentation hotspot
         usually skews to one

    Even so: these are PROPOSALS for a human to confirm, never a silent
    replacement for the zones file. The function returns evidence with every
    proposal so it can be argued with.

    ponytail: grid histogram + connected components, no clustering library.
    Good enough to find a door; swap for DBSCAN if a venue needs finer shapes.
"""
from __future__ import annotations

import math
from collections import defaultdict

MIN_LIFE_S = 1.5
MIN_TRAVEL_FRAC = 0.06
CELL_FRAC = 0.07
MIN_ENDPOINTS = 4
EDGE_FRAC = 0.18          # within this fraction of a border = "at the frame edge"


def track_endpoints(frame_log, canon=None, frame_wh=None,
                    min_life_s=MIN_LIFE_S, min_travel_frac=MIN_TRAVEL_FRAC):
    """-> (kept, stats). One birth/death per track, for tracks that are evidence.

    Uses the FOOT point (bottom-centre), not the box centre: a door is a place
    on the floor, and the feet are where the person actually is.
    """
    canon = canon or {}
    per = defaultdict(list)
    for _idx, t, boxes in frame_log:
        for tid, x1, y1, x2, y2 in boxes:
            per[canon.get(tid, tid)].append(
                (float(t), (float(x1) + float(x2)) / 2.0, float(y2)))

    if frame_wh:
        fw, fh = float(frame_wh[0]), float(frame_wh[1])
    else:
        fw = max((p[1] for v in per.values() for p in v), default=1280.0)
        fh = max((p[2] for v in per.values() for p in v), default=720.0)
    diag = math.hypot(fw, fh)

    kept, n_short, n_still = [], 0, 0
    for tid, rows in per.items():
        rows.sort()
        life = rows[-1][0] - rows[0][0]
        if life < min_life_s:
            n_short += 1
            continue
        b, d = (rows[0][1], rows[0][2]), (rows[-1][1], rows[-1][2])
        travel = math.hypot(d[0] - b[0], d[1] - b[1])
        if travel < min_travel_frac * diag:
            n_still += 1
            continue
        kept.append({"track_id": tid, "birth": b, "death": d,
                     "life_s": round(life, 1), "travel_px": round(travel, 1)})
    return kept, {"tracks": len(per), "used": len(kept),
                  "dropped_short": n_short, "dropped_still": n_still,
                  "frame_wh": (fw, fh)}


def _clusters(points, frame_wh, cell_frac=CELL_FRAC, min_count=MIN_ENDPOINTS):
    """Grid histogram -> connected components -> padded bounding boxes."""
    if not points:
        return []
    fw, fh = frame_wh
    cell = max(8.0, cell_frac * max(fw, fh))
    grid = defaultdict(list)
    for p in points:
        grid[(int(p[0] // cell), int(p[1] // cell))].append(p)

    seen, out = set(), []
    for key in grid:
        if key in seen:
            continue
        stack, comp = [key], []
        seen.add(key)
        while stack:                                  # 8-neighbour flood fill
            cx, cy = stack.pop()
            comp.extend(grid[(cx, cy)])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nk = (cx + dx, cy + dy)
                    if nk in grid and nk not in seen:
                        seen.add(nk)
                        stack.append(nk)
        if len(comp) < min_count:
            continue
        xs = [p[0] for p in comp]
        ys = [p[1] for p in comp]
        pad = cell * 0.35
        out.append({"box": (max(0.0, min(xs) - pad), max(0.0, min(ys) - pad),
                            min(fw, max(xs) + pad), min(fh, max(ys) + pad)),
                    "n": len(comp),
                    "centre": (round(sum(xs) / len(xs)), round(sum(ys) / len(ys)))})
    out.sort(key=lambda c: -c["n"])
    return out


def _edge_dist_frac(box, frame_wh):
    fw, fh = frame_wh
    x1, y1, x2, y2 = box
    return min(x1 / fw, y1 / fh, (fw - x2) / fw, (fh - y2) / fh)


def learn_entry_zones(frame_log, canon=None, frame_wh=None, top=3, **kw):
    """Propose entrance polygons from where tracks are BORN and DIE.

    -> (proposals, stats). Each proposal carries its own evidence so a human
    can disagree with it; nothing here silently overwrites a zones file.
    """
    eps, stats = track_endpoints(frame_log, canon=canon, frame_wh=frame_wh, **{
        k: v for k, v in kw.items() if k in ("min_life_s", "min_travel_frac")})
    fwh = stats["frame_wh"]
    if not eps:
        return [], stats

    births = [e["birth"] for e in eps]
    deaths = [e["death"] for e in eps]
    cl = _clusters(births + deaths, fwh)

    props = []
    for c in cl:
        x1, y1, x2, y2 = c["box"]
        nb = sum(1 for p in births if x1 <= p[0] <= x2 and y1 <= p[1] <= y2)
        nd = sum(1 for p in deaths if x1 <= p[0] <= x2 and y1 <= p[1] <= y2)
        # A real door is used in BOTH directions. A fragmentation hotspot is
        # lopsided: tracks die there and are re-born as new ids, or vice versa.
        balance = min(nb, nd) / max(nb, nd) if max(nb, nd) else 0.0
        edge = _edge_dist_frac(c["box"], fwh)
        at_edge = edge <= EDGE_FRAC
        score = (nb + nd) * (0.5 + 0.5 * balance) * (1.25 if at_edge else 1.0)
        props.append({
            "polygon": [[round(x1), round(y1)], [round(x2), round(y1)],
                        [round(x2), round(y2)], [round(x1), round(y2)]],
            "box": c["box"], "centre": c["centre"], "births": nb, "deaths": nd,
            "balance": round(balance, 2), "at_frame_edge": at_edge,
            "score": round(score, 1),
            "why": (f"{nb} track(s) began and {nd} ended here "
                    f"(balance {balance:.2f}"
                    + (", at the frame edge" if at_edge else
                       ", NOT at the frame edge — could be a fragmentation "
                       "hotspot rather than a door")
                    + ")"),
        })
    props.sort(key=lambda p: -p["score"])
    return props[:top], stats


def learn_dwell_zones(frame_log, canon=None, frame_wh=None, top=3,
                      slow_frac=0.004, min_hits=30):
    """Propose zones where people STOP — waiting areas, counters, queues.

    Movement below slow_frac of the frame diagonal per second counts as
    stopped. Same grid machinery as the entrances; different evidence.
    """
    canon = canon or {}
    per = defaultdict(list)
    for _idx, t, boxes in frame_log:
        for tid, x1, y1, x2, y2 in boxes:
            per[canon.get(tid, tid)].append(
                (float(t), (float(x1) + float(x2)) / 2.0, float(y2)))
    if frame_wh:
        fw, fh = float(frame_wh[0]), float(frame_wh[1])
    else:
        fw = max((p[1] for v in per.values() for p in v), default=1280.0)
        fh = max((p[2] for v in per.values() for p in v), default=720.0)
    diag = math.hypot(fw, fh)

    slow = []
    for _tid, rows in per.items():
        rows.sort()
        for a, b in zip(rows, rows[1:]):
            dt = b[0] - a[0]
            if dt <= 0:
                continue
            if math.hypot(b[1] - a[1], b[2] - a[2]) / dt <= slow_frac * diag:
                slow.append((b[1], b[2]))

    out = []
    for c in _clusters(slow, (fw, fh), min_count=min_hits):
        x1, y1, x2, y2 = c["box"]
        out.append({
            "polygon": [[round(x1), round(y1)], [round(x2), round(y1)],
                        [round(x2), round(y2)], [round(x1), round(y2)]],
            "box": c["box"], "centre": c["centre"], "samples": c["n"],
            "why": f"{c['n']} sample(s) where somebody was standing still here",
        })
    return out[:top], {"slow_samples": len(slow), "frame_wh": (fw, fh)}


def to_zone_config(entries, dwells=(), frame_wh=None):
    """Turn proposals into a zones_*.json-shaped dict a human can edit.

    Names carry the keyword the role classifier already understands, so a
    proposal that is accepted needs no further wiring.
    """
    polys, roles = {}, {}
    for i, p in enumerate(entries, 1):
        n = "main_entrance" if i == 1 else f"entrance_{i}"
        polys[n] = p["polygon"]
        roles[n] = ["entry"]
    for i, p in enumerate(dwells, 1):
        n = "waiting_area" if i == 1 else f"waiting_area_{i}"
        polys[n] = p["polygon"]
        roles[n] = ["wait"]
    cfg = {"polygons": polys, "roles": roles,
           "_generated": "kevacv.learn_zones — PROPOSALS, confirm before use"}
    if entries:
        # A line across the widest axis of the busiest entrance, then EXTENDED
        # well past the observed cluster.
        #
        # This overshoot is the whole point. The cluster only covers where
        # people were actually SEEN crossing; the physical doorway is always at
        # least as wide, usually wider. A line drawn to the cluster is a line
        # people can walk around the ends of — which is precisely the failure
        # that produced "0 people came through the door" for a full hour.
        # Overshooting costs nothing (a line beyond the doorway is never
        # crossed); undershooting costs every arrival.
        x1, y1, x2, y2 = entries[0]["box"]
        fw, fh = (frame_wh if frame_wh else (max(x2 * 1.2, 1), max(y2 * 1.2, 1)))
        grow = 0.6
        if (x2 - x1) >= (y2 - y1):
            pad = (x2 - x1) * grow
            cy = round((y1 + y2) / 2)
            cfg["entry_line"] = [[round(max(0, x1 - pad)), cy],
                                 [round(min(fw, x2 + pad)), cy]]
        else:
            pad = (y2 - y1) * grow
            cx = round((x1 + x2) / 2)
            cfg["entry_line"] = [[cx, round(max(0, y1 - pad))],
                                 [cx, round(min(fh, y2 + pad))]]
        cfg["_entry_line_note"] = (
            "extended 60% past the observed crossings on each side — a line "
            "that stops inside the doorway lets people walk around its ends, "
            "which is the failure this module exists to prevent")
    if frame_wh:
        cfg["frame_size"] = [int(frame_wh[0]), int(frame_wh[1])]
    return cfg


def describe(entries, dwells=(), stats=None):
    L = ["LEARNED ZONES — proposed from the tracks themselves, not drawn"]
    if stats:
        L.append(f"  evidence: {stats['used']} of {stats['tracks']} tracks used "
                 f"({stats['dropped_short']} too short-lived, "
                 f"{stats['dropped_still']} never travelled)")
        if stats["tracks"] and stats["used"] / stats["tracks"] < 0.25:
            L.append("  !! fewer than a quarter of tracks were usable — this "
                     "footage fragments badly, so treat these as weak hints")
    if not entries:
        L.append("  no entrance proposed — not enough tracks that both lived "
                 "and travelled. This says the TRACKING is too fragmented to "
                 "learn from, which is itself the finding.")
    for i, p in enumerate(entries, 1):
        L.append(f"  ENTRANCE {i} at {p['centre']}  score {p['score']}")
        L.append(f"    {p['why']}")
    for i, p in enumerate(dwells, 1):
        L.append(f"  DWELL {i} at {p['centre']}: {p['why']}")
    L.append("  These are PROPOSALS. Confirm against one frame before use — a "
             "learned zone is evidence, not authority.")
    return "\n".join(L)
