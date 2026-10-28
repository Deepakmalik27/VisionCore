"""seams.py — a person who walks across a chunk boundary is still one person.

WHY THIS EXISTS
    The night is processed one chunk at a time. Each chunk starts its tracker
    from scratch, so anyone on screen at 17:59:59 gets a new identity at
    18:00:00. On a 12-hour night cut into 1-hour chunks that is eleven seams,
    and every person standing at the desk across one is counted twice.

    Worse for the metric that matters most: a receptionist on duty all evening
    is split into eleven "different people working the desk", which is exactly
    the WEAK number the report already warns about.

    run_camera() currently processes one chunk with no knowledge of the
    previous one, so this is a real hole in the codebase path.

THE PRINCIPLE
    A seam is the one moment where the geometry is unambiguous. Two chunks are
    contiguous in time, so a body alive at the end of one and born at the start
    of the next, IN THE SAME PLACE, is the same body — no appearance model
    required. That makes seam bridging far more reliable than ordinary re-id,
    and it should be done on physics first and appearance never.

WHAT IT REFUSES TO DO
    Bridge across a gap in the footage. If chunk N ends at 18:00:00 and chunk
    N+1 starts at 18:04:00, four minutes are missing and a person could have
    left and been replaced. Bridging there would invent continuity that was
    never observed — the same sin as counting unobserved time as empty.
"""
from __future__ import annotations

import math

from .log import get_logger

_log = get_logger("seams")

# A body cannot cross the seam and land far away: the two samples are one
# frame interval apart, not minutes.
MAX_SEAM_GAP_S = 2.0        # footage discontinuity above this = do not bridge
MAX_SEAM_DIST_FRAC = 0.05   # of the frame diagonal


def tails(frame_log, within_s=1.0):
    """Tracks still alive in the last `within_s` of a chunk. -> {tid: (t, x, y)}"""
    if not frame_log:
        return {}
    t_end = max(t for _, t, _ in frame_log)
    out = {}
    for _idx, t, boxes in frame_log:
        if t < t_end - within_s:
            continue
        for tid, x1, y1, x2, y2 in boxes:
            prev = out.get(tid)
            if prev is None or t > prev[0]:
                out[tid] = (t, (float(x1) + float(x2)) / 2.0, float(y2))
    return out


def heads(frame_log, within_s=1.0):
    """Tracks first seen in the first `within_s` of a chunk. -> {tid: (t, x, y)}"""
    if not frame_log:
        return {}
    t0 = min(t for _, t, _ in frame_log)
    out = {}
    for _idx, t, boxes in frame_log:
        if t > t0 + within_s:
            continue
        for tid, x1, y1, x2, y2 in boxes:
            prev = out.get(tid)
            if prev is None or t < prev[0]:
                out[tid] = (t, (float(x1) + float(x2)) / 2.0, float(y2))
    return out


def bridge(prev_tails, next_heads, frame_wh, *, prev_end_clock=None,
           next_start_clock=None, max_dist_frac=MAX_SEAM_DIST_FRAC,
           max_gap_s=MAX_SEAM_GAP_S):
    """Match bodies across a chunk boundary. -> (mapping, findings).

    mapping is {next_chunk_track_id: previous_chunk_track_id}, so the later
    chunk adopts the earlier identity and the person keeps one id all night.

    Matching is greedy nearest-first on the FOOT point, mutually exclusive: one
    tail bridges to at most one head. No appearance is consulted — at a seam
    the geometry is decisive and appearance can only add error.
    """
    findings = []
    if prev_end_clock is not None and next_start_clock is not None:
        gap = float(next_start_clock) - float(prev_end_clock)
        if gap > max_gap_s:
            findings.append(("WARN",
                             f"{gap:.1f}s of footage is missing between chunks — "
                             f"identities are NOT bridged across it. Someone "
                             f"could have left and been replaced unobserved."))
            return {}, findings
        if gap < 0:
            findings.append(("ERROR",
                             f"chunks overlap by {-gap:.1f}s — the same seconds "
                             f"appear twice and would be double counted"))
            return {}, findings

    diag = math.hypot(float(frame_wh[0]), float(frame_wh[1]))
    limit = diag * max_dist_frac
    cands = []
    for h_id, (_ht, hx, hy) in next_heads.items():
        for t_id, (_tt, tx, ty) in prev_tails.items():
            d = math.hypot(hx - tx, hy - ty)
            if d <= limit:
                cands.append((d, h_id, t_id))
    cands.sort()

    mapping, used_t, used_h = {}, set(), set()
    for d, h_id, t_id in cands:
        if h_id in used_h or t_id in used_t:
            continue
        mapping[h_id] = t_id
        used_h.add(h_id)
        used_t.add(t_id)

    unbridged_t = [t for t in prev_tails if t not in used_t]
    unbridged_h = [h for h in next_heads if h not in used_h]
    if mapping:
        _log.info(f"seam bridged {len(mapping)} identity(ies) across the "
                  f"boundary; {len(unbridged_t)} left, {len(unbridged_h)} arrived")
    if unbridged_t and unbridged_h:
        findings.append(("WARN",
                         f"{len(unbridged_t)} body(ies) vanished at the seam and "
                         f"{len(unbridged_h)} appeared, too far apart to match — "
                         f"either people really did swap, or the chunks are not "
                         f"contiguous"))
    return mapping, findings


def apply(mapping, events=None, crossings=None, frame_log=None):
    """Rewrite a chunk's ids so bridged bodies keep the earlier identity."""
    def _m(t):
        return mapping.get(t, t)
    ev = [dict(e, track_id=_m(e["track_id"])) for e in (events or [])]
    cr = [dict(c, track_id=_m(c["track_id"])) for c in (crossings or [])]
    fl = [(i, t, [(_m(tid), *rest) for tid, *rest in boxes])
          for i, t, boxes in (frame_log or [])]
    return ev, cr, fl
