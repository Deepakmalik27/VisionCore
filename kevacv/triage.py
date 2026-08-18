"""triage.py — PHASE 5a. Spend the compute where the people are.

THE PROBLEM THIS SOLVES
    Ten hours of 4K, ~3 GB per hour. Processing it end to end does not fit a
    Kaggle session, so the pipeline has been quietly analysing a 20-minute slice
    and reporting it as the chunk.

THE PRINCIPLE
    Cost must scale with EVENTS, not with wall-clock time. A reception is empty
    for most of a night. Every serious video-analytics system is built as a
    cascade — cheap filter first, expensive model only where the cheap filter
    says something is happening. NoScope, cloud-edge analytics work, and the
    DeepStream reference designs all share this shape.

    The payoff is not only speed. If 70% of the night is empty, triage lets you
    spend 3x MORE compute per occupied frame at the same total cost. Scale and
    accuracy stop being a trade-off and become the same fix.

THE HONESTY REQUIREMENT, WHICH IS THE HARD PART
    A cheap scan CAN miss someone. Sampling one frame every 6 s will not see a
    person who crosses in 3 s. So a segment planner that silently drops time is
    just a faster way to be wrong.

    Everything here therefore reports THREE kinds of time, never two:
        ANALYSED   full pipeline ran
        SKIPPED    scanned, verified empty, deliberately not analysed
        UNSEEN     never scanned at all
    and estimates the recall risk the skipping itself introduces, from the scan
    interval and how long a person is actually in shot. A skipped minute is a
    finding ("nothing happened"), not a gap — but only if it was really looked at.
"""
from __future__ import annotations


def plan_segments(scan, min_people=1, pad_s=20.0, merge_gap_s=45.0,
                  min_segment_s=10.0):
    """Choose the stretches worth the full pipeline.

    scan:        [(t_seconds, n_people), ...] from a cheap detector pass
    min_people:  a sample counts as "activity" at or above this
    pad_s:       extend each side. The scan is coarse, so someone is usually
                 already walking in before the first sample that sees them.
                 This is the main defence against clipping an arrival, and it
                 is deliberately generous: padding is cheap, a missed entry
                 is not.
    merge_gap_s: two active stretches closer than this become one. Stopping and
                 restarting the pipeline costs more than analysing the quiet
                 gap between them.
    min_segment_s: below this a segment is not worth the startup cost.

    -> (segments, stats). segments is [(t0, t1), ...] sorted and disjoint.
    """
    pts = sorted((float(t), int(n)) for t, n in scan)
    if not pts:
        return [], {"analysed_s": 0.0, "skipped_s": 0.0, "unseen_s": 0.0,
                    "scan_samples": 0, "reason": "no scan data"}

    t_first, t_last = pts[0][0], pts[-1][0]
    step = ((t_last - t_first) / (len(pts) - 1)) if len(pts) > 1 else 1.0

    active = [t for t, n in pts if n >= min_people]
    raw = [(t - pad_s, t + step + pad_s) for t in active]

    merged = []
    for a, b in raw:
        if merged and a - merged[-1][1] <= merge_gap_s:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])

    span_lo, span_hi = t_first, t_last + step
    segs = [(max(span_lo, a), min(span_hi, b)) for a, b in merged
            if min(span_hi, b) - max(span_lo, a) >= min_segment_s]

    analysed = sum(b - a for a, b in segs)
    scanned = span_hi - span_lo
    stats = {
        "analysed_s": round(analysed, 1),
        "skipped_s": round(scanned - analysed, 1),
        "unseen_s": 0.0,                      # set by coverage_report
        "scanned_s": round(scanned, 1),
        "segments": len(segs),
        "scan_samples": len(pts),
        "scan_step_s": round(step, 2),
        "active_samples": len(active),
        "saving_pct": round(100.0 * (1 - analysed / scanned), 1) if scanned else 0.0,
        "compute_multiplier": round(scanned / analysed, 2) if analysed else None,
    }
    return segs, stats


def miss_risk(scan_step_s, typical_visit_s=25.0):
    """How likely is the cheap scan to miss a person entirely?

    A visit of length V sampled every S seconds is seen unless it falls wholly
    between two samples, so the miss probability is roughly max(0, 1 - V/S).
    Crude, and deliberately so — the point is to make the risk VISIBLE and
    tie it to a number you can change, not to be exact.

    typical_visit_s is how long a person is in shot at all, not how long they
    stay in the venue. At a reception someone crossing to the desk is in frame
    for tens of seconds; a corridor camera would be far less.
    """
    if scan_step_s <= 0 or typical_visit_s <= 0:
        return {"miss_prob": 0.0, "note": "no scan"}
    p = max(0.0, 1.0 - typical_visit_s / scan_step_s) if scan_step_s > typical_visit_s else 0.0
    verdict = ("safe: every visit is sampled at least once"
               if scan_step_s <= typical_visit_s / 2 else
               "marginal: a short visit can fall between samples"
               if scan_step_s <= typical_visit_s else
               "UNSAFE: visits shorter than the scan interval are invisible")
    return {"miss_prob": round(p, 3), "scan_step_s": scan_step_s,
            "typical_visit_s": typical_visit_s, "verdict": verdict}


def coverage_report(segments, scan_span, chunk_span):
    """Account for every second of the chunk in one of three states.

    A pipeline that reports "nothing happened" for time it never looked at is
    the exact failure this whole project exists to remove, so ANALYSED,
    SKIPPED and UNSEEN are always separated.
    """
    c0, c1 = float(chunk_span[0]), float(chunk_span[1])
    total = max(0.0, c1 - c0)
    s0, s1 = (float(scan_span[0]), float(scan_span[1])) if scan_span else (c0, c0)
    scanned = max(0.0, min(s1, c1) - max(s0, c0))
    analysed = sum(max(0.0, min(b, c1) - max(a, c0)) for a, b in segments)
    skipped = max(0.0, scanned - analysed)
    unseen = max(0.0, total - scanned)
    return {
        "total_s": round(total, 1),
        "analysed_s": round(analysed, 1),
        "skipped_s": round(skipped, 1),
        "unseen_s": round(unseen, 1),
        "analysed_pct": round(100 * analysed / total, 1) if total else 0.0,
        "skipped_pct": round(100 * skipped / total, 1) if total else 0.0,
        "unseen_pct": round(100 * unseen / total, 1) if total else 0.0,
        "accounted": abs(analysed + skipped + unseen - total) < 1.0,
    }


def describe(stats, cover, risk):
    L = ["TRIAGE — cost follows events, not the clock"]
    L.append(f"  scanned {stats['scanned_s'] / 60:.1f} min at 1 frame / "
             f"{stats['scan_step_s']:.0f} s  ({stats['scan_samples']} samples, "
             f"{stats['active_samples']} with people)")
    L.append(f"  {stats['segments']} segment(s) worth the full pipeline "
             f"= {stats['analysed_s'] / 60:.1f} min")
    if stats.get("compute_multiplier"):
        L.append(f"  saving {stats['saving_pct']:.0f}% of the work -> you can afford "
                 f"{stats['compute_multiplier']:.1f}x more compute per analysed frame "
                 f"at the same total cost")
    L.append("  every second is accounted for:")
    L.append(f"     ANALYSED {cover['analysed_pct']:5.1f}%   full pipeline ran")
    L.append(f"     SKIPPED  {cover['skipped_pct']:5.1f}%   scanned, verified empty")
    L.append(f"     UNSEEN   {cover['unseen_pct']:5.1f}%   never looked at"
             + ("" if cover["unseen_pct"] < 0.05 else "   <-- report cannot speak for this"))
    if not cover["accounted"]:
        L.append("     !! the three do not sum to the chunk — do not trust this run")
    L.append(f"  scan miss risk: {risk['verdict']}")
    return "\n".join(L)
