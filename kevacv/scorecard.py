"""RUN SCORECARD — the evidence for a run, emitted BY the run.

WHY THIS EXISTS
---------------
Until now the evidence for "did this change help?" lived in three tools an
operator (or an assistant) ran by hand afterwards: the funnel in the log,
tools/track_health.py, tools/score_line_entries.py. That makes the person
running the tools the instrument, and their summary the thing you have to
trust. It is also how three constants got shipped on a story and reverted on
measurement.

So the run states its own case. One block, every number needed to accept or
reject the run, plus the same content as JSON next to the outputs so two runs
can be diffed mechanically instead of by reading prose.

DESIGN RULES
  * Never invent a verdict. A check with no ground truth reports NO-TRUTH, not
    PASS. "0 and 'cannot tell' must never look the same" is already this
    package's stated rule for arrivals; it applies to scoring too.
  * Held-out and tuned windows are reported SEPARATELY and never summed. A
    score on tuned data is not a score.
  * Every number carries where it came from, so a wrong one can be traced
    rather than argued about.
"""
from __future__ import annotations

import collections
import glob
import json
import os
from typing import Any, Dict, List, Optional

PASS, FAIL, NOTRUTH, INFO = "PASS", "FAIL", "NO-TRUTH", "INFO"

# Windows used to tune anything are never counted as evidence for it.
TUNED_WINDOWS = {"gt_entries_305_318.json"}


def _q(xs, f):
    xs = sorted(xs)
    return xs[int(f * (len(xs) - 1))] if xs else float("nan")


def _in_poly(poly, x, y):
    c, n = False, len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and \
                x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1:
            c = not c
    return c


def track_stats(tracks: Dict[Any, List], frame_w: float, frame_h: float,
                zone_x: Optional[float] = None, edge_frac: float = 0.06,
                t_first: Optional[float] = None, t_last: Optional[float] = None,
                zones: Optional[Dict[str, List]] = None) -> Dict[str, Any]:
    """Where tracks are BORN and DIE, and how much of it is unexplained.

    A person leaves the camera's view in three legitimate ways: off a FRAME
    EDGE, through a DOORWAY that sits mid-frame, or behind FURNITURE. Only the
    fourth -- vanishing in open floor, in plain view -- is the tracker failing.

    An earlier version of this counted every non-frame-edge birth as
    fragmentation and reported 61% born / 66% died mid-scene, verdict FAIL.
    Breaking it down by zone showed what that number really contained
    (measured, p0v4):

        births: dining 37%  reception 23%  OPEN FLOOR 20%  main_entrance 9%
        deaths: dining 53%  reception 19%  OPEN FLOOR 17%  main_entrance 3%

    'dining' is the doorway to the next room and 'reception' is the desk people
    walk behind -- both are people leaving, not tracks breaking. The real
    unexplained loss is the OPEN FLOOR share: ~11% and ~9%. Reporting the
    aggregate would have sent a week of tracking work after a number that was
    mostly a doorway.

    `zones` maps name -> polygon in the SAME pixel space as the tracks. Without
    it the breakdown is unavailable and only the aggregate is reported, clearly
    labelled as such.
    """
    ex, ey = edge_frac * frame_w, edge_frac * frame_h

    def at_edge(x, y):
        return x <= ex or x >= frame_w - ex or y <= ey or y >= frame_h - ey

    def where(x, y):
        for nm, poly in (zones or {}).items():
            if _in_poly(poly, x, y):
                return nm
        return "OPEN FLOOR"

    born_mid = died_mid = transit = 0
    born_by, died_by = collections.Counter(), collections.Counter()
    for _tid, p in tracks.items():
        pts = [(((a + c) / 2.0), d) for _t, a, _b, c, d in p]
        if not pts:
            continue
        b_clip = t_first is not None and abs(p[0][0] - t_first) < 1e-6
        d_clip = t_last is not None and abs(p[-1][0] - t_last) < 1e-6
        if not at_edge(*pts[0]) and not b_clip:
            born_mid += 1
            born_by[where(*pts[0])] += 1
        if not at_edge(*pts[-1]) and not d_clip:
            died_mid += 1
            died_by[where(*pts[-1])] += 1
        if zone_x is not None:
            xs = [q[0] for q in pts]
            if any((xs[i] - zone_x) * (xs[i + 1] - zone_x) < 0
                   for i in range(len(xs) - 1)):
                transit += 1
    n = max(len(tracks), 1)
    out = {"tracks": len(tracks),
           "born_mid_scene": born_mid, "born_mid_pct": 100.0 * born_mid / n,
           "died_mid_scene": died_mid, "died_mid_pct": 100.0 * died_mid / n,
           "traversed_zone_x": transit,
           "have_zone_breakdown": bool(zones)}
    if zones:
        out["born_by_zone"] = dict(born_by)
        out["died_by_zone"] = dict(died_by)
        out["born_open_floor"] = born_by.get("OPEN FLOOR", 0)
        out["died_open_floor"] = died_by.get("OPEN FLOOR", 0)
        out["born_open_pct"] = 100.0 * born_by.get("OPEN FLOOR", 0) / n
        out["died_open_pct"] = 100.0 * died_by.get("OPEN FLOOR", 0) / n
    return out


def track_quality(tracks: Dict[Any, List], fps: float = 8.0,
                  link_gap_s: float = 3.0, link_px: float = 160.0,
                  recovery_gap_frames: float = 1.5) -> Dict[str, Any]:
    """Fragmentation, recovery and ID-switch pressure — WITHOUT ground truth.

    P3 could not be graded. "Did that tracking change help?" had no answer:
    the funnel counts detections, not identities, and track_stats says where
    tracks start and stop but not whether the SAME PERSON kept one id.

    Three things are measurable with no labels at all:

    FRAGMENTS PER CHAIN
        Link a track's end to another track's start when they are close in
        time AND space -- one person the tracker dropped and re-acquired.
        Chains of 1 are clean; a chain of 4 is one person wearing four ids.
        This is the number a tracking fix has to move.

    RECOVERIES
        A single track id whose own detections contain a gap means the lost
        buffer and Re-ID reactivation DID fire and bridged it. BoT-SORT
        already does predict -> search -> recover; nothing in this pipeline
        ever reported whether it works, so the ladder looked "missing" when it
        was merely invisible.

    SWAP PRESSURE
        Two tracks alive at the same instant, closer than link_px, are a
        moment where an id switch is possible. Not proof one happened -- that
        needs labels -- but it bounds where they can be, and it moves when
        association changes.

    link_gap_s / link_px are deliberately generous: this measures fragmentation
    PRESSURE, and being too eager to link only makes the reported chains
    LONGER, i.e. it cannot flatter a bad tracker.
    """
    ends, starts, spans = [], [], {}
    for tid, p in tracks.items():
        if not p:
            continue
        pts = [(t, (a + c) / 2.0, d) for t, a, _b, c, d in p]
        starts.append((pts[0][0], pts[0][1], pts[0][2], tid))
        ends.append((pts[-1][0], pts[-1][1], pts[-1][2], tid))
        spans[tid] = (pts[0][0], pts[-1][0], pts)

    # ── recoveries: internal time gaps inside ONE id
    recovered = gaps = 0
    step = 1.0 / max(fps, 1e-6)
    for tid, (_t0, _t1, pts) in spans.items():
        hit = False
        for i in range(len(pts) - 1):
            if pts[i + 1][0] - pts[i][0] > recovery_gap_frames * step:
                gaps += 1
                hit = True
        if hit:
            recovered += 1

    # ── chains: end of A -> start of B, close in time and space
    parent = {tid: tid for tid in spans}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    links = 0
    for (te, xe, ye, a) in ends:
        best, bestd = None, None
        for (ts, xs, ys, b) in starts:
            if b == a or ts < te or ts - te > link_gap_s:
                continue
            d = ((xs - xe) ** 2 + (ys - ye) ** 2) ** 0.5
            if d <= link_px and (bestd is None or d < bestd):
                best, bestd = b, d
        if best is not None and find(best) != find(a):
            parent[find(best)] = find(a)
            links += 1

    chains: Dict[Any, int] = {}
    for tid in spans:
        chains[find(tid)] = chains.get(find(tid), 0) + 1
    sizes = sorted(chains.values(), reverse=True)

    # ── swap pressure: co-alive and close
    ids = list(spans)
    pressure = 0
    for i in range(len(ids)):
        a0, a1, _ = spans[ids[i]]
        for j in range(i + 1, len(ids)):
            b0, b1, _ = spans[ids[j]]
            if min(a1, b1) - max(a0, b0) <= 0:
                continue
            pa = spans[ids[i]][2]
            pb = spans[ids[j]][2]
            k = 0
            for (t, x, y) in pa:
                if not (b0 <= t <= b1):
                    continue
                q = min(pb, key=lambda z: abs(z[0] - t))
                if abs(q[0] - t) < step and \
                        ((q[1] - x) ** 2 + (q[2] - y) ** 2) ** 0.5 <= link_px:
                    k += 1
            if k:
                pressure += 1
    n = max(len(spans), 1)
    return {"track_ids": len(spans),
            "chains": len(chains),
            "fragments_per_chain": len(spans) / float(max(len(chains), 1)),
            "worst_chain": sizes[0] if sizes else 0,
            "links_made": links,
            "ids_with_recovery": recovered,
            "recovery_gaps": gaps,
            "recovery_pct": 100.0 * recovered / n,
            "swap_pressure_pairs": pressure}


def pixel_height(tracks: Dict[Any, List], analysis_w: float, source_w: float,
                 imgsz: float, min_px: float = 110.0) -> Dict[str, Any]:
    """A person's height where the DETECTOR sees it, not where a human does."""
    hs = [d - b for p in tracks.values() for _t, _a, b, _c, d in p]
    if not hs:
        return {"n": 0}
    to_analysis = analysis_w / float(source_w)
    to_network = min(1.0, imgsz / float(analysis_w))
    total = to_analysis * to_network
    net = [h * to_network for h in hs]
    return {"n": len(hs),
            "scale_source_to_analysis": to_analysis,
            "scale_analysis_to_network": to_network,
            "scale_total": total,
            "analysis_px_median": _q(hs, .5),
            "source_px_median": _q(hs, .5) / to_analysis,
            "network_px_p10": _q(net, .1),
            "network_px_median": _q(net, .5),
            "network_px_p90": _q(net, .9),
            "below_min_px": sum(1 for h in net if h < min_px),
            "below_min_px_pct": 100.0 * sum(1 for h in net if h < min_px) / len(net)}


def score_windows(crossings: List[Dict], line_name: str = "entry line",
                  eval_dir: str = "eval", slack_s: float = 3.0,
                  video: Optional[str] = None) -> Dict[str, Any]:
    """Entry-line IN events vs hand-read truth windows, held-out kept apart.

    `video` BINDS the score to the footage the labels were read from. Without
    it the scorecard happily scored a 7.30pm run against the 6.30pm windows and
    reported held-out 1/3 -- a number about video the labeller never saw. A
    window whose source_chunk does not appear in the run's video path is
    skipped and reported as skipped, never silently counted.
    """
    ins = sorted(float(c["t"]) for c in crossings
                 if c.get("line") == line_name
                 and str(c.get("direction", "")).lower() == "in")
    rows, held_t, held_g, tuned_t, tuned_g = [], 0, 0, 0, 0
    skipped = []
    for path in sorted(glob.glob(os.path.join(eval_dir, "gt_entries_*.json"))):
        try:
            gt = json.load(open(path))
        except (OSError, ValueError):
            continue
        chunk = gt.get("source_chunk")
        if video and chunk and chunk not in str(video):
            skipped.append({"window": os.path.basename(path),
                            "source_chunk": chunk,
                            "why": "labels are from a different chunk"})
            continue
        t0, t1 = gt["window_s"]
        got = sum(1 for t in ins if t0 - slack_s <= t <= t1 + slack_s)
        truth = gt.get("truth_count", len(gt.get("entries", [])))
        tuned = os.path.basename(path) in TUNED_WINDOWS
        rows.append({"window": os.path.basename(path), "kind": gt.get("kind", "-"),
                     "truth": truth, "got": got, "tuned": tuned})
        if tuned:
            tuned_t += truth; tuned_g += got
        else:
            held_t += truth; held_g += got
    return {"line_in_events": len(ins), "windows": rows, "skipped": skipped,
            "heldout_truth": held_t, "heldout_got": held_g,
            "tuned_truth": tuned_t, "tuned_got": tuned_g,
            "has_truth": bool(rows)}


def verdicts(card: Dict[str, Any]) -> List[Dict[str, str]]:
    """Named checks. NO-TRUTH is a first-class outcome, never a silent PASS."""
    out = []

    def add(name, state, detail):
        out.append({"check": name, "state": state, "detail": detail})

    f = card.get("funnel") or {}
    worst = max(f.items(), key=lambda kv: kv[1], default=(None, 0))
    if worst[0]:
        add("detector funnel", FAIL if worst[1] > 25 else PASS,
            f"largest single stage drop {worst[0]} {worst[1]:.1f}%")

    t = card.get("tracks") or {}
    if t.get("tracks"):
        if t.get("have_zone_breakdown"):
            bad = max(t["born_open_pct"], t["died_open_pct"])
            add("track fragmentation", FAIL if bad > 15 else PASS,
                f"unexplained (open floor): born {t['born_open_pct']:.0f}%, "
                f"died {t['died_open_pct']:.0f}%  |  total mid-scene "
                f"{t['born_mid_pct']:.0f}%/{t['died_mid_pct']:.0f}% incl. "
                f"doorways and furniture")
        else:
            add("track fragmentation", INFO,
                f"born mid-scene {t['born_mid_pct']:.0f}%, died "
                f"{t['died_mid_pct']:.0f}% — NO zone breakdown, so this "
                f"includes doorways and furniture and is NOT a failure rate")

    q = card.get("track_quality") or {}
    if q.get("track_ids"):
        fpc = q["fragments_per_chain"]
        add("track continuity", FAIL if fpc > 1.6 else PASS,
            f"{fpc:.2f} ids per person ({q['track_ids']} ids -> "
            f"{q['chains']} chains, worst {q['worst_chain']}); "
            f"{q['ids_with_recovery']} id(s) recovered across a gap")

    g = card.get("ground_plane") or {}
    if g:
        ok = g.get("ok") and 1.5 <= (g.get("camera_h_m") or 0) <= 4.0 \
            and (g.get("horizon_row") or -1) >= 0
        add("ground plane", PASS if ok else FAIL,
            f"camera {g.get('camera_h_m')}m, horizon row {g.get('horizon_row')}")

    a = card.get("arrival_confidence") or {}
    if a:
        from .confidence import CONFIRMED as _C, UNCERTAIN as _U
        add("arrival confidence",
            PASS if a["tier"] == _C else (FAIL if a["tier"] == _U else INFO),
            f"{a['tier']}: {a['why']}")

    s = card.get("entry_score") or {}
    if not s.get("has_truth"):
        sk = s.get("skipped") or []
        add("entry line vs truth", NOTRUTH,
            (f"{len(sk)} window(s) skipped — labels are from a different chunk"
             if sk else "no eval/gt_entries_*.json found"))
    elif s["heldout_truth"] == 0:
        add("entry line vs truth", NOTRUTH,
            "no HELD-OUT window contains a labelled entry")
    else:
        r = s["heldout_got"] / float(s["heldout_truth"])
        add("entry line vs truth", PASS if r >= 0.75 else FAIL,
            f"held-out {s['heldout_got']}/{s['heldout_truth']} "
            f"({r*100:.0f}%); tuned {s['tuned_got']}/{s['tuned_truth']}")
    return out


def render(card: Dict[str, Any], width: int = 78) -> str:
    L, bar = [], "=" * width
    L.append(bar); L.append("  RUN SCORECARD".ljust(width)); L.append(bar)
    b = card.get("build", {})
    pv = card.get("provenance") or {}
    L.append(f"  build {b.get('build_id','?')}   config {b.get('config','?')}"
             + (f"   fingerprint {pv['fingerprint']}" if pv.get("fingerprint")
                else ""))
    L.append(f"  video {str(b.get('video',''))[-52:]}   {b.get('seconds','?')}s")
    if b.get("changed"):
        L.append(f"  knobs changed from default: {len(b['changed'])}")
        for k, v in list(b["changed"].items())[:40]:
            L.append(f"      {k:32s} {v}")

    f = card.get("funnel") or {}
    if f:
        L.append("-" * width); L.append("  DETECTOR FUNNEL   (drop % of raw)")
        for k, v in f.items():
            flag = "  <-- dominant" if v > 25 else ""
            L.append(f"      {k:32s} {v:6.1f}%{flag}")

    p = card.get("pixel_height") or {}
    if p.get("n"):
        L.append("-" * width)
        L.append(f"  PERSON HEIGHT     source {p['source_px_median']:.0f}px"
                 f" -> network {p['network_px_median']:.0f}px"
                 f"  (x{p['scale_total']:.2f})")
        L.append(f"      below {110}px at the network: "
                 f"{p['below_min_px']} of {p['n']} = {p['below_min_px_pct']:.0f}%")

    t = card.get("tracks") or {}
    if t.get("tracks"):
        L.append("-" * width)
        L.append(f"  TRACKS {t['tracks']}   born mid-scene "
                 f"{t['born_mid_scene']} ({t['born_mid_pct']:.0f}%)   "
                 f"died mid-scene {t['died_mid_scene']} ({t['died_mid_pct']:.0f}%)")
        L.append(f"      traversed the entry line: {t['traversed_zone_x']}")
    q = card.get("track_quality") or {}
    if q.get("track_ids"):
        L.append(f"      fragments per person {q['fragments_per_chain']:.2f} "
                 f"({q['track_ids']} ids -> {q['chains']} chains, "
                 f"worst {q['worst_chain']})")
        L.append(f"      lost-buffer recoveries {q['ids_with_recovery']} id(s) "
                 f"({q['recovery_pct']:.0f}%)   "
                 f"swap-pressure pairs {q['swap_pressure_pairs']}")

    c = card.get("counts") or {}
    if c:
        L.append("-" * width)
        L.append("  COUNTS  " + "   ".join(f"{k}={v}" for k, v in c.items()))
    a = card.get("arrival_confidence") or {}
    if a:
        L.append(f"  CONFIDENCE  [{a['tier']}]")
        L.append(f"      {a['why']}")

    v = card.get("visits") or {}
    if v.get("n_visits") is not None and v.get("unique_people") is not None:
        L.append("-" * width)
        L.append(f"  VISITS   {v['unique_people']} unique people, "
                 f"{v['n_visits']} visit(s), {v['repeat_visitors']} returned")
        L.append(f"      entries {v['entries']}   exits {v['exits']}   "
                 f"closed {v['closed']}   open {v['open']}")
        if v.get("entry_missed") or v.get("exit_missed"):
            L.append(f"      incomplete: {v['entry_missed']} left without "
                     f"arriving, {v['exit_missed']} arrived without leaving")

    s = card.get("entry_score") or {}
    if s.get("windows"):
        L.append("-" * width)
        L.append(f"  ENTRY LINE vs GROUND TRUTH   ({s['line_in_events']} IN events)")
        L.append(f"      {'window':30s}{'kind':8s}{'truth':>6s}{'got':>5s}")
        for r in s["windows"]:
            tag = "   TUNED - not evidence" if r["tuned"] else ""
            L.append(f"      {r['window']:30s}{r['kind']:8s}"
                     f"{r['truth']:6d}{r['got']:5d}{tag}")
        L.append(f"      {'HELD-OUT TOTAL':30s}{'':8s}"
                 f"{s['heldout_truth']:6d}{s['heldout_got']:5d}")

    rq = card.get("review_queue")
    if rq:
        from .review import render as _rq_render
        L.append(_rq_render(rq, width))

    v = card.get("verdicts") or []
    if v:
        L.append("-" * width); L.append("  VERDICTS")
        for r in v:
            L.append(f"      [{r['state']:8s}] {r['check']:24s} {r['detail']}")
    L.append(bar)
    return "\n".join(L)


def write(card: Dict[str, Any], out_dir: str, cam: str = "CAM") -> str:
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"{cam}_scorecard.json")
    with open(p, "w") as fh:
        json.dump(card, fh, indent=1, default=str)
    return p
