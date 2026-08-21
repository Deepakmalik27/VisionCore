#!/usr/bin/env python3
"""Track lifecycle + effective network pixel height, from a run's frame log.

TWO DIAGNOSTICS THIS REPO DID NOT HAVE
--------------------------------------
1. EFFECTIVE NETWORK PIXEL HEIGHT (P0.5)
   A person 344px tall in a 3840-wide source arrives at the network ~115px
   tall, because the frame is scaled TWICE: once by frame_source(max_w=
   ANALYSIS_MAX_W) and again by the detector's imgsz. Reasoning about
   "how big is this person" in SOURCE pixels is how the note in
   cam112_fullframe.yaml concluded "nobody in this scene is small enough for
   tiling to be the tool" -- while the detector was being handed people a
   third of the size that reasoning assumed.

2. TRACK LIFECYCLE (P3.2)
   BoT-SORT already does Kalman predict + a lost buffer + Re-ID reactivation.
   What was missing is any way to SEE it: a track that dies mid-transit and a
   track that walks off frame look identical in every log this repo writes.
   Measured on CAM.112 by hand, that difference was the whole story -- three
   guests in the held-out window were detected, but their tracks were born
   left of the entry line and died right of it, so nothing ever crossed.

Neither needs a pipeline change: both read output/<run>/debug/*.json.gz.

Usage:  python3 tools/track_health.py <run> [--zone-x 1500] [--source-w 3840]
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import os
import sys


def load_tracks(run):
    p = f"output/{run}/debug/CAM.112_frames.json.gz"
    if not os.path.exists(p):
        return None, None
    frames = json.load(gzip.open(p, "rt"))
    tr = collections.defaultdict(list)
    for _fi, t, dets in frames:
        for tid, x1, y1, x2, y2 in dets:
            tr[tid].append((float(t), float(x1), float(y1), float(x2), float(y2)))
    for k in tr:
        tr[k].sort()
    return frames, tr


def network_scale(analysis_w, source_w, imgsz):
    """How much of a source pixel survives to the network, and where it goes."""
    to_analysis = analysis_w / float(source_w)
    to_network = min(1.0, imgsz / float(analysis_w))
    return to_analysis, to_network, to_analysis * to_network


def report_pixel_height(tr, analysis_w, source_w, imgsz):
    hs = sorted(y2 - y1 for p in tr.values() for _t, _x1, y1, _x2, y2 in p)
    if not hs:
        print("   no detections")
        return
    a, n, total = network_scale(analysis_w, source_w, imgsz)
    def q(f): return hs[int(f * (len(hs) - 1))]
    print(f"   scale: source -> analysis x{a:.2f}, analysis -> network x{n:.2f}"
          f"  (total x{total:.2f}, i.e. 1/{1/total:.1f})")
    print(f"   {'':14s}{'p10':>8s}{'median':>8s}{'p90':>8s}")
    for label, mul in (("in ANALYSIS px", 1.0),
                       ("in SOURCE px", 1.0 / a),
                       ("at the NETWORK", n)):
        print(f"   {label:14s}{q(.1)*mul:8.0f}{q(.5)*mul:8.0f}{q(.9)*mul:8.0f}")
    small = sum(1 for h in hs if h * n < 110)
    print(f"   below 110 network px (TILE_TARGET_MIN_PX): {small} of {len(hs)}"
          f" = {100.0*small/len(hs):.0f}%")


def report_lifecycle(tr, zone_x, frame_w=1920, frame_h=1080, edge_frac=0.06,
                     t_first=None, t_last=None):
    """Where tracks are BORN and DIE, and whether that is physically sensible.

    A person enters the scene at a FRAME EDGE and leaves at one. A track that
    is born in the middle of the room appeared from nowhere -- the detector
    missed them until then. A track that dies in the middle vanished while
    still in view. Both are fragmentation, and both are invisible in every
    other log this pipeline writes.

    Measured by hand on CAM.112's held-out busy window, this was the entire
    failure: three real guests, detected, whose tracks were born LEFT of the
    entry line (x~1160-1195) and died RIGHT of it (x~1598) -- so nothing ever
    traversed and the line scored 0 of 3.

    (An earlier version of this function also counted tracks that "changed
    side without traversing". That branch could never fire: if the first and
    last points are on opposite sides then some consecutive pair must straddle
    the line. The self-check caught it as dead code.)
    """
    ex, ey = edge_frac * frame_w, edge_frac * frame_h

    def at_edge(x, y):
        return x <= ex or x >= frame_w - ex or y <= ey or y >= frame_h - ey

    born_mid = died_mid = transit = 0
    mids = []
    for tid, p in tr.items():
        pts = [(((a + c) / 2.0), d) for _t, a, _b, c, d in p]   # (cx, foot y)
        xs = [q[0] for q in pts]
        b_edge = at_edge(*pts[0])
        d_edge = at_edge(*pts[-1])
        # A track alive at the very start / end of the clip did not "appear";
        # the window simply cut it. Do not blame the tracker for that.
        b_clipped = t_first is not None and abs(p[0][0] - t_first) < 1e-6
        d_clipped = t_last is not None and abs(p[-1][0] - t_last) < 1e-6
        if not b_edge and not b_clipped:
            born_mid += 1
        if not d_edge and not d_clipped:
            died_mid += 1
        if any((xs[i] - zone_x) * (xs[i + 1] - zone_x) < 0
               for i in range(len(xs) - 1)):
            transit += 1
        if (not b_edge and not b_clipped) or (not d_edge and not d_clipped):
            mids.append((tid, len(p), p[-1][0] - p[0][0], pts[0], pts[-1],
                         b_edge or b_clipped, d_edge or d_clipped))
    n = max(len(tr), 1)
    print(f"   tracks {len(tr)}")
    print(f"   BORN mid-scene (appeared from nowhere): {born_mid:4d}"
          f"  = {100.0*born_mid/n:.0f}%")
    print(f"   DIED mid-scene (vanished while in view): {died_mid:4d}"
          f"  = {100.0*died_mid/n:.0f}%")
    print(f"   traversed x={zone_x:.0f}: {transit}")
    if mids:
        print(f"   worst offenders:")
        print(f"      {'id':>6s} {'pts':>5s} {'secs':>6s} {'born':>14s}"
              f" {'died':>14s}  why")
        for tid, k, d, b, e, be, de in sorted(mids, key=lambda r: -r[1])[:10]:
            why = []
            if not be: why.append("born mid")
            if not de: why.append("died mid")
            print(f"      {tid:6d} {k:5d} {d:6.1f} ({b[0]:5.0f},{b[1]:5.0f})"
                  f" ({e[0]:5.0f},{e[1]:5.0f})  {' + '.join(why)}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--zone-x", type=float, default=1500.0,
                    help="critical vertical line in ANALYSIS px (the entry line)")
    ap.add_argument("--source-w", type=float, default=3840.0)
    ap.add_argument("--analysis-w", type=float, default=1920.0)
    ap.add_argument("--imgsz", type=float, default=1280.0)
    a = ap.parse_args(argv)

    frames, tr = load_tracks(a.run)
    if tr is None:
        print(f"no frame log for run {a.run!r}", file=sys.stderr)
        return 2
    print(f"\n=== {a.run} ===")
    print("\n EFFECTIVE PIXEL HEIGHT")
    report_pixel_height(tr, a.analysis_w, a.source_w, a.imgsz)
    print("\n TRACK LIFECYCLE  (critical line x=%.0f)" % a.zone_x)
    ts = [p[0][0] for p in tr.values()] + [p[-1][0] for p in tr.values()]
    report_lifecycle(tr, a.zone_x, a.analysis_w, 1080,
                     t_first=min(ts) if ts else None,
                     t_last=max(ts) if ts else None)
    return 0


def _self_check():
    a, n, total = network_scale(1920, 3840, 1280)
    assert abs(a - 0.5) < 1e-9
    assert abs(n - (1280 / 1920)) < 1e-9
    assert abs(total - 1 / 3) < 1e-6, "3840->1920->1280 must be a 3x reduction"
    # raising ANALYSIS_MAX_W to source removes the FIRST scale entirely
    a2, n2, t2 = network_scale(3840, 3840, 1920)
    assert abs(a2 - 1.0) < 1e-9 and abs(t2 - 0.5) < 1e-9
    assert t2 > total, "3840/1920 must hand the network MORE pixels than 1920/1280"
    # lifecycle: a track born and dying in the MIDDLE of the frame is
    # fragmentation; one that enters and leaves at the edges is a person.
    tr = {
        1: [(0.0, 900, 400, 1000, 600), (5.0, 1100, 400, 1200, 600)],   # mid->mid
        2: [(0.0, 0, 400, 80, 600), (5.0, 1840, 400, 1920, 600)],       # edge->edge
    }
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_lifecycle(tr, 1500.0, 1920, 1080, t_first=None, t_last=None)
    out = buf.getvalue()
    assert "BORN mid-scene (appeared from nowhere):    1" in out, out
    assert "DIED mid-scene (vanished while in view):    1" in out, out
    # and a track alive at the clip boundary must NOT be blamed
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        report_lifecycle(tr, 1500.0, 1920, 1080, t_first=0.0, t_last=5.0)
    out2 = buf2.getvalue()
    assert "BORN mid-scene (appeared from nowhere):    0" in out2, out2
    assert "DIED mid-scene (vanished while in view):    0" in out2, out2
    print("track_health self-check OK")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        raise SystemExit(main())
