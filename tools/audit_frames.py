#!/usr/bin/env python3
"""audit_frames.py — answer the recurring "is it actually broken?" questions
from the frame log, instead of from watching the video and guessing.

WHY THIS EXISTS
    A 10-minute run is 4,505 frames. Nobody watches 4,505 frames, and the two
    people who tried (me, and a second model) both reported things the data
    flatly contradicts — "ID switching everywhere" when the live tracker
    switched zero times, "the count line is in the wrong place" when the line
    was fine and the RENDER drew a different one.

    Eyes are still needed for "is that box a person?". They are the wrong tool
    for "how often does this happen across ten minutes".

WHAT IT MEASURES
    duplicates   two boxes on one body: pairwise IoU within a frame, in the
                 band that dedup NMS deliberately keeps. This is the suspected
                 root cause of identity fragmentation — a body wearing two
                 concurrent track ids can NEVER be merged by the stitcher,
                 because the stitcher refuses to merge tracks that overlap in
                 time (correctly: same-time fragments are usually two people).
    rigid        per-location box-size coefficient of variation and centre
                 jitter, as a fraction of body height. Furniture holds a
                 near-identical box; a person's box breathes. This is how the
                 statue on the reception desk is found without anyone drawing
                 a mask polygon by hand.
    identities   how many distinct ids hold the desk, vs how many people
                 plausibly did.
    presence     empty frames, blink-outs, track lifetimes.

USAGE
    python3 tools/audit_frames.py output/profC/debug/CAM.112_frames.json.gz
"""
from __future__ import annotations

import gzip
import json
import math
import statistics as st
import sys
from collections import defaultdict


def load(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as fh:
        return json.load(fh)


def _boxes(rec):
    """Frame records have drifted in shape across builds. Accept what exists
    rather than crashing on the one that does not — a shape assumption is how
    an audit tool ends up reporting zero of everything and being believed."""
    if isinstance(rec, dict):
        bs = rec.get("boxes") or rec.get("dets") or []
        t = rec.get("t", rec.get("time", rec.get("frame_idx", 0)))
    else:
        # (frame_idx, t, boxes)
        t = rec[1] if len(rec) > 1 else 0
        bs = rec[2] if len(rec) > 2 else []
    out = []
    for b in bs:
        if isinstance(b, dict):
            tid = b.get("track_id", b.get("tid"))
            xy = b.get("xyxy") or [b.get("x1"), b.get("y1"), b.get("x2"), b.get("y2")]
        else:
            tid, xy = b[0], list(b[1:5])
        if xy and all(v is not None for v in xy):
            out.append((tid, float(xy[0]), float(xy[1]), float(xy[2]), float(xy[3])))
    return float(t or 0), out


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main(path):
    frames = load(path)
    print("=" * 72)
    print(f"  {path}")
    print(f"  {len(frames)} frame records")
    print("=" * 72)

    n_boxes = 0
    per_frame = []
    life = defaultdict(list)          # tid -> [t...]
    cells = defaultdict(list)         # grid cell -> [(cx,cy,w,h)]
    dup_pairs = 0
    dup_hist = defaultdict(int)
    dup_by_pair = defaultdict(int)    # (tid_a,tid_b) -> frames overlapping

    for rec in frames:
        t, bs = _boxes(rec)
        n_boxes += len(bs)
        per_frame.append((t, len(bs)))
        for tid, x1, y1, x2, y2 in bs:
            life[tid].append(t)
            w, h = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            cells[(int(cx // 120), int(cy // 120))].append((cx, cy, w, h))
        for i in range(len(bs)):
            for j in range(i + 1, len(bs)):
                v = iou(bs[i][1:], bs[j][1:])
                if v > 0.30:
                    dup_pairs += 1
                    dup_hist[round(v, 1)] += 1
                    key = tuple(sorted((str(bs[i][0]), str(bs[j][0]))))
                    dup_by_pair[key] += 1

    counts = [n for _, n in per_frame]
    empty = sum(1 for n in counts if n == 0)
    blinks = sum(1 for i in range(1, len(counts) - 1)
                 if counts[i] == 0 and counts[i - 1] > 0 and counts[i + 1] > 0)

    print("\nPRESENCE")
    print(f"  boxes total          {n_boxes}")
    print(f"  empty frames         {empty}  ({100*empty/max(1,len(counts)):.0f}%)")
    print(f"  blink-outs           {blinks}")
    print(f"  max people in frame  {max(counts) if counts else 0}")
    hist = defaultdict(int)
    for n in counts:
        hist[n] += 1
    print(f"  boxes/frame          {dict(sorted(hist.items()))}")

    print("\nDUPLICATES  (two boxes on one body -> two concurrent ids the")
    print("             stitcher can never merge, because they overlap in time)")
    print(f"  frame-pairs with IoU>0.30   {dup_pairs}")
    print(f"  IoU histogram               {dict(sorted(dup_hist.items()))}")
    worst = sorted(dup_by_pair.items(), key=lambda kv: -kv[1])[:8]
    for (a, b), n in worst:
        print(f"    ids {a:>6} + {b:<6} overlap in {n} frames")
    if worst and worst[0][1] > 50:
        print("  ^ VERDICT: sustained. Two ids riding one body for many seconds")
        print("    is the mechanism behind identity fragmentation, and it is a")
        print("    DETECTION/NMS fault, not a tracker fault.")

    print("\nRIGID LOCATIONS  (furniture/statue: box that never breathes)")
    print("  cell        hits   size_cv   jitter/h   verdict")
    rigid = []
    for cell, obs in cells.items():
        if len(obs) < 60:
            continue
        ws = [o[2] for o in obs]
        hs = [o[3] for o in obs]
        mh = st.median(hs)
        if mh <= 0:
            continue
        # cv of width and of height SEPARATELY, then the worse of the two.
        # Pooling them (pstdev(ws + hs) / mean(ws + hs)) measures how far
        # width differs from height — a constant 60x119 statue scored 0.33 and
        # read as "person-like". The question is whether each dimension is
        # steady over time, not whether the box is square.
        def _cv(vals):
            m = st.mean(vals)
            return st.pstdev(vals) / m if m > 1e-6 else 0.0
        size_cv = max(_cv(ws), _cv(hs))
        cxs = [o[0] for o in obs]
        cys = [o[1] for o in obs]
        jit = math.hypot(st.pstdev(cxs), st.pstdev(cys)) / mh
        rigid.append((size_cv, jit, len(obs), cell))
    for size_cv, jit, n, cell in sorted(rigid)[:10]:
        v = "RIGID — furniture" if (size_cv < 0.05 and jit < 0.05) else "person-like"
        print(f"  {str(cell):<11} {n:>5}   {size_cv:>6.4f}    {jit:>6.4f}   {v}")
    if not rigid:
        print("  (no cell reached 60 hits)")

    print("\nIDENTITIES")
    lifetimes = sorted(len(v) for v in life.values())
    print(f"  distinct track ids   {len(life)}")
    if lifetimes:
        def pct(p):
            return lifetimes[min(len(lifetimes) - 1, int(p * len(lifetimes)))]
        print(f"  lifetime frames      p10 {pct(.1)}  p50 {pct(.5)}  "
              f"p90 {pct(.9)}  max {lifetimes[-1]}")
        short = sum(1 for v in lifetimes if v <= 3)
        print(f"  tracks <=3 frames    {short} of {len(lifetimes)}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
