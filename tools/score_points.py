#!/usr/bin/env python3
"""score_points.py — recall and false positives from CLICKED POINTS.

WHY THIS EXISTS
    The question that matters most here is "how many people does the pipeline
    MISS", and answering it does not need drawn boxes. It needs to know where
    the humans are. A dot is a human; a box that contains a dot found that
    human; a dot inside no box is a miss; a box containing no dot is a false
    positive.

    Drawn boxes are needed for IoU-based mAP and for fine-tuning. They are NOT
    needed for recall, and asking a person to drag 486 rectangles to answer a
    question a click answers is how ground truth never gets collected at all.

    A dot is roughly ten times faster than a rectangle, and on this footage the
    boxes are 300-700px tall, so "is the dot inside the box" is unambiguous.

USAGE
    python3 tools/score_points.py label_pkg/quick100 points.txt
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path


def load_points(p):
    pts = defaultdict(list)
    for line in Path(p).read_text().splitlines():
        f = line.split(",")
        if len(f) < 3:
            continue
        pts[int(f[0])].append((float(f[1]), float(f[2])))
    return pts


def load_boxes(p):
    bx = defaultdict(list)
    for line in Path(p).read_text().splitlines():
        f = line.split(",")
        if len(f) < 6:
            continue
        x, y, w, h = (float(v) for v in f[2:6])
        bx[int(f[0])].append((x, y, x + w, y + h))
    return bx


def main(pkg, points_file):
    pkg = Path(pkg)
    gt = load_points(points_file)
    pr = load_boxes(pkg / "predictions.txt")
    frames = sorted(gt)
    if not frames:
        sys.exit("no points in that file — did you click any people?")

    hit = miss = fp = 0
    per_frame = []
    for f in frames:
        pts, boxes = gt[f], pr.get(f, [])
        used = set()
        m = 0
        for (px, py) in pts:
            found = -1
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                if i in used:
                    continue
                if x1 <= px <= x2 and y1 <= py <= y2:
                    found = i
                    break
            if found >= 0:
                used.add(found)
                hit += 1
            else:
                miss += 1
                m += 1
        f_fp = len(boxes) - len(used)
        fp += f_fp
        per_frame.append((f, len(pts), len(boxes), m, f_fp))

    n_gt = hit + miss
    recall = hit / n_gt if n_gt else 0.0
    prec = hit / (hit + fp) if (hit + fp) else 0.0
    print("=" * 66)
    print(f"  POINT SCORE · {pkg.name} · {len(frames)} labelled frames")
    print("=" * 66)
    print(f"  people you marked      {n_gt}")
    print(f"  found by the pipeline  {hit}")
    print(f"  MISSED                 {miss}")
    print(f"  boxes on nobody        {fp}")
    print()
    print(f"  RECALL     {recall:6.1%}   <- of the people present, how many did we see")
    print(f"  PRECISION  {prec:6.1%}   <- of the boxes drawn, how many were real")
    print()
    worst = sorted(per_frame, key=lambda r: -(r[3] + r[4]))[:8]
    if worst and (worst[0][3] + worst[0][4]) > 0:
        print("  worst frames      frame  people  boxes  missed  on-nobody")
        for f, npt, nbx, m, e in worst:
            if m + e == 0:
                continue
            print(f"  {'':16}{f:>6}{npt:>8}{nbx:>7}{m:>8}{e:>11}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
