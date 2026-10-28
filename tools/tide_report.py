#!/usr/bin/env python3
"""tide_report.py — WHAT KIND of error, not just how many.

WHY THIS EXISTS
    HOTA says the pipeline is 0.80. FiftyOne says TP 5144 / FP 2219 / FN 294.
    Neither says WHY, and the fixes for each cause are different and mutually
    useless:

        a box in roughly the right place but too loose   -> tune NMS / box loss
        a box on the plant                               -> mask it
        two boxes on one person                          -> dedup
        a person nobody detected                         -> resolution, conf

    Chasing the wrong one is how a week disappears. TIDE (Bolya et al., ECCV
    2020) separates them by asking, for each false positive, what the box WOULD
    have needed to be correct.

WHAT THE CATEGORIES MEAN HERE (single class, so Cls is always 0)
    Loc    right object, box too loose/tight (0.1 <= IoU < 0.5)
    Dupe   a second box on a person already detected
    Bkg    matched nothing at all -- plant, reflection, furniture, OR a real
           person the ground truth never labelled. TIDE cannot tell those
           apart; on Task 3 we MEASURED that 2117 of 2219 were the latter.
    Miss   a labelled person with no detection near them

Usage:
    python tools/tide_report.py --run /data/outputs/final_quiet \
        --gt eval/gt_CAM_112.txt
"""
import argparse
import collections
import gzip
import json
import sys
from pathlib import Path


def load_mot(path):
    out = collections.defaultdict(list)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        f = line.strip().replace("\t", ",").split(",")
        if len(f) < 6:
            continue
        try:
            fr = int(float(f[0]))
            x, y, w, h = (float(v) for v in f[2:6])
        except ValueError:
            continue
        if w > 0 and h > 0:
            out[fr].append([x, y, w, h])
    return out


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    i = (x2 - x1) * (y2 - y1)
    return i / (aw * ah + bw * bh - i)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--iou", type=float, default=0.5)
    a = p.parse_args(argv)

    gt = load_mot(a.gt)
    flog = next(iter((Path(a.run) / "debug").glob("*_frames.json.gz")), None)
    if flog is None:
        raise SystemExit(f"no *_frames.json.gz in {a.run}/debug")
    raw = json.loads(gzip.open(flog, "rt", encoding="utf-8").read())
    pred = {}
    for e in raw:
        pred[int(e[0]) + 1] = [[b[1], b[2], b[3] - b[1], b[4] - b[2]] for b in e[2]]

    n = collections.Counter()
    for fr, gts in gt.items():
        prs = pred.get(fr, [])
        used = set()
        # greedy match at the real threshold first
        for pi, pb in enumerate(prs):
            best, bi = 0.0, -1
            for gi, gb in enumerate(gts):
                if gi in used:
                    continue
                v = iou(pb, gb)
                if v > best:
                    best, bi = v, gi
            if best >= a.iou:
                used.add(bi)
                n["TP"] += 1
            elif best >= 0.1:
                # right object, wrong box -- OR a duplicate on an already-matched one
                dup = any(iou(pb, prs[q]) > 0.5 for q in range(len(prs)) if q != pi)
                n["Dupe" if dup else "Loc"] += 1
            else:
                n["Bkg"] += 1
        n["Miss"] += len(gts) - len(used)

    tot_fp = n["Loc"] + n["Dupe"] + n["Bkg"]
    print(f"  run {a.run}")
    print(f"  gt  {a.gt}")
    print()
    print(f"  TP    {n['TP']:>6}")
    print(f"  Miss  {n['Miss']:>6}   labelled people with nothing near them")
    print(f"  FP    {tot_fp:>6}   broken down:")
    for k, why in (("Loc", "right object, box too loose (0.1<=IoU<0.5)"),
                   ("Dupe", "a SECOND box on someone already detected"),
                   ("Bkg", "matched nothing -- phantom OR unlabelled person")):
        share = 100 * n[k] / max(tot_fp, 1)
        print(f"    {k:<5} {n[k]:>6}  {share:5.1f}%  {why}")
    print()
    if n["Bkg"] > 0.6 * max(tot_fp, 1):
        print("  Bkg DOMINATES. Before masking or raising thresholds, check "
              "whether the ground truth labels everyone in frame -- on Task 3, "
              "2117 of 2219 'phantoms' were real people nobody boxed.")
    if n["Dupe"] > 0.2 * max(tot_fp, 1):
        print("  Dupe is significant -> dedup/NMS, not detection.")
    if n["Loc"] > 0.2 * max(tot_fp, 1):
        print("  Loc is significant -> box regression, not missing people.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
