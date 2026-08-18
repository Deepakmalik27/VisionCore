#!/usr/bin/env python3
"""mask_prototype.py — test whether SEGMENTATION MASKS fix the root cause.

THE ROOT CAUSE, IN ONE LINE
    The detector draws street-shaped boxes on overhead-shaped people.

    models/best.pt is fine-tuned on CrowdHuman — street-level footage where a
    standing body is roughly h/w 2.5. CAM.112 is a ceiling camera, where a
    foreshortened person is nearer h/w 1.1-1.5. The model stretches its learned
    shape onto them, so the box runs well below the feet. Measured on the one
    frame whose hand labels are trustworthy (the rest were carried forward):

        pipeline box   194 x 542 px    h/w 2.80
        hand-drawn     290 x 330 px    h/w 1.14

    Every consumer of a detection anchors on the bottom of that box: zone
    membership, entry-line crossings, ground-plane samples, and the
    expected-height model that D0 and D1 both trust. They have all been asking
    about a point below the person.

WHY A MASK SHOULD FIX IT STRUCTURALLY, NOT BY TUNING
    A box carries an aspect-ratio prior. A mask does not — it is the pixels of
    the person. So:

        the bottom of the mask IS the feet        (no calibration constant)
        no h/w assumption                        (foreshortening is irrelevant)
        two overlapping people separate cleanly  (different pixel sets)

    That turns "missed people", "sloppy boxes", the anchor error and part of the
    identity fragmentation from things we tune into things that do not arise.
    SAM2MOT (arXiv 2504.04519) reports SOTA on DanceTrack exactly this way —
    zero-shot, a pre-trained detector prompting a pre-trained segmenter — which
    is the shape of this experiment.

WHAT THIS SCRIPT DOES, AND DOES NOT
    DOES  prompt a segmenter with the boxes we already produce, derive geometry
          from the masks, and compare BOTH against hand labels.

    DOES NOT  assume masks win. That is the question. If the mask bottom is no
          closer to the real feet than the box bottom, this says so and the idea
          is dead — worth learning in an afternoon instead of after a rebuild.

    ONLY TRUST FRAMES A HUMAN ACTUALLY EDITED. The first gt.txt was 99%
    copy-forward, which invalidated a day of measurements. --frames 1 uses just
    the frame known to be good.

USAGE
    python3 tools/mask_prototype.py label_pkg/quick100 gt.txt --frames 1
    python3 tools/mask_prototype.py label_pkg/quick100 gt.txt
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_mot(p):
    d = defaultdict(list)
    for line in Path(p).read_text().splitlines():
        f = line.split(",")
        if len(f) < 6:
            continue
        x, y, w, h = (float(v) for v in f[2:6])
        d[int(f[0])].append((int(f[1]), x, y, x + w, y + h))
    return d


def get_segmenter():
    """A segmenter, or a clear message about what to install."""
    try:
        from ultralytics import SAM
        for ckpt in ("sam2.1_s.pt", "sam2_s.pt", "sam2.1_t.pt", "mobile_sam.pt"):
            try:
                return SAM(ckpt), f"ultralytics SAM ({ckpt})"
            except Exception:
                continue
    except ImportError:
        pass
    return None, ("No segmenter available. ultralytics is already a dependency "
                  "here, so the cheapest route is:\n"
                  "    ~/kv312/bin/python -c \"from ultralytics import SAM; "
                  "SAM('sam2.1_s.pt')\"\n"
                  "which downloads the checkpoint on first use.")


def mask_box(mask):
    """Tight bounding box of a boolean mask, or None if empty."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkg")
    ap.add_argument("gt")
    ap.add_argument("--frames", type=int, default=0,
                    help="only the first N labelled frames. Use 1 when the rest "
                         "of the ground truth is carried forward.")
    a = ap.parse_args()

    pkg = Path(a.pkg)
    gt = load_mot(a.gt)
    pr = load_mot(pkg / "predictions.txt")
    frames = sorted(gt)[:a.frames] if a.frames else sorted(gt)

    seg, how = get_segmenter()
    if seg is None:
        print("=" * 70)
        print("  CANNOT RUN — no segmenter")
        print("=" * 70)
        print(how)
        return 2

    import cv2
    print("=" * 70)
    print(f"  MASK vs BOX geometry · {how} · {len(frames)} frame(s)")
    print("=" * 70)

    box_err, mask_err, box_ar, mask_ar, gt_ar = [], [], [], [], []
    for fr in frames:
        img_p = pkg / "images" / f"{fr:07d}.jpg"
        if not img_p.exists():
            continue
        img = cv2.imread(str(img_p))
        boxes = [list(b[1:]) for b in pr.get(fr, [])]
        if not boxes:
            continue
        try:
            res = seg(img, bboxes=boxes, verbose=False)[0]
            masks = res.masks.data.cpu().numpy()
        except Exception as exc:
            print(f"  frame {fr}: segmentation failed: {exc!r}")
            continue

        for i, pb in enumerate(boxes):
            if i >= len(masks):
                break
            mb = mask_box(masks[i] > 0.5)
            if mb is None:
                continue
            truth = None
            for _, gx1, gy1, gx2, gy2 in gt.get(fr, []):
                cx, cy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
                if pb[0] <= cx <= pb[2] and pb[1] <= cy <= pb[3]:
                    truth = (gx1, gy1, gx2, gy2)
                    break
            if truth is None:
                continue
            box_err.append(pb[3] - truth[3])       # +ve = below the real feet
            mask_err.append(mb[3] - truth[3])
            if pb[2] - pb[0] > 0:
                box_ar.append((pb[3] - pb[1]) / (pb[2] - pb[0]))
            if mb[2] - mb[0] > 0:
                mask_ar.append((mb[3] - mb[1]) / (mb[2] - mb[0]))
            if truth[2] - truth[0] > 0:
                gt_ar.append((truth[3] - truth[1]) / (truth[2] - truth[0]))

    if not box_err:
        print("  no matched people — nothing to compare.")
        return 1

    print(f"\nmatched people: {len(box_err)}")
    print("\nFOOT ERROR — how far BELOW the real feet the bottom edge sits (px)")
    print(f"  {'':22}{'median':>9}{'min':>9}{'max':>9}")
    for name, v in (("box bottom (today)", box_err), ("MASK bottom", mask_err)):
        print(f"  {name:<22}{st.median(v):>9.1f}{min(v):>9.1f}{max(v):>9.1f}")

    b, m = abs(st.median(box_err)), abs(st.median(mask_err))
    print(f"\n  |error|   box {b:.0f}px  ->  mask {m:.0f}px")
    if m < b * 0.6:
        print("  VERDICT: masks land substantially closer to the feet.")
        print("           Worth making the detection layer mask-based.")
    elif m < b:
        print("  VERDICT: masks help but not decisively — weigh against the cost.")
    else:
        print("  VERDICT: masks do NOT fix the anchor. Idea dead, cheaply.")

    print("\nSHAPE — height/width   (a real overhead person measured ~1.14)")
    print(f"  hand-labelled   {st.median(gt_ar):.2f}")
    print(f"  pipeline box    {st.median(box_ar):.2f}")
    print(f"  MASK            {st.median(mask_ar):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
