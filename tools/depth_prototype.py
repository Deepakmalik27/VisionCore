#!/usr/bin/env python3
"""depth_prototype.py — can MONOCULAR DEPTH give us what a ToF sensor gives?

THE REASON TO TRY
    Published people-counting accuracy sorts almost entirely by whether the
    system has DEPTH:

        stereo video (2 lenses)      98%+
        depth + vision fusion        >98%
        ToF over the door            99% optimal
        2D angled camera (ours)      80-95%

    ToF units do not win because their models are better. They win because
    depth makes two people who overlap in the image obviously separate, and
    because a height reading is a physical fact rather than an inference from
    box size. That is the whole gap.

    We cannot mount a ToF sensor from here. But Depth Anything V2 predicts a
    per-pixel depth map from ONE ordinary RGB frame — the same camera we
    already have. If it is good enough, it buys the two things that matter.

WHAT WOULD ACTUALLY HELP US, IN ORDER
    1. THE FLOOR. Our root cause is that boxes are 1.6x too tall, so the
       reported "feet" sit ~260px below the person, which breaks every zone
       test and line crossing. Depth gives a floor surface. The feet are where
       the person's depth meets the floor's depth — a physical answer, not a
       calibration constant we keep getting wrong.
    2. SEPARATION. Two guests entering together merge into one box today.
       At different distances they are obviously two objects in depth.
    3. PLAUSIBILITY. "Is this box a person at this distance" becomes a real
       test, replacing the D0/D1 size guessing that has already deleted real
       guests twice.

HOW THIS CAN FAIL, AND SHOULD SAY SO
    Monocular depth is RELATIVE and up-to-scale. It can be flat and useless on
    a dim infrared frame with little texture — which is 66% of this footage. It
    also costs a second model per frame, on a pipeline already at 30 min/hour
    against a 20-minute target.

    So this measures three specific things and prints a verdict. If depth does
    not separate people who are genuinely at different distances, it cannot do
    job 2, and job 1 is unreliable too. Better to learn that in an afternoon.

USAGE
    python3 tools/depth_prototype.py label_pkg/quick100 gt.txt --frame 1
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
        d[int(f[0])].append((x, y, x + w, y + h))
    return d


def get_depth_model():
    """Depth Anything V2, or a clear install line."""
    try:
        import torch
        from transformers import pipeline
        dev = 0 if torch.cuda.is_available() else -1
        for name in ("depth-anything/Depth-Anything-V2-Small-hf",
                     "depth-anything/Depth-Anything-V2-Base-hf"):
            try:
                return pipeline("depth-estimation", model=name, device=dev), name
            except Exception:
                continue
    except ImportError:
        pass
    return None, ("Needs transformers + torch:\n"
                  "    ~/kv312/bin/pip install 'transformers>=4.45'\n"
                  "then the model downloads on first use (~100 MB for Small).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkg")
    ap.add_argument("gt")
    ap.add_argument("--frame", type=int, default=1,
                    help="use a frame whose hand labels are TRUSTWORTHY. The "
                         "first gt.txt was 99%% carried forward; only frame 1 "
                         "was genuinely drawn.")
    a = ap.parse_args()

    pkg = Path(a.pkg)
    gt = load_mot(a.gt)
    pr = load_mot(pkg / "predictions.txt")
    fr = a.frame

    model, how = get_depth_model()
    if model is None:
        print("=" * 70)
        print("  CANNOT RUN — no depth model")
        print("=" * 70)
        print(how)
        return 2

    from PIL import Image
    img_p = pkg / "images" / f"{fr:07d}.jpg"
    if not img_p.exists():
        print(f"no such frame: {img_p}")
        return 1
    img = Image.open(img_p).convert("RGB")

    import time
    t0 = time.time()
    depth = np.asarray(model(img)["depth"], dtype=np.float32)
    dt = time.time() - t0

    print("=" * 70)
    print(f"  MONOCULAR DEPTH · {how}")
    print(f"  frame {fr} · {img.size[0]}x{img.size[1]} · {dt*1000:.0f} ms")
    print("=" * 70)

    # Is the map informative at all, or flat noise on an IR frame?
    print(f"\n1. IS THERE ANY SIGNAL?")
    print(f"   depth range {depth.min():.1f} .. {depth.max():.1f}   "
          f"std {depth.std():.1f}")
    if depth.std() < 1.0:
        print("   FLAT — no usable depth on this frame. Likely too dark / "
              "textureless.")
        return 1

    # 2. Do people at different distances read as different depths?
    people = gt.get(fr, [])
    print(f"\n2. DOES IT SEPARATE PEOPLE?   ({len(people)} hand-labelled)")
    meds = []
    for i, (x1, y1, x2, y2) in enumerate(people):
        # sample the upper-middle of the body: head/shoulders are reliably ON
        # the person even when a box is loose
        cx = int((x1 + x2) / 2)
        cy = int(y1 + (y2 - y1) * 0.3)
        h = max(4, int((y2 - y1) * 0.12))
        w = max(4, int((x2 - x1) * 0.20))
        patch = depth[max(0, cy - h):cy + h, max(0, cx - w):cx + w]
        if patch.size:
            m = float(np.median(patch))
            meds.append(m)
            print(f"   person {i+1}   depth {m:7.1f}   at ({cx},{cy})")
    if len(meds) >= 2:
        spread = max(meds) - min(meds)
        print(f"   spread across people: {spread:.1f}  "
              f"(vs whole-frame std {depth.std():.1f})")
        if spread > depth.std() * 0.5:
            print("   -> people ARE at measurably different depths. Depth can "
                  "separate merged bodies.")
        else:
            print("   -> people read at nearly the SAME depth. Depth will NOT "
                  "separate them here.")

    # 3. The prize: can depth locate the floor under a person?
    print(f"\n3. CAN IT FIND THE FLOOR (i.e. the real feet)?")
    ok = 0
    for i, (x1, y1, x2, y2) in enumerate(people[:4]):
        cx = int((x1 + x2) / 2)
        col = depth[:, max(0, cx - 3):cx + 4].mean(axis=1)
        body = float(np.median(col[int(y1):int(y1 + (y2 - y1) * 0.5)]))
        # walk down from the labelled feet and find where depth stops matching
        # the body — that boundary is the person/floor transition
        below = col[int(y2):min(len(col), int(y2) + 200)]
        if below.size:
            d = abs(below - body)
            edge = int(np.argmax(d > max(1.0, depth.std() * 0.25)))
            print(f"   person {i+1}: body depth {body:6.1f}  "
                  f"floor transition {edge:3d}px below the labelled feet")
            if edge < 60:
                ok += 1
    if people:
        print(f"   -> {ok} of {min(4,len(people))} transitions land within 60px "
              f"of the hand-labelled feet")
        print("      (close = depth locates the feet; far = it does not)")

    print("\nCOST")
    print(f"   {dt*1000:.0f} ms/frame. At 8 fps over an hour that is "
          f"{dt*28800/60:.0f} min added to a run already at ~30 min.")
    print("   Running it only near the entrance, or only when a track is close")
    print("   to a line, would cut that to a fraction.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
