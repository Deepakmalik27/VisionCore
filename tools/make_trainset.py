#!/usr/bin/env python3
"""make_trainset.py — turn HUMAN labels into a YOLO fine-tuning set.

WHY THIS EXISTS, AND WHY dataset_collector.py CANNOT DO IT
    kevacv/dataset_collector.py already exports training data, by banking the
    pipeline's own high-confidence detections as pseudo-labels. For most
    purposes that is the right trick. For OUR problem it is exactly backwards.

    The defect we are trying to remove IS the shape of the model's boxes:

        pipeline box    194 x 542 px    h/w 2.80
        hand-labelled   290 x 330 px    h/w 1.14

    best.pt is fine-tuned on CrowdHuman — street-level footage where a standing
    body really is about h/w 2.5. CAM.112 is a ceiling camera, so a
    foreshortened person is nearer h/w 1.14, and the model stretches its
    learned shape over them. The bottom edge then lands ~260 px below the real
    feet, and every zone test, entry-line crossing and ground-plane sample in
    the pipeline reads that bottom edge.

    Pseudo-labels are the model's own boxes. Training on them would teach the
    wrong shape more confidently. The only thing that can correct a shape prior
    is a human saying where the person actually is.

WHAT THIS PRODUCES
    A standard YOLO detection dataset from label_pkg/<name>/images plus a
    hand-made gt.txt:

        trainset/
          images/train/*.jpg   images/val/*.jpg
          labels/train/*.txt   labels/val/*.txt      class cx cy w h, normalised
          dataset.yaml

    Then:  yolo detect train data=trainset/dataset.yaml model=models/best.pt \
               imgsz=1280 epochs=50

REFUSES TO BUILD A SET THAT WOULD TEACH THE WRONG THING
    Two guards, both learned the hard way:

    1. CARRIED-FORWARD FRAMES ARE EXCLUDED. The first gt.txt collected here was
       99% copy-forward — six boxes drawn on frame 1 and C pressed ninety-nine
       times — while the people underneath moved up to 255 px. Training on that
       would teach "a person is wherever they were a few seconds ago".

    2. IT CHECKS THE SHAPE IT IS ABOUT TO TEACH. If the labels' median h/w is
       within 20% of the pipeline's own, they are probably traced from the
       predictions rather than the people, and fine-tuning would change
       nothing. It says so and stops.

USAGE
    python3 tools/make_trainset.py label_pkg/quick100 gt.txt --out trainset
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_mot(p):
    d = defaultdict(list)
    for line in Path(p).read_text().splitlines():
        f = line.split(",")
        if len(f) < 6:
            continue
        x, y, w, h = (float(v) for v in f[2:6])
        d[int(f[0])].append((x, y, w, h))
    return d


def static_frames(gt):
    """Frames whose boxes are byte-identical to the previous frame's.

    These are carried forward, not labelled. A person moves ~5 px per frame on
    this footage, so an identical box means the label is stale.
    """
    frames = sorted(gt)
    out = set()
    for i in range(1, len(frames)):
        a, b = gt[frames[i - 1]], gt[frames[i]]
        if len(a) == len(b) and all(
                abs(p[0] - q[0]) < 0.01 and abs(p[1] - q[1]) < 0.01
                and abs(p[2] - q[2]) < 0.01 and abs(p[3] - q[3]) < 0.01
                for p, q in zip(a, b)):
            out.add(frames[i])
    return out


def img_size(p):
    """JPEG dimensions without pulling in PIL."""
    import struct
    d = Path(p).read_bytes()
    i = 2
    while i < len(d) - 9:
        if d[i] == 0xFF and d[i + 1] in (0xC0, 0xC1, 0xC2):
            h, w = struct.unpack(">HH", d[i + 5:i + 9])
            return w, h
        i += 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkg")
    ap.add_argument("gt")
    ap.add_argument("--out", default="trainset")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--allow-static", action="store_true",
                    help="include carried-forward frames anyway. Almost always "
                         "wrong — see the module docstring.")
    a = ap.parse_args()

    pkg = Path(a.pkg)
    gt = load_mot(a.gt)
    pred_p = pkg / "predictions.txt"
    if not gt:
        sys.exit("no labels in that gt file")

    stale = static_frames(gt)
    usable = [f for f in sorted(gt) if f not in stale or a.allow_static]

    print("=" * 70)
    print("  BUILD A FINE-TUNING SET FROM HUMAN LABELS")
    print("=" * 70)
    print(f"  labelled frames        {len(gt)}")
    print(f"  carried forward        {len(stale)}   "
          f"({'INCLUDED — see --allow-static' if a.allow_static else 'excluded'})")
    print(f"  usable frames          {len(usable)}")

    # A MINIMUM, not just "more than zero".
    #
    # static_frames() can only mark frames 2..N as carried forward — frame 1 has
    # nothing before it, so it always survives. A gt.txt that is entirely
    # copy-forward therefore leaves exactly ONE usable frame, and an earlier
    # version of this happily wrote a 1-image dataset and printed a soft warning.
    # A dataset that small cannot train anything; it can only waste GPU time and
    # produce a model somebody might then trust.
    MIN_USABLE = 5
    if len(usable) < MIN_USABLE:
        print(f"\n  REFUSING: only {len(usable)} genuinely-labelled frame(s) "
              f"(need >= {MIN_USABLE}).")
        if stale:
            print(f"  {len(stale)} frames are carried forward — the boxes are")
            print(f"  identical to the previous frame while the people moved.")
            print(f"  Training on those teaches 'a person is wherever they were")
            print(f"  a few seconds ago'.")
        print(f"  Re-label in POINT mode; the label tool now blocks carrying")
        print(f"  frames forward and refuses to export below 30% edited.")
        return 1

    # Guard 2: are these labels actually different from the model's boxes?
    gt_ar = [h / w for f in usable for (_, _, w, h) in gt[f] if w > 0]
    if pred_p.exists():
        pr = load_mot(pred_p)
        pr_ar = [h / w for f in usable for (_, _, w, h) in pr.get(f, []) if w > 0]
        if gt_ar and pr_ar:
            g, p = st.median(gt_ar), st.median(pr_ar)
            print(f"\n  label   h/w median {g:.2f}")
            print(f"  model   h/w median {p:.2f}")
            if abs(g - p) / max(p, 1e-6) < 0.20:
                print("\n  REFUSING: the labels are within 20% of the model's own")
                print("  shape, so they were probably traced from the predictions")
                print("  rather than the people. Fine-tuning on them would change")
                print("  nothing. The whole point is that a real overhead person")
                print("  is h/w ~1.1-1.2 while the model draws ~2.5-2.8.")
                return 1
            print(f"  -> labels differ by {100*abs(g-p)/p:.0f}% — there is a real "
                  f"shape correction to learn.")

    out = Path(a.out)
    n_val = max(1, int(len(usable) * a.val_frac))
    val = set(usable[::max(1, len(usable) // n_val)][:n_val])
    counts = {"train": 0, "val": 0}
    boxes = {"train": 0, "val": 0}
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    for f in usable:
        src = pkg / "images" / f"{f:07d}.jpg"
        if not src.exists():
            continue
        wh = img_size(src)
        if not wh:
            continue
        W, H = wh
        split = "val" if f in val else "train"
        shutil.copy2(src, out / "images" / split / src.name)
        lines = []
        for (x, y, w, h) in gt[f]:
            # YOLO wants class + normalised centre + normalised size, clipped
            cx, cy = (x + w / 2) / W, (y + h / 2) / H
            nw, nh = w / W, h / H
            if not (0 < nw <= 1 and 0 < nh <= 1):
                continue
            cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        (out / "labels" / split / f"{src.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""))
        counts[split] += 1
        boxes[split] += len(lines)

    # best.pt is a 2-class model (person, head). These labels are person-only,
    # so say that plainly rather than letting a silent class mismatch surprise
    # whoever runs the training.
    (out / "dataset.yaml").write_text(
        f"# Fine-tuning set from HUMAN labels — corrects the box SHAPE that\n"
        f"# CrowdHuman training gets wrong on a ceiling camera.\n"
        f"# NOTE: single class. models/best.pt is 2-class (person, head), so a\n"
        f"# fine-tune from it will drop the head class unless you add head\n"
        f"# labels too. The head class currently rebuilds ~749 occluded bodies\n"
        f"# per hour, so losing it is a real trade — decide deliberately.\n"
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n  0: person\n")
    (out / "provenance.json").write_text(json.dumps({
        "package": str(pkg), "gt": str(a.gt),
        "frames_labelled": len(gt), "frames_carried_forward": len(stale),
        "frames_used": counts, "boxes": boxes,
        "label_hw_median": round(st.median(gt_ar), 3) if gt_ar else None,
    }, indent=2))

    print(f"\n  wrote {out}/")
    print(f"    train  {counts['train']:>4} images  {boxes['train']:>5} boxes")
    print(f"    val    {counts['val']:>4} images  {boxes['val']:>5} boxes")
    print(f"\n  train with:")
    print(f"    yolo detect train data={out}/dataset.yaml \\")
    print(f"        model=models/best.pt imgsz=1280 epochs=50 batch=8")
    if counts["train"] < 50:
        print(f"\n  !! only {counts['train']} training images. Enough to prove the")
        print(f"     shape shifts in the right direction; NOT enough to ship.")
        print(f"     Expect to need a few hundred frames for a real model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
