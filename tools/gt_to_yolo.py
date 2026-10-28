#!/usr/bin/env python3
"""gt_to_yolo.py — turn hand-labelled ground truth into a YOLO training set.

WHY THIS IS THE HIGHEST-CEILING ITEM
    Every detector this pipeline has tried was trained on somebody else's
    footage. yolo11x is COCO -- people photographed from eye level. best.pt is
    CrowdHuman -- pedestrians on streets, height/width ~2.5. This camera is an
    oblique ceiling fisheye in infrared, where a standing person measures ~1.14.
    Measured on 2026-08-26, CrowdHuman as the primary detector scored recall
    0.44 against ground truth where COCO scored 0.89. Neither has ever seen
    anything like Delilah's reception desk.

    There is now training data that HAS:

        Task 1 (quiet)     3,483 boxes over 1,201 frames
        Task 3 (crowded)   5,438 boxes over 1,200 frames

    Domain-matched data beats architecture almost every time at this scale.
    A detector fine-tuned on these is the one lever with headroom left after
    the config is tuned out.

WHAT IT DOES NOT DO
    It does not train. It writes a dataset and a data.yaml, and it holds back
    a validation split BY CLIP rather than by frame -- adjacent frames of the
    same person are near-duplicates, so a random split leaks the answer into
    validation and reports a number that means nothing.

Usage:
    python tools/gt_to_yolo.py \
        --pair /data/videos/GT_CAM112_1500_1620_1920x1080_15fps.mp4 eval/gt_CAM_112.txt \
        --pair /data/videos/GT_CAM112_5750_5910.mp4 eval/crowded/gt_CAM_112.txt \
        --out /data/datasets/delilah_cam112
"""
import argparse
import collections
import sys
from pathlib import Path


def load_mot(path):
    """MOT 1.1 -> {frame: [(x, y, w, h), ...]}   (1-indexed frames)"""
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
            out[fr].append((x, y, w, h))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pair", nargs=2, action="append", metavar=("VIDEO", "GT"),
                   required=True, help="a video and its MOT ground truth")
    p.add_argument("--out", required=True)
    p.add_argument("--val-clip", type=int, default=None,
                   help="index of the --pair to hold out for validation "
                        "(default: the last one)")
    p.add_argument("--stride", type=int, default=1,
                   help="keep every Nth labelled frame")
    a = p.parse_args(argv)

    import cv2
    out = Path(a.out)
    val_idx = a.val_clip if a.val_clip is not None else len(a.pair) - 1
    counts = {}

    for ci, (video, gtp) in enumerate(a.pair):
        split = "val" if ci == val_idx else "train"
        gt = load_mot(gtp)
        if not gt:
            print(f"  !! {gtp} has no usable rows — skipped")
            continue
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise SystemExit(f"cannot open {video}")
        W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        stem = Path(video).stem[:28]
        n_img = n_box = 0
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1                                   # MOT frames are 1-indexed
            rows = gt.get(i)
            # A frame with no ground-truth box is NOT proof the frame is empty
            # -- it may simply be a frame nobody labelled. Training on it as a
            # negative teaches the detector that real people are background,
            # which is the most expensive possible mistake. Labelled frames only.
            if not rows or (i % a.stride):
                continue
            name = f"{stem}_{i:06d}"
            cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), frame)
            lines = []
            for x, y, w, h in rows:
                cx, cy = (x + w / 2) / W, (y + h / 2) / H
                nw, nh = w / W, h / H
                if not (0 <= cx <= 1 and 0 <= cy <= 1):
                    continue
                nw, nh = min(nw, 1.0), min(nh, 1.0)
                lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            (out / "labels" / split / f"{name}.txt").write_text(
                "\n".join(lines), encoding="utf-8")
            n_img += 1
            n_box += len(lines)
        cap.release()
        counts[f"{split}:{stem}"] = (n_img, n_box)
        print(f"  {split:<5} {stem:<30} {n_img:>5} images  {n_box:>6} boxes")

    (out / "data.yaml").write_text(
        f"# Delilah CAM.112 — hand-labelled in CVAT, exported by tools/gt_to_yolo.py\n"
        f"# VALIDATION IS SPLIT BY CLIP, NOT BY FRAME: adjacent frames of the same\n"
        f"# person are near-duplicates, so a random split leaks the answer into val\n"
        f"# and reports a number that means nothing.\n"
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names: [person]\n", encoding="utf-8")

    tr = sum(v[1] for k, v in counts.items() if k.startswith("train"))
    va = sum(v[1] for k, v in counts.items() if k.startswith("val"))
    print(f"\n  train {tr} boxes | val {va} boxes -> {out}/data.yaml")
    if not tr or not va:
        print("  !! one split is EMPTY. Pass at least two --pair clips so the "
              "holdout is a different clip from the training data.")
        return 1
    print("\n  next:")
    print(f"    yolo detect train model=yolo11x.pt data={out}/data.yaml \\")
    print(f"         epochs=60 imgsz=1920 batch=4 patience=15")
    print("  then A/B the result against yolo11x.pt on the SAME ground truth,")
    print("  exactly like every other change today.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
