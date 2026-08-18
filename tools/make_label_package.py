#!/usr/bin/env python3
"""make_label_package.py — assemble a CVAT-ready label package from run output.

WHY THIS EXISTS
    Every SUMMARY.txt this project has produced says

        "HOTA NOT MEASURED — every accuracy claim below is an estimate"

    and the 90% accuracy goal is therefore unmeasurable, not merely unmet.
    tools/gt_kit.py and kevacv/eval_harness.py were built to close that, and
    both work (tests/test_eval_scorer.py proves the scorer against planted
    errors). They have still never been run on real output, for a dull reason:

        gt_kit.seed() wants        pkg/predictions.txt  +  pkg/images/*.jpg
        a run actually writes      debug/<cam>_predictions.txt
                                   eval_frames/<cam>/*.jpg

    Different names, different directories, and — the part that silently
    breaks everything — different frame numbering. predictions.txt is
    1-INDEXED (the writer emits idx+1, MOT convention). The exported JPEGs are
    named with the RAW 0-indexed frame number. So

        eval_frames/CAM.112/0002253.jpg   IS   prediction frame 2254

    Off by one, on every box, for the whole window. Nothing would crash; the
    boxes would simply sit on the wrong frame and the resulting HOTA would be
    quietly, confidently wrong. gt_kit's own guard would have aborted the seed
    anyway, because predictions span 1..4505 while the package holds 1,352
    images — which is exactly why nobody ever got a number out of it.

WHAT THIS DOES
    Builds a self-contained package whose frames are renumbered 1..N to match
    CVAT's own ordering, carrying the offset in a manifest so a scored gt.txt
    can always be traced back to real timestamps in the source video.

USAGE
    python3 tools/make_label_package.py output/profE --out label_pkg/profE
    python3 tools/gt_kit.py seed label_pkg/profE
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def find_inputs(run_dir):
    run = Path(run_dir)
    preds = sorted((run / "debug").glob("*_predictions.txt"))
    if not preds:
        sys.exit(f"no *_predictions.txt under {run/'debug'} — run with "
                 f"--eval-export, or check the run completed")
    pred = preds[0]
    cam = pred.name.replace("_predictions.txt", "")
    img_dir = run / "eval_frames" / cam
    if not img_dir.is_dir():
        cands = [d for d in (run / "eval_frames").glob("*") if d.is_dir()]
        if len(cands) != 1:
            sys.exit(f"cannot locate eval frames for {cam} under "
                     f"{run/'eval_frames'}")
        img_dir = cands[0]
    return cam, pred, img_dir


def build(run_dir, out_dir, copy_images=True):
    cam, pred_path, img_dir = find_inputs(run_dir)
    imgs = sorted(img_dir.glob("*.jpg"))
    if not imgs:
        sys.exit(f"no jpgs in {img_dir}")

    # Image stem -> RAW 0-indexed frame number. Predictions are 1-indexed.
    # This +1 is the whole reason this script exists; do not "simplify" it.
    raw_of = {}
    for p in imgs:
        try:
            raw_of[p] = int(p.stem)
        except ValueError:
            sys.exit(f"image name is not a frame number: {p.name}")
    raw_sorted = sorted(raw_of.values())
    want_pred_frames = {r + 1 for r in raw_sorted}

    # new sequential frame number, 1..N, in image order (what CVAT will use)
    new_of_pred = {r + 1: i + 1 for i, r in enumerate(raw_sorted)}

    kept, skipped = [], 0
    for line in pred_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        try:
            f = int(parts[0])
        except ValueError:
            continue
        if f not in want_pred_frames:
            skipped += 1
            continue
        parts[0] = str(new_of_pred[f])
        kept.append(",".join(parts))

    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(raw_sorted):
        src = next(p for p, v in raw_of.items() if v == r)
        dst = out / "images" / f"{i+1:07d}.jpg"
        if copy_images and not dst.exists():
            shutil.copy2(src, dst)
    (out / "predictions.txt").write_text("\n".join(kept) + "\n", encoding="utf-8")

    manifest = {
        "camera": cam,
        "source_run": str(run_dir),
        "n_images": len(raw_sorted),
        "n_pred_rows": len(kept),
        "pred_rows_outside_window": skipped,
        # everything needed to map a scored gt.txt back to the real video
        "raw_frame_first": raw_sorted[0],
        "raw_frame_last": raw_sorted[-1],
        "new_frame_first": 1,
        "new_frame_last": len(raw_sorted),
        "mapping": "new_frame = index of raw frame in sorted order, 1-based; "
                   "prediction frame = raw frame + 1 (MOT is 1-indexed)",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                       encoding="utf-8")

    print(f"package: {out}")
    print(f"  camera            {cam}")
    print(f"  images            {len(raw_sorted)}  "
          f"(raw {raw_sorted[0]}..{raw_sorted[-1]} -> 1..{len(raw_sorted)})")
    print(f"  prediction boxes  {len(kept)} kept, {skipped} outside the window")
    if not kept:
        print("  !! NO predictions landed in this window. Either the eval "
              "window and the frame log disagree, or the +1 convention "
              "changed. Do not label against this.")
        return 1
    ids = len({ln.split(',')[1] for ln in kept})
    print(f"  track ids         {ids}")
    print(f"\nnext:  python3 tools/gt_kit.py seed {out}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="e.g. output/profE")
    ap.add_argument("--out", required=True, help="package directory to create")
    ap.add_argument("--no-copy", action="store_true",
                    help="write predictions/manifest only, skip copying jpgs")
    a = ap.parse_args()
    return build(a.run_dir, a.out, copy_images=not a.no_copy)


if __name__ == "__main__":
    sys.exit(main())
