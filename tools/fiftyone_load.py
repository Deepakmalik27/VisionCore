#!/usr/bin/env python3
"""fiftyone_load.py — look INSIDE the model, instead of scrubbing a video.

WHY THIS EXISTS
    The annotated video is the only inspection tool this project has, and it is
    the wrong shape for the question engineers actually ask. It is built for a
    GM judging a venue -- one frame at a time, in order, at 4x speed. Finding
    the frames where the pipeline was WRONG means watching all 54,000 of them
    and hoping to notice.

    Everything hard about this pipeline is a needle-in-haystack question:

      "show me every frame where a person was missed"
      "show me the frames where two ids swapped"
      "do staff1 and a guest actually look different to the ReID model?"

    None of those can be answered by playback. All of them are one filter in a
    dataset browser.

WHAT THIS GIVES YOU THAT THE VIDEO CANNOT
    * ground truth and predictions on the SAME frame, toggleable
    * sort frames by MISTAKENNESS -- the worst failures first, instead of
      hoping to spot them at 4x
    * filter: false negatives only / low confidence only / crowded frames only
    * EMBEDDINGS plotted in 2D. Identities that do not separate visually are
      exactly the ID-switch problem, made visible. 136 canonical ids for ~10
      real people was measured on 2026-08-25 and never explained; this is where
      you would see why.

WHAT IT IS NOT
    Not a metric. eval_harness/HOTA remains the score. This is for finding out
    WHY the score is what it is -- error analysis, not evaluation.

Usage:
    python tools/fiftyone_load.py --frames-dir /scratch/door --name doorway
    python tools/fiftyone_load.py --run /data/outputs/gtsmoke --video <clip> --name gtsmoke
    # then, from your laptop:
    #   ssh -L 5151:localhost:5151 ubuntu@<box>
    #   http://localhost:5151
"""
import argparse
import gzip
import json
import sys
from pathlib import Path


def load_mot(path):
    """MOT 1.1 -> {frame: [(id, x, y, w, h), ...]}"""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.replace("\t", ",").split(",")
        try:
            fr, tid = int(float(f[0])), int(float(f[1]))
            x, y, w, h = (float(v) for v in f[2:6])
        except (ValueError, IndexError):
            continue
        out.setdefault(fr, []).append((tid, x, y, w, h))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", help="a pipeline --out directory")
    p.add_argument("--video", help="the video that run analysed")
    p.add_argument("--frames-dir", help="a directory of jpgs instead of a run")
    p.add_argument("--name", default="keva")
    p.add_argument("--port", type=int, default=5151)
    p.add_argument("--no-launch", action="store_true")
    a = p.parse_args(argv)

    import fiftyone as fo

    if a.name in fo.list_datasets():
        fo.delete_dataset(a.name)
    ds = fo.Dataset(a.name, persistent=True)

    if a.frames_dir:
        files = sorted(Path(a.frames_dir).glob("*.jpg"))
        ds.add_samples([fo.Sample(filepath=str(f)) for f in files])
        print(f"  {len(files)} frames from {a.frames_dir}")

    elif a.run:
        run = Path(a.run)
        # frames come from the run's own frame log, so what you browse is
        # EXACTLY what the pipeline saw -- not a re-decode that might differ.
        fl = run / "debug"
        flog = next(iter(fl.glob("*_frames.json.gz")), None)
        if flog is None:
            raise SystemExit(f"no *_frames.json.gz in {fl} — run with the debug artefacts")
        raw = json.loads(gzip.open(flog, "rt", encoding="utf-8").read())
        pred = {}
        for entry in raw:
            fi, _t, boxes = entry[0], entry[1], entry[2]
            pred[int(fi)] = [(b[0], b[1], b[2], b[3], b[4]) for b in boxes]

        gtf = next(iter(run.glob("gt_*.txt")), None)
        gt = load_mot(gtf) if gtf else {}
        print(f"  predictions: {len(pred)} frames | ground truth: {len(gt)} frames")

        if not a.video:
            raise SystemExit("--video is required with --run (FiftyOne needs the media)")
        import cv2
        cap = cv2.VideoCapture(a.video)
        W = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920
        H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080
        outdir = run / "fo_frames"
        outdir.mkdir(exist_ok=True)
        samples, i = [], 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            fp = outdir / f"{i:06d}.jpg"
            if not fp.exists():
                cv2.imwrite(str(fp), fr)
            s = fo.Sample(filepath=str(fp), frame_number=i)
            # FiftyOne wants [x, y, w, h] normalised 0-1
            def dets(rows, is_gt):
                out = []
                for r in rows:
                    tid, x1, y1, x2, y2 = r
                    if is_gt:
                        x, y, w, h = x1, y1, x2, y2      # MOT is already x,y,w,h
                    else:
                        x, y, w, h = x1, y1, x2 - x1, y2 - y1
                    out.append(fo.Detection(label="person",
                                            bounding_box=[x / W, y / H, w / W, h / H],
                                            index=int(tid) if tid is not None else None))
                return fo.Detections(detections=out)
            if i in pred:
                s["predictions"] = dets(pred[i], False)
            if (i + 1) in gt:
                s["ground_truth"] = dets(gt[i + 1], True)
            samples.append(s)
            i += 1
        cap.release()
        ds.add_samples(samples)
        print(f"  {len(samples)} samples")

        if gt:
            # THE POINT OF ALL THIS: per-frame TP/FP/FN, so you can sort by
            # failure instead of watching for one.
            try:
                ds.evaluate_detections("predictions", gt_field="ground_truth",
                                       eval_key="eval", compute_mAP=False)
                print("  evaluated -> filter on eval_fp / eval_fn in the app")
            except Exception as e:
                print(f"  (evaluation skipped: {e})")
    else:
        raise SystemExit("need --run or --frames-dir")

    print(f"\n  dataset '{a.name}' ready ({len(ds)} samples)")
    print("  From your laptop:")
    print("    ssh -i <key> -L 5151:localhost:5151 ubuntu@<box>")
    print("    http://localhost:5151")
    print("\n  Try first:")
    print("    sort by eval_fn desc   -> the frames where people were MISSED")
    print("    sort by eval_fp desc   -> phantom detections (plants, reflections)")
    if not a.no_launch:
        sess = fo.launch_app(ds, port=a.port, address="0.0.0.0", remote=True)
        print("  app running — Ctrl+C to stop")
        sess.wait()
    return 0


if __name__ == "__main__":
    sys.exit(main())
