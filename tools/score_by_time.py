#!/usr/bin/env python3
"""Score MOT predictions against ground truth by TIMESTAMP, not frame index.

WHY THIS EXISTS
    gt.txt is MOT format: frame numbers, not seconds. Those frame numbers are
    an artefact of the sampling rate of the run that produced them -- quick100
    came from profE at 7.5 analysed fps. Re-run the pipeline at a different
    fps and every frame is renumbered, so the ground truth silently stops
    lining up and the score is garbage rather than obviously broken.

    That makes the single largest notebook drift -- FPS_TARGET 15 -> 8 --
    untestable against the only real labels we have. This converts both sides
    to seconds and matches on time, so any two runs at any two frame rates can
    be scored against the same labels.

MAPPING (measured, from label_pkg/quick100/manifest.json + profE's log)
    profE: eval window t=300..480s, 1352 analysed frames => 7.5 fps
    quick100 = profE frames 301..400, renumbered 1..100
    so gt frame k  ->  t = 300 + (299 + k) / 7.5     (k is 1-based)
                       k=1   -> 340.0 s
                       k=100 -> 353.2 s

TOLERANCE
    Two runs at different fps never sample the same instant. A prediction is
    matched to a gt frame if it is the CLOSEST in time and within tol_s.
    Default tol_s is half the coarser run's frame interval, so at most one
    prediction frame can claim each gt frame.
"""
from __future__ import annotations
import argparse, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kevacv.eval_harness import load_mot, score_sequence

GT_T0, GT_FPS, GT_OFFSET = 300.0, 7.5, 299


def gt_frame_to_t(k, t0=GT_T0, fps=GT_FPS, offset=GT_OFFSET):
    return t0 + (offset + k) / fps


def pred_frame_to_t(f, t0, fps):
    """Prediction frames are 1-based over the run's own eval window."""
    return t0 + (f - 1) / fps


def retime(gt, pred, pred_t0, pred_fps, tol_s=None):
    """-> (gt, pred_renumbered_into_gt_frame_space, stats)"""
    if tol_s is None:
        tol_s = 0.5 / min(GT_FPS, pred_fps)
    gt_times = {k: gt_frame_to_t(k) for k in gt}
    out, claimed, unmatched = {}, {}, 0
    for f, rows in pred.items():
        t = pred_frame_to_t(f, pred_t0, pred_fps)
        best, best_d = None, None
        for k, gt_t in gt_times.items():
            d = abs(gt_t - t)
            if best_d is None or d < best_d:
                best, best_d = k, d
        if best is None or best_d > tol_s:
            unmatched += 1
            continue
        # one prediction frame per gt frame: keep the temporally closest
        if best in claimed and claimed[best] <= best_d:
            continue
        claimed[best] = best_d
        out[best] = rows
    return gt, out, {"tol_s": round(tol_s, 4), "gt_frames": len(gt),
                     "pred_frames_in": len(pred), "matched": len(out),
                     "pred_frames_dropped": unmatched}


def main():
    import os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("gt")
    ap.add_argument("pred")
    ap.add_argument("--pred-t0", type=float, required=True,
                    help="start of the prediction run's eval window, seconds")
    ap.add_argument("--pred-fps", type=float, required=True,
                    help="ANALYSED fps of the prediction run (not native)")
    ap.add_argument("--tol-s", type=float, default=None)
    ap.add_argument("--json", help="write the score here")
    ap.add_argument("--force", action="store_true",
                    help="score even against a REFUSED reference (the number "
                         "is not accuracy; do not quote it)")
    a = ap.parse_args()

    # GATE THE SCORE ON THE REFERENCE.
    #
    # A bad reference does not look like an error, it looks like a result --
    # gt.txt (copy-forward, 14 unique boxes over 600 rows) produced "HOTA
    # 0.4762, +107%" twice before anyone re-checked it. So the check runs
    # here, before the number exists, not after somebody notices.
    from gt_validate import validate as _gtval
    _code, _fail, _warn, _info = _gtval(a.gt)
    for _i in _info:
        print(f"  gt: {_i}")
    for _w in _warn:
        print(f"  gt !  {_w}")
    for _f in _fail:
        print(f"  gt ✗  {_f}")
    if _code == 2 and not a.force:
        print("  REFUSED: this reference cannot produce a meaningful score. "
              "Pass --force only if you know exactly why you want it anyway, "
              "and never quote the result as accuracy.")
        return 2

    gt, pred = load_mot(a.gt), load_mot(a.pred)
    if not gt:
        sys.exit(f"{a.gt} empty or unparseable")
    gt2, pred2, stats = retime(gt, pred, a.pred_t0, a.pred_fps, a.tol_s)
    print(f"  time-matched: {stats}")
    if stats["matched"] < 0.5 * stats["gt_frames"]:
        print(f"  !! only {stats['matched']}/{stats['gt_frames']} gt frames got "
              f"a prediction within {stats['tol_s']}s. Check --pred-t0/--pred-fps "
              f"-- a wrong mapping scores as catastrophic failure, not as an error.")

    # ALIGNMENT SWEEP, not just a count.
    #
    # A count of matched frames is NOT enough: a wrong --pred-t0 of 300s
    # matched 100/100 gt frames -- at the wrong instants -- and scored
    # HOTA 0.078 / TP 3, which reads as a catastrophic model rather than as
    # operator error. The correct offset scored 0.259. So probe neighbouring
    # offsets and shout if one of them fits materially better.
    _probe = {}
    # The range must cover a WHOLE-WINDOW offset, not just jitter: the real
    # error was -300s (predictions started at t=0, --pred-t0 said 300), and a
    # +/-60s sweep sailed straight past it.
    for _d in (-1800.0, -900.0, -600.0, -300.0, -120.0, -60.0, -30.0, -10.0,
               0.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 900.0, 1800.0):
        _, _p, _ = retime(gt, pred, a.pred_t0 + _d, a.pred_fps, a.tol_s)
        if not _p:
            continue
        _probe[_d] = score_sequence(gt, _p).get("TP", 0)
    if _probe:
        _best_d = max(_probe, key=_probe.get)
        _here = _probe.get(0.0, 0)
        if _best_d != 0.0 and _probe[_best_d] > max(_here * 1.5, _here + 10):
            print(f"  !! ALIGNMENT SUSPECT: offsetting --pred-t0 by {_best_d:+.0f}s "
                  f"raises TP from {_here} to {_probe[_best_d]}. The frame->time "
                  f"mapping is probably wrong; fix it before reading the score "
                  f"as a model result.")
            print(f"     probe (offset -> TP): "
                  f"{ {k: v for k, v in sorted(_probe.items())} }")
    r = score_sequence(gt2, pred2)
    for k in ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "precision", "recall",
              "TP", "FP", "FN", "ID_switches", "n_gt_ids", "n_pr_ids"):
        v = r.get(k)
        if isinstance(v, float):
            print(f"  {k:<14} {v:.4f}")
        elif v is not None:
            print(f"  {k:<14} {v}")
    if a.json:
        json.dump({**{k: r[k] for k in r if isinstance(r[k], (int, float))},
                   "_timematch": stats}, open(a.json, "w"), indent=1)
        print(f"  -> {a.json}")


if __name__ == "__main__":
    sys.exit(main() or 0)
