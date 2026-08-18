"""eval_harness.py — PHASE 2. Turn "we think it improved" into a number.

Standalone and dependency-free (numpy only). Import it from the notebook, or run
it from a shell on exported artifacts. Nothing here touches the pipeline; it
only ever reads predictions and ground truth.

WHY THIS EXISTS
    Every accuracy change from v42 to v55 was judged by watching the video and
    feeling better about it. Thirty reasonable changes, none attributable. This
    file is the gate: from here on, a change ships only if a scored delta says
    it helped.

WHAT IT MEASURES
    HOTA = sqrt(DetA x AssA)   the metric the field reports
      DetA  did we FIND the people          -> detection problem
      AssA  did we KEEP them the same person -> association problem
    These fail independently and need opposite fixes, which is exactly why a
    single accuracy number was never enough to tell us what to do next.
    Also IDF1, MOTA, precision, recall, ID switches.

EVERYTHING IS DEBUGGABLE — the point is never to hand back a bare number:
    * score_sequence() returns per-frame TP/FP/FN and every ID switch with its
      frame and the two ids involved
    * explain() prints the worst frames and what went wrong in each
    * dump_errors_csv() writes every error with its box, to overlay on frames
    * compare() prints an A/B delta between two runs AND the config diff, so a
      change is never confused with noise from a different setting
    * self-validated: test_eval_harness.py checks the metric code against cases
      whose answers are known by construction (perfect tracker, all-swapped,
      half-missed, ...) - a metric you have not tested is not a measurement

HOTA IMPLEMENTATION NOTE (read before quoting the number)
    This is the TrackEval algorithm, reimplemented: global Jaccard alignment
    over the whole sequence, association-aware Hungarian per frame, alpha swept
    0.05..0.95. It is validated against constructed cases with analytically
    known answers. It is NOT the official TrackEval binary. For A/B comparison
    (our main use) any small systematic bias cancels between the two runs. If a
    number ever leaves this project, re-run it through the official TrackEval.
"""
from __future__ import annotations

from .log import get_logger, stage, banner

_log = get_logger("eval_harness")


import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _lsa
    _HAVE_SCIPY = True
except Exception:                                    # pragma: no cover
    _HAVE_SCIPY = False

# Rounded on purpose: np.arange with a float step accumulates error, and the
# alpha==0.5 lookup below (which produces every per-frame debug detail) would
# then silently find nothing.
ALPHAS = np.round(np.arange(0.05, 1.0, 0.05), 2)


# ---------------------------------------------------------------------------
# MOT I/O
# ---------------------------------------------------------------------------
def load_mot(path):
    """MOT 1.1 / MOT16: frame,id,x,y,w,h,conf,...  -> {frame: [(id,x,y,w,h)]}

    Lenient on purpose: CVAT, MOTChallenge and our own exporter all differ
    slightly in trailing columns and in whether conf is present. A ground-truth
    file that silently fails to parse would be the worst possible bug here, so
    parse failures are collected and reported rather than swallowed.
    """
    out, bad = defaultdict(list), []
    p = Path(path)
    for ln, line in enumerate(p.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.replace("\t", ",").split(",")
        try:
            fr, tid = int(float(f[0])), int(float(f[1]))
            x, y, w, h = (float(v) for v in f[2:6])
        except (ValueError, IndexError):
            bad.append((ln, line[:80]))
            continue
        # MOT gt files use column 7 as a "consider this box" flag; 0 = ignore.
        if len(f) > 6:
            try:
                if float(f[6]) == 0 and len(f) >= 9:
                    continue
            except ValueError:
                pass
        if w <= 0 or h <= 0:
            bad.append((ln, f"non-positive box: {line[:60]}"))
            continue
        out[fr].append((tid, x, y, w, h))
    if bad:
        _log.error(f"  !! {p.name}: {len(bad)} unparseable line(s), first few:")
        for ln, txt in bad[:3]:
            _log.info(f"     line {ln}: {txt}")
    return dict(out)


def write_mot(path, rows):
    """rows: iterable of (frame, id, x, y, w, h)."""
    Path(path).write_text("\n".join(
        f"{int(fr)},{int(tid)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1"
        for fr, tid, x, y, w, h in rows))
    return Path(path)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def iou_matrix(a, b):
    """a, b: lists of (x, y, w, h) -> IoU matrix. Empty-safe."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=float)
    A = np.asarray(a, dtype=float)
    B = np.asarray(b, dtype=float)
    ax1, ay1 = A[:, 0][:, None], A[:, 1][:, None]
    ax2, ay2 = ax1 + A[:, 2][:, None], ay1 + A[:, 3][:, None]
    bx1, by1 = B[:, 0][None, :], B[:, 1][None, :]
    bx2, by2 = bx1 + B[:, 2][None, :], by1 + B[:, 3][None, :]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    union = (A[:, 2] * A[:, 3])[:, None] + (B[:, 2] * B[:, 3])[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def _hungarian(cost):
    """Maximise-free wrapper. scipy if present, else a small O(n^3) fallback so
    the harness never silently declines to score."""
    if _HAVE_SCIPY:
        r, c = _lsa(cost)
        return list(r), list(c)
    n, m = cost.shape                       # pragma: no cover - fallback path
    size = max(n, m)
    C = np.full((size, size), cost.max() + 1.0 if cost.size else 1.0)
    C[:n, :m] = cost
    used_c, rows, cols = set(), [], []
    order = np.argsort(C.min(axis=1))
    for i in order:                          # greedy fallback, documented as such
        j = int(np.argmin(np.where([k in used_c for k in range(size)],
                                   np.inf, C[i])))
        if j < m and i < n:
            rows.append(int(i)); cols.append(j)
        used_c.add(j)
    return rows, cols


# ---------------------------------------------------------------------------
# the metric
# ---------------------------------------------------------------------------
def score_sequence(gt, pr, alphas=ALPHAS, keep_detail=True):
    """HOTA / DetA / AssA / IDF1 / MOTA + everything needed to debug them.

    gt, pr: {frame: [(id, x, y, w, h)]}
    """
    frames = sorted(set(gt) | set(pr))
    gt_count = defaultdict(int)      # gt_id -> frames present
    pr_count = defaultdict(int)
    pot = defaultdict(float)         # (gt_id, pr_id) -> summed similarity

    # pass 1 — global alignment. HOTA matches with knowledge of how often two
    # ids co-occur across the WHOLE sequence, which is what stops a one-frame
    # coincidence from being treated like a real identity link.
    sims = {}
    for f in frames:
        g, p = gt.get(f, []), pr.get(f, [])
        for tid, *_ in g:
            gt_count[tid] += 1
        for tid, *_ in p:
            pr_count[tid] += 1
        S = iou_matrix([b[1:] for b in g], [b[1:] for b in p])
        sims[f] = S
        for i, (gi, *_) in enumerate(g):
            for j, (pj, *_) in enumerate(p):
                if S[i, j] > 0:
                    pot[(gi, pj)] += S[i, j]

    align = {}
    for (gi, pj), s in pot.items():
        denom = gt_count[gi] + pr_count[pj] - s
        align[(gi, pj)] = s / denom if denom > 0 else 0.0

    res = {"per_alpha": [], "frames": len(frames)}
    detail_at_half = None

    for a in alphas:
        TP = FP = FN = 0
        tpa = defaultdict(int)                  # (gt,pr) -> matched frames
        matched_gt = defaultdict(int)
        matched_pr = defaultdict(int)
        per_frame, last_match, idsw_list = [], {}, []
        for f in frames:
            g, p = gt.get(f, []), pr.get(f, [])
            S = sims[f]
            pairs = []
            if len(g) and len(p):
                # association-aware score, exactly as HOTA specifies
                score = np.zeros_like(S)
                for i, (gi, *_) in enumerate(g):
                    for j, (pj, *_) in enumerate(p):
                        score[i, j] = S[i, j] * align.get((gi, pj), 0.0)
                ri, ci = _hungarian(-score)
                for i, j in zip(ri, ci):
                    if i < len(g) and j < len(p) and S[i, j] >= a:
                        pairs.append((i, j))
            f_tp = len(pairs)
            f_fn = len(g) - f_tp
            f_fp = len(p) - f_tp
            TP += f_tp; FN += f_fn; FP += f_fp
            for i, j in pairs:
                gi, pj = g[i][0], p[j][0]
                tpa[(gi, pj)] += 1
                matched_gt[gi] += 1
                matched_pr[pj] += 1
                if gi in last_match and last_match[gi] != pj:
                    idsw_list.append({"frame": f, "gt_id": gi,
                                      "was": last_match[gi], "now": pj})
                last_match[gi] = pj
            if keep_detail:
                per_frame.append({"frame": f, "tp": f_tp, "fp": f_fp, "fn": f_fn,
                                  "n_gt": len(g), "n_pr": len(p)})

        det_a = TP / (TP + FN + FP) if (TP + FN + FP) else 0.0
        ass_sum = 0.0
        for (gi, pj), c in tpa.items():
            fna = gt_count[gi] - c
            fpa = pr_count[pj] - c
            ass_sum += c * (c / (c + fna + fpa)) if (c + fna + fpa) else 0.0
        ass_a = ass_sum / TP if TP else 0.0
        res["per_alpha"].append({"alpha": round(float(a), 2), "DetA": det_a,
                                 "AssA": ass_a, "HOTA": math.sqrt(det_a * ass_a),
                                 "TP": TP, "FP": FP, "FN": FN,
                                 "IDSW": len(idsw_list)})
        if abs(a - 0.5) < 1e-6:
            detail_at_half = {"per_frame": per_frame, "idsw": idsw_list,
                              "TP": TP, "FP": FP, "FN": FN, "tpa": dict(tpa)}

    res["HOTA"] = float(np.mean([r["HOTA"] for r in res["per_alpha"]]))
    res["DetA"] = float(np.mean([r["DetA"] for r in res["per_alpha"]]))
    res["AssA"] = float(np.mean([r["AssA"] for r in res["per_alpha"]]))

    # IDF1 / MOTA at the conventional alpha=0.5
    h = next(r for r in res["per_alpha"] if abs(r["alpha"] - 0.5) < 1e-6)
    n_gt = sum(gt_count.values())
    res["precision"] = h["TP"] / (h["TP"] + h["FP"]) if (h["TP"] + h["FP"]) else 0.0
    res["recall"] = h["TP"] / (h["TP"] + h["FN"]) if (h["TP"] + h["FN"]) else 0.0
    res["ID_switches"] = h["IDSW"]
    res["MOTA"] = 1.0 - (h["FN"] + h["FP"] + h["IDSW"]) / n_gt if n_gt else 0.0
    res["TP"], res["FP"], res["FN"] = h["TP"], h["FP"], h["FN"]

    # IDF1 = identity F1 over the best global id-to-id assignment
    if tpa:
        gids = sorted(gt_count); pids = sorted(pr_count)
        M = np.zeros((len(gids), len(pids)))
        for (gi, pj), c in detail_at_half["tpa"].items():
            M[gids.index(gi), pids.index(pj)] = c
        ri, ci = _hungarian(-M)
        idtp = sum(M[i, j] for i, j in zip(ri, ci)
                   if i < len(gids) and j < len(pids))
        idfn = n_gt - idtp
        idfp = sum(pr_count.values()) - idtp
        res["IDF1"] = 2 * idtp / (2 * idtp + idfn + idfp) if idtp else 0.0
    else:
        res["IDF1"] = 0.0

    res["n_gt_boxes"] = n_gt
    res["n_pr_boxes"] = sum(pr_count.values())
    res["n_gt_ids"] = len(gt_count)
    res["n_pr_ids"] = len(pr_count)
    res["_detail"] = detail_at_half
    return res


# ---------------------------------------------------------------------------
# debugging surface — a bare number is never the deliverable
# ---------------------------------------------------------------------------
def explain(res, label="", worst_n=8):
    """Print the number AND where it came from. Reads the diagnosis out loud so
    the next action is obvious instead of a guess."""
    bar = "=" * 74
    _log.info(bar)
    _log.info(f"  SCORE{(' · ' + label) if label else ''}")
    _log.info(bar)
    _log.info(f"    HOTA   {res['HOTA']:.4f}   = sqrt(DetA x AssA)")
    _log.info(f"    DetA   {res['DetA']:.4f}   did we FIND the people")
    _log.info(f"    AssA   {res['AssA']:.4f}   did we KEEP them the same person")
    _log.info(f"    IDF1   {res['IDF1']:.4f}   MOTA {res['MOTA']:.4f}")
    _log.info(f"    prec   {res['precision']:.4f}   recall {res['recall']:.4f}")
    _log.info(f"    TP {res['TP']}  FP {res['FP']}  FN {res['FN']}  "
          f"ID switches {res['ID_switches']}")
    _log.info(f"    gt: {res['n_gt_boxes']} boxes / {res['n_gt_ids']} people   "
          f"pred: {res['n_pr_boxes']} boxes / {res['n_pr_ids']} identities")

    # the actionable part
    d, a = res["DetA"], res["AssA"]
    _log.info("    " + "-" * 68)
    if d < 0.4:
        _log.info("    -> DETECTION is the bottleneck. Raise resolution / lower the conf")
        _log.info("       floor / use SAHI. Tracker work will not help yet.")
    elif a < d - 0.12:
        _log.info("    -> ASSOCIATION is the bottleneck. Raise fps, fix the re-id gate,")
        _log.info("       upgrade the tracker. More detections will not help.")
    elif d < 0.6:
        _log.info("    -> Both are mediocre; detection is still the cheaper win.")
    else:
        _log.info("    -> Balanced. Further gains need a better detector AND tracker.")
    if res["n_pr_ids"] > res["n_gt_ids"] * 1.5:
        _log.info(f"    -> {res['n_pr_ids']} predicted identities for {res['n_gt_ids']} real "
              f"people: FRAGMENTATION. Every unique-people count is inflated.")
    elif res["n_pr_ids"] * 1.5 < res["n_gt_ids"]:
        _log.info(f"    -> only {res['n_pr_ids']} identities for {res['n_gt_ids']} real people: "
              f"OVER-MERGING. Counts are deflated and journeys are fiction.")

    det = res.get("_detail") or {}
    pf = sorted(det.get("per_frame", []), key=lambda r: -(r["fp"] + r["fn"]))[:worst_n]
    if pf:
        _log.info("    " + "-" * 68)
        _log.info(f"    worst frames (alpha=0.5)   frame   gt  pred   TP  FP  FN")
        for r in pf:
            if r["fp"] + r["fn"] == 0:
                continue
            _log.info(f"    {'':26s}{r['frame']:>6d}{r['n_gt']:>5d}{r['n_pr']:>6d}"
                  f"{r['tp']:>5d}{r['fp']:>4d}{r['fn']:>4d}")
    sw = det.get("idsw", [])[:worst_n]
    if sw:
        _log.info(f"    ID switches (first {len(sw)}):")
        for s in sw:
            _log.info(f"      frame {s['frame']}: real person {s['gt_id']} was "
                  f"id {s['was']}, became id {s['now']}")
    _log.info(bar)
    return res


def dump_errors_csv(gt, pr, res, path, alpha=0.5):
    """Every FP and FN with its box, so they can be drawn on the frames. This is
    what turns 'recall 0.62' into 'we are missing the far-left table'."""
    rows = []
    for f in sorted(set(gt) | set(pr)):
        g, p = gt.get(f, []), pr.get(f, [])
        S = iou_matrix([b[1:] for b in g], [b[1:] for b in p])
        gm, pm = set(), set()
        if len(g) and len(p):
            ri, ci = _hungarian(-S)
            for i, j in zip(ri, ci):
                if i < len(g) and j < len(p) and S[i, j] >= alpha:
                    gm.add(i); pm.add(j)
        for i, b in enumerate(g):
            if i not in gm:
                rows.append([f, "FN", b[0], *[round(v, 1) for v in b[1:]]])
        for j, b in enumerate(p):
            if j not in pm:
                rows.append([f, "FP", b[0], *[round(v, 1) for v in b[1:]]])
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "kind", "id", "x", "y", "w", "h"])
        w.writerows(rows)
    _log.info(f"  -> {Path(path).name}  ({len(rows)} errors: overlay these on the frames)")
    return Path(path)


def save_baseline(res, path, config=None, label=""):
    """Freeze a score so the NEXT change has something to be compared against."""
    payload = {"label": label,
               "metrics": {k: v for k, v in res.items() if not k.startswith("_")},
               "config": config or {}}
    Path(path).write_text(json.dumps(payload, indent=2, default=str))
    _log.info(f"  -> baseline saved: {Path(path).name}")
    return Path(path)


def compare(before, after, label_a="before", label_b="after"):
    """A/B delta AND the config diff. Without the config diff a delta is not
    attributable - that is the exact trap CALIBRATION_AUTO_APPLY put us in."""
    ma = before.get("metrics", before)
    mb = after.get("metrics", after)
    ca = before.get("config", {}) or {}
    cb = after.get("config", {}) or {}
    _log.info("=" * 74)
    _log.info(f"  A/B   {label_a}  ->  {label_b}")
    _log.info("=" * 74)
    _log.info(f"    {'metric':14s}{label_a:>12s}{label_b:>12s}{'delta':>12s}")
    verdicts = []
    for k in ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "precision", "recall"):
        if k not in ma or k not in mb:
            continue
        d = mb[k] - ma[k]
        flag = "  ++" if d > 0.01 else ("  --" if d < -0.01 else "    ")
        _log.info(f"    {k:14s}{ma[k]:>12.4f}{mb[k]:>12.4f}{d:>+12.4f}{flag}")
        verdicts.append((k, d))
    for k in ("ID_switches", "FP", "FN", "n_pr_ids"):
        if k in ma and k in mb:
            _log.info(f"    {k:14s}{ma[k]:>12d}{mb[k]:>12d}{mb[k]-ma[k]:>+12d}"
                  + ("  ++" if mb[k] < ma[k] else ("  --" if mb[k] > ma[k] else "")))
    changed = {k: (ca.get(k), cb.get(k)) for k in set(ca) | set(cb)
               if ca.get(k) != cb.get(k)}
    _log.info("    " + "-" * 68)
    if changed:
        _log.info(f"    config changed ({len(changed)}):")
        for k, (x, y) in sorted(changed.items()):
            _log.info(f"      {k}: {x}  ->  {y}")
    else:
        _log.info("    config identical — any delta here is run-to-run NOISE, not a fix.")
    dh = mb.get("HOTA", 0) - ma.get("HOTA", 0)
    _log.info("    " + "-" * 68)
    if abs(dh) < 0.005:
        _log.info(f"    VERDICT: HOTA moved {dh:+.4f} — inside noise. Not a win, do not ship.")
    elif dh > 0:
        _log.info(f"    VERDICT: HOTA {dh:+.4f}. Real improvement. Keep it.")
    else:
        _log.info(f"    VERDICT: HOTA {dh:+.4f}. This made it WORSE. Revert.")
    _log.info("=" * 74)
    return {"delta_HOTA": dh, "config_changed": changed}


# ---------------------------------------------------------------------------
# per-condition scoring — day / night / IR-switch must be scored SEPARATELY
# ---------------------------------------------------------------------------
def score_conditions(pairs, out_dir=None):
    """pairs: {condition_name: (gt_path, pred_path)}

    One aggregate number hides the thing we most need to see. Our night path
    disables colour evidence entirely and has never been measured; averaged in
    with daylight it would look fine while being broken.
    """
    results = {}
    for name, (gtp, prp) in pairs.items():
        gt, pr = load_mot(gtp), load_mot(prp)
        if not gt:
            _log.error(f"  !! {name}: ground truth is empty ({gtp}) — SKIPPED")
            continue
        res = explain(score_sequence(gt, pr), label=name)
        results[name] = res
        if out_dir:
            out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
            dump_errors_csv(gt, pr, res, out / f"{name}_errors.csv")
            save_baseline(res, out / f"{name}_score.json", label=name)
    if len(results) > 1:
        _log.info("=" * 74)
        _log.info("  ACROSS CONDITIONS  (a single average would have hidden this)")
        _log.info("=" * 74)
        _log.info(f"    {'condition':22s}{'HOTA':>9s}{'DetA':>9s}{'AssA':>9s}{'IDF1':>9s}")
        for n, r in results.items():
            _log.info(f"    {n:22s}{r['HOTA']:>9.4f}{r['DetA']:>9.4f}"
                  f"{r['AssA']:>9.4f}{r['IDF1']:>9.4f}")
        worst = min(results.items(), key=lambda kv: kv[1]["HOTA"])
        best = max(results.items(), key=lambda kv: kv[1]["HOTA"])
        gap = best[1]["HOTA"] - worst[1]["HOTA"]
        _log.info("    " + "-" * 68)
        _log.info(f"    weakest condition: {worst[0]} (HOTA {worst[1]['HOTA']:.4f}), "
              f"{gap:.4f} below {best[0]}")
        if gap > 0.10:
            _log.info(f"    -> that gap is where the error lives. Fix {worst[0]} before "
                  f"tuning anything that is already working.")
    return results


if __name__ == "__main__":                            # pragma: no cover
    import sys
    if len(sys.argv) >= 3:
        explain(score_sequence(load_mot(sys.argv[1]), load_mot(sys.argv[2])),
                label=f"{Path(sys.argv[1]).name} vs {Path(sys.argv[2]).name}")
    else:
        _log.info(__doc__)
        _log.info("usage: python eval_harness.py <gt.txt> <predictions.txt>")
