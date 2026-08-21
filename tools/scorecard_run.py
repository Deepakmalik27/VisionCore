#!/usr/bin/env python3
"""Build a RUN SCORECARD from a finished run's outputs.

Standalone so it works on runs that already happened. The same kevacv.scorecard
functions are meant to be called from inside the pipeline, so a live run emits
this block itself rather than relying on anyone to run a tool afterwards.

Usage:  python3 tools/scorecard_run.py <run> [<run> ...] [--compare]
"""
from __future__ import annotations

import collections
import gzip
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kevacv import scorecard as SC   # noqa: E402

CAM = "CAM.112"


def _log_for(run):
    for p in (f"/tmp/{run}.log", f"output/{run}/run.log"):
        if os.path.exists(p):
            return open(p, errors="ignore").read()
    return ""


def _funnel_from_log(txt):
    """The funnel table the engine already prints -> {stage: drop % of raw}."""
    out, raw = collections.OrderedDict(), None
    for line in txt.splitlines():
        m = re.search(r"\|\s+(\S.*?)\s{2,}(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)%\s+(\d+)\s*$", line)
        if not m:
            continue
        stage, _in, _out, _drop, pct, _emptied = m.groups()
        stage = stage.strip()
        if stage in ("stage", "SURVIVED"):
            continue
        out[stage] = float(pct)
    return out


def _plane_from_log(txt):
    m = None
    for mm in re.finditer(r"implied camera height ([\d.]+) m, horizon at row (-?\d+)", txt):
        m = mm
    if not m:
        return {}
    return {"ok": True, "camera_h_m": float(m.group(1)),
            "horizon_row": int(m.group(2))}


def _counts_from_log(txt):
    c = {}
    m = re.search(r"arrivals: line=(\d+) region=(\d+) trust=(\w+)", txt)
    if m:
        c.update(line=int(m.group(1)), region=int(m.group(2)), trust=m.group(3))
    m = re.search(r"guests=(\d+)", txt)
    if m:
        c["guests"] = int(m.group(1))
    return c


def _changed_from_log(txt):
    ch = {}
    for line in txt.splitlines():
        m = re.match(r"\s{2}([A-Z_0-9]+)\s+(.+?)\s+\[analysis\.[a-z_0-9]+\]\s*$", line)
        if m and "(unchanged)" not in line:
            ch[m.group(1)] = m.group(2).strip()
    return ch


def build(run):
    txt = _log_for(run)
    fp = f"output/{run}/debug/{CAM}_frames.json.gz"
    tracks, t_first, t_last = {}, None, None
    if os.path.exists(fp):
        tr = collections.defaultdict(list)
        for _fi, t, dets in json.load(gzip.open(fp, "rt")):
            for tid, x1, y1, x2, y2 in dets:
                tr[tid].append((float(t), float(x1), float(y1), float(x2), float(y2)))
        for k in tr:
            tr[k].sort()
        tracks = dict(tr)
        ts = [p[0][0] for p in tracks.values()] + [p[-1][0] for p in tracks.values()]
        t_first, t_last = (min(ts), max(ts)) if ts else (None, None)

    cp = f"output/{run}/debug/{CAM}_crossings.json"
    crossings = []
    if os.path.exists(cp):
        d = json.load(open(cp))
        crossings = d if isinstance(d, list) else d.get("crossings", d)

    ch = _changed_from_log(txt)
    aw = float(ch.get("ANALYSIS_MAX_W", "1920 -> 1920").split("->")[-1].strip() or 1920)
    imgsz = float(ch.get("YOLO_IMGSZ", "1280 -> 1280").split("->")[-1].strip() or 1280)
    zone_x = 1500.0 * (aw / 1920.0)

    card = {
        "run": run,
        "build": {"build_id": (re.search(r"BUILD (\w+)", txt) or [None, "?"])[1],
                  "config": (re.search(r"RUN CONFIG — (\S+)", txt) or [None, "?"])[1],
                  "seconds": (re.search(r"duration_s=(\d+)", txt) or [None, "?"])[1],
                  "changed": ch},
        "funnel": _funnel_from_log(txt),
        "ground_plane": _plane_from_log(txt),
        "counts": _counts_from_log(txt),
    }
    if tracks:
        # Zone polygons, so the fragmentation verdict separates a person
        # leaving through the dining doorway from a track lost in open floor.
        # Without them this path fell back to the aggregate and reported
        # "NOT a failure rate" while the pipeline reported the real number --
        # two code paths, two different answers for the same run.
        zpolys = None
        try:
            import json as _j
            zf = None
            for cand in ("zones/CAM.112_zone_v5.json", "zones/CAM.112_zone_v4.json",
                         "zones/CAM.112_zone.json"):
                if os.path.exists(cand):
                    zf = cand
                    break
            if zf:
                _z = _j.load(open(zf))
                _fw, _fh = _z.get("frame_size", [3840, 2160])[:2]
                _ah = aw * float(_fh) / float(_fw)
                zpolys = {nm: [(px * aw / _fw, py * _ah / _fh) for px, py in pts]
                          for nm, pts in (_z.get("polygons") or {}).items()}
        except Exception:
            zpolys = None
        card["tracks"] = SC.track_stats(tracks, aw, aw * 9 / 16.0, zone_x=zone_x,
                                        t_first=t_first, t_last=t_last,
                                        zones=zpolys)
        card["pixel_height"] = SC.pixel_height(tracks, aw, 3840.0, imgsz)
        card["track_quality"] = SC.track_quality(tracks, fps=8.0)
    # SOURCE video, for binding ground truth to the footage it was read from.
    # Taken from SUMMARY.txt, which already records it under "source" and is
    # written by the run itself. Grepping the LOG for "*.mp4" matched the
    # ANNOTATED OUTPUT instead, which made every window look like it came from
    # a different chunk and silently zeroed the score.
    video = None
    sp = f"output/{run}/SUMMARY.txt"
    if os.path.exists(sp):
        m = re.search(r"^\s*source\s+(.+?\.mp4)\s*$",
                      open(sp, errors="ignore").read(), re.M)
        if m:
            video = m.group(1).strip()
    if video is None:
        m = re.search(r"--video\s+(.+?\.mp4)", txt)
        video = m.group(1) if m else None
    card["build"]["video"] = video
    from kevacv.visits import build_visits
    card["visits"] = build_visits(crossings, line_name="entry line")
    card["entry_score"] = SC.score_windows(crossings, video=video)
    from kevacv.review import build_queue
    card["review_queue"] = build_queue(
        arrival_confidence=card.get("arrival_confidence"),
        visits=card.get("visits"), camera=CAM)
    card["verdicts"] = SC.verdicts(card)
    return card


def main(argv):
    runs = [a for a in argv if not a.startswith("--")]
    cards = []
    for r in runs:
        c = build(r)
        cards.append(c)
        out = f"output/{r}"
        if os.path.isdir(out):
            SC.write(c, out, CAM)
        if "--compare" not in argv:
            print(SC.render(c))
    if "--compare" in argv and len(cards) >= 2:
        keys = [("counts", "line"), ("counts", "region"), ("counts", "guests"),
                ("counts", "trust"),
                ("entry_score", "heldout_got"), ("entry_score", "heldout_truth"),
                ("entry_score", "tuned_got"),
                ("tracks", "tracks"), ("tracks", "born_mid_pct"),
                ("tracks", "died_mid_pct"), ("tracks", "traversed_zone_x"),
                ("pixel_height", "network_px_median"),
                ("track_quality", "fragments_per_chain"),
                ("track_quality", "chains"),
                ("track_quality", "worst_chain"),
                ("track_quality", "ids_with_recovery"),
                ("track_quality", "swap_pressure_pairs")]
        w = max(len(c["run"]) for c in cards) + 2
        print("\n" + "=" * (26 + w * len(cards)))
        print("  RUN COMPARISON")
        print("=" * (26 + w * len(cards)))
        print(f"  {'metric':24s}" + "".join(f"{c['run']:>{w}s}" for c in cards))
        for sec, k in keys:
            vals = []
            for c in cards:
                v = (c.get(sec) or {}).get(k, "-")
                if isinstance(v, float):
                    vals.append(f"{v:.2f}" if k == "fragments_per_chain"
                                else f"{v:.0f}")
                else:
                    vals.append(str(v))
            print(f"  {sec + '.' + k:24s}" + "".join(f"{v:>{w}s}" for v in vals))
        print("=" * (26 + w * len(cards)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
