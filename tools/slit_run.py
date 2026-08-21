#!/usr/bin/env python3
"""Run the slit counter over a long stretch, and render an annotated video.

CHUNKING, and why it is not optional: foreground() models the doorway with a
per-column MEDIAN over the window. That was validated on 13-second windows.
Over 20 minutes the daylight changes completely and a single median describes
none of it, so the run is cut into short chunks and each gets its own
background. Chunk length is kept close to the validated window size.

Events within MERGE_S of each other in the SAME direction across a chunk
boundary are merged, so one person crossing on the seam is not counted twice.
"""
from __future__ import annotations
import argparse, json, sys, os
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.slit_count import (count, debounce, new_funnel,
                              A_DEFAULT, B_DEFAULT)


def line_from_zones(path, frame_wh=None):
    """Read entry_line from a zones json, scaled to the actual frame."""
    cfg = json.load(open(path))
    ln = cfg.get("entry_line")
    if not ln:
        lines = cfg.get("entry_lines") or {}
        # PICK THE VENUE DOOR, NOT THE FIRST KEY.
        # `next(iter(lines.values()))` returned 'dining entry' on CAM.112 --
        # the first key in the file -- so passing --zones silently counted the
        # INTERIOR dining threshold instead of the street door, with no warning.
        for _k in ("entry line", "entry_line", "main_entrance", "main entry"):
            if _k in lines:
                ln = lines[_k]
                break
        else:
            _venue = [k for k in lines
                      if not any(w in k.lower() for w in
                                 ("dining", "staff", "kitchen", "interior"))]
            if len(_venue) != 1:
                raise SystemExit(
                    f"{path}: cannot tell which of {sorted(lines)} is the venue "
                    f"door. Name it 'entry line', or pass the line explicitly.")
            ln = lines[_venue[0]]
            print(f"  entry line chosen by elimination: {_venue[0]!r}")
    if not ln:
        raise SystemExit(f"{path} has no entry_line")
    A, B = np.array(ln[0], float), np.array(ln[1], float)
    ref = cfg.get("frame_size")
    if ref and frame_wh and tuple(ref) != tuple(frame_wh):
        sx, sy = frame_wh[0] / ref[0], frame_wh[1] / ref[1]
        A, B = A * [sx, sy], B * [sx, sy]
    return A, B

CHUNK_S = 15.0
MERGE_S = 1.0


def run(video, t0, t1, chunk_s=CHUNK_S, A=A_DEFAULT, B=B_DEFAULT, funnel=None):
    evs = []
    t = t0
    while t < t1:
        b = min(t + chunk_s, t1)
        try:
            _n, e = count(video, t, b, A=A, B=B, funnel=funnel)
        except Exception as exc:                 # one bad chunk must not kill it
            print(f"  !! chunk {t:.0f}-{b:.0f}s failed: {exc}", file=sys.stderr)
            e = []
        evs += e
        pct = (b - t0) / max(t1 - t0, 1e-9) * 100
        print(f"\r  {pct:5.1f}%  t={b:6.0f}s  events={len(evs)}", end="",
              flush=True)
        t = b
    print()
    evs.sort(key=lambda x: x["t"])
    merged = []
    for e in evs:
        if merged and e["dir"] == merged[-1]["dir"] \
                and e["t"] - merged[-1]["t"] <= MERGE_S:
            continue                              # same person on a seam
        merged.append(e)
    # SAME-DIRECTION merging above is not enough: a REVERSAL that straddles a
    # seam was never examined, because count()'s debounce only ever sees one
    # chunk. 1019.0 IN <-> 1020.7 OUT survived the whole 20-minute run for
    # exactly that reason. Re-run the identical guard on the concatenated list.
    return debounce(merged)


def render(video, t0, t1, evs, out, fps_out=10, w=960, h=540,
           LINE=(A_DEFAULT, B_DEFAULT), SRC_W=1920, SRC_H=1080):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000.0)
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (w, h))
    step = max(1, int(round(fps / fps_out)))
    A, B = np.array(LINE[0]), np.array(LINE[1])
    d = B - A
    n = np.array([-d[1], d[0]], float); n /= np.linalg.norm(n)
    sx, sy = w / float(SRC_W), h / float(SRC_H)
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = t0 + i / fps
        if t > t1:
            break
        if i % step == 0:
            src_w, src_h = fr.shape[1], fr.shape[0]
            fr = cv2.resize(fr, (w, h))
            for o, col in ((-22, (80, 200, 255)), (22, (255, 200, 80))):
                p, q = A + n * o, B + n * o
                cv2.line(fr, (int(p[0]*sx), int(p[1]*sy)),
                         (int(q[0]*sx), int(q[1]*sy)), col, 2)
            nin = sum(1 for e in evs if e["t"] <= t and e["dir"] == "IN")
            nout = sum(1 for e in evs if e["t"] <= t and e["dir"] == "OUT")
            cv2.rectangle(fr, (0, 0), (w, 52), (0, 0, 0), -1)
            cv2.putText(fr, f"IN {nin}", (12, 36), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, (90, 255, 90), 3)
            cv2.putText(fr, f"OUT {nout}", (170, 36), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, (90, 160, 255), 3)
            cv2.putText(fr, f"t={t:7.1f}s", (w - 210, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (220, 220, 220), 2)
            hot = [e for e in evs if 0 <= t - e["t"] <= 1.2]
            if hot:
                e = hot[-1]
                col = (90, 255, 90) if e["dir"] == "IN" else (90, 160, 255)
                cv2.rectangle(fr, (2, 54), (w - 2, h - 2), col, 6)
                cv2.putText(fr, e["dir"], (w // 2 - 50, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.6, col, 4)
            vw.write(fr)
        i += 1
    cap.release(); vw.release()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=1200.0)
    ap.add_argument("--out", default="output/slit20")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--zones", help="zones json with entry_line")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    _c = cv2.VideoCapture(a.video)
    SRC_W = int(_c.get(cv2.CAP_PROP_FRAME_WIDTH))
    SRC_H = int(_c.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _c.release()
    if a.zones:
        A, B = line_from_zones(a.zones, (SRC_W, SRC_H))
    else:
        # A_DEFAULT/B_DEFAULT are 1920x1080 coordinates. On the 3840x2160
        # source they must be doubled -- the published runs only worked because
        # the numbers happened to already be at source scale. Without this,
        # running with no --zones puts the slit across the RECEPTION DESK
        # (verified: 5 spurious OUT events at the desk in one 15s chunk).
        _sx, _sy = SRC_W / 1920.0, SRC_H / 1080.0
        A = np.array(A_DEFAULT, float) * [_sx, _sy]
        B = np.array(B_DEFAULT, float) * [_sx, _sy]
        if (_sx, _sy) != (1.0, 1.0):
            print(f"  !! no --zones: scaled the built-in line by "
                  f"{_sx:.2f}x{_sy:.2f} for a {SRC_W}x{SRC_H} source. "
                  f"Pass --zones to use the real door.")
    print(f"frame {SRC_W}x{SRC_H}   entry line ({A[0]:.0f},{A[1]:.0f}) -> "
          f"({B[0]:.0f},{B[1]:.0f})")
    print(f"counting {a.start:.0f}-{a.end:.0f}s in {CHUNK_S:.0f}s chunks")
    fun = new_funnel()
    evs = run(a.video, a.start, a.end, A=A, B=B, funnel=fun)
    nin = sum(1 for e in evs if e["dir"] == "IN")
    nout = sum(1 for e in evs if e["dir"] == "OUT")
    json.dump(evs, open(f"{a.out}/events.json", "w"), indent=1, default=float)
    json.dump(fun, open(f"{a.out}/funnel.json", "w"), indent=1, default=float)
    print(f"\n  IN  {nin}\n  OUT {nout}\n  total events {len(evs)}")
    print(f"  -> {a.out}/events.json")
    # WHERE THE BLOBS DIED. Printed, not buried in the json, because the one
    # question every past slit regression turned on -- which stage ate the
    # people -- was never answerable from the run's own output.
    counts = fun.get("counts", {})
    born = counts.get("blobs", 0) + counts.get("under_min_area", 0)
    print(f"\n  BLOB FUNNEL   {born} connected components ->"
          f" {counts.get('kept', 0)} events")
    for stage in ("under_min_area", "one_tripwire", "past_doorway_x",
                  "too_wide", "debounce"):
        n = counts.get(stage, 0)
        share = 100.0 * n / born if born else 0.0
        widths = [s["w"] for s in fun.get("samples", {}).get(stage, [])]
        extra = (f"   widths {min(widths)}-{max(widths)}" if widths else "")
        print(f"    -{n:<6} {share:5.1f}%  {stage}{extra}")
    print(f"  -> {a.out}/funnel.json")
    if not a.no_render:
        print("rendering...")
        render(a.video, a.start, a.end, evs, f"{a.out}/annotated.mp4",
               LINE=(A, B), SRC_W=SRC_W, SRC_H=SRC_H)
        print(f"  -> {a.out}/annotated.mp4")
