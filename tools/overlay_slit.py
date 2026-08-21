#!/usr/bin/env python3
"""Overlay slit-counter IN/OUT onto the pipeline's annotated video.

Two independent counters in one frame: the box-based pipeline (zones, roles,
per-frame HUD) and the slit counter (line crossings, no tracking). They fail
differently, so seeing both together shows where each one is inventing or
missing people.
"""
import argparse, json
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True, help="pipeline annotated mp4")
ap.add_argument("--events", required=True, help="slit events.json")
ap.add_argument("--zones", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--t0", type=float, default=0.0)
a = ap.parse_args()

evs = json.load(open(a.events))
cfg = json.load(open(a.zones))
ln = cfg.get("entry_line") or next(iter((cfg.get("entry_lines") or {}).values()))
ref = cfg.get("frame_size")

cap = cv2.VideoCapture(a.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
sx, sy = (W / ref[0], H / ref[1]) if ref else (1.0, 1.0)
A = np.array([ln[0][0] * sx, ln[0][1] * sy])
B = np.array([ln[1][0] * sx, ln[1][1] * sy])
d = B - A
n = np.array([-d[1], d[0]], float); n /= (np.linalg.norm(n) + 1e-9)
scale = np.hypot(*d) / max(np.hypot(ln[1][0] - ln[0][0], ln[1][1] - ln[0][1]), 1)

vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
i = 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    t = a.t0 + i / fps
    for o, col in ((-22 * scale, (80, 200, 255)), (22 * scale, (255, 200, 80))):
        p, q = A + n * o, B + n * o
        cv2.line(fr, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), col, 2)
    cv2.line(fr, (int(A[0]), int(A[1])), (int(B[0]), int(B[1])), (0, 0, 255), 3)
    nin = sum(1 for e in evs if e["t"] <= t and e["dir"] == "IN")
    nout = sum(1 for e in evs if e["t"] <= t and e["dir"] == "OUT")
    bh = 46
    cv2.rectangle(fr, (0, H - bh), (W, H), (0, 0, 0), -1)
    cv2.putText(fr, "SLIT COUNTER", (10, H - 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 2)
    cv2.putText(fr, f"IN {nin}", (200, H - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (90, 255, 90), 2)
    cv2.putText(fr, f"OUT {nout}", (310, H - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (90, 160, 255), 2)
    cv2.putText(fr, f"t={t:6.1f}s", (W - 170, H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
    hot = [e for e in evs if 0 <= t - e["t"] <= 1.2]
    if hot:
        e = hot[-1]
        col = (90, 255, 90) if e["dir"] == "IN" else (90, 160, 255)
        cv2.rectangle(fr, (2, 2), (W - 2, H - bh - 2), col, 5)
        cv2.putText(fr, f"{e['dir']}", (W // 2 - 45, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, col, 4)
    vw.write(fr)
    i += 1
cap.release(); vw.release()
print(f"wrote {a.out}  ({i} frames, {W}x{H} @ {fps:.1f}fps)")
