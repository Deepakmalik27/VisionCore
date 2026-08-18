"""contact_sheet.py — every identity the run found, as one picture.

WHY THIS EXISTS INSTEAD OF WATCHING THE VIDEO
    The annotated video answers "what happened at 19:07". It is the wrong tool
    for the question this pipeline actually turns on, which is:

        the report says 56 people. Are they 56 PEOPLE?

    Answering that from an hour of footage means scrubbing until you happen to
    see two boxes wearing different ids on the same person. Answering it from
    one contact sheet takes about four seconds: duplicates are obvious the
    instant two tiles are the same human, and so are phantoms — a plant, a
    reflection, a chair with an id on it.

    Over-counting is this system's dominant error (64 lockstep pairs, an
    appearance model that cannot separate people under infrared), and this is
    the cheapest instrument that measures it.

HOW IT AVOIDS DECODING THE VIDEO AGAIN
    It reads debug/<cam>_predictions.txt — CANONICAL ids, written by every run
    — picks the single largest box per identity, and seeks straight to that
    timestamp. 56 seeks instead of 54,081 frames, so it runs in seconds and
    works on any run already finished.

    Largest box = closest to camera = most pixels on the person. If a tile is
    unreadable at its own best moment, no threshold or backbone was ever going
    to separate that identity, and the honest fix is the camera.

Usage
    python tools/contact_sheet.py output/debug/CAM.112_predictions.txt \\
        "data/CAM.112 ... .mp4" [--fps 15] [--out output/snaps]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

TILE_W, TILE_H = 128, 256          # the aspect ReID backbones are trained on
COLS = 8


def best_box_per_id(pred_path):
    """-> {id: (frame, x, y, w, h)} — the largest box each identity ever had."""
    best = {}
    for line in Path(pred_path).read_text(encoding="utf-8").splitlines():
        parts = line.split(",")
        if len(parts) < 6:
            continue
        try:
            f, tid = int(parts[0]), parts[1]
            x, y, w, h = (float(v) for v in parts[2:6])
        except ValueError:
            continue
        if w <= 0 or h <= 0:
            continue
        if tid not in best or w * h > best[tid][3] * best[tid][4]:
            best[tid] = (f, x, y, w, h)
    return best


def frame_at(video, t_s):
    """One frame, by seeking — never by decoding forward to it."""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t_s:.3f}", "-i", str(video),
           "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"]
    out = subprocess.run(cmd, capture_output=True).stdout
    if not out:
        return None
    return cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)


def build(pred_path, video, fps, out_dir, analysed_w=1280):
    best = best_box_per_id(pred_path)
    if not best:
        print(f"  no rows in {pred_path}")
        return 2
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"  {len(best)} identities -> seeking {len(best)} frames")

    tiles, missed = [], 0
    for tid, (f, x, y, w, h) in sorted(best.items(), key=lambda kv: str(kv[0])):
        t = max(0.0, (f - 1) / float(fps))         # MOT frames are 1-indexed
        img = frame_at(video, t)
        if img is None:
            missed += 1
            continue
        # boxes are in ANALYSED pixels; the source is larger, and cropping the
        # source gives a sharp tile instead of an upscaled blur
        s = img.shape[1] / float(analysed_w)
        x1, y1 = int(max(0, x * s)), int(max(0, y * s))
        x2 = int(min(img.shape[1], (x + w) * s))
        y2 = int(min(img.shape[0], (y + h) * s))
        if x2 - x1 < 8 or y2 - y1 < 8:
            missed += 1
            continue
        crop = cv2.resize(img[y1:y2, x1:x2], (TILE_W, TILE_H))
        cv2.imwrite(str(out / f"{tid}.jpg"), crop)
        # the pixel height is on the tile because it is the honest ceiling on
        # what any ReID model could have done with this identity
        cv2.rectangle(crop, (0, 0), (TILE_W, 18), (0, 0, 0), -1)
        cv2.putText(crop, f"{tid}  {y2 - y1}px", (3, 13),
                    cv2.FONT_HERSHEY_SIMPLEX, .38, (255, 255, 255), 1,
                    cv2.LINE_AA)
        tiles.append(crop)

    if not tiles:
        print("  every seek failed — is the video path right?")
        return 2
    rows = (len(tiles) + COLS - 1) // COLS
    sheet = np.full((rows * TILE_H, COLS * TILE_W, 3), 30, np.uint8)
    for i, t_img in enumerate(tiles):
        r, c = divmod(i, COLS)
        sheet[r * TILE_H:(r + 1) * TILE_H, c * TILE_W:(c + 1) * TILE_W] = t_img
    sheet_path = out.parent / "contact_sheet.jpg"
    cv2.imwrite(str(sheet_path), sheet)
    print(f"  {len(tiles)} tiles -> {sheet_path}")
    print(f"  individual crops -> {out}")
    if missed:
        print(f"  {missed} identity(ies) had no readable crop")
    print()
    print("  Look for: the same person twice (over-count), and tiles that are")
    print("  not people at all (plants, reflections, chairs holding an id).")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("predictions", help="output/debug/<cam>_predictions.txt")
    p.add_argument("video", help="the source chunk that was analysed")
    p.add_argument("--fps", type=float, default=15.0,
                   help="analysed fps — must match the run (default 15)")
    p.add_argument("--analysed-width", type=int, default=1280)
    p.add_argument("--out", default=str(ROOT / "output" / "snaps"))
    a = p.parse_args(argv)
    for label, path in (("predictions", a.predictions), ("video", a.video)):
        if not Path(path).exists():
            print(f"  MISSING {label}: {path}")
            return 2
    return build(a.predictions, a.video, a.fps, a.out, a.analysed_width)


if __name__ == "__main__":
    raise SystemExit(main())
