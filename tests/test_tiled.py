"""Tests for kevacv/tiled.py — detect at native resolution, not on a downscale.

Built from the measured problem: 3840x2160 decoded, 1280x720 analysed, a guest
at the door ~60px tall, crops upscaled to 256, separability 0.658. Every test
asks whether slicing recovers real pixels without inventing duplicate people.

Run: python test_tiled.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kevacv.tiled import (cost_estimate, height_roi, nms, slice_grid,
                          tiled_predict)

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


class FakeFrame:
    """Minimal stand-in for a numpy image: knows its shape and can be sliced."""
    def __init__(self, w, h, x0=0, y0=0):
        self.shape = (h, w)
        self.x0, self.y0 = x0, y0

    def __getitem__(self, key):
        ys, xs = key
        return FakeFrame(xs.stop - xs.start, ys.stop - ys.start,
                         self.x0 + xs.start, self.y0 + ys.start)


print("=" * 74)
print("  the grid covers the frame and never runs off the edge")
print("=" * 74)
g = slice_grid(1280, 720, tile=640, overlap=0.2)
check(all(x1 <= 1280 and y1 <= 720 for _, _, x1, y1 in g),
      "no tile extends past the frame", f"{len(g)} tiles")
check(all(x0 >= 0 and y0 >= 0 for x0, y0, _, _ in g), "and none starts negative")
check(min(x0 for x0, _, _, _ in g) == 0 and min(y0 for _, y0, _, _ in g) == 0,
      "the grid starts at the origin")
check(max(x1 for _, _, x1, _ in g) == 1280, "and reaches the right edge")
check(max(y1 for _, _, _, y1 in g) == 720, "and the bottom edge")
check(len(set(g)) == len(g), "no duplicate tiles after edge clamping")
check(slice_grid(100, 100, tile=640)[0] == (0, 0, 100, 100),
      "a frame smaller than one tile yields exactly one tile")
check(slice_grid(1280, 720, roi=(0, 0, 0, 0)) == [], "an empty roi yields none")

print()
print("=" * 74)
print("  the ROI keeps the cost down: tile only where people are SMALL")
print("=" * 74)
# this run's own fit: expected_h = 0.820*foot_y - 37
roi = height_roi(1280, 720, slope=0.820, intercept=-37.0, min_px=150)
check(roi is not None and roi[3] < 720, "a band, not the whole frame", str(roi))
check(height_roi(1280, 720, 0.820, -37.0, min_px=10) is None,
      "a sliver at the top of frame -> None, not a tile bill for 93px",
      "intercept=-37 means the fit predicts NEGATIVE height above y~45")
check(height_roi(1280, 720, 0.820, -37.0, min_px=10, min_band_px=10) is not None,
      "and the sliver guard is tunable, not hardcoded policy")
check(height_roi(1280, 720, 0.0, 100.0, min_px=150) is None,
      "a degenerate ground fit -> None rather than a nonsense band")

full = cost_estimate(1280, 720)["calls_per_frame"]
band = cost_estimate(1280, 720, roi=roi)["calls_per_frame"]
check(band < full, "tiling only the far band costs less than tiling everything",
      f"{band} vs {full} calls/frame")
print(f"    -> full grid {full} calls/frame, ROI band {band} calls/frame")

print()
print("=" * 74)
print("  boxes come back in FULL-FRAME coordinates")
print("=" * 74)


def one_box_middle(img):
    """A detector that always finds one box in the centre of whatever it sees."""
    h, w = img.shape
    return [(w * 0.4, h * 0.4, w * 0.6, h * 0.6, 0.9, 0)]


frame = FakeFrame(1280, 720)
boxes, stats = tiled_predict(frame, one_box_middle, tile=640, overlap=0.2)
check(stats["tiles"] > 1, "the frame was actually sliced", str(stats["tiles"]))
check(all(0 <= b[0] <= 1280 and 0 <= b[1] <= 720 for b in boxes),
      "every returned box lies inside the frame")
check(any(b[0] > 640 for b in boxes),
      "a detection from a right-hand tile is shifted right, not left at 0")

print()
print("=" * 74)
print("  the same person seen in two overlapping tiles is ONE person")
print("=" * 74)
dup = nms([(100, 100, 200, 300, 0.9, 0), (104, 98, 205, 297, 0.7, 0)])
check(len(dup) == 1 and dup[0][4] == 0.9,
      "overlapping duplicates collapse, highest confidence wins", str(len(dup)))
check(len(nms([(0, 0, 50, 50, 0.9, 0), (500, 500, 560, 620, 0.8, 0)])) == 2,
      "two genuinely separate people are both kept")
check(len(nms([(100, 100, 200, 300, 0.9, 0), (104, 98, 205, 297, 0.7, 1)])) == 2,
      "different CLASSES are never merged (person vs head)")
check(nms([]) == [], "no boxes -> no crash")

print()
print("=" * 74)
print("  a body cut in half by a seam is dropped, not reported as a short person")
print("=" * 74)


def box_at_left_edge(img):
    h, w = img.shape
    return [(0, 100, 60, 300, 0.9, 0)]      # flush against the tile's left edge


boxes, _ = tiled_predict(FakeFrame(1280, 720), box_at_left_edge, tile=640,
                         overlap=0.2, full_frame=False)
check(all(b[0] < 5 for b in boxes),
      "only the tile whose left edge IS the frame edge survives",
      f"{len(boxes)} box(es)")
kept_all, _ = tiled_predict(FakeFrame(1280, 720), box_at_left_edge, tile=640,
                            overlap=0.2, full_frame=False,
                            drop_seam_boxes=False)
check(len(kept_all) > len(boxes),
      "and disabling the guard keeps the half-bodies, as a control")

print()
print("=" * 74)
print("  the full-frame pass is kept so close-up people are not lost")
print("=" * 74)


def big_box(img):
    h, w = img.shape
    return [(10, 10, w - 10, h - 10, 0.95, 0)] if w > 1000 else []


boxes, _ = tiled_predict(FakeFrame(1280, 720), big_box, full_frame=True)
check(len(boxes) == 1, "a person larger than a tile is found by the full pass",
      "tiles alone would only ever see pieces of them")
boxes, _ = tiled_predict(FakeFrame(1280, 720), big_box, full_frame=False)
check(len(boxes) == 0, "and is genuinely invisible without it — hence the default")

print()
print("=" * 74)
print("  degenerate input")
print("=" * 74)
check(tiled_predict(FakeFrame(640, 640), lambda i: None)[0] == [],
      "a detector returning None is treated as no detections")
try:
    slice_grid(1280, 720, tile=0)
    check(False, "tile=0 raises")
except ValueError:
    check(True, "tile=0 raises ValueError rather than looping forever")

print()
print("=" * 74)
if FAILED:
    print(f"  {len(FAILED)} FAILED:")
    for f in FAILED:
        print(f"    - {f}")
    sys.exit(1)
print("  ALL PASS")
print("=" * 74)

# ── BATCHED TILING — same boxes, one GPU call ───────────────────────────────
# The 2026-08-14 run took 1h58m for 27k frames because tiled_predict called the
# detector once per tile: nine batch-of-1 inferences per frame, with the GPU
# idle between them. Batching them is free speed ONLY if the boxes are
# identical, so that is what this asserts.
def _bt_check(ok, what, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {what}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        raise SystemExit(1)


class _F:
    shape = (1080, 1920, 3)

    def __getitem__(self, k):
        return _F()


def _one(img):
    return [(10.0, 20.0, 60.0, 140.0, 0.9, 0)]


print()
print("=" * 74)
print("  batched tiling returns EXACTLY the sequential result")
print("=" * 74)
_seq, _ss = tiled_predict(_F(), _one, tile=640, full_frame=True)
_bat, _sb = tiled_predict(_F(), _one, tile=640, full_frame=True,
                          predict_batch_fn=lambda ims: [_one(i) for i in ims])
_bt_check(_seq == _bat, "batched output is identical to sequential",
          f"{len(_seq)} boxes both ways")
_bt_check(_ss["tiles"] == _sb["tiles"], "same tile count")
_bt_check(_sb.get("batched") is True, "and the stats say it batched")

# A detector returning fewer results than images must look like "found
# nothing", never like "those tiles were never requested".
_short, _ = tiled_predict(_F(), _one, tile=640, full_frame=True,
                          predict_batch_fn=lambda ims: [_one(ims[0])])
_bt_check(len(_short) == 1, "a short batch result is padded, not zipped away",
          "zip() would silently drop every tile after the first")
print("  batched-tiling checks PASS")
