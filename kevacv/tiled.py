"""tiled.py — run the detector at native resolution on slices, not on a
downscaled whole frame.

WHY THIS EXISTS
    The pipeline decodes 3840x2160 and analyses 1280x720. A guest at the door
    is then ~60 px tall. The banked crop is upscaled to 128x256 before it ever
    reaches the ReID backbone, so the embedding is built mostly from
    interpolation — and the diagnostic that should have caught this measured
    the resize target (fixed in V76). Measured separability on this footage is
    0.658 balanced accuracy, which no threshold repairs.

    Slicing Aided Hyper Inference (Akyon et al., arXiv:2202.06934) is the
    standard answer: cut the full-resolution frame into overlapping tiles, run
    the detector on each tile at native scale, map the boxes back, and merge.
    Reported +6.8% to +14.5% AP on small-object benchmarks.

WHY NOT `pip install sahi`
    This kernel pins numpy==2.0.2, force-reinstalls scipy to restore Cython
    C-extensions, and hand-attaches 38 string ufuncs into numpy._core.umath to
    keep InsightFace importable. Adding a dependency that pulls its own numpy
    /torch constraints into that is a poor trade for ~150 lines of geometry.
    This module is pure Python + whatever the caller's detector already uses.

THE ROI IDEA (why this is cheaper than 4-9x)
    Naive slicing multiplies detector cost by the tile count. But small people
    only occur where the GROUND PLANE says they are small — the far half of
    the frame. This run already fits `expected height = 0.820*foot_y - 37px`.
    Tile only the band where predicted height is below the size you care
    about, and keep one cheap full-frame pass for everyone near the camera.
    On a typical oblique reception view that is 2-3 tiles, not 9.

CONTRACT
    tiled_predict() takes a `predict_fn(image_array) -> [(x1,y1,x2,y2,conf,cls)]`
    and returns the same shape in FULL-FRAME coordinates. It never imports
    cv2, torch or ultralytics, so it is testable with a fake detector.
"""
from __future__ import annotations

DEFAULT_TILE = 640
DEFAULT_OVERLAP = 0.2
DEFAULT_IOU = 0.55
# A box hugging a tile edge is probably a body the slice cut in half. Boxes
# that touch an INTERIOR seam are dropped in favour of whatever the
# neighbouring tile (which saw the whole body) produced.
EDGE_MARGIN_PX = 2


def slice_grid(width, height, tile=DEFAULT_TILE, overlap=DEFAULT_OVERLAP,
               roi=None):
    """-> [(x0, y0, x1, y1), ...] tiles covering `roi` (default: whole frame).

    Tiles are clamped to the frame, so the last row/column overlaps more
    rather than running past the edge — a slice that extends beyond the image
    would need padding, and padding invents pixels the detector then scores.
    """
    if tile <= 0:
        raise ValueError("tile must be positive")
    rx0, ry0, rx1, ry1 = roi or (0, 0, width, height)
    rx0, ry0 = max(0, int(rx0)), max(0, int(ry0))
    rx1, ry1 = min(int(width), int(rx1)), min(int(height), int(ry1))
    if rx1 <= rx0 or ry1 <= ry0:
        return []

    step = max(1, int(tile * (1.0 - overlap)))
    xs, ys = [], []
    x = rx0
    while True:
        xs.append(min(x, max(rx0, rx1 - tile)))
        if x + tile >= rx1:
            break
        x += step
    y = ry0
    while True:
        ys.append(min(y, max(ry0, ry1 - tile)))
        if y + tile >= ry1:
            break
        y += step

    seen, out = set(), []
    for yy in ys:
        for xx in xs:
            x1, y1 = min(xx + tile, rx1), min(yy + tile, ry1)
            key = (xx, yy, x1, y1)
            if key in seen:
                continue          # clamping can produce duplicates
            seen.add(key)
            out.append(key)
    return out


def height_roi(width, height, slope, intercept, min_px, pad=0.05,
               min_band_px=120):
    """The band of the frame where a person is predicted SHORTER than min_px.

    Uses the run's own fitted ground model, `expected_h = slope*foot_y +
    intercept`. Returns None when slicing would buy nothing, so the caller
    skips the tiles rather than paying for them.

    Two ways it declines, and the second matters more than it looks:

      * slope <= 0 — the ground fit is degenerate, so its band is meaningless.
      * the band is thinner than min_band_px. This run fits intercept = -37,
        which means predicted height is NEGATIVE above foot_y ~= 45: the model
        is extrapolating outside the data it was fitted on. Without this guard
        a small min_px returns a ~90 px sliver at the top of frame, and we
        would pay for tiles over a strip that cannot contain a detectable
        person in the first place.
    """
    if slope <= 0:
        return None
    # expected_h(y) grows with y (further down the frame = closer = bigger)
    y_at_min = (min_px - intercept) / slope
    if y_at_min <= 0:
        return None                      # even the top of frame is big enough
    y_cut = min(float(height), y_at_min + pad * height)
    if y_cut < min_band_px:
        return None
    return (0, 0, int(width), int(round(y_cut)))


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    bb = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    denom = aa + bb - inter
    return inter / denom if denom > 0 else 0.0


def nms(boxes, iou_thresh=DEFAULT_IOU):
    """Greedy NMS over (x1,y1,x2,y2,conf,cls). Highest confidence wins.

    Suppression is per-class: two different classes overlapping is not a
    duplicate, and merging them would silently delete one of them.
    """
    kept = []
    for box in sorted(boxes, key=lambda b: -b[4]):
        if all(_iou(box, k) < iou_thresh or box[5] != k[5] for k in kept):
            kept.append(box)
    return kept


def _touches_interior_seam(box, tile, frame_wh, margin=EDGE_MARGIN_PX):
    """True if the box hugs a tile edge that is NOT also the frame edge."""
    tx0, ty0, tx1, ty1 = tile
    fw, fh = frame_wh
    x1, y1, x2, y2 = box[:4]
    if abs(x1 - tx0) <= margin and tx0 > 0:
        return True
    if abs(y1 - ty0) <= margin and ty0 > 0:
        return True
    if abs(x2 - tx1) <= margin and tx1 < fw:
        return True
    if abs(y2 - ty1) <= margin and ty1 < fh:
        return True
    return False


def tiled_predict(frame, predict_fn, tile=DEFAULT_TILE,
                  overlap=DEFAULT_OVERLAP, roi=None, iou_thresh=DEFAULT_IOU,
                  full_frame=True, drop_seam_boxes=True, predict_batch_fn=None):
    """Detect on overlapping native-resolution slices; return full-frame boxes.

    frame        anything sliceable as frame[y0:y1, x0:x1] with a .shape
    predict_fn   image -> iterable of (x1, y1, x2, y2, conf, cls)
    full_frame   also run one whole-frame pass. Keep it: a person close to the
                 camera can be larger than a tile, and tiles alone would only
                 ever see pieces of them.

    predict_batch_fn  OPTIONAL. [image, ...] -> [[box, ...], ...], one list per
                 image, in order. When supplied, the whole-frame pass and every
                 tile go to the detector in ONE call instead of N sequential
                 ones.

                 FREE SPEED, NOT A TRADE. Same model, same weights, same
                 images, same conf/iou — a detector letterboxes and scores each
                 image of a batch independently, so the boxes are identical to
                 calling it N times. What changes is that the GPU gets one
                 batch of 9 instead of nine batches of 1 and stops idling
                 between them. The 2026-08-14 run spent 1h58m on 27k frames
                 doing 9 sequential calls each.
    """
    h, w = frame.shape[0], frame.shape[1]
    out = []
    tiles = slice_grid(w, h, tile, overlap, roi)

    if predict_batch_fn is not None:
        imgs, origins = [], []
        if full_frame:
            imgs.append(frame)
            origins.append(None)           # None = already in full-frame coords
        for (x0, y0, x1, y1) in tiles:
            imgs.append(frame[y0:y1, x0:x1])
            origins.append((x0, y0, x1, y1))
        res = list(predict_batch_fn(imgs) or [])
        # PAD, never zip away the difference: a short result list must look
        # like "that tile found nothing", never like "that tile was skipped".
        res += [[]] * (len(imgs) - len(res))
        for boxes, origin in zip(res, origins):
            for b in boxes or []:
                if origin is None:
                    out.append(tuple(b))
                    continue
                ox, oy, ox1, oy1 = origin
                shifted = (b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy, b[4], b[5])
                if drop_seam_boxes and _touches_interior_seam(
                        shifted, (ox, oy, ox1, oy1), (w, h)):
                    continue
                out.append(shifted)
        return nms(out, iou_thresh), {"tiles": len(tiles), "raw": len(out),
                                      "batched": True}

    if full_frame:
        for b in predict_fn(frame) or []:
            out.append(tuple(b))

    for (x0, y0, x1, y1) in tiles:
        sub = frame[y0:y1, x0:x1]
        for b in predict_fn(sub) or []:
            bx1, by1, bx2, by2, conf, cls = b[0], b[1], b[2], b[3], b[4], b[5]
            shifted = (bx1 + x0, by1 + y0, bx2 + x0, by2 + y0, conf, cls)
            if drop_seam_boxes and _touches_interior_seam(
                    shifted, (x0, y0, x1, y1), (w, h)):
                continue          # a body the slice cut in half
            out.append(shifted)

    return nms(out, iou_thresh), {"tiles": len(tiles), "raw": len(out)}


def cost_estimate(width, height, tile=DEFAULT_TILE, overlap=DEFAULT_OVERLAP,
                  roi=None, full_frame=True):
    """Detector calls per frame, so the compute trade is visible up front."""
    n = len(slice_grid(width, height, tile, overlap, roi)) + (1 if full_frame else 0)
    return {"calls_per_frame": n, "vs_baseline": f"{n:.1f}x"}
