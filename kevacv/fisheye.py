"""fisheye.py — undo the lens before asking the detector what it sees.

WHY THIS EXISTS
    CAM.112 is a ceiling fisheye, and nothing in this pipeline knew that.
    `grep -rl "fisheye|undistort|barrel" kevacv/ tools/ docs/` returned zero
    hits while the audit's top complaints were all the same failure wearing
    different hats:

      * "bounding boxes are badly localized / much larger than the person"
      * "huge box around the doorway", "huge box over empty floor"
      * P3 covering the ENTIRE right column of frame_60s -- around a REAL
        guest walking in the door

    Under a fisheye a standing person is imaged along the image RADIUS, so
    near the edge they are rotated. An axis-aligned box around a rotated
    person is enormous, and it swallows the plant, the doorway and the wall
    behind them. The detector was not wrong; the box shape was.

    This matters more here than it looks. Those boxes feed the size filter,
    which predicts how tall a person at a given footline can be. Fed enormous
    boxes it over-fires, the D1 guard measures the drop rate and RELAXES its
    own tolerance 2.5x -> 5.0x, and from then on nothing filters them at all.
    A geometry problem became a self-amplifying one.

WHY PATCHES, NOT A WHOLE-FRAME DEWARP
    Undistorting the whole frame is the obvious move and it is the wrong one
    here: every polygon and every door line in zones/CAM.112_zone.json is in
    SOURCE pixels. A whole-frame dewarp silently invalidates all of them, and
    that is precisely the class of silent geometry failure this camera has
    already produced twice.

    So: dewarp, detect, and map the boxes BACK to source coordinates. Zones,
    lines, the ground plane and every stored coordinate keep working
    untouched, and the detector is the only thing that sees rectified pixels.
    This is the published approach for top-view fisheye people detection
    (Tamura et al., "Efficient Pedestrian Detection in Top-View Fisheye
    Images Using Compositions of Perspective View Patches", arXiv:2009.02711).

    It also mirrors kevacv/tiled.py exactly: same predict_fn contract, same
    map-back-and-merge shape, same "no cv2, no torch, testable with a fake
    detector" rule. The two compose -- dewarp, then tile the dewarped view.

THE MODEL, AND WHY ONE PARAMETER
    A single-parameter division model (Fitzgibbon, CVPR 2001):

        r_source = r_rectified * (1 + k * r_rectified^2)

    with radius normalised by the half-diagonal, so k is resolution
    independent and lives in roughly [-0.6, 0.6].

    One parameter, because a checkerboard calibration of this camera does not
    exist and is not going to -- it is a ceiling dome in a working restaurant.
    What DOES exist is straight lines in the scene: door frames, shelf edges,
    and a checkered floor whose tile rows are straight by construction. fit_k()
    finds the k that makes those lines straightest. That is a measurement, not
    a slider, and it uses the same floor the ground plane is calibrated from.

    ponytail: one radial term, no tangential, no per-axis focal length. A dome
    camera's dominant distortion is radial by an order of magnitude. Add k2 if
    fit_k's residual stops improving before the lines look straight.
"""
from __future__ import annotations

import math

# Straightness better than this (in normalised radius units, so ~0.2% of the
# half-diagonal) is below what you can see, and below what hand-clicked points
# can resolve anyway. Used to stop the fit congratulating itself.
STRAIGHT_ENOUGH = 0.002
K_SEARCH = (-0.60, 0.60)
BOX_EDGE_SAMPLES = 5      # per side; a mapped rectangle is a CURVE, not a line


def horizon_r(k):
    """Largest SOURCE radius (normalised) this k can image, or None if
    unbounded.

    For k < 0 -- which is what a real barrel/fisheye lens IS in this
    parameterisation, since it pulls the edges inward -- f(r) = r*(1+k*r^2)
    rises, peaks at r_turn = sqrt(-1/(3k)), and falls. Source radii above
    f(r_turn) are not produced by the model at all, and the frame CORNERS
    (r ~ 1.0) can sit outside it for |k| >~ 0.11.

    Callers get a clamp, not an exception: a corner pixel of a dome camera is
    usually black surround anyway, and refusing to rectify a frame because its
    corner is out of domain would be the tail wagging the dog. But it is
    stated here rather than discovered, because "silently wrong at the edges"
    is precisely this camera's history.
    """
    if k >= 0:
        return None
    r_turn = math.sqrt(-1.0 / (3.0 * k))
    return r_turn * (1.0 + k * r_turn * r_turn)


def in_domain(x, y, k, size):
    """Is this SOURCE pixel one the model can invert exactly?"""
    h = horizon_r(k)
    if h is None:
        return True
    cx, cy, s = _centre_scale(size)
    return math.hypot((x - cx) / s, (y - cy) / s) < h


def _centre_scale(size):
    """-> (cx, cy, s). s is the half-diagonal, so r is ~1.0 at the corners and
    k means the same thing at 1920x1080 as at 3840x2160."""
    w, h = size
    return w / 2.0, h / 2.0, math.hypot(w, h) / 2.0


def to_source(x, y, k, size):
    """Rectified pixel -> SOURCE pixel. The easy direction: closed form.

    This is what maps a detection back onto the original frame, so zones and
    lines never have to move.
    """
    cx, cy, s = _centre_scale(size)
    u, v = (x - cx) / s, (y - cy) / s
    f = 1.0 + k * (u * u + v * v)
    return cx + u * f * s, cy + v * f * s


def to_rectified(x, y, k, size, iters=60):
    """SOURCE pixel -> rectified pixel. Numeric inverse of to_source.

    BRACKETING IS THE WHOLE JOB, and getting it wrong is silent.
    f(r) = r*(1 + k*r^2) is only monotonic while f'(r) = 1 + 3*k*r^2 > 0:

        k >= 0   increasing everywhere, and f(r) >= r, so the answer is
                 bracketed by [0, r_src] exactly.
        k <  0   f(r) <= r, so the answer is ABOVE r_src -- and f turns over
                 at r_turn = sqrt(-1/(3k)). A bisection allowed past r_turn
                 straddles two roots and converges to nonsense.

    This module's own test caught that: a naive [0, 2*r_src] bracket was wrong
    by 924 px at k=-0.18 while passing at every positive k. Nothing downstream
    could have noticed -- the boxes would simply have been in the wrong place.
    """
    cx, cy, s = _centre_scale(size)
    u, v = (x - cx) / s, (y - cy) / s
    r_src = math.hypot(u, v)
    if r_src < 1e-12 or abs(k) < 1e-12:
        return float(x), float(y)

    def f(r):
        return r * (1.0 + k * r * r)

    if k > 0:
        lo, hi = 0.0, r_src                     # f(r) >= r  =>  r_rect <= r_src
    else:
        r_turn = math.sqrt(-1.0 / (3.0 * k))    # last radius the lens can image
        if r_src >= f(r_turn):
            # Beyond what this model can produce: the pixel is outside the
            # lens's imaged circle. Clamp rather than invent a radius.
            lo = hi = r_turn
        else:
            lo, hi = r_src, r_turn              # f(r) <= r  =>  r_rect >= r_src
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if f(mid) < r_src:
            lo = mid
        else:
            hi = mid
    r_rect = (lo + hi) / 2.0
    scale = r_rect / r_src
    return cx + u * scale * s, cy + v * scale * s


def box_to_source(box, k, size):
    """A box found on the rectified image -> the axis-aligned box that
    contains it on the SOURCE image.

    Sampling the PERIMETER, not just the four corners: rectifying bends
    straight edges, so a box's mapped outline bulges between its corners. Using
    corners alone loses the bulge and clips the person -- reintroducing, in
    miniature, the bad-box problem this module exists to remove.
    """
    x1, y1, x2, y2 = box[:4]
    n = max(2, BOX_EDGE_SAMPLES)
    pts = []
    for i in range(n):
        t = i / (n - 1.0)
        pts += [(x1 + (x2 - x1) * t, y1), (x1 + (x2 - x1) * t, y2),
                (x1, y1 + (y2 - y1) * t), (x2, y1 + (y2 - y1) * t)]
    mapped = [to_source(px, py, k, size) for px, py in pts]
    xs = [p[0] for p in mapped]
    ys = [p[1] for p in mapped]
    return (min(xs), min(ys), max(xs), max(ys)) + tuple(box[4:])


def straightness(points, k, size):
    """RMS deviation from a straight line, after rectifying `points`.

    `points` are clicked along something PHYSICALLY straight in the source
    image -- a door frame, a shelf edge, one row of floor tiles. If k is right
    they rectify onto a line; the residual is how wrong k is. Returns in
    normalised units (fraction of the half-diagonal), so it is comparable
    across cameras and resolutions.
    """
    pts = [to_rectified(x, y, k, size) for x, y in points]
    if len(pts) < 3:
        return 0.0
    n = float(len(pts))
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    syy = sum((p[1] - my) ** 2 for p in pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    # Total-least-squares line through the centroid: the normal is the
    # eigenvector of the scatter matrix with the SMALLER eigenvalue. Ordinary
    # least squares would blow up on a vertical door frame, which is exactly
    # the kind of line worth clicking.
    theta = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    nx, ny = -math.sin(theta), math.cos(theta)
    tx, ty = math.cos(theta), math.sin(theta)      # along the line
    resid = [((p[0] - mx) * nx + (p[1] - my) * ny) for p in pts]
    along = [((p[0] - mx) * tx + (p[1] - my) * ty) for p in pts]

    # NORMALISE BY THE CHAIN'S OWN LENGTH, not by the frame diagonal.
    #
    # THE BUG THIS FIXES, found 2026-08-13 when a fit landed on exactly the
    # search bound (+0.6000): dividing by a FIXED scale is not scale-invariant
    # in k. For k > 0 every point is pulled toward the centre, so the whole
    # chain shrinks and its absolute deviation shrinks with it -- a larger k
    # always scored "straighter" regardless of the lens. The objective was
    # monotonic, so golden-section walked to the edge of the range every time
    # and reported a confident number.
    #
    # A ratio of two lengths measured on the SAME rectified chain cannot be
    # gamed by scaling it. This is the difference between measuring bend and
    # measuring zoom.
    extent = math.sqrt(sum(a * a for a in along) / n)
    if extent < 1e-9:
        return 0.0
    return math.sqrt(sum(r * r for r in resid) / n) / extent


def fit_k(lines, size, bounds=K_SEARCH, iters=60):
    """-> (k, rms). The k that makes every supplied line straightest.

    `lines` is a list of point-lists, each clicked along one physically
    straight edge. Two lines in different parts of the frame beat one long
    one: a single line through the centre is nearly straight for EVERY k, so
    it cannot constrain anything -- the same "the check proves nothing" trap
    that four points hit in calibrate_plane.py.

    Golden-section search: the objective is smooth and unimodal in k over a
    real lens's range, and this needs no derivative and no scipy.
    """
    usable = [ln for ln in (lines or []) if len(ln) >= 3]
    if not usable:
        return 0.0, None

    def cost(k):
        return sum(straightness(ln, k, size) ** 2 for ln in usable) / len(usable)

    lo, hi = bounds
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = hi - gr * (hi - lo), lo + gr * (hi - lo)
    fa, fb = cost(a), cost(b)
    for _ in range(iters):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - gr * (hi - lo)
            fa = cost(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + gr * (hi - lo)
            fb = cost(b)
    k = (lo + hi) / 2.0
    return k, math.sqrt(cost(k))


def dewarped_predict(frame, predict_fn, k, remap_fn=None):
    """Rectify, detect, map the boxes back. Same contract as tiled_predict.

    predict_fn(image) -> [(x1,y1,x2,y2,conf,cls), ...]
    -> the same list, in SOURCE frame coordinates.

    remap_fn(frame, k) -> rectified image. Injected so this module never
    imports cv2 and stays testable with a fake detector and a fake lens;
    engine passes remap_cv2.
    """
    if abs(k) < 1e-9:
        return list(predict_fn(frame))          # no lens, no work
    h, w = frame.shape[:2] if hasattr(frame, "shape") else (None, None)
    if w is None:
        raise TypeError("dewarped_predict needs an array-like frame")
    rect = (remap_fn or _require_cv2_remap)(frame, k)
    return [box_to_source(b, k, (w, h)) for b in predict_fn(rect)]


def _require_cv2_remap(frame, k):
    return remap_cv2(frame, k)


_MAP_CACHE = {}


def remap_cv2(frame, k):
    """The one place cv2 is touched. Maps are cached per (size, k) because
    building them per frame costs more than the detector does."""
    import cv2
    import numpy as np
    h, w = frame.shape[:2]
    key = (w, h, round(float(k), 6))
    maps = _MAP_CACHE.get(key)
    if maps is None:
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy, s = _centre_scale((w, h))
        u, v = (xs - cx) / s, (ys - cy) / s
        f = 1.0 + k * (u * u + v * v)
        maps = ((cx + u * f * s).astype(np.float32),
                (cy + v * f * s).astype(np.float32))
        if len(_MAP_CACHE) > 4:                 # one camera, a couple of ks
            _MAP_CACHE.clear()
        _MAP_CACHE[key] = maps
    return cv2.remap(frame, maps[0], maps[1], interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)
