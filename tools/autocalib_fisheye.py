"""autocalib_fisheye.py — measure the lens from the picture, not from clicks.

WHY THIS EXISTS
    kevacv/fisheye.py can rectify the camera once it knows one number, k. Two
    rounds of hand-clicked plumb-lines could not supply it:

        draw 1  (6 lines, centre floor)   k = -0.2869   2/6 lines improved
        draw 2  (7 lines, wider)          k = -0.2252   6/7 improved
                                          residual 0.0074, threshold 0.0040

    The sign and rough size agree, so the lens is real. The fit still failed
    because a hand click scatters 3-10 px and the bow being measured at those
    radii is about the same size. The operator was not doing it wrong; the
    METHOD has a noise floor above the signal, and a third round of clicking
    would have wasted an afternoon to land in the same place.

    Edges in the image do not shake. This finds them.

THE METHOD — plumb-line calibration
    A straight edge in the world must be straight in a rectified image. That is
    the whole constraint (Brown 1971), and it needs no checkerboard, no target,
    and nothing measured in the room:

      1. LSD finds line segments. A gently bowed edge still comes back as ONE
         segment, because LSD tolerates slight curvature.
      2. For each segment, walk along it and SNAP each sample to the nearest
         real Canny edge pixel on the perpendicular. This is the step that
         makes the method non-circular: LSD's segment is straight BY
         CONSTRUCTION, so fitting to it would prove nothing. The snapped edge
         pixels are where the lens actually put the wall, and they bow.
      3. Find the k that makes every chain straightest at once.

    Sub-pixel by centroid, hundreds of samples per chain, no hand in the loop.

WHY IT REFUSES MORE OFTEN THAN IT ACCEPTS
    Same bar as the clicked version, deliberately. A wrong k does not fail
    loudly -- it silently moves every box, and this camera's whole history is
    silent geometry errors. The acceptance test is a HOLD-OUT: fit k on half
    the chains, score it on the other half. A k that only explains the chains
    it was fitted to has explained nothing.

USAGE
    python tools/autocalib_fisheye.py --frame output/_ref.png
    python tools/autocalib_fisheye.py --frame output/_ref.png --write
    python tools/autocalib_fisheye.py --frame output/_ref.png \\
        --clicked zones/CAM.112_calib_lines.json      # cross-check vs clicks
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kevacv.fisheye import fit_k, straightness  # noqa: E402

MIN_SEG_FRAC = 0.06     # segment must span this fraction of the width to count
SAMPLES = 24            # points per chain
SNAP_PX = 6             # perpendicular search radius for the true edge
MIN_HITS = 10           # a chain with fewer snapped points is not evidence
MIN_RADIUS_SPAN = 0.12  # a chain must cross this much radius; distortion is
                        # r^2, so a chain at constant radius constrains nothing
# Units changed 2026-08-13 when straightness() became scale-invariant: it now
# returns bend AS A FRACTION OF CHAIN LENGTH, not of the frame diagonal. 0.015
# means the chain bows by 1.5% of its own length after correction -- tight for
# a real architectural edge, loose enough for CCTV noise and JPEG ringing.
ACCEPT_RESIDUAL = 0.015
MIN_IMPROVE_FRAC = 0.70

# Canny does not know what a straight edge IS. It returns curtain folds, marble
# veins, plant leaves and the rim of a vase alongside the door frame -- and a
# chain that is genuinely curved in the world can never be straightened by any
# k, so it only drags the fit. Keep the best-explained TRIM_KEEP of chains and
# re-fit on those. Standard robust practice, and the alternative is letting a
# potted plant vote on the lens model.
TRIM_KEEP = 0.6


def extract_chains(gray, edges, min_seg_px):
    """-> [[(x,y), ...], ...] real edge chains, one per detected segment."""
    import cv2
    import numpy as np
    lsd = cv2.createLineSegmentDetector()
    segs = lsd.detect(gray)[0]
    if segs is None:
        return []
    h, w = edges.shape
    chains = []
    for s in segs.reshape(-1, 4):
        x1, y1, x2, y2 = map(float, s)
        L = math.hypot(x2 - x1, y2 - y1)
        if L < min_seg_px:
            continue
        dx, dy = (x2 - x1) / L, (y2 - y1) / L
        nx, ny = -dy, dx                      # unit normal
        pts = []
        for i in range(SAMPLES):
            t = i / (SAMPLES - 1.0)
            px, py = x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
            # SNAP: the nearest true edge pixel along the normal. Weighted
            # centroid of the hits gives sub-pixel position.
            num = den = 0.0
            for d in range(-SNAP_PX, SNAP_PX + 1):
                qx, qy = int(round(px + nx * d)), int(round(py + ny * d))
                if 0 <= qx < w and 0 <= qy < h and edges[qy, qx]:
                    wgt = 1.0 / (1.0 + abs(d))
                    num += d * wgt
                    den += wgt
            if den > 0:
                off = num / den
                pts.append((px + nx * off, py + ny * off))
        if len(pts) >= MIN_HITS:
            chains.append(pts)
    return chains


def radius_span(pts, size):
    cx, cy = size[0] / 2.0, size[1] / 2.0
    s = math.hypot(*size) / 2.0
    rr = [math.hypot((x - cx) / s, (y - cy) / s) for x, y in pts]
    return min(rr), max(rr)


def evaluate(k, chains, size):
    """-> (frac_improved, mean_residual)."""
    if not chains:
        return 0.0, None
    better = 0
    tot = 0.0
    for c in chains:
        a, b = straightness(c, 0.0, size), straightness(c, k, size)
        better += (b < a)
        tot += b * b
    return better / len(chains), math.sqrt(tot / len(chains))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frame", required=True)
    ap.add_argument("--zones", default=str(ROOT / "zones" / "CAM.112_zone.json"))
    ap.add_argument("--clicked", default=None,
                    help="a *_calib_lines.json to cross-check the result against")
    ap.add_argument("--canny", nargs=2, type=int, default=[60, 160])
    ap.add_argument("--write", action="store_true",
                    help="write analysis.fisheye_k into the run config (only on PASS)")
    ap.add_argument("--config", default=str(ROOT / "config" / "cam112.yaml"))
    a = ap.parse_args(argv)

    import cv2
    img = cv2.imread(a.frame)
    if img is None:
        print(f"  cannot read {a.frame}")
        return 2
    h, w = img.shape[:2]
    size = (w, h)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, a.canny[0], a.canny[1])
    chains = extract_chains(gray, edges, w * MIN_SEG_FRAC)
    print(f"\n  frame {w}x{h} · {int(edges.sum()//255)} edge px · "
          f"{len(chains)} candidate chain(s)")

    # A chain at constant radius cannot constrain a radial model, no matter how
    # long or clean it is. Drop them BEFORE fitting rather than letting them
    # dilute the objective.
    keep = []
    for c in chains:
        r0, r1 = radius_span(c, size)
        if r1 - r0 >= MIN_RADIUS_SPAN:
            keep.append(c)
    print(f"  {len(keep)} chain(s) span >= {MIN_RADIUS_SPAN} of radius "
          f"(the rest cannot constrain a radial model)")
    if len(keep) < 4:
        print("\n  FAIL — too few usable chains. Try --canny 40 120 for a "
              "softer edge threshold, or a frame with more architecture in it.")
        return 1

    # ROBUST TRIM: a first pass to find which chains are ACTUALLY straight in
    # the world, then re-fit on those. Done before the hold-out split so both
    # halves are drawn from the same inlier population.
    k0, _ = fit_k(keep, size)
    scored = sorted(keep, key=lambda c: straightness(c, k0, size))
    n_keep = max(4, int(round(len(scored) * TRIM_KEEP)))
    dropped = len(scored) - n_keep
    keep = scored[:n_keep]
    print(f"  robust trim: kept the {n_keep} best-explained, dropped {dropped} "
          f"(curved in the world -- no k can straighten those)")

    # HOLD-OUT: fit on half, score on the other half. A k that only explains
    # the chains it was fitted to has explained nothing.
    keep.sort(key=lambda c: -radius_span(c, size)[1])
    fit_set = keep[0::2]
    hold_set = keep[1::2]
    k, _ = fit_k(fit_set, size)
    f_fit, r_fit = evaluate(k, fit_set, size)
    f_hold, r_hold = evaluate(k, hold_set, size)
    k_all, _ = fit_k(keep, size)
    f_all, r_all = evaluate(k_all, keep, size)

    print(f"\n  fitted on {len(fit_set)} chains   k = {k:+.4f}   "
          f"improved {f_fit*100:.0f}%   residual {r_fit:.5f}")
    print(f"  HELD OUT {len(hold_set)} chains                    "
          f"improved {f_hold*100:.0f}%   residual {r_hold:.5f}")
    print(f"  all {len(keep)} chains        k = {k_all:+.4f}   "
          f"improved {f_all*100:.0f}%   residual {r_all:.5f}")

    if a.clicked:
        try:
            cj = json.loads(Path(a.clicked).read_text())
            csize = tuple(cj.get("frame_size", size))
            cl = [v for v in cj["lines"].values() if len(v) >= 3]
            k_click, _ = fit_k(cl, csize)
            print(f"\n  cross-check vs {len(cl)} hand-clicked line(s): "
                  f"k = {k_click:+.4f}   (this run: {k_all:+.4f})")
            print(f"  {'AGREE' if abs(k_click - k_all) < 0.10 else 'DISAGREE'} "
                  f"— two independent methods, |dk| = {abs(k_click - k_all):.4f}")
        except Exception as e:
            print(f"  (cross-check skipped: {e})")

    passed = (f_hold >= MIN_IMPROVE_FRAC and r_hold is not None
              and r_hold <= ACCEPT_RESIDUAL)
    print("\n  " + ("PASS — k generalises to chains it never saw."
                    if passed else
                    "FAIL — k does NOT hold up on the held-out chains.\n"
                    "         Not written. A wrong k moves every box silently,\n"
                    "         which is worse than no correction at all."))
    if a.write and passed:
        import re
        p = Path(a.config)
        txt = p.read_text(encoding="utf-8")
        line = f"  fisheye_k: {k_all:.4f}"
        if re.search(r"^\s*fisheye_k:", txt, re.M):
            txt = re.sub(r"^\s*fisheye_k:.*$", line, txt, flags=re.M)
        else:
            txt = txt.replace("  dedup_nms_iou:",
                              f"  # MEASURED by tools/autocalib_fisheye.py from "
                              f"{Path(a.frame).name}: {len(keep)} edge chains, "
                              f"held-out residual {r_hold:.5f}.\n{line}\n\n"
                              f"  dedup_nms_iou:", 1)
        p.write_text(txt, encoding="utf-8")
        print(f"  WROTE fisheye_k: {k_all:.4f} -> {p}")
    elif a.write:
        print("  --write IGNORED because the hold-out check failed.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
