#!/usr/bin/env python3
"""Pixel-accurate distractor masks, MEASURED from the video, per modality.

WHY THIS EXISTS -- the "photoshop instead of polygons" question
--------------------------------------------------------------
The operator asked whether a pixel-accurate, edge-to-edge traced mask would
beat the 6-point polygons in zones/*.json. Two measurements decide it:

  1. The current "plant area mask" on CAM.112 is a RECTANGLE-ish hexagon
     covering 18.3% of the frame, and 93% of the main entry line lies inside
     it. _drop_masked deletes every detection whose feet land in a mask, so
     the door could never produce a crossing. A tight mask traced to the
     plant's actual silhouette would not reach the floor beside the door.
     => on footprint, pixel-accurate WINS, and by a lot.

  2. Only ~24% of the edges in a colour frame survive into an IR frame, and
     this camera flips modality ~96 times per 20 minutes. A mask hand-traced
     on a daylight still therefore describes a DIFFERENT ROOM at night.
     => one hand-traced mask LOSES. You need one per modality.

So: pixel-accurate yes, hand-traced no. Trace it from the pixels instead,
separately for colour and IR, which is what this tool does.

WHAT COUNTS AS A DISTRACTOR
---------------------------
"Static" alone is useless -- the floor and walls are static too, and masking
the floor is precisely the bug we are fixing. A false-person distractor is
static AND textured: a plant, a poster, a mirror frame, a coat stand. Plain
floor is static and FLAT. So:

    distractor = (low temporal variance) AND (high local edge energy)

Both thresholds are taken as PERCENTILES of this camera's own pixels, not
typed in, and both are printed so a run can be argued with.

THE SAFETY PROPERTY (the whole point)
-------------------------------------
This tool REFUSES to write a mask that covers an entry line, because that is
the failure it was built in response to. A mask that silently deletes the
door is worse than no mask at all.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# A FIXED saturation bar does not survive contact with a real file.
# Measured on keva_5min_FIXED.mp4, 40 frames (mean HSV saturation per frame):
#     IR frames      20.3 - 24.1
#     colour frames 133.2 - 176.5
# An IR frame is NOT saturation 0 once h264 4:2:0 chroma noise and any burnt-in
# overlay are in the picture. A typed IR_SAT_MAX = 12.0 classified all 1,097 IR
# frames as colour and built the IR mask from ZERO samples -- silently, because
# "0 frames" looked like a short video rather than a broken test.
#
# The gap is enormous (24 -> 133) but its LOCATION is per-video, so split this
# video's own frames instead of asserting a number. Absolute bars remain only
# as a sanity envelope for the single-modality case.
IR_SAT_ABS_MAX = 60.0      # nothing above this is plausibly infrared
COLOUR_SAT_ABS_MIN = 60.0
MIN_CLUSTER_SEP = 3.0      # split accepted only if between-gap > 3x within-spread


def sat_mean(frame_bgr) -> float:
    return float(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)[:, :, 1].mean())


def split_modalities(sats):
    """Return (threshold, bimodal). 1-D 2-means on this video's saturations.

    Refuses to invent a split when the video is one modality throughout: an
    all-colour video would otherwise be cut into 'dim colour' and 'bright
    colour' and produce a confident, meaningless IR mask.
    """
    v = np.asarray(sorted(float(x) for x in sats), np.float64)
    if v.size < 4:
        return float(np.mean(v)) if v.size else 0.0, False
    best = None
    for i in range(1, v.size):
        lo, hi = v[:i], v[i:]
        if lo.size < 2 or hi.size < 2:
            continue
        within = float(lo.std() + hi.std())
        gap = float(hi.mean() - lo.mean())
        score = gap / max(within, 1e-6)
        if best is None or score > best[0]:
            best = (score, float((lo.max() + hi.min()) / 2.0))
    if best is None:
        return float(v.mean()), False
    return best[1], best[0] >= MIN_CLUSTER_SEP


def modality_of(sat, threshold, bimodal) -> str:
    """'ir' or 'colour' for one frame's mean saturation."""
    if bimodal:
        return "ir" if sat <= threshold else "colour"
    return "ir" if sat <= IR_SAT_ABS_MAX else "colour"


class _Welford:
    """Streaming per-pixel mean/variance, so we never hold the frame stack."""

    def __init__(self):
        self.n = 0
        self.mean = None
        self.m2 = None

    def add(self, gray_f32):
        self.n += 1
        if self.mean is None:
            self.mean = gray_f32.copy()
            self.m2 = np.zeros_like(gray_f32)
            return
        d = gray_f32 - self.mean
        self.mean += d / self.n
        self.m2 += d * (gray_f32 - self.mean)

    @property
    def std(self):
        if self.n < 2:
            return None
        return np.sqrt(self.m2 / (self.n - 1))


def edge_energy(gray_u8):
    """Local texture. Sobel magnitude, blurred so single-pixel noise doesn't
    read as texture -- a plant is textured over an AREA, sensor noise is not."""
    gx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.GaussianBlur(cv2.magnitude(gx, gy), (0, 0), 3.0)


# ABSOLUTE floors, because a percentile alone is not a threshold.
# Caught by _self_check: on a frame that is mostly flat, percentile(edges, 75)
# can be ~0, so "edges >= thr" is true EVERYWHERE and the mask swallows the
# floor -- reproducing the exact door-deleting bug this tool exists to prevent.
# A percentile says "textured RELATIVE to this frame"; we also need "textured
# at all". Same in reverse for variance.
#
# CALIBRATION KNOBS, not laws of nature. Measured scale on 8-bit grey with
# Sobel-3 blurred at sigma 3: a plain painted wall with sensor noise sits at
# magnitude ~2-10, a plant/poster edge at ~50-300. A pixel nothing walks
# through has temporal std ~1-4 over minutes; one on a walking path ~20-60.
# Re-derive per camera if a new site disagrees; both are printed every run.
EDGE_FLOOR_DEFAULT = 25.0
STD_CEIL_DEFAULT = 8.0


def build_mask(std, edges, std_pct, edge_pct, min_area_frac, shape,
               edge_floor=EDGE_FLOOR_DEFAULT, std_ceil=STD_CEIL_DEFAULT):
    """distractor = quiet over time AND textured in space."""
    std_thr = min(float(np.percentile(std, std_pct)), float(std_ceil))
    edge_thr = max(float(np.percentile(edges, edge_pct)), float(edge_floor))
    raw = ((std <= std_thr) & (edges >= edge_thr)).astype(np.uint8)

    # Close pinholes inside a leafy plant, then drop confetti. A distractor is
    # a connected OBJECT; scattered pixels are noise however well they score.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    min_area = max(1.0, min_area_frac * shape[0] * shape[1])
    out = np.zeros_like(raw)
    kept, banded, boxes = [], [], []
    H, W = shape
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        # BURNT-IN OVERLAY GUARD. Measured on keva_5min_FIXED.mp4: the single
        # largest "distractor" found was x 0-1915, y 5-320 -- the HUD band and
        # the NVR's burnt-in timestamp, not anything in the room. An overlay is
        # perfectly static and highly textured, so it scores better than any
        # real object and can dominate the whole mask.
        # A room object does not span the full frame width in a thin strip at
        # an edge. Drop those and say so, rather than silently masking a
        # caption and calling it a plant.
        if w >= 0.90 * W and h <= 0.30 * H and (y <= 0.05 * H or y + h >= 0.95 * H):
            banded.append((int(x), int(y), int(w), int(h)))
            continue
        out[lab == i] = 1
        kept.append(int(stats[i, cv2.CC_STAT_AREA]))
        boxes.append((int(x), int(y), int(w), int(h)))
    boxes.sort(key=lambda b: -b[2] * b[3])
    return out, dict(std_thr=std_thr, edge_thr=edge_thr,
                     components=len(kept), areas=sorted(kept, reverse=True)[:8],
                     overlay_bands=banded, boxes=boxes)


def line_hits(mask, p1, p2, samples=201):
    """Fraction of an entry line's length that lands on the mask."""
    h, w = mask.shape
    hit = 0
    for t in np.linspace(0.0, 1.0, samples):
        x = int(round(p1[0] + (p2[0] - p1[0]) * t))
        y = int(round(p1[1] + (p2[1] - p1[1]) * t))
        if 0 <= x < w and 0 <= y < h and mask[y, x]:
            hit += 1
    return hit / float(samples)


def scaled_lines(zones_cfg, shape):
    """Entry lines in MASK pixels. Zone files are authored at source
    resolution; a mask built from analysed frames is smaller. Comparing the
    two without scaling is how a check passes while the bug is still there."""
    h, w = shape
    fw, fh = (zones_cfg.get("frame_size") or [w, h])[:2]
    sx, sy = w / float(fw), h / float(fh)
    out = {}
    for name, pts in (zones_cfg.get("entry_lines") or {}).items():
        try:
            (x1, y1), (x2, y2) = pts[0], pts[1]
        except (TypeError, ValueError, IndexError):
            continue
        out[name] = ((x1 * sx, y1 * sy), (x2 * sx, y2 * sy))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--zones", help="zone json; entry lines are PROTECTED")
    ap.add_argument("--out-dir", default="zones")
    ap.add_argument("--cam", default="CAM")
    ap.add_argument("--frames", type=int, default=120, help="frames sampled per video")
    ap.add_argument("--max-w", type=int, default=1920,
                    help="match ANALYSIS_MAX_W — the mask must live in the "
                         "coordinate system the detector runs in")
    ap.add_argument("--std-pct", type=float, default=25.0,
                    help="quietest N%% of pixels count as static")
    ap.add_argument("--edge-pct", type=float, default=75.0,
                    help="most textured N%% count as textured")
    ap.add_argument("--min-area-frac", type=float, default=0.002)
    ap.add_argument("--edge-floor", type=float, default=EDGE_FLOOR_DEFAULT,
                    help="absolute Sobel magnitude a pixel must reach to count "
                         "as textured, whatever the percentile says")
    ap.add_argument("--std-ceil", type=float, default=STD_CEIL_DEFAULT,
                    help="absolute temporal std a pixel must stay under to "
                         "count as static, whatever the percentile says")
    ap.add_argument("--max-line-overlap", type=float, default=0.02,
                    help="REFUSE to write if any entry line is covered more "
                         "than this. Default 2%%: essentially, never.")
    ap.add_argument("--write", action="store_true", help="write files")
    a = ap.parse_args(argv)

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        print(f"!! cannot open {a.video}", file=sys.stderr)
        return 2
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = set(np.linspace(0, max(total - 1, 0), min(a.frames, total)).astype(int).tolist())

    def _scan(fn):
        """SEEK to each sampled frame instead of decoding the whole file.

        The sequential version read every frame twice. That is fine for a
        5-minute clip and catastrophic for the real input: the source chunks
        are 1-hour 4K files of ~3.4 GB, i.e. ~108,000 frames decoded twice to
        look at 120 of them. Seeking costs 120 keyframe jumps.
        """
        for j in sorted(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(j))
            ok, fr = cap.read()
            if not ok:
                continue
            if fr.shape[1] > a.max_w:
                fr = cv2.resize(fr, (a.max_w,
                                     int(round(fr.shape[0] * a.max_w / fr.shape[1]))),
                                interpolation=cv2.INTER_AREA)
            fn(fr)

    # Pass 1 -- what modalities does this video even contain?
    sats = []
    _scan(lambda fr: sats.append(sat_mean(fr)))
    thr, bimodal = split_modalities(sats)
    print(f"\nmodality split: threshold {thr:.1f} "
          f"({'BIMODAL — this video flips' if bimodal else 'single modality'})"
          f"   saturation {min(sats):.1f}..{max(sats):.1f}")

    # Pass 2 -- accumulate per-pixel statistics, per modality.
    acc = {"ir": _Welford(), "colour": _Welford()}
    last = {}

    def _accum(fr):
        m = modality_of(sat_mean(fr), thr, bimodal)
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        acc[m].add(g.astype(np.float32))
        last[m] = g

    _scan(_accum)
    cap.release()

    zones_cfg = json.load(open(a.zones)) if a.zones else {}
    out_dir = Path(a.out_dir)
    results, refused = {}, []

    print(f"\nsampled {len(idxs)} of {total} frames from {Path(a.video).name}")
    for m in ("colour", "ir"):
        w = acc[m]
        if w.n < 8:
            print(f"\n[{m}] only {w.n} frame(s) -- SKIPPED "
                  f"(a variance estimate needs a sample, not an anecdote)")
            continue
        std = w.std
        mask, info = build_mask(std, edge_energy(last[m]),
                                a.std_pct, a.edge_pct, a.min_area_frac, std.shape,
                                edge_floor=a.edge_floor, std_ceil=a.std_ceil)
        cov = float(mask.mean())
        print(f"\n[{m}] {w.n} frames  {std.shape[1]}x{std.shape[0]}")
        print(f"   std<= {info['std_thr']:.2f} (p{a.std_pct:g})   "
              f"edge>= {info['edge_thr']:.1f} (p{a.edge_pct:g})")
        print(f"   coverage {cov*100:.1f}% of frame   "
              f"{info['components']} component(s)   top areas {info['areas']}")
        for (bx, by, bw, bh) in info["overlay_bands"]:
            print(f"   !! dropped a full-width band at x{bx}-{bx+bw} y{by}-{by+bh} "
                  f"— that is a burnt-in overlay/HUD, not a room object. "
                  f"Build masks from the RAW source, not an annotated render.")
        for i, (cx, cy, cw, ch) in enumerate(info["boxes"][:5]):
            print(f"      component {i+1}: x{cx}-{cx+cw} y{cy}-{cy+ch}")

        for name, (p1, p2) in scaled_lines(zones_cfg, std.shape).items():
            f = line_hits(mask, p1, p2)
            flag = "OK" if f <= a.max_line_overlap else "REFUSE"
            print(f"   entry line {name!r:16s} covered {f*100:5.1f}%  {flag}")
            if f > a.max_line_overlap:
                refused.append((m, name, f))
        results[m] = (mask, cov)

    if refused:
        print("\n!! NOT WRITING. A mask covering an entry line deletes the "
              "detections that line exists to count:")
        for m, name, f in refused:
            print(f"     {m}: {name!r} covered {f*100:.1f}%")
        print("   Lower --std-pct / raise --edge-pct, or shrink the mask, and re-run.")
        return 1

    if not a.write:
        print("\n(dry run -- pass --write to save)")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for m, (mask, cov) in results.items():
        p = out_dir / f"{a.cam}_static_mask_{m}.png"
        cv2.imwrite(str(p), (mask * 255).astype(np.uint8))
        print(f"wrote {p}  ({cov*100:.1f}% coverage)")
    return 0


def _self_check():
    """One runnable check: a flat static floor must NOT be masked, a textured
    static object MUST be, and the line guard must actually refuse."""
    rng = np.random.default_rng(0)
    h, w = 200, 300
    std = np.full((h, w), 3.0, np.float32)          # quiet: sensor noise only
    edges = rng.uniform(1.0, 8.0, (h, w)).astype(np.float32)   # wall + noise
    edges[50:150, 100:200] = 500.0                  # ...except one textured block
    mask, info = build_mask(std, edges, 50.0, 75.0, 0.002, (h, w))
    assert mask[100, 150] == 1, "textured static object was not masked"
    assert mask[10, 10] == 0, "flat static floor got masked -- this is the door bug"

    # A moving textured object must survive: high variance disqualifies it.
    std2 = std.copy(); std2[50:150, 100:200] = 90.0
    mask2, _ = build_mask(std2, edges, 50.0, 75.0, 0.002, (h, w))
    assert mask2[100, 150] == 0, "a MOVING object was masked -- that deletes people"

    # The guard: a line straight through the block must report ~full coverage.
    assert line_hits(mask, (150, 50), (150, 149)) > 0.9
    assert line_hits(mask, (10, 10), (10, 40)) == 0.0

    # Modality split: the REAL measured values from keva_5min_FIXED.mp4.
    ir_like = [20.3, 19.7, 20.3, 21.5, 23.5, 22.7, 24.1, 21.6]
    col_like = [133.2, 152.2, 158.0, 163.9, 166.8, 176.5, 151.0, 161.7]
    t, bi = split_modalities(ir_like + col_like)
    assert bi, "a video that visibly flips IR<->colour was called single-modality"
    assert 24.1 < t < 133.2, f"split landed at {t}, outside the measured gap"
    assert modality_of(20.3, t, bi) == "ir"
    assert modality_of(163.9, t, bi) == "colour"
    # And it must NOT invent a split in a single-modality video.
    _, bi2 = split_modalities([140.0, 143.1, 139.4, 145.2, 141.0, 142.7, 144.9, 138.8])
    assert not bi2, "invented an IR cluster inside an all-colour video"

    print("build_static_mask self-check OK")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        raise SystemExit(main())
