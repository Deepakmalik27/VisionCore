"""camera_health.py — PHASE 4 / U1. "Is this still the same camera view?"

THE BUG THIS CLOSES
    Nothing in the pipeline ever checked that the camera is still pointing where
    it was pointing when the zones were drawn. I grepped for it: the 'drift'
    hits are VFR clock drift and stationary-merge pixels, 'stabili' is a
    comment. There is no such check.

    If a camera is knocked, re-aimed, or is a PTZ that returns to a slightly
    different preset, EVERY zone polygon is silently in the wrong place. The run
    completes, the charts render, the report prints confident numbers, and all
    of them are wrong. Over a ten-hour night and across venues this is not a
    hypothetical.

THE RULE (Prabh's call, and the right one)
    Detect it and REFUSE TO SCORE. A missing number is recoverable; a confident
    wrong number gets quoted in a meeting. So this module never silently
    corrects and never silently continues — it returns a verdict, and the
    caller marks every zone-dependent metric INVALID.

WHY THE TOLERANCE IS MEASURED ON THE ZONES, NOT ON THE FRAME
    What matters is not "did the image move" but "did the image move enough to
    put my polygons in the wrong place". A 2-pixel shift is irrelevant. A small
    ROTATION is irrelevant at the image centre and severe at the edges, which is
    exactly where a doorway usually is. So the tolerance is applied to the
    displacement of the ACTUAL ZONE VERTICES under the estimated transform.

FOUR MORE WAYS THE VIDEO IS NOT WHAT YOU THINK IT IS
    Cheap to check, all of them silent failures today:
      out of focus   lens knocked, dirty, or auto-focus hunting
      blinded        direct sun or a light shining into the lens
      lens blocked   something in front of the camera
      frozen stream  the NVR wrote the same frame repeatedly (a real export bug)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception:                                     # pragma: no cover
    cv2 = None

# Defaults are expressed as FRACTIONS of the frame, never as raw pixels, so the
# same profile is correct on 720p and on 4K without being retuned.
#
# HOW THIS NUMBER WAS CHOSEN (it started as a guess at 0.012 and was wrong):
# A tolerance has to sit above the estimator's own noise and below the point
# where a zone-boundary decision can flip. Both ends were MEASURED, not guessed:
#
#   noise floor      0.03 px   worst false shift with the camera genuinely still,
#                              across 14 people walking through, lights dimmed
#                              45%, jpeg q=30 and sensor noise combined. RANSAC
#                              on background features is far steadier than
#                              expected — this end is not the constraint.
#   answer flips    ~10 px     one third of a person's shoulder width at
#                              mid-room depth (a ~108 px tall person is ~31 px
#                              wide at 720p). Beyond this, someone standing at
#                              the edge of the reception polygon can be counted
#                              on the wrong side of it.
#
# 0.8% of the diagonal is 11.7 px at 720p, 21 px at 1080p, 42 px at 4K — about
# 390x the noise floor and just at the answer-flip point. The original 0.012
# was 17.6 px at 720p, i.e. ALREADY past the point where answers change.
DEFAULT_ZONE_TOL_FRAC = 0.008
DEFAULT_MIN_INLIERS = 12
BLUR_VAR_FLOOR = 40.0             # Laplacian variance below this = out of focus
BRIGHT_HI, BRIGHT_LO = 242.0, 12.0
EDGE_DENSITY_FLOOR = 0.004        # fraction of pixels that are edges
FROZEN_DIFF_FLOOR = 0.2           # mean abs diff below this = identical frame


def _prep(frame, max_w=960):
    """Grayscale, downscaled, contrast-normalised.

    Equalising matters more than it looks: the same room in daylight and under
    infrared produces very different raw intensities, and without normalisation
    the feature matcher would report 'the camera moved' every dusk.
    """
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    if g.shape[1] > max_w:
        s = max_w / g.shape[1]
        g = cv2.resize(g, (int(g.shape[1] * s), int(g.shape[0] * s)))
    return cv2.equalizeHist(g)


class CameraHealth:
    """Reference view + the checks that say whether a frame still matches it."""

    def __init__(self, ref_gray, ref_shape, zone_tol_frac=DEFAULT_ZONE_TOL_FRAC,
                 min_inliers=DEFAULT_MIN_INLIERS):
        self.ref = ref_gray
        self.ref_shape = tuple(ref_shape)          # (w, h) of the ORIGINAL frame
        self.zone_tol_frac = zone_tol_frac
        self.min_inliers = min_inliers
        self._orb = cv2.ORB_create(nfeatures=1500) if cv2 else None
        self._kp, self._des = (self._orb.detectAndCompute(ref_gray, None)
                               if self._orb is not None else (None, None))
        self._scale = ref_gray.shape[1] / max(self.ref_shape[0], 1)

    # -- lifecycle -----------------------------------------------------------
    @classmethod
    def from_frame(cls, frame, **kw):
        h, w = frame.shape[:2]
        return cls(_prep(frame), (w, h), **kw)

    def save(self, path):
        """Persist so chunk 7 is compared against chunk 1's view, not its own.

        V73: append the extension, never with_suffix(). Callers build the path
        as `viewref_{camera_id}` with no extension, and this venue's camera_id
        ends "...5.30.00pm CDT" — pathlib treats ".00pm CDT" as the suffix and
        REPLACES it. save() then wrote one filename and load() looked for
        another, so the camera-moved guard silently compared against nothing on
        every run. The 68b97311f9 log shows the mangled name:
            viewref_CAM.112 (PP.09_12) 7-28-2026, 4.30.00pm CDT - ..., 5.30.png
        """
        p = Path(path)
        cv2.imwrite(str(Path(str(p) + ".png")), self.ref)
        Path(str(p) + ".json").write_text(json.dumps({
            "ref_shape": list(self.ref_shape),
            "zone_tol_frac": self.zone_tol_frac,
            "min_inliers": self.min_inliers}))
        return p

    @classmethod
    def load(cls, path):
        p = Path(path)
        img = cv2.imread(str(Path(str(p) + ".png")), cv2.IMREAD_GRAYSCALE)  # V73
        if img is None:
            return None
        meta = json.loads(Path(str(p) + ".json").read_text())               # V73
        return cls(img, meta["ref_shape"], meta.get("zone_tol_frac", DEFAULT_ZONE_TOL_FRAC),
                   meta.get("min_inliers", DEFAULT_MIN_INLIERS))

    # -- the checks ----------------------------------------------------------
    def estimate_shift(self, frame):
        """Affine transform from the reference view to this frame.

        ORB + RANSAC on purpose. People walking through the scene are outliers
        by definition, and RANSAC is what makes 'the background agrees' the
        thing being measured rather than 'the pixels agree'.
        """
        cur = _prep(frame)
        if self._des is None or self._orb is None:
            return None, 0
        kp2, des2 = self._orb.detectAndCompute(cur, None)
        if des2 is None or len(kp2) < 4 or len(self._kp) < 4:
            return None, 0
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(self._des, des2)
        if len(matches) < 4:
            return None, 0
        matches = sorted(matches, key=lambda m: m.distance)[:400]
        src = np.float32([self._kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                             ransacReprojThreshold=3.0,
                                             maxIters=3000, confidence=0.995)
        n_inl = int(inl.sum()) if inl is not None else 0
        return M, n_inl

    def zone_displacement(self, M, polygons):
        """Worst zone-vertex displacement in ORIGINAL-frame pixels.

        The number that actually decides whether the report is valid, because a
        zone is what the metrics are computed over.
        """
        if M is None:
            return None
        pts = []
        for poly in (polygons or {}).values():
            pts.extend([(float(x), float(y)) for x, y in np.asarray(poly).reshape(-1, 2)])
        if not pts:
            w, h = self.ref_shape                    # fall back to frame corners
            pts = [(0, 0), (w, 0), (w, h), (0, h)]
        worst = 0.0
        for x, y in pts:
            xs, ys = x * self._scale, y * self._scale        # into working scale
            nx = M[0, 0] * xs + M[0, 1] * ys + M[0, 2]
            ny = M[1, 0] * xs + M[1, 1] * ys + M[1, 2]
            worst = max(worst, math.hypot(nx - xs, ny - ys) / max(self._scale, 1e-9))
        return worst

    def image_quality(self, frame, prev_frame=None):
        """The other four ways the video is not what you think it is."""
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        small = cv2.resize(g, (320, 180))
        blur = float(cv2.Laplacian(small, cv2.CV_64F).var())
        bright = float(small.mean())
        edges = float((cv2.Canny(small, 50, 150) > 0).mean())
        frozen = None
        if prev_frame is not None:
            pg = (cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                  if prev_frame.ndim == 3 else prev_frame)
            frozen = float(cv2.absdiff(small, cv2.resize(pg, (320, 180))).mean())
        problems = []
        if blur < BLUR_VAR_FLOOR:
            problems.append(f"OUT OF FOCUS (sharpness {blur:.0f} < {BLUR_VAR_FLOOR:.0f})")
        if bright > BRIGHT_HI:
            problems.append(f"BLINDED / over-exposed (brightness {bright:.0f})")
        elif bright < BRIGHT_LO:
            problems.append(f"TOO DARK to detect anything (brightness {bright:.0f})")
        if edges < EDGE_DENSITY_FLOOR:
            problems.append(f"LENS BLOCKED (edge density {edges:.4f})")
        if frozen is not None and frozen < FROZEN_DIFF_FLOOR:
            problems.append(f"FROZEN STREAM (consecutive frames identical, diff {frozen:.3f})")
        return {"blur": blur, "brightness": bright, "edge_density": edges,
                "frozen_diff": frozen, "problems": problems}

    def check(self, frame, polygons=None, prev_frame=None):
        """Full verdict. `valid` False means: do NOT publish zone-based numbers."""
        w, h = frame.shape[1], frame.shape[0]
        q = self.image_quality(frame, prev_frame)
        res = {"resolution": (w, h), "ref_resolution": self.ref_shape,
               "quality": q, "moved": False, "valid": True, "reasons": [],
               "zone_shift_px": None, "inliers": 0, "rotation_deg": None,
               "scale": None}

        # U2 — resolution change. Zones are pixel coordinates; a different frame
        # size means they point at different parts of the room.
        if (w, h) != self.ref_shape:
            res["valid"] = False
            res["reasons"].append(
                f"RESOLUTION CHANGED {self.ref_shape[0]}x{self.ref_shape[1]} -> "
                f"{w}x{h}; zone coordinates no longer refer to the same places")
            return res

        M, n_inl = self.estimate_shift(frame)
        res["inliers"] = n_inl
        if M is None or n_inl < self.min_inliers:
            res["valid"] = False
            res["reasons"].append(
                f"CANNOT VERIFY the view ({n_inl} matching background features, "
                f"need {self.min_inliers}). Either the scene changed completely "
                f"or the image is too degraded to compare.")
            return res

        res["rotation_deg"] = float(math.degrees(math.atan2(M[1, 0], M[0, 0])))
        res["scale"] = float(math.hypot(M[0, 0], M[1, 0]))
        shift = self.zone_displacement(M, polygons)
        res["zone_shift_px"] = shift
        tol = self.zone_tol_frac * math.hypot(*self.ref_shape)
        res["zone_tol_px"] = tol
        if shift is not None and shift > tol:
            res["moved"] = True
            res["valid"] = False
            res["reasons"].append(
                f"CAMERA MOVED: zone corners are {shift:.0f} px out of place "
                f"(tolerance {tol:.0f} px, rotation {res['rotation_deg']:+.2f} deg, "
                f"scale {res['scale']:.3f}). Every zone-based number would be "
                f"measuring the wrong part of the room. Re-draw the zones on a "
                f"current frame, or restore the camera to its original aim.")
        if q["problems"]:
            res["valid"] = False
            res["reasons"].extend(q["problems"])
        return res


def verdict_line(res):
    """One line for the run log. Green means the numbers may be published."""
    if res["valid"]:
        s = res.get("zone_shift_px")
        return (f"OK  camera view unchanged"
                + (f" (zones off by {s:.1f} px, tolerance "
                   f"{res.get('zone_tol_px', 0):.0f} px, {res['inliers']} features)"
                   if s is not None else ""))
    return "INVALID  " + " | ".join(res["reasons"])
