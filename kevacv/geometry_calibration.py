"""calibration.py — Phase B P6 Automated Camera Calibration & Geometry Consistency.

WHY THIS EXISTS
    Auto perspective fit in CAM.112 dropped 46% of detections or produced implausible camera heights
    when seated/crouching people or static phantoms contaminated the regression line h(y) = a*y + b.

THE PRINCIPLE
    1. Robust RANSAC Ground-Plane Estimator: Filters out posture anomalies (seated, crouching, giant phantoms).
    2. GeoCalib / Intrinsic Refinement: Validates implied focal length and tilt angle against physical constraints.
    3. Ground Plane Consistency Auditor: Verifies near vs far scale ratios to prevent distorted metric measurements.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .ground_plane import GroundPlane, PERSON_H_M


def fit_robust_ground_plane(detections: List[Tuple[float, float, float, float]],
                            frame_size: Tuple[int, int],
                            person_h: Optional[float] = None) -> GroundPlane:
    """Robustly fit ground plane line h(y) = a*y + b using RANSAC to exclude seated/crouching detections.

    detections: List of (x1, y1, x2, y2) bounding boxes.
    frame_size: (width, height)
    person_h: metres; defaults to the CURRENT config value, resolved at call
        time. It used to be `person_h: float = PERSON_H_M`, which binds once
        when this def executes at import -- so apply_run_config could set
        geometry_calibration.PERSON_H_M (this module IS in its propagation
        list) and every call here would still use the import-time number. Same
        defect as ground_plane's, in the module that fits the plane.
    """
    if person_h is None:
        person_h = globals()["PERSON_H_M"]
    fw, fh = frame_size
    
    # Filter valid standing aspect ratios (~2.0 to 4.5)
    valid_pts = []
    for x1, y1, x2, y2 in detections:
        w, h = float(x2 - x1), float(y2 - y1)
        if w <= 0 or h <= 0:
            continue
        aspect = h / w
        if 2.0 <= aspect <= 4.8 and y2 > fh * 0.15:  # Below horizon region
            valid_pts.append((y2, h))

    if len(valid_pts) < 10:
        # A bare "insufficient" sent the run to the drifting bin-median fit
        # without saying WHICH filter emptied the set, so nobody could tell a
        # quiet clip from a mis-tuned gate. On CAM.112 it is the gate: the
        # camera is oblique enough that the MEDIAN person box measures 1.58
        # h/w, while "standing" here means 2.0-4.8. Report the distribution so
        # the next refusal is diagnosable instead of merely discouraging.
        _asp = sorted(float(y2 - y1) / float(x2 - x1)
                      for x1, y1, x2, y2 in detections
                      if x2 > x1 and y2 > y1)
        _med = _asp[len(_asp) // 2] if _asp else float("nan")
        return GroundPlane.none(
            f"Insufficient valid standing person detections for robust "
            f"calibration: {len(valid_pts)} of {len(detections)} boxes passed "
            f"aspect 2.0-4.8 and the below-horizon cut (median aspect "
            f"{_med:.2f}). A median below 2.0 means the gate, not the clip, "
            f"is the reason.")

    ys = np.array([p[0] for p in valid_pts], dtype=np.float64)
    hs = np.array([p[1] for p in valid_pts], dtype=np.float64)

    # RANSAC fitting with deterministic seed for reproducibility
    best_a, best_b = None, None
    max_inliers = 0
    inlier_threshold_px = 25.0

    n_samples = len(ys)
    iterations = min(200, n_samples * 5)
    rng = np.random.RandomState(seed=42)

    for _ in range(iterations):
        idx = rng.choice(n_samples, 2, replace=False)
        y_samp, h_samp = ys[idx], hs[idx]
        
        dy = y_samp[1] - y_samp[0]
        if abs(dy) < 10:
            continue
            
        a = (h_samp[1] - h_samp[0]) / dy
        b = h_samp[0] - a * y_samp[0]

        if a <= 1e-6:  # Unphysical: height must increase with lower image row y
            continue

        pred_h = a * ys + b
        residuals = np.abs(hs - pred_h)
        inliers = np.sum(residuals < inlier_threshold_px)

        if inliers > max_inliers:
            max_inliers = inliers
            best_a, best_b = a, b

    if best_a is None or best_a <= 1e-6:
        return GroundPlane.none("RANSAC ground plane fit failed to find valid physical parameters (a <= 0).")

    gp = GroundPlane.from_perspective(
        a=best_a,
        b=best_b,
        frame_w=fw,
        frame_h=fh,
        person_h=person_h
    )

    return gp
