"""observe.py — one row per person per frame, for the intelligence layer.

WHY THIS MODULE IS PURE
    Every function here takes numbers and returns numbers: no file, no
    network, no model. That is what makes the row format testable without a
    video, and it is why the DB write lives in tools/ingest_obs.py instead.

WHAT IT IS NOT
    It does not decide staff vs customer, IN vs OUT, or who is whom. Those are
    the intelligence layer's, per the 2026-08-20 design discussion: this repo
    stops at "detection is correct".
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .helpers import anchor_point, uses_centre_anchor


def zone_and_confidence(box, polygons, staff_zones, feet_visible):
    """Which zone this body is in, and how sure the geometry is.

    conf = clamp(0.5 + edge_distance/box_width, 0, 1), halved when the feet
    are not visible (the footline is then inferred, not seen). Dead centre of
    a zone -> ~1.0; one pixel over the line -> ~0.5 and falling. Every term is
    measurable in the frame, which is the point -- a confidence nobody can
    recompute is a decoration.

    The anchor rule matches engine.py exactly (feet everywhere except staff
    zones, where the counter clips the body), via the same two helpers, so a
    row can never disagree with the zone the run itself counted.
    """
    best_name, best_d, best_w = None, None, 1.0
    x1, y1, x2, y2 = (float(v) for v in box)
    width = max(x2 - x1, 1.0)
    for name, poly in polygons.items():
        pt = anchor_point((x1, y1, x2, y2), uses_centre_anchor(name, staff_zones))
        d = cv2.pointPolygonTest(
            np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2),
            (float(pt[0]), float(pt[1])), True)     # signed px, + inside
        if best_d is None or d > best_d:
            best_name, best_d, best_w = name, d, width
    if best_name is None or best_d < 0:
        return None, 0.0
    conf = min(1.0, max(0.0, 0.5 + best_d / best_w))
    return best_name, round(conf * (1.0 if feet_visible else 0.5), 3)


def _median(vals):
    v = sorted(vals)
    n = len(v)
    if not n:
        return None
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0


def motion(samples, ground=None, stationary_px_s=8.0):
    """Speed and heading of one track, from its recent foot anchors.

    samples: [(t_seconds, (x, y))] oldest first, typically the last 4 analysed
    frames. Per-frame deltas at 8 fps are mostly box jitter, so the reported
    speed is the MEDIAN of the per-interval speeds -- one bad frame cannot
    invent a sprint.

    heading_deg is a compass bearing in image space (0 = up, 90 = right) over
    the whole window, and is None while stationary: a still body has no
    direction, and emitting one would be a number with no referent.

    IN/OUT is deliberately NOT decided here. It depends on the camera angle
    and belongs to the intelligence layer.
    """
    if len(samples) < 2:
        return {"speed_px_s": None, "speed_mps": None,
                "heading_deg": None, "stationary": None}
    speeds, metres = [], []
    for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
        dt = float(t1) - float(t0)
        if dt <= 0:
            continue
        speeds.append(math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / dt)
        if ground is not None and getattr(ground, "ok", False):
            mps = ground.speed_mps(p0, p1, dt)
            if mps is not None:
                metres.append(mps)
    if not speeds:
        return {"speed_px_s": None, "speed_mps": None,
                "heading_deg": None, "stationary": None}
    px_s = _median(speeds)
    still = px_s < float(stationary_px_s)
    heading = None
    if not still:
        dx = samples[-1][1][0] - samples[0][1][0]
        dy = samples[-1][1][1] - samples[0][1][1]
        heading = (math.degrees(math.atan2(dx, -dy))) % 360.0
    return {"speed_px_s": round(px_s, 3),
            "speed_mps": (round(_median(metres), 4) if metres else None),
            "heading_deg": (round(heading, 1) if heading is not None else None),
            "stationary": bool(still)}


def pose_field(last, t):
    """The pose column, with its own staleness.

    Pose runs on a stride (POSE_STRIDE analysed frames); the rows in between
    repeat the last activity. age_s is what keeps that honest -- the consumer
    can filter on freshness instead of assuming every row was measured. Never
    sampled is NULL, not "unknown": "unknown" is a pose model's answer when it
    saw a body and could not tell, and conflating the two would let a row that
    was never looked at masquerade as a failed classification.
    """
    if not last:
        return {"available": False, "activity": None, "age_s": None}
    t0, activity = last
    return {"available": True, "activity": activity,
            "age_s": round(float(t) - float(t0), 3)}


OBS_COLUMNS = (
    "run_id", "camera_id", "frame_idx", "raw_track_id", "canon_id",
    "ts", "t_s", "x1", "y1", "x2", "y2", "det_conf",
    "foot_x", "foot_y", "feet_visible", "is_ir",
    "zone", "zone_conf",
    "speed_px_s", "speed_mps", "heading_deg", "stationary",
    "emb_id", "pose_activity", "pose_age_s",
)

EMB_COLUMNS = ("emb_id", "run_id", "camera_id", "raw_track_id", "frame_idx",
               "blur_score", "vec")


def _f(v):
    """numpy scalar -> float. json.dumps chokes on np.float32, and the failure
    lands in a worker thread where it is counted as a sink error and easy to
    miss."""
    return None if v is None else float(v)


def build_row(*, run_id, camera_id, frame_idx, t_s, ts, box, det_conf,
              raw_track_id, canon_id, emb_id, polygons, staff_zones,
              feet_visible, is_ir, samples, ground, pose_last,
              stationary_px_s=8.0):
    x1, y1, x2, y2 = (float(v) for v in box)
    zone, zconf = zone_and_confidence(box, polygons, staff_zones, feet_visible)
    mv = motion(samples, ground=ground, stationary_px_s=stationary_px_s)
    po = pose_field(pose_last, t_s)
    foot = samples[-1][1] if samples else anchor_point((x1, y1, x2, y2))
    return {
        "kind": "obs",
        "run_id": str(run_id), "camera_id": str(camera_id),
        "frame_idx": int(frame_idx),
        "raw_track_id": str(raw_track_id),
        "canon_id": (None if canon_id is None else str(canon_id)),
        "ts": (None if ts is None else str(ts)), "t_s": _f(t_s),
        "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
        "det_conf": _f(det_conf),
        "foot_x": _f(foot[0]), "foot_y": _f(foot[1]),
        "feet_visible": bool(feet_visible), "is_ir": bool(is_ir),
        "zone": zone, "zone_conf": _f(zconf),
        "speed_px_s": _f(mv["speed_px_s"]), "speed_mps": _f(mv["speed_mps"]),
        "heading_deg": _f(mv["heading_deg"]), "stationary": mv["stationary"],
        "emb_id": (None if emb_id is None else str(emb_id)),
        "pose_activity": po["activity"], "pose_age_s": _f(po["age_s"]),
    }


def emb_row(*, run_id, camera_id, raw_track_id, frame_idx, emb_id, vec,
            blur_score=None):
    return {"kind": "emb", "emb_id": str(emb_id), "run_id": str(run_id),
            "camera_id": str(camera_id), "raw_track_id": str(raw_track_id),
            "frame_idx": int(frame_idx), "blur_score": _f(blur_score),
            "vec": [float(v) for v in np.asarray(vec).ravel()]}
