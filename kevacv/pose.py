"""Pose / activity — a layer ON TOP of identity, never inside it.

WHY THIS SHAPE
--------------
accuracy.txt asks for pose, and is explicit about where it belongs:

    "Pose + action recognition should be a separate layer from tracking.
     Don't make your tracker responsible for understanding actions."
    "Run it only on selected tracks, not every person on every frame."

follow_up.txt is equally explicit that it comes AFTER identity is stable, and
the measurements on this camera agree: pose answers "what is this body doing",
not "who is this" or "did they come in", so it cannot improve a single count
this project is currently wrong about.

So this module:
  * takes GLOBAL PERSON IDS and a frame source, never raw detections
  * runs on SELECTED tracks only, on a stride, with a hard budget
  * is OFF by default and imports its model lazily, so a pipeline that never
    enables it pays nothing and cannot break on a missing dependency

WHAT IT ANSWERS
---------------
standing / walking / bending / sitting, from keypoint geometry rather than a
second learned model. Two joints decide most of it:

    torso angle from vertical   upright vs bent
    hip height vs knee height   standing vs seated
    foot displacement over time walking vs still

Deliberately rule-based. A learned action model needs labelled action clips,
and this project has 3 labelled ENTRIES -- adding a model whose training data
does not exist would be the same mistake as the accuracy claims that had to be
withdrawn.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

STANDING, WALKING, BENDING, SITTING, UNKNOWN = (
    "standing", "walking", "bending", "sitting", "unknown")

# COCO-17 keypoint indices, the layout yolo*-pose emits.
KP = {"nose": 0, "l_sh": 5, "r_sh": 6, "l_hip": 11, "r_hip": 12,
      "l_knee": 13, "r_knee": 14, "l_ank": 15, "r_ank": 16}


def _mid(kps, a, b):
    pa, pb = kps[KP[a]], kps[KP[b]]
    if pa is None or pb is None:
        return None
    return ((pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0)


def classify(kps: Sequence, foot_move_px: float = 0.0,
             walk_px: float = 12.0, bend_deg: float = 45.0,
             sit_ratio: float = 0.25) -> Dict[str, Any]:
    """One person's keypoints -> an activity, with the reason.

    kps: 17 (x, y) pairs or None. foot_move_px: displacement of the foot point
    since the previous sampled frame, which is what separates standing from
    walking -- a single frame cannot.
    """
    sh = _mid(kps, "l_sh", "r_sh")
    hip = _mid(kps, "l_hip", "r_hip")
    knee = _mid(kps, "l_knee", "r_knee")
    if sh is None or hip is None:
        return {"activity": UNKNOWN, "why": "shoulders or hips not visible"}

    dx, dy = hip[0] - sh[0], hip[1] - sh[1]
    torso = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))

    if knee is not None:
        torso_len = math.hypot(dx, dy)
        thigh = abs(knee[1] - hip[1])
        if torso_len > 1e-6 and thigh / torso_len < sit_ratio:
            return {"activity": SITTING, "torso_deg": torso,
                    "why": f"knees are level with the hips "
                           f"(thigh/torso {thigh/torso_len:.2f})"}

    if torso >= bend_deg:
        return {"activity": BENDING, "torso_deg": torso,
                "why": f"torso {torso:.0f}deg from vertical"}
    if foot_move_px >= walk_px:
        return {"activity": WALKING, "torso_deg": torso,
                "why": f"feet moved {foot_move_px:.0f}px since the last sample"}
    return {"activity": STANDING, "torso_deg": torso,
            "why": f"upright ({torso:.0f}deg) and feet still"}


def select_tracks(tracks: Dict[Any, List], max_tracks: int = 8,
                  min_seconds: float = 2.0) -> List[Any]:
    """WHICH tracks are worth the compute. Longest-lived first.

    Pose on every person on every frame is the cost accuracy.txt warns about.
    A track that lived under min_seconds cannot support an activity claim
    anyway -- standing vs walking needs at least two samples.
    """
    out = []
    for tid, p in tracks.items():
        if len(p) < 2:
            continue
        dur = float(p[-1][0]) - float(p[0][0])
        if dur >= min_seconds:
            out.append((dur, tid))
    out.sort(reverse=True)
    return [tid for _d, tid in out[:max_tracks]]


def timeline(samples: List[Dict[str, Any]], min_run: int = 2) -> List[Dict[str, Any]]:
    """Per-sample activities -> stable spans, so one bad frame is not an event.

    "Never make an important decision from one frame" (accuracy.txt). A span
    shorter than min_run samples is absorbed into its neighbour rather than
    published as an activity change.
    """
    spans: List[Dict[str, Any]] = []
    for s in samples:
        a = s.get("activity", UNKNOWN)
        if spans and spans[-1]["activity"] == a:
            spans[-1]["t_end"] = s["t"]
            spans[-1]["n"] += 1
        else:
            spans.append({"activity": a, "t_start": s["t"], "t_end": s["t"],
                          "n": 1})
    if len(spans) <= 1:
        return spans
    out = [spans[0]]
    for sp in spans[1:]:
        if sp["n"] < min_run and out:
            # absorb the blip into the span before it
            out[-1]["t_end"] = sp["t_end"]
            out[-1]["n"] += sp["n"]
        elif out and out[-1]["activity"] == sp["activity"]:
            # AND re-merge across it. Without this, STANDING / blip / STANDING
            # absorbed the blip and then published TWO standing spans -- an
            # "activity change" from standing to standing, which is exactly the
            # single-frame decision this function exists to prevent.
            out[-1]["t_end"] = sp["t_end"]
            out[-1]["n"] += sp["n"]
        else:
            out.append(sp)
    return out


# ── THE PRODUCER ────────────────────────────────────────────────────────────
# Everything above this line CONSUMES keypoints. Until 2026-08-20 nothing in
# this repository PRODUCED any: classify() and timeline() were reachable, and
# the pipeline called select_tracks() and wrote a note saying inference "is a
# second pass and is not run inline". So enable_pose:true would have changed
# no pixel and no number -- the module was a consumer with no producer, and it
# was reported as "wired". This is that producer.

_MODEL = None
_MODEL_KEY = None

# COCO-17 limb pairs, for drawing. Head links omitted deliberately: on this
# camera the face is rarely resolvable and drawing noise as anatomy makes the
# overlay look more certain than the data is.
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def get_model(weights: str = "yolo11n-pose.pt", device=None):
    """Lazily load and cache the pose model. Returns None if unavailable, so a
    missing weight file degrades the overlay instead of killing the run."""
    global _MODEL, _MODEL_KEY
    key = (str(weights), str(device))
    if _MODEL is not None and _MODEL_KEY == key:
        return _MODEL
    try:
        from ultralytics import YOLO
        m = YOLO(str(weights))
        if device is not None:
            m.to(device)
        _MODEL, _MODEL_KEY = m, key
        return _MODEL
    except Exception:
        return None


def keypoints_for_frame(frame, model, conf: float = 0.35, imgsz: int = 640):
    """-> [{"box": (x1,y1,x2,y2), "kps": [(x,y)|None x17]}] for one frame.

    A keypoint below its own confidence is returned as None rather than as a
    low-confidence coordinate, because classify() already treats None as
    "not visible" and a wrong coordinate is worse than a missing one.
    """
    if model is None or frame is None:
        return []
    try:
        res = model.predict(frame, conf=conf, imgsz=imgsz, classes=[0],
                            verbose=False)
    except Exception:
        return []
    out = []
    for r in res:
        kobj = getattr(r, "keypoints", None)
        bobj = getattr(r, "boxes", None)
        if kobj is None or bobj is None or kobj.xy is None:
            continue
        xy = kobj.xy.cpu().numpy()
        cf = (kobj.conf.cpu().numpy() if getattr(kobj, "conf", None) is not None
              else None)
        bx = bobj.xyxy.cpu().numpy()
        for i in range(len(xy)):
            pts = []
            for j in range(len(xy[i])):
                if cf is not None and float(cf[i][j]) < 0.30:
                    pts.append(None)
                else:
                    pts.append((float(xy[i][j][0]), float(xy[i][j][1])))
            out.append({"box": tuple(float(v) for v in bx[i]), "kps": pts})
    return out


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def match_to_boxes(poses, boxes, min_iou: float = 0.30):
    """-> {box_index: kps}. Greedy best-IoU, one pose per box.

    Matching by IoU rather than trusting the pose model's own person boxes
    keeps identity with the TRACKER, which is the thing that has been reasoned
    about. A pose detection that matches no track is dropped, not promoted
    into a person -- this layer must never be able to change a count.
    """
    pairs = sorted(((_iou(p["box"], b), pi, bi)
                    for pi, p in enumerate(poses)
                    for bi, b in enumerate(boxes)),
                   key=lambda t: -t[0])
    used_p, used_b, out = set(), set(), {}
    for iou, pi, bi in pairs:
        if iou < min_iou or pi in used_p or bi in used_b:
            continue
        used_p.add(pi)
        used_b.add(bi)
        out[bi] = poses[pi]["kps"]
    return out


def draw(frame, kps, colour=(60, 220, 255), radius: int = 3):
    """Draw one skeleton in place. Missing joints simply are not drawn."""
    import cv2
    for a, b in SKELETON:
        pa, pb = kps[a], kps[b]
        if pa is None or pb is None:
            continue
        cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                 colour, 2, cv2.LINE_AA)
    for p in kps:
        if p is not None:
            cv2.circle(frame, (int(p[0]), int(p[1])), radius, colour, -1,
                       cv2.LINE_AA)


def demo():
    """Self-check: the producer's contracts, without needing a model."""
    # match_to_boxes keeps identity with the tracker and drops unmatched poses
    poses = [{"box": (0, 0, 10, 20), "kps": [None] * 17},
             {"box": (500, 500, 510, 520), "kps": [(1.0, 1.0)] * 17}]
    m = match_to_boxes(poses, [(0, 0, 10, 20)])
    assert set(m) == {0}, m               # only the overlapping box matched
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    # one pose cannot be spent on two boxes
    m2 = match_to_boxes([poses[0]], [(0, 0, 10, 20), (0, 0, 10, 20)])
    assert len(m2) == 1, m2
    # a missing model degrades, never raises
    assert keypoints_for_frame(None, None) == []
    assert get_model("this-file-does-not-exist.pt") is None
    print("pose producer self-check OK")


if __name__ == "__main__":
    demo()
