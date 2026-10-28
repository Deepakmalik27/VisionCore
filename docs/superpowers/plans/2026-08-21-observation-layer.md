# Observation Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Emit one row per person per frame (zone, motion, pose, re-ID vector) from `engine.process_video` into Supabase Postgres, so Deepak's intelligence layer can build per-person journeys.

**Architecture:** A pure row-builder module (`kevacv/observe.py`) with no I/O, called from the existing frame loop, feeding the existing non-blocking `EventQueue` into an append-only JSONL. A separate offline tool loads that JSONL into Postgres. The CV loop never touches the network.

**Tech Stack:** Python 3.10+, numpy, OpenCV (`cv2.pointPolygonTest`), ultralytics (`yolo11n-pose`, already a dep), `psycopg[binary]` (new, ingest tool only), Supabase Postgres.

**Spec:** `docs/superpowers/specs/2026-08-21-observation-layer-design.md`

## Global Constraints

- **No git commits or pushes by the implementer.** This repo's rule: all git writes are Prabh's. Every task ends by leaving changes in the working tree and reporting what changed. (Note: an IDE on this machine has auto-committed other repos before — if that happens here, report it, do not push.)
- **Never read or edit `.env`.** Credentials are read at runtime via `os.environ` only.
- **Off by default.** `ENABLE_OBSERVATIONS = False` in `kevacv/config.py`. A run with the flag off must be byte-identical in behaviour to today.
- **This layer adds an output; it must not change a single existing count.** If a person count, crossing, or merge decision moves, that is a bug in this work.
- **Test style is this repo's, not pytest.** Plain script, `check()` helper, `main()` returning an exit code, run as `python3 tests/test_observe.py`. Copy the shape from `tests/test_build_id.py`.
- **No fabricated numbers.** Every emitted confidence/speed must be computable from the frame; unavailable values are `None`, never a plausible-looking default.
- **Reuse, don't reimplement:** `helpers.anchor_point`, `helpers.uses_centre_anchor`, `pose.keypoints_for_frame`, `pose.match_to_boxes`, `pose.classify`, `ground_plane.GroundPlane.speed_mps`, `event_queue.EventQueue`, `event_queue.jsonl_sink` all already exist and are the intended building blocks.

---

### Task 1: Zone assignment and honest zone confidence

**Files:**
- Create: `kevacv/observe.py`
- Test: `tests/test_observe.py`

**Interfaces:**
- Consumes: `kevacv.helpers.anchor_point(box, centre=False)`, `kevacv.helpers.uses_centre_anchor(zone_name, staff_zones)`
- Produces: `observe.zone_and_confidence(box, polygons, staff_zones, feet_visible) -> (zone_name|None, conf_float)`

- [ ] **Step 1: Write the failing test**

```python
"""test_observe.py — the per-frame observation row.

Run: python3 tests/test_observe.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from kevacv import observe

FAILED = []

def check(cond, msg):
    print(("  ✅ " if cond else "  ❌ ") + msg)
    if not cond:
        FAILED.append(msg)

SQUARE = {"dining": np.array([[100, 100], [300, 100], [300, 300], [100, 300]], dtype=float)}

def _box_at(x, y, w=40, h=100):
    """A box whose FOOT anchor lands on (x, y)."""
    return (x - w / 2.0, y - h, x + w / 2.0, y)

def test_deep_inside_beats_near_edge():
    mid, _c1 = observe.zone_and_confidence(_box_at(200, 200), SQUARE, set(), True)
    edge, _c2 = observe.zone_and_confidence(_box_at(200, 295), SQUARE, set(), True)
    check(mid == "dining" and edge == "dining", "both anchors land in dining")
    check(_c1 > _c2, f"centre {_c1} is more confident than edge {_c2}")

def test_outside_is_no_zone():
    z, c = observe.zone_and_confidence(_box_at(50, 50), SQUARE, set(), True)
    check(z is None and c == 0.0, "anchor outside every polygon -> zone None, conf 0.0")

def test_hidden_feet_halve_confidence():
    _z, seen = observe.zone_and_confidence(_box_at(200, 200), SQUARE, set(), True)
    _z, hid = observe.zone_and_confidence(_box_at(200, 200), SQUARE, set(), False)
    check(abs(hid - seen / 2.0) < 1e-6, f"hidden feet halve conf: {seen} -> {hid}")

def test_confidence_is_bounded():
    _z, c = observe.zone_and_confidence(_box_at(200, 200, w=2), SQUARE, set(), True)
    check(0.0 <= c <= 1.0, f"a tiny box cannot exceed 1.0 (got {c})")

def main():
    for fn in (test_deep_inside_beats_near_edge, test_outside_is_no_zone,
               test_hidden_feet_halve_confidence, test_confidence_is_bounded):
        print(f"\n{fn.__name__}")
        fn()
    print()
    if FAILED:
        print(f"FAILED ({len(FAILED)}): " + ", ".join(FAILED))
        return 1
    print("all green")
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 tests/test_observe.py`
Expected: FAIL — `ModuleNotFoundError` / `AttributeError: module 'kevacv.observe' has no attribute 'zone_and_confidence'`

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 tests/test_observe.py`
Expected: PASS, "all green"

- [ ] **Step 5: Report, do not commit**

Print `git status --short` and report the two files to Prabh. No `git add`, no `git commit`.

---

### Task 2: Motion — smoothed speed, heading, stationary

**Files:**
- Modify: `kevacv/observe.py`
- Modify: `tests/test_observe.py`

**Interfaces:**
- Consumes: `kevacv.ground_plane.GroundPlane.speed_mps(p, q, dt)` and `.ok()` (may be a `GroundPlane.none()` instance)
- Produces: `observe.motion(samples, ground=None, stationary_px_s=8.0) -> {"speed_px_s": float, "speed_mps": float|None, "heading_deg": float|None, "stationary": bool}` where `samples` is `[(t_seconds, (x, y)), ...]` oldest-first, at most 4 entries

- [ ] **Step 1: Write the failing test**

Add to `tests/test_observe.py` (and add each new function to the tuple in `main()`):

```python
def test_stationary_track_has_no_heading():
    s = [(0.0, (100.0, 200.0)), (0.125, (100.0, 200.0)), (0.25, (100.0, 200.0))]
    m = observe.motion(s)
    check(m["speed_px_s"] == 0.0, "identical anchors -> 0 px/s")
    check(m["stationary"] is True, "and stationary True")
    check(m["heading_deg"] is None, "and no heading (a still body has no direction)")

def test_single_sample_is_unknown_not_zero():
    m = observe.motion([(0.0, (10.0, 10.0))])
    check(m["speed_px_s"] is None, "one sample cannot measure speed -> None")
    check(m["stationary"] is None, "and cannot claim stationary either")

def test_moving_right_reads_east():
    s = [(0.0, (100.0, 200.0)), (0.125, (110.0, 200.0)), (0.25, (120.0, 200.0))]
    m = observe.motion(s)
    check(abs(m["speed_px_s"] - 80.0) < 1e-6, f"10px per 0.125s = 80 px/s (got {m['speed_px_s']})")
    check(m["stationary"] is False, "80 px/s is not stationary")
    check(abs(m["heading_deg"] - 90.0) < 1e-6, f"straight right = 90 deg (got {m['heading_deg']})")

def test_one_jittery_interval_does_not_dominate():
    """The oldest anchor is the bad one: intervals are [784, 8, 8] px/s, so the
    MEDIAN reads 8 and the mean would read 267. One outlying interval is what a
    median can absorb -- a spike in the MIDDLE corrupts two intervals and is
    deliberately not claimed to be fixed here."""
    steady = [(0.0, (0.0, 0.0)), (0.125, (1.0, 0.0)), (0.25, (2.0, 0.0)), (0.375, (3.0, 0.0))]
    spike = [(0.0, (99.0, 0.0)), (0.125, (1.0, 0.0)), (0.25, (2.0, 0.0)), (0.375, (3.0, 0.0))]
    check(observe.motion(spike)["speed_px_s"] == observe.motion(steady)["speed_px_s"],
          "median over the window absorbs one outlying interval")

def test_speed_mps_is_none_without_a_ground_plane():
    from kevacv.ground_plane import GroundPlane
    s = [(0.0, (100.0, 200.0)), (0.125, (110.0, 200.0))]
    m = observe.motion(s, ground=GroundPlane.none("test"))
    check(m["speed_mps"] is None, "no calibration -> metres NOT invented from pixels")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 tests/test_observe.py`
Expected: FAIL — `AttributeError: module 'kevacv.observe' has no attribute 'motion'`

- [ ] **Step 3: Write minimal implementation**

Append to `kevacv/observe.py`:

```python
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
        if ground is not None and getattr(ground, "ok", lambda: False)():
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 tests/test_observe.py`
Expected: PASS, "all green"

- [ ] **Step 5: Report, do not commit**

---

### Task 3: Pose sampling with honest staleness

**Files:**
- Modify: `kevacv/observe.py`
- Modify: `tests/test_observe.py`

**Interfaces:**
- Consumes: `kevacv.pose.keypoints_for_frame(frame, model, conf, imgsz)`, `kevacv.pose.match_to_boxes(poses, boxes, min_iou)`, `kevacv.pose.classify(kps, foot_move_px)`
- Produces: `observe.pose_field(last, t) -> {"available": bool, "activity": str|None, "age_s": float|None}` where `last` is `(t_sampled, activity)` or `None`

- [ ] **Step 1: Write the failing test**

```python
def test_pose_never_sampled_is_null_not_unknown():
    p = observe.pose_field(None, 12.0)
    check(p["available"] is False, "no sample -> available False")
    check(p["activity"] is None and p["age_s"] is None,
          "and NULL activity/age, not a guessed 'unknown'")

def test_pose_fresh_sample_has_zero_age():
    p = observe.pose_field((12.0, "standing"), 12.0)
    check(p == {"available": True, "activity": "standing", "age_s": 0.0}, str(p))

def test_pose_carry_forward_states_its_age():
    p = observe.pose_field((12.0, "sitting"), 12.875)
    check(p["activity"] == "sitting", "activity carries forward between strides")
    check(abs(p["age_s"] - 0.875) < 1e-6, f"age is reported, not hidden (got {p['age_s']})")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 tests/test_observe.py`
Expected: FAIL — `AttributeError: ... has no attribute 'pose_field'`

- [ ] **Step 3: Write minimal implementation**

Append to `kevacv/observe.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 tests/test_observe.py`
Expected: PASS, "all green"

- [ ] **Step 5: Report, do not commit**

---

### Task 4: `build_row` — the wire format

**Files:**
- Modify: `kevacv/observe.py`
- Modify: `tests/test_observe.py`

**Interfaces:**
- Consumes: Tasks 1-3 (`zone_and_confidence`, `motion`, `pose_field`)
- Produces:
  `observe.build_row(*, run_id, camera_id, frame_idx, t_s, ts, box, det_conf, raw_track_id, canon_id, emb_id, polygons, staff_zones, feet_visible, is_ir, samples, ground, pose_last, stationary_px_s) -> dict` with `kind="obs"` and exactly the column names in the spec's `vision_observations`
  `observe.emb_row(*, run_id, camera_id, raw_track_id, frame_idx, emb_id, vec, blur_score) -> dict` with `kind="emb"`
  `observe.OBS_COLUMNS` / `observe.EMB_COLUMNS` — tuples the ingest tool imports, so the writer and the reader cannot drift apart

- [ ] **Step 1: Write the failing test**

```python
def _row(**kw):
    base = dict(run_id="r1", camera_id="cam01", frame_idx=10, t_s=1.25,
                ts="2026-08-20T20:15:32.400000+00:00", box=(180.0, 100.0, 220.0, 200.0),
                det_conf=0.94, raw_track_id=103, canon_id=None, emb_id=None,
                polygons=SQUARE, staff_zones=set(), feet_visible=True, is_ir=False,
                samples=[(1.125, (200.0, 200.0)), (1.25, (200.0, 200.0))],
                ground=None, pose_last=None, stationary_px_s=8.0)
    base.update(kw)
    return observe.build_row(**base)

def test_row_has_exactly_the_declared_columns():
    r = _row()
    check(set(r) == set(observe.OBS_COLUMNS) | {"kind"},
          f"row keys match OBS_COLUMNS; diff={set(r) ^ (set(observe.OBS_COLUMNS) | {'kind'})}")
    check(r["kind"] == "obs", "tagged obs so one JSONL can carry both kinds")

def test_row_keeps_both_ids():
    r = _row(canon_id="guest_7")
    check(r["raw_track_id"] == "103" and r["canon_id"] == "guest_7",
          "raw AND canonical id both survive; raw is stringified for a text PK")

def test_row_is_json_serialisable():
    import json
    json.dumps(_row(det_conf=np.float32(0.94)))   # numpy scalars must not leak
    check(True, "numpy scalars are cast before they reach the JSONL")

def test_emb_row_carries_a_real_vector():
    e = observe.emb_row(run_id="r1", camera_id="cam01", raw_track_id=103,
                        frame_idx=10, emb_id="r1_103_10", vec=np.zeros(512, dtype=np.float32),
                        blur_score=41.2)
    check(len(e["vec"]) == 512 and isinstance(e["vec"][0], float),
          "vector is stored as plain floats, not an opaque id")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 tests/test_observe.py`
Expected: FAIL — `AttributeError: ... has no attribute 'build_row'`

- [ ] **Step 3: Write minimal implementation**

Append to `kevacv/observe.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 tests/test_observe.py`
Expected: PASS, "all green"

- [ ] **Step 5: Report, do not commit**

---

### Task 5: Wire into the frame loop behind a flag

**Files:**
- Modify: `kevacv/config.py` (new block after the pose block at ~line 396)
- Modify: `kevacv/engine.py` (state init near `frame_log = []` at ~line 2583; emit inside the per-box loop at ~line 3541-3561)
- Modify: `kevacv/pipeline.py` (queue lifecycle around the `analyse_fn` call at ~line 510)
- Test: manual — `ENABLE_OBSERVATIONS=False` run unchanged; flag-on smoke test is Task 8

**Interfaces:**
- Consumes: `observe.build_row`, `observe.emb_row`, `kevacv.event_queue.EventQueue`, `kevacv.event_queue.jsonl_sink`
- Produces: `engine.OBS_QUEUE` (module global, `None` or an object with `.put(dict) -> bool`); `output/<run>/observations.jsonl`; `card["observations"]` = `EventQueue.stats()`

- [ ] **Step 1: Add the config block**

In `kevacv/config.py`, directly after the pose block:

```python
# ── observation layer (per-frame rows for the intelligence layer) ───────────
# One row per person per frame -> EventQueue -> observations.jsonl ->
# tools/ingest_obs.py -> Postgres. OFF by default: it is an output nobody
# needs on a tuning run, and at 8 fps it is ~145k rows/camera-hour.
ENABLE_OBSERVATIONS = False
OBS_QUEUE_MAXSIZE = 20000      # ~2.5 s of backlog at 8 fps x 5 people
OBS_STATIONARY_PX_S = 8.0      # ~1 px/frame at 8 fps
OBS_MOTION_WINDOW = 4          # anchors kept per track for the median
OBS_EMB_STRIDE = 8             # embeddings per track: 1/s, not 8/s
```

- [ ] **Step 2: Add engine state**

In `engine.process_video`, next to `frame_log = []` (~line 2583):

```python
    # OBSERVATION LAYER. OBS_QUEUE is injected by pipeline.py (same pattern as
    # BASE/OUTPUT_DIR) so the engine never opens a file or a socket itself.
    _obs_q = globals().get("OBS_QUEUE")
    _obs_on = bool(globals().get("ENABLE_OBSERVATIONS", False)) and _obs_q is not None
    _obs_hist = defaultdict(list)      # raw tid -> [(t, (x, y))], newest last
    _obs_pose = {}                     # raw tid -> (t_sampled, activity)
    _obs_run_id = f"{camera_id}_{chunk_tag or 'run'}"
    _obs_emb_at = {}                   # raw tid -> frame_idx of last embedding
```

- [ ] **Step 3: Emit rows inside the existing per-box loop**

Inside `for (bx1, by1, bx2, by2), tid, cid_ in zip(...)` in the frame loop, immediately after `rec.vote_role(tid, role_vote)` and before the loop ends (~line 3560). `bc` (the foot anchor) is already computed just above:

```python
            if _obs_on:
                _raw = tid
                _h = _obs_hist[_raw]
                _h.append((t, bc))
                del _h[:-int(globals().get("OBS_MOTION_WINDOW", 4))]
                # Feet are "visible" when the box bottom is inside the frame
                # and the box is not clipped by a desk mask. Both are already
                # known here; the transcript's "only the head is visible, so
                # he is behind the desk" case is what this flags.
                _feet_vis = (by2 < frame_h - 2)
                _emb_id = None
                _vec = _vecs.get(_i_by_tid.get(_raw)) if _vecs else None
                _stride_f = int(globals().get("OBS_EMB_STRIDE", 8))
                if _vec is not None and (frame_idx - _obs_emb_at.get(_raw, -10**9)) >= _stride_f:
                    _obs_emb_at[_raw] = frame_idx
                    _emb_id = f"{_obs_run_id}_{_raw}_{frame_idx}"
                    _obs_q.put(observe.emb_row(
                        run_id=_obs_run_id, camera_id=camera_id,
                        raw_track_id=_raw, frame_idx=frame_idx,
                        emb_id=_emb_id, vec=_vec, blur_score=None))
                _obs_q.put(observe.build_row(
                    run_id=_obs_run_id, camera_id=camera_id,
                    frame_idx=frame_idx, t_s=t, ts=_wall_ts(t),
                    box=(bx1, by1, bx2, by2), det_conf=_conf_by_tid.get(_raw),
                    raw_track_id=_raw,
                    canon_id=(_identity_memory.raw_to_canon.get(_raw)
                              if _identity_memory is not None else None),
                    emb_id=_emb_id, polygons=polygons,
                    staff_zones=staff_zones_here, feet_visible=_feet_vis,
                    is_ir=bool(_frame_ir.get(frame_idx)),
                    samples=list(_h), ground=_ground,
                    pose_last=_obs_pose.get(_raw),
                    stationary_px_s=float(globals().get("OBS_STATIONARY_PX_S", 8.0))))
```

Three lookups above must be built where the data actually is, in the same
frame, just before this loop — they exist under different names in the
detection arrays, so bind them explicitly rather than guessing:

```python
        # index/confidence by raw track id, for the observation rows
        _i_by_tid = {_safe_id(_tt): _ii for _ii, _tt in enumerate(dets.tracker_id)}
        _conf_by_tid = {_safe_id(_tt): float(_cc) for _tt, _cc
                        in zip(dets.tracker_id, dets.confidence)}
```

and `_wall_ts(t)` — wall-clock for the row, or None when the clock was never
verified (never guess a timestamp; `clock.py` exists because a run once
stamped 19:30 onto 16:30 footage):

```python
    def _wall_ts(t_rel):
        _b = globals().get("VIDEO_START_DT")     # set by pipeline/provenance
        if _b is None:
            return None
        return (_b + timedelta(seconds=float(t_rel))).isoformat()
```

`_vecs` is assigned only inside the re-id branch (~line 3452), so a run with
re-id disabled would raise `NameError` here. Initialise `_vecs = {}` once per
frame ABOVE that branch rather than guarding at every use.

Add `from . import observe` to engine's imports and confirm `timedelta` is
imported there (it is used by `clock.py`; import it in engine if absent).

- [ ] **Step 4: Wire the queue in pipeline.py**

Around the `analyse_fn` call in `run_camera` (~line 510), mirroring the
existing `ENABLE_EVENT_QUEUE` block at line ~1073:

```python
    _obs_eq = None
    if getattr(_Eg, "ENABLE_OBSERVATIONS", False):
        from .event_queue import EventQueue as _OEQ, jsonl_sink as _osink
        _obs_path = out / "observations.jsonl"
        _obs_eq = _OEQ(sink=_osink(str(_obs_path)),
                       maxsize=int(getattr(_Eg, "OBS_QUEUE_MAXSIZE", 20000)))
        _obs_eq.start()
        _Eg.OBS_QUEUE = _obs_eq
```

and immediately after the analyse call returns (in a `finally`, so a crashing
run still flushes what it had):

```python
    finally:
        if _obs_eq is not None:
            _Eg.OBS_QUEUE = None
            _st = _obs_eq.close()
            card["observations"] = {**_st, "path": str(_obs_path)}
            _log.info(f"\U0001f4dd observations -> {_obs_path.name}: "
                      f"{_st['written']} written, {_st['dropped']} dropped, "
                      f"{_st['sink_errors']} sink error(s)")
            if _st["lost"]:
                _log.error(f"!! observation queue LOST {_st['lost']} row(s) — "
                           f"the JSONL is INCOMPLETE for this run")
```

- [ ] **Step 5: Prove the flag-off path is untouched**

Run: `./run.sh --tests`
Expected: the suite is as green as it was before this task (record the
before/after pass counts; this repo has known-failing suites, so compare, do
not assume zero).

Then: `python3 -c "import kevacv.engine"` — expected: no import error.

- [ ] **Step 6: Report, do not commit**

Report the diff summary and both test counts. Flag explicitly if any suite
that passed before now fails — that would mean this task changed a count,
which the Global Constraints forbid.

---

### Task 6: `tools/ingest_obs.py` — JSONL into Postgres

**Files:**
- Create: `tools/ingest_obs.py`
- Test: built-in `--selftest` (this repo's tool convention, see `tools/ingest_db.py`)

**Interfaces:**
- Consumes: `observe.OBS_COLUMNS`, `observe.EMB_COLUMNS`
- Produces: CLI `python3 tools/ingest_obs.py --jsonl output/<run>/observations.jsonl [--sqlite path | --pg]`, env var `SUPABASE_DB_URL`, tables `vision_runs` / `vision_observations` / `vision_embeddings`

- [ ] **Step 1: Write the failing selftest**

Create `tools/ingest_obs.py` containing only the docstring and:

```python
def selftest():
    import sqlite3, tempfile, pathlib, json
    rows = [{"kind": "obs", "run_id": "r1", "camera_id": "cam01", "frame_idx": i,
             "raw_track_id": "103", "canon_id": None, "ts": None, "t_s": i * 0.125,
             "x1": 1, "y1": 2, "x2": 3, "y2": 4, "det_conf": 0.9,
             "foot_x": 2.0, "foot_y": 4.0, "feet_visible": True, "is_ir": False,
             "zone": "dining", "zone_conf": 0.8, "speed_px_s": 0.0,
             "speed_mps": None, "heading_deg": None, "stationary": True,
             "emb_id": None, "pose_activity": None, "pose_age_s": None}
            for i in range(3)]
    d = pathlib.Path(tempfile.mkdtemp())
    p = d / "observations.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    conn = sqlite3.connect(":memory:")
    create_tables(conn, dialect="sqlite", vec_type="blob")
    n = ingest(conn, str(p), dialect="sqlite")
    assert n["obs"] == 3, n
    # Re-ingest a SHORTER run: stale rows must not survive. This is the exact
    # bug tools/ingest_db.py's selftest guards, and INSERT OR REPLACE misses.
    p.write_text(json.dumps(rows[0]) + "\n")
    ingest(conn, str(p), dialect="sqlite")
    got = conn.execute("SELECT count(*) FROM vision_observations").fetchone()[0]
    assert got == 1, got
    print("selftest ok")
    return 0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python3 tools/ingest_obs.py --selftest`
Expected: FAIL — `NameError: name 'create_tables' is not defined`

- [ ] **Step 3: Implement the tool**

```python
#!/usr/bin/env python3
"""ingest_obs.py — observations.jsonl into Postgres (or SQLite, for a local look).

WHY A FILE FIRST AND NOT A DIRECT INSERT
    "The video pipeline should never wait for PostgreSQL." A dropped
    connection at minute 52 must not cost an hour of GPU time, and a file
    makes ingestion replayable. Same reasoning, and the same delete-then-
    insert discipline, as tools/ingest_db.py.

USAGE
    export SUPABASE_DB_URL=postgresql://...        # never read from .env here
    python3 tools/ingest_obs.py --jsonl output/run1/observations.jsonl
    python3 tools/ingest_obs.py --jsonl ... --sqlite output/obs.db
    python3 tools/ingest_obs.py --selftest
"""
import argparse, json, os, pathlib, sys, struct

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from kevacv.observe import OBS_COLUMNS, EMB_COLUMNS      # noqa: E402

TYPES_PG = {"frame_idx": "integer", "x1": "integer", "y1": "integer",
            "x2": "integer", "y2": "integer", "t_s": "real",
            "det_conf": "real", "foot_x": "real", "foot_y": "real",
            "zone_conf": "real", "speed_px_s": "real", "speed_mps": "real",
            "heading_deg": "real", "pose_age_s": "real", "blur_score": "real",
            "fps_analysed": "real", "frames_analysed": "integer",
            "feet_visible": "boolean", "is_ir": "boolean",
            "stationary": "boolean", "ts": "timestamptz"}


def _coltype(name, dialect):
    t = TYPES_PG.get(name, "text")
    if dialect == "sqlite":
        return {"timestamptz": "text", "boolean": "integer"}.get(t, t)
    return t


def create_tables(conn, dialect="pg", vec_type="bytea"):
    obs = ", ".join(f"{c} {_coltype(c, dialect)}" for c in OBS_COLUMNS)
    emb = ", ".join((f"{c} {vec_type}" if c == "vec"
                     else f"{c} {_coltype(c, dialect)}") for c in EMB_COLUMNS)
    conn.execute(f"CREATE TABLE IF NOT EXISTS vision_observations ({obs}, "
                 f"PRIMARY KEY (run_id, frame_idx, raw_track_id))")
    conn.execute(f"CREATE TABLE IF NOT EXISTS vision_embeddings ({emb}, "
                 f"PRIMARY KEY (emb_id))")
    conn.execute("CREATE INDEX IF NOT EXISTS obs_track ON vision_observations "
                 "(run_id, raw_track_id, t_s)")
    conn.execute("CREATE INDEX IF NOT EXISTS obs_zone ON vision_observations "
                 "(run_id, zone, t_s)")


def _ph(dialect, n):
    return ", ".join(("?" if dialect == "sqlite" else "%s") for _ in range(n))


def _pack(vec, vec_type):
    if vec_type == "vector":
        return "[" + ",".join(f"{float(v):.6f}" for v in vec) + "]"
    return struct.pack(f"<{len(vec)}f", *[float(v) for v in vec])


def ingest(conn, jsonl, dialect="pg", vec_type="bytea"):
    obs, emb, runs = [], [], set()
    with open(jsonl) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            runs.add(r.get("run_id"))
            if r.get("kind") == "emb":
                r["vec"] = _pack(r["vec"], vec_type)
                emb.append([r.get(c) for c in EMB_COLUMNS])
            else:
                obs.append([r.get(c) for c in OBS_COLUMNS])
    for run in runs:
        conn.execute(f"DELETE FROM vision_observations WHERE run_id = "
                     f"{_ph(dialect, 1)}", (run,))
        conn.execute(f"DELETE FROM vision_embeddings WHERE run_id = "
                     f"{_ph(dialect, 1)}", (run,))
    if obs:
        conn.executemany(
            f"INSERT INTO vision_observations ({', '.join(OBS_COLUMNS)}) "
            f"VALUES ({_ph(dialect, len(OBS_COLUMNS))})", obs)
    if emb:
        conn.executemany(
            f"INSERT INTO vision_embeddings ({', '.join(EMB_COLUMNS)}) "
            f"VALUES ({_ph(dialect, len(EMB_COLUMNS))})", emb)
    conn.commit()
    return {"obs": len(obs), "emb": len(emb), "runs": sorted(r for r in runs if r)}


def _pg_connect(url):
    import psycopg
    conn = psycopg.connect(url)
    # pgvector turns Deepak's re-match into one SQL query instead of a full
    # pull into Python. Use it when the project has it; never CREATE EXTENSION
    # from here -- that is a database owner's decision, not an ingest tool's.
    has_vec = conn.execute(
        "SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone()
    return conn, ("vector" if has_vec else "bytea")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl")
    p.add_argument("--sqlite")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.jsonl:
        p.error("--jsonl is required")
    if a.sqlite:
        import sqlite3
        conn, dialect, vec_type = sqlite3.connect(a.sqlite), "sqlite", "blob"
    else:
        url = os.environ.get("SUPABASE_DB_URL")
        if not url:
            print("SUPABASE_DB_URL is not set. Export it in this shell "
                  "(this tool never reads .env).", file=sys.stderr)
            return 2
        conn, vec_type = _pg_connect(url)
        dialect = "pg"
    create_tables(conn, dialect=dialect, vec_type=vec_type)
    n = ingest(conn, a.jsonl, dialect=dialect, vec_type=vec_type)
    print(f"  {n['obs']} observations, {n['emb']} embeddings, "
          f"runs: {', '.join(n['runs'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Note: `vision_runs` is written by Task 7, which is where the run-level facts
(fps, git sha, zone config hash) are collected. Do not stub it here.

- [ ] **Step 4: Run the selftest**

Run: `python3 tools/ingest_obs.py --selftest`
Expected: `selftest ok`

- [ ] **Step 5: Pin the dependency**

In `Dockerfile`, inside the existing pinned `pip install` block (line ~40),
add `psycopg[binary]` with an exact `==` version — run
`python3 -m pip download "psycopg[binary]" --no-deps -d /tmp/pg` to learn the
version that actually resolves, and pin that. Do not write a range; the file's
own comment says ">=" is how the fleet drifts.

- [ ] **Step 6: Report, do not commit**

---

### Task 7: Run-level provenance row

**Files:**
- Modify: `tools/ingest_obs.py`
- Modify: `kevacv/pipeline.py` (the same block added in Task 5)

**Interfaces:**
- Consumes: `kevacv.build_id.manifest()`, `card["observations"]`
- Produces: a `{"kind": "run", ...}` first line in `observations.jsonl`; table `vision_runs`

- [ ] **Step 1: Write the failing selftest addition**

In `selftest()`, prepend a run row to the fixture and assert it lands:

```python
    run_row = {"kind": "run", "run_id": "r1", "camera_id": "cam01",
               "video_sha": "abc123", "fps_analysed": 8.0,
               "started_at": None, "frames_analysed": 3,
               "zones_cfg_hash": "z1", "git_sha": "deadbeef"}
    p.write_text("\n".join(json.dumps(r) for r in [run_row] + rows) + "\n")
    ...
    assert conn.execute("SELECT count(*) FROM vision_runs").fetchone()[0] == 1
    # A re-ingest must UPDATE the run row, not duplicate it
    ingest(conn, str(p), dialect="sqlite")
    assert conn.execute("SELECT count(*) FROM vision_runs").fetchone()[0] == 1
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python3 tools/ingest_obs.py --selftest`
Expected: FAIL — `sqlite3.OperationalError: no such table: vision_runs`

- [ ] **Step 3: Implement**

In `tools/ingest_obs.py`:

```python
RUN_COLUMNS = ("run_id", "camera_id", "video_sha", "fps_analysed",
               "started_at", "frames_analysed", "zones_cfg_hash", "git_sha")
```

Add to `create_tables`:

```python
    runs = ", ".join(f"{c} {'timestamptz' if c == 'started_at' and dialect == 'pg' else _coltype(c, dialect)}"
                     for c in RUN_COLUMNS)
    conn.execute(f"CREATE TABLE IF NOT EXISTS vision_runs ({runs}, "
                 f"PRIMARY KEY (run_id))")
```

In `ingest`, collect `kind == "run"` lines into `run_rows`, and inside the
per-run delete loop also `DELETE FROM vision_runs WHERE run_id = ?` before
inserting them — same delete-then-insert rule, so a re-ingest updates rather
than duplicates. Return `{"obs": ..., "emb": ..., "runs": ...}` unchanged plus
`"run_rows": len(run_rows)`.

In `kevacv/pipeline.py`, immediately after `_obs_eq.start()`:

```python
        from .build_id import manifest as _mf
        _obs_eq.put({"kind": "run", "run_id": f"{camera_id}_{chunk_tag or 'run'}",
                     "camera_id": camera_id,
                     "video_sha": _prov.get("sha256") if isinstance(_prov, dict) else None,
                     "fps_analysed": float(getattr(_Eg, "FPS_TARGET", 0) or 0),
                     "started_at": None, "frames_analysed": None,
                     "zones_cfg_hash": None,
                     "git_sha": (_mf() or {}).get("build_id")})
```

Fill `video_sha`, `started_at`, `frames_analysed` and `zones_cfg_hash` from the
provenance/clock values already computed in `run_camera` — grep for the
existing `verify_provenance` result in that function and use its fields rather
than recomputing a hash. Any value genuinely unavailable stays `None`; do not
substitute a placeholder string.

- [ ] **Step 4: Run the selftest**

Run: `python3 tools/ingest_obs.py --selftest`
Expected: `selftest ok`

- [ ] **Step 5: Report, do not commit**

---

### Task 8: End-to-end acceptance on real footage

**Files:**
- Modify: none (or small fixes surfaced by the run)
- Test: a real 60-second clip

**Interfaces:**
- Consumes: everything above
- Produces: a populated `vision_observations` in Supabase, and the numbers to send Deepak

- [ ] **Step 1: Run 60 seconds with the flag on**

```bash
cd /mnt/c/Users/prabh/Documents/Aurika/keva_vision
./run.sh --dry --video <clip.mp4> --zones zones/<cam>.json
```
with `ENABLE_OBSERVATIONS: true` and `ENABLE_POSE: true` set in the run-config
yaml passed to `apply_run_config` (add them to a copy of `config/cam112.yaml`,
do not edit the baseline config in place).

Expected: the log line `📝 observations -> observations.jsonl: N written, 0 dropped, 0 sink error(s)`.

- [ ] **Step 2: Check the shape of what came out**

```bash
head -2 output/<run>/observations.jsonl
python3 - <<'EOF'
import json, collections
rows = [json.loads(l) for l in open("output/<run>/observations.jsonl")]
obs = [r for r in rows if r["kind"] == "obs"]
print("rows", len(obs), "tracks", len({r["raw_track_id"] for r in obs}))
print("zones", collections.Counter(r["zone"] for r in obs))
print("pose", collections.Counter(r["pose_activity"] for r in obs))
print("null speed", sum(1 for r in obs if r["speed_px_s"] is None))
EOF
```

Expected, and each of these is a real check, not a formality:
- `tracks` matches the person count in `SUMMARY.txt` for the same clip.
- `zones` is not overwhelmingly `None` — a mostly-None column means the anchor
  and the polygons disagree, which is the exact staff-mismatch bug the
  transcript describes.
- `null speed` equals roughly the track count (only each track's first frame).
- `pose_activity` is not 100% `None` when `ENABLE_POSE` is on.

- [ ] **Step 3: Ingest to Postgres**

```bash
export SUPABASE_DB_URL='postgresql://...'      # typed in the shell, not from .env
python3 tools/ingest_obs.py --jsonl output/<run>/observations.jsonl
psql "$SUPABASE_DB_URL" -c "SELECT count(*), count(distinct raw_track_id), \
  count(distinct zone) FROM vision_observations"
```

Expected: counts match Step 2.

- [ ] **Step 4: Re-ingest and prove idempotency**

Run the same ingest command twice more. Expected: the row count does not
change. (This is the failure mode `ingest_db.py` was written to avoid.)

- [ ] **Step 5: Report to Prabh with the numbers**

Report: rows written, tracks, zone distribution, pose coverage, queue drops,
and the wall-clock cost of the run with the flag on versus off (pose on a
stride should be a modest increase — if it is not, say so rather than
absorbing it).

- [ ] **Step 6: Do not commit**

Leave everything in the working tree and hand Prabh the file list.

---

## Deliberately not in this plan

- Staff vs customer classification, IN/OUT counting, journey stitching,
  cross-camera matching — all Deepak's intelligence layer.
- Live streaming of rows during the run (JSONL + post-run ingest covers the
  stated need; a DB worker thread is a change to `EventQueue`'s sink, not a
  rewrite, if that need ever appears).
- Partitioning / retention for 24h multi-camera runs. Revisit when a run
  actually exceeds one camera-hour.
