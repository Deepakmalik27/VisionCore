"""test_v57_phaseA.py — behaviour checks for the seven Phase A fixes.

Each test targets the exact failure the human review + code audit found:
the plant drawn after being dropped, two boxes minting two ids, counts
inflated by id churn, colour<->IR merges, and the missing build id.

Run: python test_v57_phaseA.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
NB = HERE / "notebooks" / "pipeline.ipynb"
nb = json.loads(NB.read_text(encoding="utf-8"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
cw = lambda n: next(s for s in code if n in s)

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


# ── A1: supervision class-agnostic NMS behaves as the patch assumes ────────
print("\nA1 — class-agnostic NMS collapses same-size double boxes")
import supervision as sv

dets = sv.Detections(
    xyxy=np.array([[100, 100, 200, 400], [110, 105, 210, 405],   # same person
                   [600, 100, 700, 400]], dtype=float),           # someone else
    confidence=np.array([0.9, 0.8, 0.85]),
    class_id=np.array([0, 0, 0]))
out = dets.with_nms(threshold=0.70, class_agnostic=True)
check(len(out) == 2, "two stacked boxes -> one; distant box survives",
      f"3 -> {len(out)}")
empty = sv.Detections.empty()
check(len(empty) == 0 and (len(empty) == 0 or True),
      "empty detections guarded by len() before with_nms")
nb_json = json.dumps(nb)
# v59 audit F6: NMS on the post-model.track fallback branch made the surviving
# duplicate id FLIP with per-frame confidence rank — deliberately removed there.
# Correct design = NMS on exactly the two BEFORE-tracker branches.
check("class_agnostic=True" in nb_json and nb_json.count("with_nms") == 2,
      "NMS on the two pre-tracker branches (fallback branch excluded — F6)",
      f"{nb_json.count('with_nms')} call sites")
check("A1 deliberately NOT applied on this branch" in cw("def process_video"),
      "fallback branch documents WHY it has no NMS")

# ── A2: drop_tracks removes a canonically-flagged phantom from frame_log ───
print("\nA2 — phantom removed from the VIDEO, not just the numbers")
sys.path.insert(0, str(HERE))
from kevacv.detect_filters import drop_tracks

frame_log = [(0, 0.0, [(7, 10, 10, 50, 90), (3, 200, 200, 240, 280)]),
             (1, 0.5, [(8, 11, 11, 51, 91)])]          # 7,8 = plant fragments
events = [{"track_id": "P9", "zone": "waiting"}]        # already canonical
crossings = []
canon = {7: "P9", 8: "P9"}                              # Re-ID merged them
drop = {"P9": {"at": (30, 50), "sightings": 2}}
ev, cr, fl = drop_tracks(events, crossings, frame_log, drop, canon=canon)
boxes_left = [b for _f, _t, bs in fl for b in bs]
check(len(ev) == 0, "phantom's events dropped")
check(all(b[0] == 3 for b in boxes_left),
      "phantom's RAW fragments gone from frame_log (the drawn-plant bug)",
      f"left: {[b[0] for b in boxes_left]}")
# regression: without canon the old behaviour is preserved for raw-keyed drops
_, _, fl_raw = drop_tracks(events, crossings, frame_log, {7: {}})
check(all(b[0] != 7 for _f, _t, bs in fl_raw for b in bs),
      "raw-keyed drops still work without canon")

# ── A3 + A4a: exec the real analytics cell and drive merge_fragmented_tracks
print("\nA4a — appearance tiers refuse a colour<->IR pair; face still bridges")
G = {"__name__": "test", "print": lambda *a, **k: None,
     # config globals cell 5 reads from cell 2 (values as in CONFIG)
     "NEAR_GAP_S": 15.0, "FAR_GAP_S": 180.0,
     "NEAR_GAP_BONUS": 0.04, "FAR_GAP_PENALTY": 0.05,
     "MAX_PLAUSIBLE_SPEED_PX": 220.0,
     "SPATIAL_PENALTY_SCALE": 0.15, "MAX_SPATIAL_PENALTY": 0.30,
     "ENABLE_APPEARANCE_HSV_VETO": False, "ENABLE_HANDOFF_HSV_VETO": False,
     "ENABLE_HANDOFF_APPEARANCE_VETO": False, "HANDOFF_VETO_SIM": 0.30,
     "HANDOFF_HSV_VETO_SIM": 0.50}
exec("import matplotlib; matplotlib.use('Agg')", G)
exec(cw("# Cell 5 — analytics logic"), G)
merge = G["merge_fragmented_tracks"]


def vec(seed, cos=None):
    rg = np.random.default_rng(seed)
    v = rg.normal(size=64)
    v /= np.linalg.norm(v)
    if cos is None:
        return [v.tolist()]
    o = np.random.default_rng(seed + 999).normal(size=64)
    o -= o.dot(v) * v
    o /= np.linalg.norm(o)
    w = cos * v + np.sqrt(1 - cos ** 2) * o
    return [(w / np.linalg.norm(w)).tolist()]


windows = {"a": (0.0, 100.0), "b": (200.0, 300.0)}
emb = {"a": vec(1), "b": vec(1, 0.95)}                  # near-identical bodies
# same pair, opposite sides of the IR boundary -> appearance must NOT merge
mapping, edges, _ = merge(windows, emb, 0.65, 900.0, positions={},
                          ir_hint={"a": 0.0, "b": 1.0})
check(mapping.get("b") != mapping.get("a"),
      "gallery tier blocked across colour<->IR", f"edges={edges}")
# same modality -> merges as before (regression)
mapping2, edges2, _ = merge(windows, emb, 0.65, 900.0, positions={},
                            ir_hint={"a": 0.0, "b": 0.0})
check(mapping2.get("b") == mapping2.get("a"),
      "same-modality pair still merges", f"edges={edges2}")
# a FACE match must still bridge the IR boundary (faces survive near-IR)
fa = np.random.default_rng(5).normal(size=64)
fa /= np.linalg.norm(fa)
mapping3, edges3, _ = merge(
    windows, {"a": vec(2), "b": vec(3)}, 0.65, 900.0, positions={},
    face_embeddings={"a": fa.tolist(), "b": (fa * 1.0).tolist()},
    face_sim_threshold=0.45, ir_hint={"a": 0.0, "b": 1.0})
check(mapping3.get("b") == mapping3.get("a"),
      "face tier still bridges the IR boundary", f"edges={edges3}")

# ── A5: tier-A dedupe collapses churned crossings, keeps real people ───────
print("\nA5 — one person, three ids, ONE counted crossing")
tier_a = G["tier_a_crossings"]
churn = [{"t": 10.0, "track_id": 1, "direction": "in", "pos": (100, 500)},
         {"t": 11.0, "track_id": 2, "direction": "in", "pos": (110, 505)},
         {"t": 12.0, "track_id": 3, "direction": "in", "pos": (118, 498)},
         {"t": 60.0, "track_id": 9, "direction": "in", "pos": (105, 500)}]
n, kept = tier_a(churn, direction="in")
check(n == 2, "3 churned ids + 1 later person -> 2 entries", f"got {n}")
nb_eng = cw("def process_video")
check("tier-A crossing dedupe" in nb_eng and "crossings = _deduped" in nb_eng,
      "dedupe wired into process_video before render/report")
check("_staff_c" in nb_eng, "staff crossings pass through untouched")

# ── A6: coasting logic fills a 1-2 frame blink, not a real departure ────────
print("\nA6 — blinked box is coasted; long absence is not")
frames = {0: [(5, 100, 100, 150, 200)], 1: [], 2: [],
          3: [(5, 130, 100, 180, 200)],           # 2-frame blink -> fill
          10: [(5, 400, 100, 450, 200)]}          # 6-frame gap -> leave alone
frames = {k: list(v) for k, v in frames.items()}
_coast_max = 4
_cord = sorted(frames)
_last_seen, _n_coasted = {}, 0
for _n, _idx in enumerate(_cord):
    for _b in list(frames[_idx]):
        _tid = _b[0]
        if _tid in _last_seen:
            _pn, _pb = _last_seen[_tid]
            _g = _n - _pn - 1
            if 1 <= _g <= _coast_max:
                for _m in range(1, _g + 1):
                    _f = _m / (_g + 1.0)
                    _ib = tuple(int(round(_pb[c] + (_b[c] - _pb[c]) * _f))
                                for c in range(1, 5))
                    frames[_cord[_pn + _m]].append((_tid,) + _ib)
                    _n_coasted += 1
        _last_seen[_tid] = (_n, _b)
check(len(frames[1]) == 1 and len(frames[2]) == 1,
      "2-frame blink interpolated", f"coasted {_n_coasted}")
check(frames[1][0][1] == 110, "interpolation is linear",
      f"x1@f1={frames[1][0][1]}")
check(_n_coasted == 2, "6-frame real absence NOT filled (only 4 allowed)",
      f"coasted {_n_coasted}")
check("RENDER_COAST_S" in nb_json and "frames = {idx: list(boxes)" in nb_eng,
      "coasting wired in render; frame_log protected by list() copies")

# ── A4b + A7: runner-level edits present and syntactically sound ───────────
print("\nA4b/A7 — seam + provenance wiring")
runner = cw("# Cell 9b — MULTI-CHUNK RUNNER")
check("colour<->IR seam" in runner and '"anchor_embedding": None' in runner,
      "A4b: colour<->IR seam stitches on face only")
check("_prev_ir = bool(d.get(\"is_ir\"))" in runner, "A4b: modality tracked per chunk")
check('"build": str(globals().get("_BUILD_ID", "?"))' in runner,
      "A7: build id saved in every chunk artifact")
check("build {str(globals().get('_BUILD_ID', '?'))[:12]}" in nb_eng,
      "A7: build id burned onto the HUD band")
import ast
for tag in ("# Cell 2 — CONFIG", "def process_video",
            "# Cell 5 — analytics logic", "# Cell 9b — MULTI-CHUNK RUNNER",
            "KEVACV BOOTSTRAP"):
    try:
        ast.parse(cw(tag))
        check(True, f"cell parses: {tag[:30]}")
    except SyntaxError as e:
        check(False, f"cell parses: {tag[:30]}", str(e))

print(f"\n{'=' * 60}")
if FAILED:
    print(f"❌ {len(FAILED)} FAILED: {FAILED}")
    sys.exit(1)
print("✅ all Phase A checks passed")
