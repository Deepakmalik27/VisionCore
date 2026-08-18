"""Tests for patch_v56_phase3.py — metres instead of pixels.

The ground plane itself is validated against a synthetic camera in
test_ground_plane.py. This file tests what the pipeline DOES with it:

  G2  the live re-id gate and the hand-off merge must ask a metric question
      when a plane exists, and fall back to pixels when it does not
  G3  Tier A must count one person who fragments into three track ids ONCE,
      and two genuinely different people TWICE — that is the entire point
  G4  groups must chain (a family spreads wider than the radius, but each
      member is close to the one before)

Run: python test_v56_phase3.py
"""
import ast
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kevacv.ground_plane import GroundPlane, synth_camera

NB = Path(__file__).resolve().parent.parent / "notebooks" / "pipeline.ipynb"
nb = json.loads(NB.read_text(encoding="utf-8"))
CODE = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
SRC = "\n".join(CODE)

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


def grab(start, end):
    i = SRC.index(start)
    return SRC[i:SRC.index(end, i)]


# ---- pull the real functions out of the notebook ---------------------------
ns = {"math": math, "np": np, "numpy": np}
for k, v in [("NEAR_GAP_S", 15.0), ("FAR_GAP_S", 180.0), ("NEAR_GAP_BONUS", 0.04),
             ("FAR_GAP_PENALTY", 0.05), ("MAX_PLAUSIBLE_SPEED_PX", 220.0),
             ("SPATIAL_PENALTY_SCALE", 0.15), ("MAX_SPATIAL_PENALTY", 0.30),
             ("OCCLUSION_IOU", 0.30), ("OCCLUSION_CONTAIN", 0.60)]:
    ns[k] = v
exec(grab("def _cosine(a, b):", "\n# ------"), ns)
exec(grab("def _dynamic_accept_thresh", "\ndef _boxes_occluding"), ns)
ns["_NOVEC"] = object()   # sentinel the class closes over
exec(grab("class _IdentityMemory:", "\ndef _hsv_embed_single"), ns)
exec(grab("def tier_a_crossings", "\ndef remap_events"), ns)

# a real ground plane from a synthetic level camera
FRAME, CAM_H, FOCAL = (1920, 1080), 3.0, 1200.0
proj = synth_camera(CAM_H, FOCAL, FRAME)
PLANE = GroundPlane.from_perspective(1.7 / CAM_H, -(1.7 / CAM_H) * FRAME[1] / 2,
                                     FRAME[0], FRAME[1], focal_px=FOCAL)
check(PLANE.ok, "test fixture: ground plane built", PLANE.describe()[:60])


def at(X, Z):
    """floor position -> image foot point"""
    return proj(X, Z, 0.0)


print()
print("=" * 74)
print("  G3 — Tier A counting, the whole reason it exists")
print("=" * 74)
tier_a = ns["tier_a_crossings"]

# ONE person crossing once, whose track id fragments three ways at the door.
# Tier B (unique ids) says 3. The truth is 1.
frag = [{"t": 100.0, "track_id": 11, "direction": "in", "pos": at(0.0, 8.0)},
        {"t": 100.5, "track_id": 12, "direction": "in", "pos": at(0.15, 8.1)},
        {"t": 101.2, "track_id": 13, "direction": "in", "pos": at(-0.1, 8.2)}]
n, kept = tier_a(frag, plane=PLANE)
check(n == 1, "one person, 3 fragmented ids -> Tier A counts 1", f"got {n}")
check(len({c['track_id'] for c in frag}) == 3, "  (Tier B would have said 3)")

# two genuinely different people, 3 m apart, crossing at the same moment
two = [{"t": 200.0, "track_id": 21, "direction": "in", "pos": at(-1.5, 8.0)},
       {"t": 200.2, "track_id": 22, "direction": "in", "pos": at(1.5, 8.0)}]
n, _ = tier_a(two, plane=PLANE)
check(n == 2, "two people 3 m apart -> Tier A counts 2", f"got {n}")

# the same person legitimately entering twice, ten minutes apart
twice = [{"t": 300.0, "track_id": 31, "direction": "in", "pos": at(0.0, 8.0)},
         {"t": 900.0, "track_id": 31, "direction": "in", "pos": at(0.0, 8.0)}]
n, _ = tier_a(twice, plane=PLANE)
check(n == 2, "same person entering twice, 10 min apart -> counts 2", f"got {n}")

# direction and staff filtering
mixed = [{"t": 10.0, "track_id": 1, "direction": "in", "pos": at(0, 8)},
         {"t": 11.0, "track_id": 2, "direction": "out", "pos": at(3, 8)},
         {"t": 12.0, "track_id": "receptionist_sarah", "direction": "in",
          "pos": at(-3, 8)}]
n, _ = tier_a(mixed, plane=PLANE, roles={"receptionist_sarah": "staff"})
check(n == 1, "outward crossings and staff are excluded", f"got {n}")
n_out, _ = tier_a(mixed, plane=PLANE, direction="out")
check(n_out == 1, "outward direction can be counted separately", f"got {n_out}")

# no plane -> pixel fallback must still dedupe, never crash
n, _ = tier_a(frag, plane=GroundPlane.none())
check(n == 1, "no ground plane -> pixel fallback still collapses the fragments",
      f"got {n}")
# no positions at all -> degrade to unique-id behaviour (the OLD answer, never worse)
noposs = [dict(c, pos=None) for c in frag]
n, _ = tier_a(noposs, plane=PLANE)
check(n == 3, "no positions -> falls back to unique-id counting (old behaviour)",
      f"got {n}")
check(tier_a([], plane=PLANE)[0] == 0, "empty input -> 0, no crash")

# the far end of the room: the SAME 1.2 m dedupe radius must still work there,
# where it is only ~40 px instead of ~290 px
far = [{"t": 500.0, "track_id": 41, "direction": "in", "pos": at(0.0, 25.0)},
       {"t": 500.6, "track_id": 42, "direction": "in", "pos": at(0.3, 25.1)}]
n, _ = tier_a(far, plane=PLANE)
d_px = math.hypot(far[0]["pos"][0] - far[1]["pos"][0],
                  far[0]["pos"][1] - far[1]["pos"][1])
check(n == 1, "fragments 25 m away are still collapsed", f"only {d_px:.0f} px apart")
near_two = [{"t": 600.0, "track_id": 51, "direction": "in", "pos": at(-1.0, 5.0)},
            {"t": 600.4, "track_id": 52, "direction": "in", "pos": at(1.0, 5.0)}]
n, _ = tier_a(near_two, plane=PLANE)
check(n == 2, "two people 2 m apart near the camera are NOT collapsed", f"got {n}")
print("    -> the same 1.2 m rule, correct at both ends. In pixels it could not be.")

print()
print("=" * 74)
print("  G4 — parties, not individuals")
print("=" * 74)
groups = ns["detect_groups"]
family = [{"t": 100.0 + i * 3, "track_id": 60 + i, "pos": at(-1.0 + i * 1.1, 9.0)}
          for i in range(5)]                       # spread over 4.4 m, chained
g = groups(family, plane=PLANE)
check(len(g) == 1 and g[0]["size"] == 5,
      "a family of 5 spread over 4.4 m chains into ONE party",
      f"{len(g)} group(s), sizes {[x['size'] for x in g]}")
solo = [{"t": 100.0, "track_id": 70, "pos": at(-4.0, 9.0)},
        {"t": 104.0, "track_id": 71, "pos": at(4.0, 9.0)}]
g = groups(solo, plane=PLANE)
check(len(g) == 2, "two people 8 m apart are two parties", f"{len(g)}")
apart_in_time = [{"t": 100.0, "track_id": 80, "pos": at(0.0, 9.0)},
                 {"t": 400.0, "track_id": 81, "pos": at(0.2, 9.0)}]
g = groups(apart_in_time, plane=PLANE)
check(len(g) == 2, "same spot 5 minutes apart is not a party", f"{len(g)}")
check(groups([], plane=PLANE) == [], "no arrivals -> no groups")
check(sum(x["size"] for x in groups(family + solo, plane=PLANE)) == 7,
      "every arrival lands in exactly one party")

print()
print("=" * 74)
print("  G2 — the live re-id gate now asks a metric question")
print("=" * 74)
IM = ns["_IdentityMemory"]
VEC = [1.0, 0.0, 0.0]


def mem(plane):
    m = IM(embed_fn=lambda c: VEC, sim_threshold=0.5, max_dist_px=140.0,
           memory_ttl_s=600.0, plane=plane, max_speed_mps=2.2)
    return m


def box_at(X, Z, w_px=60, h_px=160):
    x, y = at(X, Z)
    return (x - w_px / 2, y - h_px, x + w_px / 2, y)


# Near the camera a normal walking step is a LOT of pixels. The old 140 px gate
# refused it; a metric gate accepts it because a person really can walk 0.9 m.
m = mem(PLANE)
m.resolve("A", None, 0.9, box_at(-0.45, 6.0), 0.0, vec=VEC)
same = m.resolve("B", None, 0.9, box_at(0.45, 6.0), 0.5, vec=VEC)
px_moved = math.hypot(at(-0.45, 6.0)[0] - at(0.45, 6.0)[0], 0)
check(same == "A", "0.9 m step near the camera is accepted as the same person",
      f"{px_moved:.0f} px moved — the old 140 px gate would have refused it")

# ...and far away, where 140 px is an impossible teleport, the metric gate refuses
m2 = mem(PLANE)
m2.resolve("C", None, 0.9, box_at(-4.0, 24.0), 0.0, vec=VEC)
other = m2.resolve("D", None, 0.9, box_at(4.0, 24.0), 0.2, vec=VEC)
px_far = abs(at(-4.0, 24.0)[0] - at(4.0, 24.0)[0])
check(other == "D", "8 m jump in 0.2 s far away is REFUSED (nobody moves that fast)",
      f"only {px_far:.0f} px — a pixel gate would have merged them")

# with no plane the old pixel behaviour must be exactly preserved
m3 = mem(GroundPlane.none())
m3.resolve("E", None, 0.9, (100, 100, 160, 260), 0.0, vec=VEC)
r = m3.resolve("F", None, 0.9, (120, 100, 180, 260), 0.5, vec=VEC)
check(r == "E", "no plane -> pixel gate still merges a small step")
m4 = mem(GroundPlane.none())
m4.resolve("G", None, 0.9, (100, 100, 160, 260), 0.0, vec=VEC)
r = m4.resolve("H", None, 0.9, (900, 100, 960, 260), 0.5, vec=VEC)
check(r == "H", "no plane -> pixel gate still refuses an 800 px jump")

print()
print("=" * 74)
print("  wiring, config and notebook integrity")
print("=" * 74)
for needle, why in [
    ("[KEVACV_BOOTSTRAP]", "kevacv bootstrap provides the ground plane"),
    ("plane=_ground, handoff_m=REID_HANDOFF_M", "stitcher receives the plane"),
    ('"pos": _bc_of.get(_sid)', "crossings carry a position"),
    ("_refresh_ground()", "plane is (re)built during the run"),
    ("MAX_WALK_SPEED_MPS", "metric walking speed in config"),
    ("GREET_PROXIMITY_M", "metric greet distance in config"),
    ("TIER A · geometry", "Tier A appears in the answers table"),
    ("TIER B · identity", "Tier B is reported beside it, not instead of it"),
    ("Tier A vs Tier B disagree", "a disagreement between the two is flagged"),
    ("Arriving parties (groups)", "parties are reported"),
    ("staff_contacts(run.get(\"frame_log\", []), roles, plane=_plane)",
     "greet proximity uses the plane"),
]:
    check(needle in SRC, why)

check("ground_points" in SRC, "the exact 4-point upgrade path is documented in config")
check(SRC.count("def tier_a_crossings") == 1, "tier_a_crossings defined once")
check(SRC.count("class GroundPlane") == 1, "GroundPlane defined once (no duplication)")

syn = []
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    if s.strip().startswith(("!", "%")):
        continue
    try:
        ast.parse(s)
    except SyntaxError as e:
        syn.append((i, e))
check(not syn, "every code cell still parses", f"{len(syn)} error(s)")
for i, e in syn:
    print(f"      cell {i}: {e}")
# Phase 6 REMOVES cells on purpose (4 embedded copies -> 1 bootstrap), so an
# ever-growing cell count is the wrong invariant. What matters is that every
# pipeline cell is still there.
for _m in ("def process_video", "def render_annotated", "def answer_for_run",
           "def reception_report", "def build_report"):
    check(_m in SRC, f"pipeline cell still present: {_m}")

print()
print("=" * 74)
if FAILED:
    print(f"  {len(FAILED)} FAILED:")
    for f in FAILED:
        print(f"    - {f}")
    sys.exit(1)
print("  ALL PASS")
print("=" * 74)
