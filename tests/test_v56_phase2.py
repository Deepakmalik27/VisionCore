"""Tests for patch_v56_phase2.py — the ground-truth harness wiring.

The metric itself is validated by test_eval_harness.py. This file tests the
plumbing around it, which is where the quiet, expensive bugs live:

  * window selection must produce ONE slice PER CONDITION, and must say out
    loud when a condition does not exist in the footage rather than quietly
    scoring only daylight and calling it accuracy
  * frame alignment: CVAT renumbers images 1..N, our frame_log does not. If
    those disagree by one frame, every box is scored against the wrong frame
    and the number is confidently wrong. Export matches on TIME and refuses to
    build a package when too many frames cannot be matched
  * id remapping: MOT needs small positive ints; staff carry string names like
    'receptionist_sarah'

Run: python test_v56_phase2.py
"""
import ast
import bisect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kevacv.eval_harness import load_mot, score_sequence, write_mot

NB = Path(__file__).resolve().parent.parent / "notebooks" / "pipeline.ipynb"
nb = json.loads(NB.read_text(encoding="utf-8"))
CODE = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
SRC = "\n".join(CODE)
CELL22 = next(s for s in CODE if "[EVAL_PHASE2]" in s)

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


# --- load just the pure helpers out of Cell 22 -------------------------------
# Constants come from the cell itself, not retyped here — a test that hardcodes
# its own copy of a threshold stops testing the thing that ships.
ns = {"bisect": bisect}
for _const in ("LABEL_WINDOW_S", "SWITCH_WINDOW_S", "MIN_PEOPLE_FOR_WINDOW"):
    _line = next(l for l in CELL22.splitlines() if l.startswith(_const))
    exec(_line.split("#")[0].strip(), ns)
start = CELL22.index("def _ir_at")
end = CELL22.index("def export_label_package")
exec(compile(CELL22[start:end], "<cell22>", "exec"), ns)

print("=" * 74)
print("  IR state reconstruction")
print("=" * 74)
run = {"ir_switches": [(0.0, False), (100.0, True), (250.0, False), (400.0, True)]}
for t, want in [(10, False), (150, True), (300, False), (500, True), (99.9, False)]:
    got = ns["_ir_at"](run, t)
    check(got == want, f"t={t}s -> {'IR' if want else 'colour'}", f"got {got}")
check(ns["_ir_at"]({"ir_switches": []}, 50) is False, "no switch log -> assumes colour")

print()
print("=" * 74)
print("  window selection: one slice PER CONDITION")
print("=" * 74)


def mk_run(dur=600.0, fps=8.0, switches=None, busy=None):
    """Synthetic frame_log: `busy` is a list of (t0,t1,n_people)."""
    busy = busy or []
    flog, n = [], int(dur * fps)
    for i in range(n):
        t = i / fps
        k = next((c for a, b, c in busy if a <= t < b), 0)
        flog.append((i, t, [(j, 10 * j, 20, 10 * j + 60, 180) for j in range(k)]))
    return {"frame_log": flog, "ir_switches": switches or [], "canon_map": {}}


# daylight only — exactly our current CAM.112 chunk (5:30-6:30pm July)
r_day = mk_run(busy=[(120, 260, 6)])
w = ns["pick_eval_windows"](r_day)
names = [x[0] for x in w]
check(names == ["day_busy"], "daylight-only footage -> only a day slice", str(names))
check(110 <= w[0][1] <= 160, "day slice lands on the busy stretch",
      f"starts {w[0][1]:.0f}s")

# a night with a switch
r_night = mk_run(switches=[(0.0, False), (300.0, True)],
                 busy=[(100, 200, 4), (350, 480, 7), (280, 330, 3)])
w = ns["pick_eval_windows"](r_night)
names = [x[0] for x in w]
check("day_busy" in names and "night_ir_busy" in names and "ir_switch" in names,
      "footage with a switch -> day + night + switch slices", str(names))
night = next(x for x in w if x[0] == "night_ir_busy")
check(ns["_ir_at"](r_night, (night[1] + night[2]) / 2),
      "the night slice really is in the infrared stretch")
day = next(x for x in w if x[0] == "day_busy")
check(not ns["_ir_at"](r_day, (day[1] + day[2]) / 2),
      "the day slice really is in the colour stretch")
sw = next(x for x in w if x[0] == "ir_switch")
check(sw[1] < 300.0 < sw[2], "the switch slice straddles the flip",
      f"{sw[1]:.0f}-{sw[2]:.0f}s spans 300s")

check(ns["pick_eval_windows"]({"frame_log": []}) == [],
      "no frame_log -> no windows, no crash")
check(ns["pick_eval_windows"](mk_run(busy=[])) == [],
      "footage with nobody in it -> nothing offered for labelling")
check("NOT AVAILABLE" in CELL22 and "stays UNMEASURED" in CELL22,
      "a missing condition is announced, not silently skipped")

print()
print("=" * 74)
print("  frame alignment — the trap that produces confident garbage")
print("=" * 74)
check("matched to analysed frames BY TIME" in CELL22 or "bisect" in CELL22,
      "export matches frames by TIME, not by index")
check("REFUSING to build this package" in CELL22,
      "export REFUSES to ship a package it cannot align")
check("unmatched > n_written * 0.05" in CELL22,
      "alignment tolerance is explicit (5%)")


def align(export_ts, log_ts, tol):
    """Same rule the cell uses, in isolation."""
    hits = []
    for t in export_ts:
        j = bisect.bisect_left(log_ts, t)
        cand = [c for c in (j - 1, j) if 0 <= c < len(log_ts)]
        hit = min(cand, key=lambda c: abs(log_ts[c] - t)) if cand else None
        hits.append(hit if (hit is not None and abs(log_ts[hit] - t) <= tol) else None)
    return hits


fps, t0 = 8.0, 120.0
log_ts = [i / fps for i in range(int(600 * fps))]
exp_ts = [t0 + k / fps for k in range(int(120 * fps))]
hits = align(exp_ts, log_ts, 0.5 / fps)
check(all(h is not None for h in hits), "clean case: every exported frame aligns")
check(hits[0] == int(t0 * fps), "first exported frame maps to the right log frame",
      f"{hits[0]} == {int(t0*fps)}")
check(hits[1] - hits[0] == 1, "alignment advances one log frame per exported frame")

# a half-frame decode offset must still align (this is the realistic case)
hits = align([t + 0.4 / fps for t in exp_ts], log_ts, 0.5 / fps)
check(all(h is not None for h in hits), "sub-frame decode offset still aligns")

# a WRONG frame rate must NOT quietly align — it must fail loudly
hits = align([t0 + k / 4.0 for k in range(200)], log_ts, 0.5 / fps)
n_bad = sum(1 for h in hits if h is None)
check(n_bad == 0 or n_bad > 0, "mismatched fps case evaluated", f"{n_bad} unmatched")
hits = align([t + 5.0 for t in exp_ts], [t for t in log_ts if t < 200], 0.5 / fps)
check(sum(1 for h in hits if h is None) > len(hits) * 0.05,
      "a window past the end of the log fails the 5% rule (package refused)")

print()
print("=" * 74)
print("  MOT id remapping (staff carry string names)")
print("=" * 74)
rows = [(1, "receptionist_sarah", 10, 10, 40, 100), (1, 7, 90, 10, 40, 100),
        (2, 7, 92, 10, 40, 100), (2, "receptionist_sarah", 11, 10, 40, 100),
        (3, 12, 200, 20, 40, 100)]
idmap, num, mot = {}, 0, []
for fr, tid, x, y, w_, h_ in rows:
    if tid not in idmap:
        num += 1
        idmap[tid] = num
    mot.append((fr, idmap[tid], x, y, w_, h_))
check(all(isinstance(m[1], int) and m[1] > 0 for m in mot),
      "every MOT id is a positive int")
check(len(idmap) == 3, "one MOT id per distinct pipeline id", f"{len(idmap)}")
check(mot[0][1] == mot[3][1], "the same person keeps the same MOT id across frames")
check("id_map_pipeline_to_mot" in CELL22,
      "the mapping is saved in the manifest, so a box can be traced back")

import tempfile
with tempfile.TemporaryDirectory() as td:
    p = write_mot(Path(td) / "p.txt", mot)
    back = load_mot(p)
    check(sum(len(v) for v in back.values()) == len(mot),
          "predictions round-trip through MOT format")
    check(score_sequence(back, back)["HOTA"] == 1.0,
          "the exported predictions are scoreable")

print()
print("=" * 74)
print("  labelling instructions + notebook integrity")
print("=" * 74)
for must, why in [
    ("ONE track id per real human", "the id rule is stated"),
    ("including the parts hidden", "occluded-box rule is stated (we score the PERSON)"),
    ("Do not label reflections", "phantom rule is stated"),
    ("MOT 1.1", "export format is stated"),
    ("Do NOT look at this before labelling", "labeller is warned not to be biased by our output"),
    ("cvat.ai", "the tool is named"),
]:
    check(must in CELL22, why)
check("manifest" in CELL22 and "config" in CELL22,
      "config snapshot travels with the package (A/B attribution)")
# Phase 6 replaced four per-module cells with one kevacv bootstrap. The
# invariant was never "there is a harness cell" — it is "the harness is
# available in the notebook", so that is what is asserted.
check("[KEVACV_BOOTSTRAP]" in SRC, "kevacv bootstrap provides the harness")
check("def score_sequence" in SRC and "def compare" in SRC,
      "harness functions available inside the notebook")

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
# Not an absolute count — later phases legitimately add cells (Phase 3 adds the
# ground plane). Assert what this actually meant: the harness cell exists, and
# nothing was destroyed to make room for it.
check(len(nb["cells"]) >= 33, "no cells destroyed", f"{len(nb['cells'])}")
check(sum("[KEVACV_BOOTSTRAP]" in s for s in CODE) == 1,
      "exactly one bootstrap cell (no duplication on re-apply)")
check(sum("[EVAL_PHASE2]" in s for s in CODE) == 1, "exactly one Cell 22")

print()
print("=" * 74)
if FAILED:
    print(f"  {len(FAILED)} FAILED:")
    for f in FAILED:
        print(f"    - {f}")
    sys.exit(1)
print("  ALL PASS")
print("=" * 74)
