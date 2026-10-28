"""test_v68 — certifies the IR debounce + staff-pin uniqueness patch.

Runs the ACTUAL patched notebook source (extracted), not a re-implementation.
"""
import json
import re
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebooks" / "pipeline.ipynb"
nb = json.loads(NB.read_text(encoding="utf-8"))
src = next("".join(c["source"]) for c in nb["cells"]
           if c["cell_type"] == "code" and "V68b" in "".join(c["source"]))

# ── 1. extract + run the debounce block against synthetic chroma ────────────
m = re.search(r"( +)_chroma71 = .*?_frame_ir\[_fi\] = _ir_now\n", src, re.S)
assert m, "debounce block not found"
block = "".join(l[12:] + "\n" for l in m.group(0).splitlines())  # dedent loop body

def run_stream(chromas, debounce=24):
    g = {"_ir_state": [None], "_ir_pend": [0, None], "_ir_switches": [],
         "_frame_ir": {}, "globals": lambda: {"IR_DEBOUNCE_FRAMES": debounce},
         "IR_CHROMA_THRESHOLD": 10.0, "cv2": None}
    code = block.replace("_frame_chroma(None, small=_smallc)", "_chroma_in")
    code = code.replace("_smallc = cv2.resize(_fr, (160, 90))", "pass")
    for i, ch in enumerate(chromas):
        g.update(_chroma_in=ch, _t=float(i), _fi=i)
        exec(code, g)
    return g

# flapping chroma around the threshold — ONE entry only (the initial state)
g = run_stream([9.0, 11.0] * 200)
assert len(g["_ir_switches"]) == 1, f"flap not debounced: {len(g['_ir_switches'])}"

# V71c hysteresis: in IR, chroma must exceed 1.35x threshold to leave.
# 12.0 > 10 but < 13.5 -> stays IR forever
g = run_stream([5.0] * 50 + [12.0] * 100)
assert len(g["_ir_switches"]) == 1, f"hysteresis leak: {g['_ir_switches']}"
assert g["_frame_ir"][140] is True

# clearly colour (14 > 13.5) -> real switch out of IR after the dwell
g = run_stream([5.0] * 50 + [14.0] * 100)
assert len(g["_ir_switches"]) == 2, f"real exit missed: {g['_ir_switches']}"
assert g["_ir_switches"][1][1] is False
assert g["_ir_switches"][1][0] >= 50 + 23

# real switch INTO ir from colour: threshold is plain 1.0x
g = run_stream([20.0] * 50 + [5.0] * 100)
assert len(g["_ir_switches"]) == 2 and g["_ir_switches"][1][1] is True
assert g["_frame_ir"][55] is False and g["_frame_ir"][90] is True

# ── 2. extract + run the V68c overlap veto ──────────────────────────────────
m2 = re.search(r"( +)# V68c: one body.*?_kept\.append\(_v68t\)\n", src, re.S)
assert m2, "V68c block not found"
veto = "".join(l[20:] + "\n" for l in m2.group(0).splitlines())

def run_veto(hits, scores, windows):
    g = {"_sweep_hits": dict(hits), "_sweep_scores": dict(scores),
         "windows": windows}
    exec(veto, g)
    return g["_sweep_hits"]

W = {9: (0, 100), 202: (150, 300), 246: (250, 400), 264: (500, 600)}
S = {9: .9, 202: .8, 246: .95, 264: .7}
H = {t: "sarah" for t in W}
out = run_veto(H, S, W)
# 202 overlaps 246 (250-300); 246 scores higher -> 202 evicted, rest kept
assert set(out) == {9, 246, 264}, f"veto wrong: {sorted(out)}"

# non-overlapping tracks are never evicted even with low scores
out = run_veto({1: "a", 2: "a"}, {1: .5, 2: .4}, {1: (0, 10), 2: (20, 30)})
assert set(out) == {1, 2}

# two different staff members at the same time are both fine
out = run_veto({1: "a", 2: "b"}, {1: .5, 2: .4}, {1: (0, 10), 2: (0, 10)})
assert set(out) == {1, 2}

print("test_v68: ALL PASS (debounce kills flapping, keeps real switches; "
      "staff pin is unique per overlapping window)")
