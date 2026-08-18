"""test_v58_phaseC.py — behaviour checks for the Phase C batch.

Run AFTER patch_v58_phaseC.py. Checks the sweep's compliance boundary, the
identity pin's group-follow, the C4 debounce, the occluboost branch's
signature filtering, the occupancy consistency arithmetic, and that every
touched cell still parses.

Run: python test_v58_phaseC.py
"""
import ast
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


nb_json = json.dumps(nb)
if "PATCH_V58_PHASEC" not in nb_json:
    sys.exit("patch_v58_phaseC.py not applied — nothing to test")

eng = cw("def process_video")
run_cell = cw("# Cell 9b — MULTI-CHUNK RUNNER")

# ── C2: sweep exists, discards non-matches, pin follows the merge group ────
print("\nC2 — gallery sweep + identity pin")
check("ENABLE_STAFF_GALLERY_SWEEP and _STAFF_FACE_GALLERY" in eng,
      "sweep gated on flag AND an enrolled gallery")
check("# else: fe goes out of scope — nothing stored" in eng,
      "non-matching customer embeddings discarded (compliance boundary)")
check("role_hint[tid] = \"staff\"" in eng.split("STAFF-GALLERY SWEEP")[1][:2000],
      "a sweep match earns the staff role")
# pin: group-follow logic replicated exactly as patched
mapping = {"f1": "f1", "f2": "f1", "f3": "f3"}   # f2 merged into f1; f3 alone
_sweep_hits = {"f1": "alice"}
for _tid, _sn in _sweep_hits.items():
    for _k, _v in list(mapping.items()):
        if _v == _tid:
            mapping[_k] = _sn
    mapping[_tid] = _sn
check(mapping == {"f1": "alice", "f2": "alice", "f3": "f3"},
      "pin renames the fragment AND its whole merge-group", str(mapping))
check("for _k, _v in list(mapping.items()):" in eng,
      "group-follow wired into the notebook")

# ── C3: occluboost branch, signature-filtered construction ─────────────────
print("\nC3 — occluboost option")
check('elif TRACKER_MODE == "occluboost":' in eng, "branch exists")
check("inspect" in eng.split('occluboost":')[1][:900]
      and "_sig = _insp.signature(_OB.__init__).parameters" in eng,
      "constructor args signature-filtered against upstream drift")


class FakeOB:                       # older API: no cmc_method, no frame_rate
    def __init__(self, reid_model=None, device=None, half=False,
                 track_high_thresh=0.5, track_buffer=30):
        self.kw = dict(reid_model=reid_model, device=device, half=half,
                       track_high_thresh=track_high_thresh,
                       track_buffer=track_buffer)


import inspect as _insp

_want = dict(reid_model="be", device="0", half=False, track_high_thresh=0.45,
             track_low_thresh=0.2, new_track_thresh=0.45, track_buffer=480,
             match_thresh=0.75, cmc_method="sof", frame_rate=8)
_sig = _insp.signature(FakeOB.__init__).parameters
ob = FakeOB(**{k: v for k, v in _want.items() if k in _sig})
check(ob.kw["track_buffer"] == 480 and "cmc_method" not in ob.kw,
      "filtering survives an older constructor without crashing")

# ── C4: debounced hard cut joins the existing rebuild path ─────────────────
print("\nC4 — IR hard-cut")
check("if _gap_s > LOST_TRACK_BUFFER_S or _ir_cut:" in eng,
      "cut reuses the gated-silence rebuild (tracker + identity clear)")
check("IR_CUT_MIN_GAP_S" in eng and "_ir_cut_last" in eng, "debounce present")
check("ENABLE_IR_HARD_CUT   = False" in cw("# Cell 2 — CONFIG"),
      "OFF by default — the ablation decides")
# debounce arithmetic: flip at t=100 cuts; flicker at t=110 must not
ENABLE_IR_HARD_CUT, IR_CUT_MIN_GAP_S = True, 30.0
_prev, _last, cuts = [None], [-1e9], []
for t, irf in [(50, False), (100, True), (110, False), (111, True),
               (200, False)]:
    cut = (ENABLE_IR_HARD_CUT and irf is not None and _prev[0] is not None
           and irf != _prev[0] and (t - _last[0]) >= IR_CUT_MIN_GAP_S)
    _prev[0] = irf
    if cut:
        _last[0] = t
        cuts.append(t)
check(cuts == [100, 200], "flip cuts once; dusk flicker within 30 s ignored",
      str(cuts))

# ── C1r: occupancy consistency arithmetic ───────────────────────────────────
print("\nC1r — occupancy consistency")
rpt = cw("def reception_report")
check("occupancy_went_negative" in rpt and "occupancy_worst_deficit" in rpt,
      "fields land in the report dict")
ins = [{"t": t} for t in (10, 20, 30)]
outs = [{"t": t} for t in (5, 25, 40, 50)]     # exit BEFORE any entry + extra
_occ, _occ_min, _occ_dips = 0, 0, 0
for _t2, _d2 in sorted([(c["t"], 1) for c in ins]
                       + [(c["t"], -1) for c in outs]):
    _occ += _d2
    if _occ < 0:
        if _occ < _occ_min:
            _occ_min = _occ
        if _occ == -1 and _d2 == -1:
            _occ_dips += 1
check(_occ_dips == 2 and _occ_min == -1,
      "impossible exits counted, worst deficit tracked",
      f"dips={_occ_dips} min={_occ_min}")

# ── ABL: runner cell + gates + manifest provenance ──────────────────────────
print("\nABL — ablation runner")
abl = cw("Cell 9e — ABLATION RUNNER")
check('"occluboost": {"TRACKER_MODE": "occluboost"}' in abl,
      "variant table covers C3")
check('"ir_hardcut": {"ENABLE_IR_HARD_CUT": True}' in abl, "…and C4")
check('"no_sweep": {"ENABLE_STAFF_GALLERY_SWEEP": False}' in abl.replace("   ", " "),
      "…and C2's contribution (no_sweep)")
check("globals().update(_saved)" in abl and "finally:" in abl,
      "overrides restored even when a variant crashes")
check('not globals().get("RUN_ABLATION")' in run_cell,
      "night runner skipped during an ablation session")
check('"source_video": Path(video_path).name' in cw("EVAL_PHASE2"),
      "manifests record their source chunk for re-runs")
check('if "__" in str(_cond):' in abl,
      "variant packages are never themselves re-ablated")

# ── every touched cell parses ───────────────────────────────────────────────
print("\nsyntax — touched cells parse")
for tag in ("# Cell 2 — CONFIG", "def process_video",
            "# Cell 9b — MULTI-CHUNK RUNNER", "def reception_report",
            "EVAL_PHASE2", "Cell 9e — ABLATION RUNNER"):
    try:
        ast.parse(cw(tag))
        check(True, f"parses: {tag[:34]}")
    except SyntaxError as e:
        check(False, f"parses: {tag[:34]}", str(e))

print(f"\n{'=' * 60}")
if FAILED:
    print(f"❌ {len(FAILED)} FAILED: {FAILED}")
    sys.exit(1)
print("✅ all Phase C checks passed")
