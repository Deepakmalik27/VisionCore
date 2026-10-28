"""test_overlay_p2.py — the annotated video must SHOW what it decided.

WHY THIS FILE EXISTS
    The annotated video is the only tool anyone actually uses to judge this
    pipeline by eye, and until 2026-08-25 it drew a box, an id and a role --
    none of which are the thing that decided the count.

    Three bugs in this project's own audit history were invisible in the very
    video that was reviewed to find them:

      * the foot anchor was measured 350 px wrong (2026-08-19). The anchor is
        the single point that decides zone membership, and it was never drawn.
      * "staff" from a face-gallery match and "staff" from a zone-dwell guess
        render identically, and only one of them is evidence.
      * D0 removes ~26% of every detection. The funnel says HOW MANY. Nothing
        said WHERE, so frame-wide garbage and near-field guests looked the same.

    These are checks on the OVERLAY, not on the counting. They fail if the
    diagnostic layer silently stops being drawn -- which is exactly how it
    became absent in the first place.

Run: python tests/test_overlay_p2.py
"""
import ast
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


# ── 1. the renderer must ACCEPT the diagnostic inputs, with safe defaults ────
print()
print("=" * 74)
print("  render_annotated accepts the diagnostic inputs")
print("=" * 74)

src = (ROOT / "kevacv" / "engine.py").read_text(encoding="utf-8")
tree = ast.parse(src)
fn = next(n for n in ast.walk(tree)
          if isinstance(n, ast.FunctionDef) and n.name == "render_annotated")
kwonly = {a.arg for a in fn.args.kwonlyargs} | {a.arg for a in fn.args.args}
check("frame_ir" in kwonly, "render_annotated takes frame_ir")
check("killed_boxes" in kwonly, "render_annotated takes killed_boxes")

# both must default to None: a caller that does not supply them (any older
# call site, any test) must still render rather than raise.
defaults = dict(zip([a.arg for a in fn.args.args][-len(fn.args.defaults):],
                    fn.args.defaults)) if fn.args.defaults else {}
for name in ("frame_ir", "killed_boxes"):
    d = defaults.get(name)
    check(isinstance(d, ast.Constant) and d.value is None,
          f"{name} defaults to None (optional, never required)")

# ── 2. the engine must actually PASS them at the single call site ───────────
call_ok = "frame_ir=_frame_ir" in src and "killed_boxes=_killed_boxes" in src
check(call_ok, "process_video hands both maps to the renderer",
      "otherwise the overlay is dead code")

# ── 3. ghosts are captured from the filter chain ────────────────────────────
print()
print("=" * 74)
print("  the filter chain records what it killed")
print("=" * 74)
check("_raw_for_ghost" in src, "raw detections are captured at chain entry")
check("_killed_boxes[round(float(t), 3)]" in src,
      "kills are keyed by t, the one key render_annotated also has")
check("_KILLED_CAP_FRAMES" in src and "_KILLED_CAP_PER_FRAME" in src,
      "capture is CAPPED — a diagnostic must not OOM an 8-hour run")

# ghosts must be drawn BEFORE the live boxes, or a dead box paints over a
# real person and the video lies in the worst possible direction.
i_ghost = src.index("GHOSTS FIRST")
i_live = src.index("for _bi, (tid, x1, y1, x2, y2) in enumerate(boxes):")
check(i_ghost < i_live, "ghosts are drawn BEFORE live boxes",
      "a killed box must never paint over a real person")

# ── 4. the anchor crosshair uses the SAME point the trail/zones use ─────────
print()
print("=" * 74)
print("  the anchor that decides the zone is the anchor that is drawn")
print("=" * 74)
seg = src[src.index("THE ANCHOR."): src.index("WHICH ZONE, AND HOW SURE")]
check("_ax, _ay = int(bc[0]), int(bc[1])" in seg,
      "crosshair is drawn at bc — the same anchor_point() the trail uses",
      "drawing a DIFFERENT point would be worse than drawing none")

# ── 5. zone confidence comes from observe.py, not a second implementation ───
_i2 = src.index("WHICH ZONE, AND HOW SURE")
seg2 = src[_i2: _i2 + 1200]
check("from .observe import zone_and_confidence" in seg2,
      "zone+confidence is imported from observe.py",
      "a second copy could disagree with observations.jsonl")

# and that function must still have the signature the renderer calls it with
from kevacv.observe import zone_and_confidence  # noqa: E402
import inspect  # noqa: E402
params = list(inspect.signature(zone_and_confidence).parameters)
check(params[:4] == ["box", "polygons", "staff_zones", "feet_visible"],
      "observe.zone_and_confidence signature matches the call", str(params))

# it must actually run on a real polygon and return a bounded confidence
POLY = {"reception": [[100, 100], [400, 100], [400, 400], [100, 400]]}
name, conf = zone_and_confidence((200, 200, 260, 380), POLY, set(), True)
check(name == "reception" and 0.0 <= conf <= 1.0,
      "a body inside a zone resolves to that zone", f"{name} conf={conf}")
out_name, out_conf = zone_and_confidence((900, 900, 960, 980), POLY, set(), True)
check(out_name is None and out_conf == 0.0,
      "a body OUTSIDE every zone returns None, not a nearest guess",
      f"{out_name} conf={out_conf}")
# feet not visible must HALVE the confidence -- the footline is then inferred
_, c_seen = zone_and_confidence((200, 200, 260, 380), POLY, set(), True)
_, c_blind = zone_and_confidence((200, 200, 260, 380), POLY, set(), False)
# BOTH values are rounded to 3dp by observe.py, so the halving can only be
# asserted to within twice that rounding. A tighter epsilon fails on float
# representation alone (0.417 - 0.416 == 0.0010000000000000009), which is a
# test bug wearing the costume of a real one.
check(abs(c_blind * 2.0 - c_seen) <= 0.005,
      "occluded feet halve the zone confidence", f"{c_seen} -> {c_blind}")

# ── 6. why-staff must distinguish evidence from a guess ────────────────────
print()
print("=" * 74)
print("  'staff' says WHY it thinks so")
print("=" * 74)
_i3 = src.index('_why = ("face"')
seg3 = src[_i3 - 400: _i3 + 300]
check('"face"' in seg3 and '"zone"' in seg3,
      "the badge separates a gallery match from a zone-dwell guess")
check("isdigit()" in seg3,
      "it uses the id type — a non-numeric id came from the face gallery")

# ── 7. the HUD must show modality, because half the run behaves differently ─
print()
print("=" * 74)
print("  the HUD carries the modality")
print("=" * 74)
check('_ir_chip = "IR" if _ir_now else "COLOUR"' in src,
      "every frame is labelled IR or COLOUR")
check("frame_ir or {}" in src,
      "a missing frame_ir map degrades to COLOUR, it does not raise")

# ── 8. the dashed helper the ghosts rely on must exist with that signature ──
from kevacv.engine import _dashed_poly  # noqa: E402
p = list(inspect.signature(_dashed_poly).parameters)
check(p[:3] == ["frame", "poly", "colour"] and "dash" in p and "gap" in p,
      "_dashed_poly signature matches the ghost call", str(p))
img = np.zeros((200, 200, 3), dtype=np.uint8)
_dashed_poly(img, [(20, 20), (180, 20), (180, 180), (20, 180)], (120, 120, 130),
             dash=9, gap=7, thick=1)
check(img.any(), "a ghost outline actually marks pixels")


print()
print("=" * 74)


def test_script_level_checks_passed():
    assert not FAILED, f"overlay checks failed: {FAILED}"


if __name__ == "__main__":
    if FAILED:
        print(f"  {len(FAILED)} FAILED:")
        for f in FAILED:
            print(f"    - {f}")
        sys.exit(1)
    print("  ALL PASS")
    print("=" * 74)
