"""Tests for patch_v56_phase11.py — the annotated video has to be readable.

Structural checks that the patch landed, plus the drawing helpers actually
EXECUTED against real pixels. "It should look cleaner" is not a claim a patch
script can make on its own — a dashed line that renders solid, or a "dimmed"
colour that comes back brighter, would pass every text check and change
nothing on screen.

Run: python test_v56_phase11.py
"""
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent.parent
NB = HERE / "notebooks" / "pipeline.ipynb"
MARKER = "PHASE11_RENDER_LEGIBILITY"

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


nb = json.loads(NB.read_text(encoding="utf-8"))
eng = next("".join(c["source"]) for c in nb["cells"]
           if c["cell_type"] == "code" and "def render_annotated" in "".join(c["source"]))

print("=" * 74)
print("  the patch landed")
print("=" * 74)
check(MARKER in eng, "phase-11 marker present")
check("_dashed_poly(frame, poly, _zc)" in eng, "zones are drawn DASHED")
check("cv2.polylines(frame, [poly], True, zone_bgr" not in eng,
      "and the old solid zone outline is GONE",
      "a solid rectangle must mean exactly one thing: a person")
check("cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)" in eng,
      "person boxes are still solid and thick — the only solid rectangles")
check("IGNORED (static phantom)" in eng, "D3's phantom regions are drawn struck through")
check("phantoms=_phantoms" in eng, "and the engine passes them to the renderer")
check("phantoms=None" in eng, "with a safe default when there are none")
check("_ly - _ly0 > 6" in eng, "a displaced label gets a leader line")
check("if n not in _mask_z" in eng, "mask zones are excluded from HUD counts")

print()
print("=" * 74)
print("  the helpers actually work on real pixels")
print("=" * 74)
ns = {"cv2": cv2, "np": np}
for fn in ("_dim", "_dashed_poly"):
    m = re.search(rf"^def {fn}\(.*?(?=^def |\Z)", eng, re.S | re.M)
    if not m:
        check(False, f"could not extract {fn} from the notebook")
        continue
    exec(compile(m.group(0), fn, "exec"), ns)

check("_dim" in ns and "_dashed_poly" in ns, "both helpers extracted and compiled")

# _dim must actually darken, and must stay a valid BGR triple
out = ns["_dim"]((200, 100, 50))
check(all(0 <= c <= 255 for c in out) and len(out) == 3, "_dim returns a valid BGR", str(out))
check(all(o < i for o, i in zip(out, (200, 100, 50))), "_dim genuinely darkens", str(out))
check(ns["_dim"]((255, 255, 255)) != (255, 255, 255), "even pure white recedes")

# a dashed square must leave GAPS — this is the whole point
poly = np.array([[50, 50], [350, 50], [350, 250], [50, 250]], dtype=np.int32)
dashed = np.zeros((300, 400, 3), np.uint8)
ns["_dashed_poly"](dashed, poly, (255, 255, 255))
solid = np.zeros((300, 400, 3), np.uint8)
# LINE_AA on BOTH. _dashed_poly antialiases, and an antialiased line spreads
# faint pixels either side of its centre — measured against a hard-edged
# LINE_8 reference the DASHED outline came back with 172% of the solid one's
# ink, which says nothing about dashing and everything about the comparison.
cv2.polylines(solid, [poly], True, (255, 255, 255), 1, cv2.LINE_AA)
d_on = int((dashed.max(axis=2) > 40).sum())
s_on = int((solid.max(axis=2) > 40).sum())
check(d_on > 0, "the dashed outline draws something at all", f"{d_on}px")
check(d_on < s_on * 0.85, "and it is genuinely broken, not solid",
      f"dashed {d_on}px vs solid {s_on}px")
check(d_on > s_on * 0.25, "but still reads as an outline, not a dotted whisper",
      f"{100*d_on/s_on:.0f}% of solid")

# the top edge must contain at least one real gap
row = dashed[50, 50:351].max(axis=1) > 40
runs = "".join("1" if v else "0" for v in row)
check("0" in runs.strip("0"), "there is a real gap along an edge",
      f"{runs.count('10')} dash starts")

# it must survive the shapes a zones file really contains
for name, p in [("triangle", [[10, 10], [200, 30], [100, 200]]),
                ("degenerate (all one point)", [[5, 5], [5, 5], [5, 5]]),
                ("two points", [[10, 10], [200, 200]])]:
    try:
        ns["_dashed_poly"](np.zeros((300, 400, 3), np.uint8),
                           np.array(p, dtype=np.int32), (255, 255, 255))
        ok = True
    except Exception as e:
        ok = False
        print(f"      {name}: {e}")
    check(ok, f"handles a {name} polygon without crashing")

print()
print("=" * 74)
# PYTEST VISIBILITY. This file's checks run at IMPORT and reported only via
# an exit code. A module-level sys.exit aborts pytest COLLECTION, which hid
# 59 of 74 test files -- and simply guarding the exit would have hidden the
# FAILURES instead. So the same condition is also asserted as a real test.
def test_script_level_checks_passed():
    assert not (FAILED), "module-level checks in this file failed"


if __name__ == "__main__":
    if FAILED:
        print(f"  {len(FAILED)} FAILED:")
        for f in FAILED:
            print(f"    - {f}")
        sys.exit(1)
print("  ALL PASS")
print("=" * 74)
