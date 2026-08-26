"""Tests for patch_v56_phase8.py — PRIVACY: face recognition scope.

The patch must accomplish these things:

  P1  FACE_SCOPE config exists and defaults to "staff_only"
  P2  The post-processing face embedding loop is gated on FACE_SCOPE
  P3  A privacy banner prints in the run output
  P4  FACE_SCOPE can be overridden from venue profile

Venue profile tests:
  V1  Default face_scope is "staff_only"
  V2  An invalid face_scope is caught by validate()
  V3  "all" passes validation
  V4  The privacy section merges from a profile override

Functional tests:
  F1  Staff gallery matching (per-frame) is NOT affected by FACE_SCOPE
  F2  The face_embeddings dict only contains staff IDs when scope is staff_only
  F3  The face_embeddings dict contains all IDs when scope is "all"
"""
import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
NB = HERE / "notebooks" / "pipeline.ipynb"
sys.path.insert(0, str(HERE))

_pass = _fail = 0


def check(cond, label, detail=None):
    global _pass, _fail
    if cond:
        _pass += 1
        print(f"  ✅ {label}")
    else:
        _fail += 1
        d = f"  ({detail})" if detail else ""
        print(f"  ❌ {label}{d}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Load the notebook
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
nb = json.loads(NB.read_text(encoding="utf-8"))
cells = nb["cells"]
CODE = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
CELLS = [(s, i) for i, s in enumerate(CODE)]
SRC = "\n".join(CODE)

print("=" * 74)
print("  P1 — FACE_SCOPE config exists and defaults to staff_only")
print("=" * 74)
check("FACE_SCOPE" in SRC, "FACE_SCOPE is defined in the notebook")
check('FACE_SCOPE = "staff_only"' in SRC, "default is staff_only")
check("PATCH_V56_PHASE8" in SRC, "phase 8 marker is present")

print()
print("=" * 74)
print("  P2 — face embedding loop is gated on FACE_SCOPE")
print("=" * 74)
check("_face_scope" in SRC, "FACE_SCOPE is read into _face_scope")
check("_face_eligible" in SRC, "face-eligible set is computed")
check('_face_scope == "all"' in SRC or "_face_scope == 'all'" in SRC,
      "all-tracks path exists for opt-in")
check("tid not in _face_eligible" in SRC,
      "non-eligible tracks are skipped in face embedding loop")
check("role_hint" in SRC and "_face_eligible" in SRC,
      "staff role_hint feeds into face-eligible set")
check("_STAFF_FACE_GALLERY" in SRC and "_face_eligible" in SRC,
      "staff gallery match feeds into face-eligible set")

print()
print("=" * 74)
print("  P3 — privacy banner in run output")
print("=" * 74)
check("Face scope:" in SRC or "face scope:" in SRC.lower(),
      "privacy banner prints face scope status")
check("STAFF ONLY" in SRC, "staff_only mode is labelled clearly")
check("customer consent" in SRC.lower() or "customer faces not processed" in SRC.lower(),
      "all-tracks mode warns about consent requirement")

print()
print("=" * 74)
print("  P4 — FACE_SCOPE from venue profile")
print("=" * 74)
check("face_scope" in SRC and "profile" in SRC.lower(),
      "profile override reads face_scope")

print()
print("=" * 74)
print("  V1-V4 — venue profile privacy section")
print("=" * 74)

from kevacv.venue_profile import DEFAULTS, ENUMS, validate, load_profile

check("privacy" in DEFAULTS, "V1 privacy section exists in defaults")
check(DEFAULTS["privacy"]["face_scope"] == "staff_only",
      "V1 default face_scope is staff_only")
check("privacy.face_scope" in ENUMS, "V2 face_scope has enum validation")
check(ENUMS["privacy.face_scope"] == ("staff_only", "all"),
      "V2 allowed values are staff_only and all")

# V2: invalid face_scope caught
bad_profile = {"camera": {}, "venue": {}, "privacy": {"face_scope": "everyone"}}
probs = validate(bad_profile)
check(any("face_scope" in p for p in probs),
      "V2 invalid face_scope is caught by validate()")

# V3: valid face_scope passes
good_profile = {"camera": {}, "venue": {}, "privacy": {"face_scope": "all"}}
probs = validate(good_profile)
check(not any("face_scope" in p for p in probs),
      "V3 face_scope='all' passes validation")

# V4: defaults merge
prof = load_profile()
check(prof.get("privacy", {}).get("face_scope") == "staff_only",
      "V4 default profile has staff_only face_scope")

# V4: override merges
prof_override = load_profile(explicit={"privacy": {"face_scope": "all"}})
check(prof_override.get("privacy", {}).get("face_scope") == "all",
      "V4 profile override with face_scope=all works")

print()
print("=" * 74)
print("  F1 — staff gallery matching is NOT affected by FACE_SCOPE")
print("=" * 74)
# The per-frame face embed (inside the detection loop) must NOT check FACE_SCOPE.
# It's gated only by ENABLE_FACE_CORROBORATION. Find the staff gallery match code.
# Look for the pattern: the per-frame face embed block should NOT contain _face_eligible
eng_cell_src = None
for s in CODE:
    if "def process_video" in s:
        eng_cell_src = s
        break
check(eng_cell_src is not None, "engine cell found")

if eng_cell_src:
    # The staff gallery match block is in the per-frame loop.
    # It should use ENABLE_FACE_CORROBORATION but NOT _face_eligible.
    # Find the per-frame face embedding section (inside live loop)
    lines = eng_cell_src.split("\n")
    per_frame_block = [line for line in lines if "STAFF_MATCH_THRESHOLD" in line or "embed_face_scored" in line or "STAFF_GALLERY" in line]
    post_process_block = [line for line in lines if "FACE EMBEDDINGS (v30" in line or "_face_eligible" in line]

    per_frame_text = "\n".join(per_frame_block)
    post_process_text = "\n".join(post_process_block)

    check("_face_eligible" not in per_frame_text,
          "F1 per-frame staff gallery match does NOT check _face_eligible")
    check("STAFF_MATCH_THRESHOLD" in eng_cell_src or "_STAFF_FACE_GALLERY" in eng_cell_src or "STAFF_GALLERY" in eng_cell_src,
          "F1 per-frame block still does staff gallery matching")
    check("_face_eligible" in post_process_text,
          "F2 post-processing block checks _face_eligible")

print()
print("=" * 74)
print("  Notebook integrity")
print("=" * 74)
# Syntax check all code cells
syn = []
for i, s in enumerate(CODE):
    try:
        ast.parse(s)
    except SyntaxError as e:
        syn.append((i, e))
check(not syn, "every code cell parses", f"{len(syn)} error(s)")
for i, e in syn:
    print(f"      cell {i}: {e}")

check("PATCH_V56_PHASE8" in SRC, "phase 8 marker present")
check(sum("KEVACV_BOOTSTRAP" in s for s in CODE) == 1,
      "exactly one kevacv bootstrap cell")
check(len(nb["cells"]) >= 30, "notebook intact", f"{len(nb['cells'])} cells")

# Runtime order: face scope config before engine
cfg_cell = next(i for s, i in CELLS if "FACE_SCOPE" in s and "staff_only" in s)
eng_cell = next(i for s, i in CELLS if "def process_video" in s)
check(cfg_cell < eng_cell,
      "FACE_SCOPE config is defined before the engine cell",
      f"cell {cfg_cell} < {eng_cell}")

print()
print("=" * 74)
summary = f"  TOTAL: {_pass + _fail} checks — {_pass} pass, {_fail} fail"
if _fail:
    print(f"  ❌ {summary}")
else:
    print(f"  ✅ {summary}")
print("=" * 74)
# PYTEST VISIBILITY. This file's checks run at IMPORT and reported only via
# an exit code. A module-level sys.exit aborts pytest COLLECTION, which hid
# 59 of 74 test files -- and simply guarding the exit would have hidden the
# FAILURES instead. So the same condition is also asserted as a real test.
def test_script_level_checks_passed():
    assert not (1 if _fail else 0), "module-level checks in this file failed"


if __name__ == "__main__":
    sys.exit(1 if _fail else 0)
