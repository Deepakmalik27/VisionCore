#!/usr/bin/env python3
"""verify_fresh.py — prove THIS run uses THIS code, before burning GPU hours.

WHY THIS EXISTS
    "We might be improving one folder while another folder is what actually
    runs." That has already happened in this project twice:

      * build_id.py exists because five days of runs produced byte-identical
        results while the source on a laptop changed daily -- the GPU box was
        running a COPY of kevacv/ made once by setup_pod.sh.
      * the notebook carried a FROZEN second copy of the package, and the two
        drifted until a NameError landed an hour into a run.

    A fix you cannot prove is running is a fix you do not have. This asserts,
    from the filesystem and from the imported modules, that every fix made on
    2026-08-25/26 is present in the code that is ABOUT to run -- and it fails
    loudly rather than letting a 2-hour run produce stale numbers.

WHAT IT CANNOT PROVE
    That the fixes are CORRECT. Only that they are present and reachable.
    Correctness is what tests/ and the ground truth are for.

Run: python tools/verify_fresh.py            (exit 0 = fresh, 1 = stale)
"""
import hashlib
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


print()
print("=" * 78)
print("  1. WHICH FILES ARE WE ACTUALLY IMPORTING?")
print("=" * 78)
# Import the package and report where each module came from. If a stale copy is
# on sys.path ahead of this checkout, this is where it shows.
import kevacv  # noqa: E402

pkg_dir = Path(kevacv.__file__).resolve().parent
print(f"  kevacv package  : {pkg_dir}")
check(pkg_dir == (ROOT / "kevacv").resolve(),
      "the imported kevacv IS this checkout",
      f"expected {(ROOT / 'kevacv').resolve()}")

mods = {}
for name in ("engine", "pipeline", "observe", "config", "build_id", "coverage",
             "validity", "visits"):
    m = importlib.import_module(f"kevacv.{name}")
    f = Path(m.__file__).resolve()
    mods[name] = f
    if f.parent != pkg_dir:
        check(False, f"kevacv.{name} loaded from a DIFFERENT directory", str(f))

# a single fingerprint over the modules that matter, so two runs can be
# compared by one number rather than by eight paths
h = hashlib.sha256()
for name in sorted(mods):
    h.update(mods[name].read_bytes())
FINGERPRINT = h.hexdigest()[:12]
print(f"  code fingerprint: {FINGERPRINT}   <- record this next to the run")

print()
print("=" * 78)
print("  2. IS EVERY 2026-08-25/26 FIX PRESENT IN THE CODE ABOUT TO RUN?")
print("=" * 78)

ENG = (pkg_dir / "engine.py").read_text(encoding="utf-8")
PIPE = (pkg_dir / "pipeline.py").read_text(encoding="utf-8")
OBS = (pkg_dir / "observe.py").read_text(encoding="utf-8")
CFG = (pkg_dir / "config.py").read_text(encoding="utf-8")
SLIT = (ROOT / "tools" / "slit_count.py").read_text(encoding="utf-8")

check("_import_tracker" in ENG, "boxmot-19 tracker resolver",
      "without it the run dies at tracker construction on boxmot>=19")
check("_numeric_tracker_ids" in ENG and "zone.trigger(_sv_dets)" in ENG,
      "named-staff-id alias for supervision",
      "without it sv.LineZone.trigger raises the moment a face matches")
check("ROLE LOCK" in ENG and "_face_named" in ENG,
      "ROLE LOCK (a face match outranks a zone vote)",
      "without it staff are filed as ungreeted guests")
check("_coverage = _CoverageGrid" in ENG and "_canary = _DetectorCanary" in ENG,
      "coverage map + detector canary are constructed")
check("_coverage.add_detection(bc," in ENG, "coverage grid is FED")
check("_canary.observe(t," in ENG, "canary is FED")
check("frame_ir=_frame_ir" in ENG and "killed_boxes=_killed_boxes" in ENG,
      "overlay diagnostics reach render_annotated")
check("GHOSTS FIRST" in ENG, "ghost boxes for killed detections")
check("THE ANCHOR." in ENG, "foot-anchor crosshair")
check('_ir_chip = "IR" if _ir_now else "COLOUR"' in ENG, "IR/COLOUR HUD chip")
check("make_run_id" in ENG and "make_run_id" in PIPE,
      "run_id collision fix, in BOTH sites")
check("short_emb_id" in OBS and "short_emb_id" in ENG,
      "short embedding ids (were 90 chars of filename)")
check("staff_zone_s" in OBS and "role_source" in OBS,
      "staff EVIDENCE columns (not a verdict)")
check("_gt_candidates" in PIPE, "camera-scoped ground truth",
      "without it a bare gt.txt scores one camera against another")
check("coverage_findings" in PIPE and "canary_findings" in PIPE,
      "coverage/canary findings reach result['findings']")
check("RUN_CONFIG_PATH" in CFG, "config path recorded for run identity")
check("MIN_WIDTH" in SLIT and "too_narrow" in SLIT,
      "slit shape gate (95% of false positives)")

print()
print("=" * 78)
print("  3. IS ANY OTHER COPY OF kevacv ON THIS MACHINE?")
print("=" * 78)
# A second checkout that happens to be earlier on sys.path is the exact failure
# build_id.py was written for.
others = []
for p in sys.path:
    try:
        cand = Path(p) / "kevacv" / "engine.py"
    except Exception:
        continue
    if cand.exists() and cand.resolve() != (pkg_dir / "engine.py"):
        others.append(cand.resolve())
if others:
    for o in others:
        print(f"       {o}")
check(not others, "no competing kevacv on sys.path",
      f"{len(others)} other copy(ies)" if others else "")

print()
print("=" * 78)
print(f"  CODE FINGERPRINT {FINGERPRINT}")
if FAILED:
    print(f"  STALE — {len(FAILED)} check(s) failed. DO NOT trust a run from this tree:")
    for f in FAILED:
        print(f"    - {f}")
    print("=" * 78)
    sys.exit(1)
print("  FRESH — every 2026-08-25/26 fix is present in the code that will run.")
print("=" * 78)
