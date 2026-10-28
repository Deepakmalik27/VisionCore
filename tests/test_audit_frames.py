"""test_audit_frames.py — the frame auditor must separate furniture from a
person who is standing still.

WHY THIS EXISTS
    The statue on the reception desk and the receptionist behind it occupy
    adjacent pixels and both hold position. Every "is it a person?" heuristic
    that leans on movement fails here. The discriminator that works is whether
    the BOX BREATHES: a detector fed identical pixels returns an identical
    box; a real body's box changes as they turn, lean and gesture.

    The first version of this metric pooled widths and heights into one
    coefficient of variation, which measures how far width differs from height
    rather than how steady either is over time. A perfectly rigid 60x119
    statue scored 0.33 and read as "person-like" — the tool would have
    reported the scene clean and been believed.
"""
import gzip
import json
import os
import random
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
random.seed(7)

frames = []
for i in range(300):
    bs = []
    # STATUE: detector sees identical pixels, returns an identical box.
    bs.append([900 + (i % 3), 1500.0 + (i % 2) * 0.5, 1501.0, 1560.0, 1620.0])
    # RECEPTIONIST: holds position, but the box breathes.
    w = 180 + random.randint(-25, 25)
    h = 430 + random.randint(-45, 45)
    cx = 2500 + random.randint(-18, 18)
    cy = 1200 + random.randint(-14, 14)
    bs.append([1, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    # DUPLICATE PAIR: two ids riding one body, IoU ~0.7. The stitcher can
    # never merge these (they overlap in time), so they fragment identity.
    bs.append([7, 200.0, 900.0, 400.0, 1400.0])
    bs.append([8, 230.0, 930.0, 430.0, 1430.0])
    frames.append([i, i / 8.0, bs])

p = tempfile.mktemp(suffix=".json.gz")
with gzip.open(p, "wt") as fh:
    json.dump(frames, fh)
out = subprocess.run([sys.executable, os.path.join(ROOT, "tools/audit_frames.py"), p],
                     capture_output=True, text=True).stdout
os.unlink(p)

rows = [l for l in out.splitlines() if l.strip().startswith("(")]
statue = [l for l in rows if "RIGID" in l and "(12, 13)" in l]
person = [l for l in rows if "person-like" in l]

fail = 0


def check(ok, what):
    global fail
    print(f"  {'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        fail = 1


print("=" * 72)
print("  frame auditor: furniture vs a person standing still")
print("=" * 72)
check(bool(statue), "statue flagged RIGID")
check(bool(person), "standing receptionist NOT flagged as furniture")
check("7 + 8" in " ".join(out.split()), "duplicate id pair reported")
check("DETECTION/NMS fault" in out, "sustained duplicates named as a detection fault")
if statue and person:
    import re
    s_cv = float(re.split(r"\s+", statue[0].strip())[3])
    p_cv = min(float(re.split(r"\s+", l.strip())[3]) for l in person)
    check(p_cv > 4 * s_cv,
          f"margin is real: person cv {p_cv:.4f} > 4x statue cv {s_cv:.4f}")

print()
print("  ALL PASS" if not fail else "  FAILURES ABOVE")
# PYTEST VISIBILITY. This file's checks run at IMPORT and reported only via
# an exit code. A module-level sys.exit aborts pytest COLLECTION, which hid
# 59 of 74 test files -- and simply guarding the exit would have hidden the
# FAILURES instead. So the same condition is also asserted as a real test.
def test_script_level_checks_passed():
    assert not (fail), "module-level checks in this file failed"


if __name__ == "__main__":
    sys.exit(fail)
