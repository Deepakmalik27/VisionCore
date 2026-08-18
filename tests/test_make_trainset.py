"""test_make_trainset.py — a fine-tuning set must never teach the wrong thing.

WHY THIS EXISTS
    The whole point of fine-tuning here is to correct ONE defect: the detector
    draws street-shaped boxes (h/w ~2.5, from CrowdHuman) on overhead-shaped
    people (h/w ~1.14). That wrong shape puts the box bottom ~260px below the
    real feet, and every zone test and entry-line crossing reads that bottom.

    Two ways to build a training set that makes things WORSE:

    1. CARRIED-FORWARD LABELS. The first gt.txt collected on this project was
       99% copy-forward — six boxes drawn on frame 1 and C pressed ninety-nine
       times — while the people underneath moved up to 255px. Training on that
       teaches "a person is wherever they were a few seconds ago".

    2. LABELS TRACED FROM THE MODEL. If a labeller nudges the predicted boxes
       instead of drawing the people, the labels carry the model's own shape and
       fine-tuning changes nothing while looking like progress.

    Both are silent. Neither raises an error. So both get a guard.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "label_pkg" / "quick100"
fail = 0


def check(ok, what, detail=""):
    global fail
    print(f"  {'PASS' if ok else 'FAIL'}  {what}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        fail = 1


def run(gt_rows, out):
    p = Path(tempfile.mktemp(suffix=".txt"))
    p.write_text("\n".join(gt_rows) + "\n")
    r = subprocess.run([sys.executable, str(ROOT / "tools/make_trainset.py"),
                        str(PKG), str(p), "--out", out],
                       capture_output=True, text=True)
    return r.returncode, r.stdout


print("=" * 74)
print("  fine-tuning set: refuses the two sets that would teach the wrong thing")
print("=" * 74)

if not PKG.exists():
    print(f"  SKIP  {PKG} not present")
    sys.exit(0)

# 1. every frame identical -> only frame 1 survives -> too few to train
rows = [f"{f},{i+1},100.00,200.00,290.00,330.00,1,1,1"
        for f in range(1, 21) for i in range(3)]
rc, out = run(rows, "/tmp/ts_t1")
check(rc == 1 and "REFUSING" in out,
      "all-identical labels are refused, not turned into a 1-image set",
      "static_frames cannot flag frame 1, so a count guard is required")

# 2. labels that carry the MODEL's shape -> nothing to learn
pred = {}
for ln in (PKG / "predictions.txt").read_text().splitlines():
    f = ln.split(",")
    if len(f) >= 6:
        pred.setdefault(int(f[0]), []).append(f)
rows = []
for fr in sorted(pred)[:30]:
    for j, f in enumerate(pred[fr]):
        rows.append(f"{fr},{j+1},{float(f[2])+(fr%5):.2f},"
                    f"{float(f[3])+(fr%3):.2f},{f[4]},{f[5]},1,1,1")
rc, out = run(rows, "/tmp/ts_t2")
check(rc == 1 and "traced from the predictions" in out,
      "labels within 20% of the model's own shape are refused")

# 3. genuinely varied, human-shaped labels -> builds
rows = [f"{fr},{i+1},{200+i*300+fr*3}.00,{300+(i%2)*200+fr*2}.00,"
        f"290.00,330.00,1,1,1"
        for fr in range(1, 41) for i in range(4)]
rc, out = run(rows, "/tmp/ts_t3")
check(rc == 0 and "wrote" in out, "good varied labels build a set")

# 4. the YOLO labels must be valid and normalised, or training silently learns
#    garbage from out-of-range coordinates
bad = tot = 0
for f in sorted(Path("/tmp/ts_t3/labels/train").glob("*.txt")):
    for ln in f.read_text().splitlines():
        if not ln.strip():
            continue
        tot += 1
        parts = ln.split()
        if len(parts) != 5 or not all(0.0 <= float(v) <= 1.0 for v in parts[1:]):
            bad += 1
check(tot > 0 and bad == 0, "every YOLO label is class + 4 normalised floats",
      f"{tot} labels, {bad} bad")

# 5. the dataset.yaml must warn about the 2-class -> 1-class drop, because
#    best.pt has a head class that rebuilds ~749 occluded bodies per hour
y = Path("/tmp/ts_t3/dataset.yaml").read_text()
check("head" in y.lower(), "dataset.yaml flags the head-class trade-off")

print()
print("  ALL PASS" if not fail else "  FAILURES ABOVE")
sys.exit(fail)
