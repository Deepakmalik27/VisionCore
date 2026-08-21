"""test_gt_kit.py — the ground-truth kit on a synthetic package.

The worst bug this kit could have is a silent frame misalignment: a score that
is confidently wrong. So the tests build a package where the truth is known
exactly and check the scores land where arithmetic says they must.

Run: python test_gt_kit.py
"""
import json
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from kevacv.eval_harness import load_mot, score_sequence, write_mot

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


import tempfile

tmp = Path(tempfile.mkdtemp())
pkg = tmp / "CAM.112_day_busy"
(pkg / "images").mkdir(parents=True)
for i in range(1, 11):
    (pkg / "images" / f"{i:06d}.jpg").write_bytes(b"\xff\xd8\xff\xd9")  # stub jpg

# two people walking; person 2 suffers an id switch at frame 6 in PREDICTIONS
gt_rows, pred_rows = [], []
for f in range(1, 11):
    gt_rows.append((f, 1, 100 + 10 * f, 100, 50, 120))
    gt_rows.append((f, 2, 400 - 10 * f, 200, 50, 120))
    pred_rows.append((f, 1, 100 + 10 * f, 100, 50, 120))          # perfect
    pred_rows.append((f, 2 if f <= 5 else 7, 400 - 10 * f, 200, 50, 120))
write_mot(pkg / "predictions.txt", pred_rows)
(pkg / "manifest.json").write_text(json.dumps(
    {"window_clock": ["18:00:00", "18:02:00"], "is_infrared": False,
     "n_pred_ids": 3, "config": {"TRACKER_MODE": "botsort-reid"}}))

# ── seed: CVAT archive layout + alignment guard ─────────────────────────────
print("\nseed — CVAT MOT 1.1 archive")
r = subprocess.run([sys.executable, "tools/gt_kit.py", "seed", str(pkg)],
                   capture_output=True, text=True, cwd=HERE)
check(r.returncode == 0, "seed runs", r.stderr[-200:] if r.returncode else "")
zp = pkg / "cvat_seed.zip"
check(zp.exists(), "cvat_seed.zip written")
with zipfile.ZipFile(zp) as z:
    names = set(z.namelist())
    check(names == {"gt/gt.txt", "gt/labels.txt"},
          "archive layout gt/gt.txt + gt/labels.txt", str(names))
    body = z.read("gt/gt.txt").decode()
    first = body.splitlines()[0].split(",")
    check(len(first) == 9 and first[6:] == ["1", "1", "1"],
          "rows are 9-column MOT gt (not_ignored,class,visibility)", first)
    check(z.read("gt/labels.txt").decode() == "person", "label is 'person'")

# misaligned package (predictions reference frame 99, only 10 images) aborts
bad = tmp / "bad_pkg"
(bad / "images").mkdir(parents=True)
(bad / "images" / "000001.jpg").write_bytes(b"\xff\xd8\xff\xd9")
write_mot(bad / "predictions.txt", [(99, 1, 0, 0, 10, 10)])
r = subprocess.run([sys.executable, "tools/gt_kit.py", "seed", str(bad)],
                   capture_output=True, text=True, cwd=HERE)
check(r.returncode != 0, "misaligned seed REFUSES instead of shipping a lie")

# ── score: known arithmetic ─────────────────────────────────────────────────
print("\nscore — the numbers land where arithmetic says")
gt_path = tmp / "CAM.112_day_busy_gt.txt"
write_mot(gt_path, gt_rows)
res = score_sequence(load_mot(gt_path), load_mot(pkg / "predictions.txt"))
check(res["DetA"] > 0.99, "every box found -> DetA ~ 1", f"DetA={res['DetA']:.3f}")
check(res["AssA"] < 0.99, "one id switch -> AssA < 1", f"AssA={res['AssA']:.3f}")
check(res["HOTA"] < res["DetA"], "HOTA punished by the switch",
      f"HOTA={res['HOTA']:.3f}")

perfect = score_sequence(load_mot(gt_path), load_mot(gt_path))
check(perfect["HOTA"] > 0.999, "gt vs itself -> HOTA 1.0",
      f"{perfect['HOTA']:.4f}")

# ── score CLI + freeze + compare round-trip ────────────────────────────────
print("\nCLI — score --freeze, then compare detects a regression")
r = subprocess.run([sys.executable, "tools/gt_kit.py", "score", str(pkg),
                    str(gt_path), "--freeze"],
                   capture_output=True, text=True, cwd=HERE)
check(r.returncode == 0 and "BASELINE FROZEN" in r.stdout, "score --freeze runs",
      r.stderr[-200:] if r.returncode else "")
check((pkg / "baseline_score.json").exists(), "baseline saved")
check((pkg / "errors.csv").exists(), "per-frame errors CSV saved")

# a 'worse' run: person 2 fragments twice more
worse_rows = [(f, tid if tid != 7 or f <= 7 else 9, x, y, w, h)
              for f, tid, x, y, w, h in pred_rows]
pkg2 = tmp / "CAM.112_day_busy_worse"
(pkg2 / "images").mkdir(parents=True)
write_mot(pkg2 / "predictions.txt", worse_rows)
from kevacv.eval_harness import explain, save_baseline
res2 = explain(score_sequence(load_mot(gt_path),
                              load_mot(pkg2 / "predictions.txt")), label="worse")
save_baseline(res2, pkg2 / "score.json", label="worse")
r = subprocess.run([sys.executable, "tools/gt_kit.py", "compare",
                    str(pkg / "baseline_score.json"), str(pkg2 / "score.json")],
                   capture_output=True, text=True, cwd=HERE)
check(r.returncode == 0, "compare runs")
check(res2["AssA"] < res["AssA"], "extra fragmentation scores worse",
      f"{res['AssA']:.3f} -> {res2['AssA']:.3f}")

print(f"\n{'=' * 60}")
# PYTEST VISIBILITY. This file's checks run at IMPORT and reported only via
# an exit code. A module-level sys.exit aborts pytest COLLECTION, which hid
# 59 of 74 test files -- and simply guarding the exit would have hidden the
# FAILURES instead. So the same condition is also asserted as a real test.
def test_script_level_checks_passed():
    assert not (FAILED), "module-level checks in this file failed"


if __name__ == "__main__":
    if FAILED:
        print(f"❌ {len(FAILED)} FAILED: {FAILED}")
        sys.exit(1)
print("✅ all gt_kit checks passed")
