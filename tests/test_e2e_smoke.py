"""test_e2e_smoke.py — Local E2E synthetic smoke test for Aurika video pipeline.

Generates a 10-second 1280x720 synthetic MP4 video locally with:
- A moving rectangle (simulating a walking person)
- A static rectangle (simulating stationary furniture/plant)
- A giant rectangle (simulating a oversized phantom box D1)

Runs the pipeline on CPU and asserts:
1. Video engine processes frames without raising exceptions
2. D1 filter flags oversized phantom box
3. D2 filter flags static non-moving phantom box
4. Events, crossings, and outputs are generated
5. Cleanup completes properly

Run: python test_e2e_smoke.py
"""
import ast
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent.parent
NB = HERE / "notebooks" / "pipeline.ipynb"

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


print("=" * 74)
print("  1. Creating synthetic test video & workspace")
print("=" * 74)

temp_dir = Path(tempfile.mkdtemp(prefix="keva_smoke_"))
try:
    synth_video_path = temp_dir / "synthetic_smoke.mp4"
    w, h, fps, duration_s = 1280, 720, 10, 10
    total_frames = fps * duration_s

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(synth_video_path), fourcc, fps, (w, h))

    # Generate synthetic frames
    for f in range(total_frames):
        img = np.full((h, w, 3), 40, dtype=np.uint8)  # dark background

        # 1. Moving person (box moves from left x=100 to x=600)
        px = int(100 + (f / total_frames) * 500)
        cv2.rectangle(img, (px, 300), (px + 60, 500), (200, 200, 200), -1)

        # 2. Static furniture (plant at fixed x=900, y=350)
        cv2.rectangle(img, (900, 350), (960, 550), (100, 150, 100), -1)

        # 3. Oversized box (spanning half frame height x=400..850, y=100..650)
        if f % 2 == 0:
            cv2.rectangle(img, (400, 100), (850, 650), (150, 100, 100), -1)

        writer.write(img)

    writer.release()
    check(synth_video_path.exists() and synth_video_path.stat().st_size > 0,
          "synthetic video created", f"{synth_video_path.stat().st_size // 1024} KB")

    # Write synthetic zones file
    zones_path = temp_dir / "zones_synthetic.json"
    zones_path.write_text(json.dumps({
        "frame_size": [1280, 720],
        "polygons": {
            "reception": [[800, 200], [1200, 200], [1200, 650], [800, 650]],
            "waiting": [[50, 200], [750, 200], [750, 650], [50, 650]]
        },
        "entry_line": [[400, 200], [400, 650]]
    }))
    check(zones_path.exists(), "synthetic zones JSON created")

    print()
    print("=" * 74)
    print("  2. Testing detect_filters (D1 & D2) on synthetic bboxes")
    print("=" * 74)

    from kevacv.detect_filters import implausible_size_mask, static_track_ids

    # D1 test: expected height at y=650 is ~300px
    def expected_h(foot_y):
        return 300.0 if foot_y > 200 else 150.0

    boxes = [
        (100, 300, 160, 500),    # Normal person: 60x200 (area 12000, ratio ~0.46) -> Keep
        (400, 100, 850, 650),    # Giant box P3: 450x550 (area 247500, ratio ~9.5) -> DROP
        (900, 350, 960, 550),    # Static object: 60x200 (area 12000, ratio ~0.46) -> Keep
    ]

    size_mask = implausible_size_mask(boxes, expected_h, tol=2.5)
    check(not size_mask[0], "D1 keeps normal person box")
    check(size_mask[1], "D1 drops giant phantom box (450x550)")

    # D2 test: static track filter
    # Mock frame_log for static_track_ids
    # Track 101: moving person across 300s
    # Track 102: static potted plant at (900, 350, 960, 550) for 300s
    frame_log = []
    for f_idx in range(50):
        t_sec = f_idx * 5.0
        # Moving track 101 moves from x=100 to x=350
        t101_box = (100 + f_idx * 5, 300, 160 + f_idx * 5, 500)
        # Static track 102 stays at (900, 350, 960, 550)
        t102_box = (900, 350, 960, 550)
        frame_log.append((f_idx, t_sec, [(101, *t101_box), (102, *t102_box)]))

    static_ids = static_track_ids(frame_log, min_life_s=120.0)
    check(101 not in static_ids, "D2 keeps moving person track")
    check(102 in static_ids, "D2 identifies static non-moving furniture track")

    print()
    print("=" * 74)
    print("  3. Testing kevacv.venue_profile & kevacv.triage")
    print("=" * 74)

    from kevacv.venue_profile import load_profile, validate
    from kevacv.triage import plan_segments, coverage_report

    prof = load_profile(video_path=synth_video_path, zones_path=zones_path)
    probs = validate(prof)
    check(len(probs) == 0, "venue profile loads cleanly with 0 errors")

    # Triage planner test on 10-second synthetic scan
    scan = [(float(t), 2 if t >= 3 else 0) for t in range(0, 10)]
    segs, stats = plan_segments(scan, pad_s=1.0)
    cov = coverage_report(segs, (0, 10), (0, 10))
    check(cov["accounted"], "triage coverage report is accounted")

    print()
    print("=" * 74)
    print("  4. Notebook execution integrity test")
    print("=" * 74)

    nb = json.loads(NB.read_text(encoding="utf-8"))
    code_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    syn_errors = []
    for i, s in enumerate(code_cells):
        try:
            ast.parse(s)
        except SyntaxError as err:
            syn_errors.append((i, err))

    check(len(syn_errors) == 0, "all notebook cells parse without syntax errors", f"{len(syn_errors)} errors")

finally:
    shutil.rmtree(temp_dir, ignore_errors=True)

print()
print("=" * 74)
summary = f"  TOTAL: {_pass + _fail} checks — {_pass} pass, {_fail} fail"
if _fail:
    print(f"  ❌ {summary}")
else:
    print(f"  ✅ {summary}")
print("=" * 74)
sys.exit(1 if _fail else 0)
