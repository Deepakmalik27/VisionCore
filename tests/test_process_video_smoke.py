"""test_process_video_smoke.py — actually RUN the engine, on a fake video.

WHY THIS EXISTS
    process_video() is 1,000 lines that had never been executed by any test.
    Every other suite here checks pure helpers, so the only thing that ever
    exercised the frame loop was a 30-minute run on a rented GPU.

    That let a use-before-assignment ship: a capability summary was built
    before _has_head was set, ~170 lines earlier in the same function. Nothing
    caught it. AST checks cannot — the name IS bound in the function, just
    later — and Python only raises when the line executes.

        UnboundLocalError: cannot access local variable '_has_head'
        where it is not associated with a value

    It cost a download, a model load and 10 seconds; on a slower path it would
    have cost the hour. The class of bug is "code that only runs on real
    footage", and the only defence is to run it on some.

WHAT THIS DOES
    Writes a few seconds of synthetic video with moving rectangles, a zones
    file over it, and calls process_video for real — same code path as the GPU
    box, minus the GPU. It asserts almost nothing about the ANSWERS, because
    detections on noise are meaningless. It asserts the function completes and
    returns the keys the pipeline reads, which is exactly the property that
    was broken.

SKIPS, LOUDLY, when torch/ultralytics/weights are unavailable — a laptop
without them should not fail the suite, but must not silently "pass" either.

Run: python tests/test_process_video_smoke.py
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAILED = []
SKIPPED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


def make_video(path, seconds=3, fps=30, w=640, h=360):
    """Moving rectangles on a textured background. Not people — the point is
    to drive the loop, not to detect anything."""
    import cv2
    import numpy as np
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        return False
    rng = np.random.RandomState(0)
    bg = rng.randint(40, 90, (h, w, 3), dtype=np.uint8)
    for i in range(int(seconds * fps)):
        fr = bg.copy()
        for k, speed in enumerate((3, -2)):
            x = int((80 + i * speed) % (w - 60))
            y = h // 2 - 40 + k * 20
            cv2.rectangle(fr, (x, y), (x + 40, y + 90), (200, 200, 200), -1)
        vw.write(fr)
    vw.release()
    return path.exists() and path.stat().st_size > 0


def make_zones(path, w=640, h=360):
    import json
    path.write_text(json.dumps({
        "frame_size": [w, h],
        "entry_line": [[int(w * 0.5), h], [int(w * 0.5), int(h * 0.4)]],
        "polygons": {
            "main_entrance": [[int(w*.55), int(h*.3)], [w, int(h*.3)], [w, h], [int(w*.55), h]],
            "reception":     [[0, int(h*.3)], [int(w*.45), int(h*.3)], [int(w*.45), h], [0, h]],
            "waiting_area":  [[int(w*.2), 0], [int(w*.8), 0], [int(w*.8), int(h*.28)], [int(w*.2), int(h*.28)]],
        },
        "roles": {"reception": ["staff"], "waiting_area": ["wait"],
                  "main_entrance": ["entry"]},
    }), encoding="utf-8")


def main():
    print(__doc__.strip().splitlines()[0])
    try:
        import cv2  # noqa: F401
        import torch  # noqa: F401
        from ultralytics import YOLO  # noqa: F401
    except Exception as e:
        print(f"\n  SKIPPED — engine dependencies unavailable: "
              f"{type(e).__name__}: {e}")
        SKIPPED.append("deps")
        return 0
    weights = ROOT / "models" / "best.pt"
    if not weights.exists():
        print(f"\n  SKIPPED — no detector weights at {weights}")
        SKIPPED.append("weights")
        return 0

    from kevacv.pipeline import bind_runtime

    # ignore_cleanup_errors: on Windows the decoder still holds the mp4 when
    # the block exits, and a teardown PermissionError would mask a green run.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        tmp = Path(td)
        vid, zones = tmp / "clip.mp4", tmp / "zones.json"
        if not make_video(vid):
            print("\n  SKIPPED — no mp4 writer available")
            SKIPPED.append("writer")
            return 0
        make_zones(zones)

        bind_runtime(base=ROOT, output_dir=tmp / "out", device="cpu",
                     detector=str(weights), input_root=tmp,
                     staff_gallery=str(ROOT / "staff_gallery"))
        from kevacv import engine as E
        E.RENDER_VIDEO = False          # the writer is not what this tests
        E.EVAL_EXPORT = False
        E.ENABLE_DATASET_EXPORT = False
        # LEAN MODE. boxmot/CLIP weights and InsightFace are a GPU-box
        # concern; requiring them here would mean this test only ever runs on
        # the machine it is least needed on. ByteTrack needs neither, and the
        # frame loop, the filter chain, the funnel, zones, crossings, the
        # staff override and the capability ledger all still execute — which
        # is where the bug this file exists for actually lived.
        E.TRACKER_MODE = "bytetrack"
        E.ENABLE_REID_STITCH = False
        E.ENABLE_LIVE_IDENTITY_MEMORY = False
        E.ENABLE_FACE_CORROBORATION = False
        E.ENABLE_LIVE_SEPARABILITY = False
        # Production settings make this test unusable: imgsz 1280 at batch 24
        # on CPU takes many minutes for three seconds of video, and a test
        # nobody can afford to run is a test that does not run. The code path
        # is identical at 320/4 — only the arithmetic is smaller.
        E.YOLO_IMGSZ = 320
        E.DET_BATCH = 4
        E.DETECTOR_HALF = False         # fp16 is a GPU feature; CPU rejects it

        print("\nprocess_video runs to completion")
        try:
            run = E.process_video(camera_id="TEST", video_path=str(vid),
                                  zones_path=str(zones), device="cpu")
        except Exception as e:
            check(False, "process_video completed",
                  f"{type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return 1
        check(True, "process_video completed without raising")

        print("\nit returns what the pipeline reads")
        for key in ("camera_id", "events", "roles", "crossings", "zone_roles",
                    "detection_funnel", "capabilities"):
            check(key in run, f"run['{key}'] present")

        print("\nthe new subsystems reported themselves")
        caps = run.get("capabilities") or {}
        check(bool(caps.get("rows")), "capability ledger populated",
              f"{len(caps.get('rows') or [])} row(s)")
        funnel = run.get("detection_funnel") or {}
        check("stages" in funnel, "detection funnel populated",
              f"{len(funnel.get('stages') or [])} stage(s)")
        names = [r["name"] for r in caps.get("rows", [])]
        for want in ("detector", "tracker", "zones"):
            check(want in names, f"capability row '{want}'")

        # ── the TILED path, executed ────────────────────────────────────────
        # _detect_tiled replaces the detector call rather than filtering its
        # output, so it is a code path the default run never touches. Two
        # use-before-assignment bugs already shipped in this function from
        # exactly that — code that only executes under a flag nobody set.
        print("\ntiled detection runs (SAHI path)")
        E.ENABLE_TILED_DETECT = True
        E.TILE_PX = 256          # the synthetic frame is 640x360
        try:
            run_t = E.process_video(camera_id="TEST_TILED", video_path=str(vid),
                                    zones_path=str(zones), device="cpu")
            check(True, "process_video completed with tiling on")
            f_t = run_t.get("detection_funnel") or {}
            check(bool(f_t.get("stages")), "funnel populated on the tiled path")
            rows = {r["name"]: r for r in (run_t.get("capabilities") or {}).get("rows", [])}
            check("tiled detect (SAHI)" in rows,
                  "capabilities reports tiling as active")
            check(rows.get("tiled detect (SAHI)", {}).get("status") == "OK",
                  "and marks it OK rather than degraded",
                  str(rows.get("tiled detect (SAHI)", {}).get("status")))
        except Exception as e:
            check(False, "process_video completed with tiling on",
                  f"{type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            E.ENABLE_TILED_DETECT = False

    print()
    if FAILED:
        print(f"FAILED ({len(FAILED)}): " + ", ".join(FAILED))
        return 1
    print("all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
