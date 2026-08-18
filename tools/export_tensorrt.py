"""export_tensorrt.py — compile best.pt into a TensorRT engine for THIS GPU.

WHAT IT BUYS, AND WHY IT IS THE LAST FREE ONE
    Same weights, same graph, same outputs. TensorRT fuses conv+bn+activation
    into single kernels, picks the fastest algorithm for this exact card and
    input size, and removes PyTorch's per-layer dispatch. Typically 1.3-2x on
    the detector.

    It is the last speedup on this pipeline that costs nothing. Everything
    after it -- lowering imgsz, sampling fewer frames, a lighter ReID backbone
    -- trades detection or identity accuracy for time, and there is currently
    no ground truth to measure what the trade cost.

THE ENGINE IS NOT PORTABLE
    A .engine is compiled for one GPU architecture, one TensorRT version and
    one input shape. It will not load on a different instance type, and a
    silent fallback to best.pt would make a "TensorRT run" indistinguishable
    from a normal one in the log. So the pipeline is pointed at the engine
    EXPLICITLY with --detector; nothing auto-detects it.

VERIFY BEFORE TRUSTING
    Export is a model rewrite. `--verify` runs both on the same frames and
    reports box-count and coordinate agreement, because "same outputs" is a
    claim about a conversion nobody watched, not a guarantee.

Usage
    python tools/export_tensorrt.py                       # export + verify
    python tools/export_tensorrt.py --imgsz 1280 --batch 24
    python tools/run_pipeline.py ... --detector models/best.engine
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def verify_real(pt_path, engine_path, imgsz, frames_dir, n=60, conf=0.35,
                max_loss_pct=1.0, small_px=120):
    """Do the two models agree ON REAL FOOTAGE, and where do they disagree?

    WHY THIS EXISTS ALONGSIDE verify()
        verify() compares the two models on RANDOM NOISE, and says so itself:
        "random noise produces few detections, so agreement here is necessary
        and not sufficient". It is a smoke test for "did the engine load and
        produce plausible output", not evidence that the pipeline still sees
        the same people.

        The thing quantisation actually costs is SMALL-OBJECT recall, and on
        this camera the small objects are the guests at the main entrance —
        measured at 45x82px in the frame log. That is already the pipeline's
        largest open defect. An engine that runs 3x faster while dropping
        those would look like a win in every timing number and make the real
        problem worse.

        So: real frames, IoU matching, and a separate line for what was lost
        among the SMALL boxes specifically.
    """
    import cv2
    import numpy as np
    from ultralytics import YOLO
    paths = sorted(Path(frames_dir).glob("*.jpg"))
    if not paths:
        print(f"  !! no jpgs in {frames_dir} — skipping real-frame validation")
        return True
    step = max(1, len(paths) // n)
    picked = paths[::step][:n]
    frames = [cv2.imread(str(p)) for p in picked]
    frames = [f for f in frames if f is not None]
    print(f"\n  verifying on {len(frames)} REAL frames from {frames_dir}")

    out, times = {}, {}
    for label, path in (("pytorch", pt_path), ("tensorrt", engine_path)):
        m = YOLO(str(path))
        t0 = time.time()
        res = [m.predict(f, imgsz=imgsz, conf=conf, verbose=False)[0]
               for f in frames]
        times[label] = time.time() - t0
        out[label] = [r.boxes.xyxy.cpu().numpy() if r.boxes is not None
                      else np.empty((0, 4)) for r in res]

    n_base = sum(len(b) for b in out["pytorch"])
    n_trt = sum(len(b) for b in out["tensorrt"])
    lost = gained = lost_small = n_small = 0
    for bb, tt in zip(out["pytorch"], out["tensorrt"]):
        used = set()
        for b in bb:
            is_small = (b[3] - b[1]) < small_px
            n_small += is_small
            best, bi = 0.0, -1
            for i, t in enumerate(tt):
                if i in used:
                    continue
                v = _iou(b, t)
                if v > best:
                    best, bi = v, i
            if best >= 0.5:
                used.add(bi)
            else:
                lost += 1
                lost_small += is_small
        gained += len(tt) - len(used)

    pct = 100.0 * lost / max(1, n_base)
    pct_small = 100.0 * lost_small / max(1, n_small)
    print(f"    pytorch  {n_base:>5} boxes  {times['pytorch']:.1f}s")
    print(f"    tensorrt {n_trt:>5} boxes  {times['tensorrt']:.1f}s  "
          f"({times['pytorch']/max(times['tensorrt'],1e-6):.2f}x)")
    print(f"    LOST {lost} ({pct:.2f}%)   gained {gained}")
    print(f"    of the {n_small} boxes shorter than {small_px}px — the "
          f"entrance guests — {lost_small} lost ({pct_small:.2f}%)")
    ok = pct <= max_loss_pct and pct_small <= max(max_loss_pct * 2, 2.0)
    if not ok:
        print(f"  REFUSED: too many detections lost. A faster run that sees "
              f"fewer people is not an improvement — this is the defect the "
              f"pipeline already has.")
    else:
        print(f"  ACCEPTED: loss within bar ({max_loss_pct}%).")
    return ok


def verify(pt_path, engine_path, imgsz, n=8):
    """Do the two models agree? -> True/False, and say how they differ.

    SMOKE TEST ONLY — random noise. See verify_real() for the check that
    decides whether the engine is safe to ship."""
    import numpy as np
    from ultralytics import YOLO
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
              for _ in range(n)]
    print("\n  verifying — same frames through both models")
    outs = {}
    for label, path in (("pytorch", pt_path), ("tensorrt", engine_path)):
        m = YOLO(str(path))
        t0 = time.time()
        res = m.predict(frames, imgsz=imgsz, conf=0.25, verbose=False)
        dt = time.time() - t0
        outs[label] = [r.boxes.xyxy.cpu().numpy() if r.boxes is not None
                       else np.empty((0, 4)) for r in res]
        print(f"    {label:<9} {sum(len(b) for b in outs[label]):>4} boxes  "
              f"{dt:.2f}s for {n} frames")

    ok = True
    for i, (a, b) in enumerate(zip(outs["pytorch"], outs["tensorrt"])):
        if len(a) != len(b):
            print(f"    frame {i}: box COUNT differs ({len(a)} vs {len(b)})")
            ok = False
            continue
        if len(a) and float(np.abs(np.sort(a, 0) - np.sort(b, 0)).max()) > 2.0:
            print(f"    frame {i}: coordinates differ by more than 2px")
            ok = False
    # random noise produces few detections, so agreement here is necessary and
    # not sufficient. The real check is a scored run against ground truth.
    print(f"  {'AGREE' if ok else 'DISAGREE'} on synthetic frames — confirm on "
          f"real footage with a --dry run before trusting a full night")
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default=str(ROOT / "models" / "best.pt"))
    p.add_argument("--imgsz", type=int, default=1280,
                   help="MUST match the run's imgsz — the engine is compiled "
                        "for one input size")
    p.add_argument("--batch", type=int, default=24,
                   help="MUST match DET_BATCH")
    p.add_argument("--fp32", action="store_true",
                   help="compile fp32. Default is fp16, matching DETECTOR_HALF")
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--frames", default=None,
                   help="directory of REAL jpgs (e.g. output/hour/eval_frames/"
                        "CAM.112) to validate against. Without this the only "
                        "check is random noise, which cannot detect the loss "
                        "that actually matters: the 45x82px guests at the "
                        "entrance. Strongly recommended.")
    p.add_argument("--max-loss-pct", type=float, default=1.0,
                   help="fail if more than this %% of baseline detections "
                        "disappear on real frames")
    a = p.parse_args(argv)

    w = Path(a.weights)
    if not w.exists():
        print(f"  MISSING weights: {w}")
        return 2
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        print("  TensorRT is not installed. On the DLAMI:")
        print("    pip install tensorrt")
        print("  (it is a large wheel and must match the CUDA runtime)")
        return 2

    from ultralytics import YOLO
    print("=" * 70)
    print(f"  EXPORT  {w.name} -> TensorRT")
    print(f"  imgsz={a.imgsz}  batch={a.batch}  "
          f"precision={'fp32' if a.fp32 else 'fp16'}")
    print("  This compiles for THIS GPU and takes several minutes.")
    print("=" * 70)
    t0 = time.time()
    out = YOLO(str(w)).export(format="engine", imgsz=a.imgsz, batch=a.batch,
                              half=not a.fp32, dynamic=True, verbose=False)
    engine = Path(out)
    print(f"\n  wrote {engine}  ({engine.stat().st_size / 1e6:.0f} MB) "
          f"in {time.time() - t0:.0f}s")

    _safe = True
    if not a.no_verify:
        verify(w, engine, a.imgsz)          # smoke test: did it load at all
        if a.frames:
            _safe = verify_real(w, engine, a.imgsz, a.frames,
                                max_loss_pct=a.max_loss_pct)
        else:
            print("\n  !! NO --frames given. The engine was checked against "
                  "random noise only,")
            print("     which cannot see the loss that matters here. Re-run "
                  "with e.g.")
            print("       --frames output/hour/eval_frames/CAM.112")

    if not _safe:
        print()
        print("  NOT RECOMMENDED FOR USE. The engine exists but loses real "
              "detections.")
        return 1

    print()
    print("  Use it — nothing auto-detects it, on purpose:")
    print(f"    python tools/run_pipeline.py ... --detector {engine}")
    print()
    print("  If you move to a different instance type, re-export. An engine")
    print("  built for one GPU will not load on another.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
