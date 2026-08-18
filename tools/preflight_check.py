"""preflight_check.py — verify every precondition BEFORE spending 30 minutes.

WHY THIS EXISTS
    Three runs have now been spent discovering, at the end, something that was
    knowable at the start: a stale zone map, a full disk, a CPU-only face
    model, a build that never reached the box. Each cost half an hour and
    produced numbers nobody could use.

    This asks every question that has already gone wrong once, in about two
    seconds, and prints PASS or FAIL for each. Nothing here is a claim by
    anybody -- it reads the actual files and the actual environment.

    It does NOT run the pipeline and it changes nothing.

Run:  python tools/preflight_check.py
      python tools/preflight_check.py --out output/run2
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAILS, WARNS = [], []


def ok(label, detail=""):
    print(f"  [ PASS ] {label}" + (f"   {detail}" if detail else ""))


def bad(label, detail="", fix=""):
    print(f"  [ FAIL ] {label}" + (f"   {detail}" if detail else ""))
    if fix:
        print(f"           fix: {fix}")
    FAILS.append(label)


def warn(label, detail="", fix=""):
    print(f"  [ WARN ] {label}" + (f"   {detail}" if detail else ""))
    if fix:
        print(f"           {fix}")
    WARNS.append(label)


def section(t):
    print(f"\n{'=' * 74}\n  {t}\n{'=' * 74}")


def check_build():
    section("1. WHICH CODE IS THIS")
    sys.path.insert(0, str(ROOT / "kevacv"))
    import build_id
    bid = build_id.compute(ROOT / "kevacv")
    ok("build id computed", bid)
    print(f"           the annotated video's HUD must show this exact string.")
    print(f"           if it does not, the run used different code.")
    return bid


def check_zones(zones_path):
    section("2. ZONES — the thing that made IN/OUT read 0/0")
    p = Path(zones_path)
    if not p.exists():
        bad("zone file exists", str(p), "check the path")
        return
    cfg = json.loads(p.read_text(encoding="utf-8"))
    fw, fh = cfg.get("frame_size", (0, 0))
    if not fw:
        bad("frame_size present", "absent",
            "without it every polygon is scaled by a guess")
        return
    ok("frame_size", f"{fw}x{fh}")
    s = 1280.0 / fw

    roles = cfg.get("roles") or {}
    polys = cfg.get("polygons") or {}
    entry = [z for z, r in roles.items() if "entry" in (r or [])]
    if not entry:
        bad("a zone has the ENTRY role", "none",
            "arrivals cannot be counted by any method")
    else:
        ok("entry zone", ", ".join(entry))

    line = cfg.get("entry_line")
    if not line or len(line) != 2:
        bad("entry_line present", str(line))
    else:
        (x1, y1), (x2, y2) = line
        L = math.hypot(x2 - x1, y2 - y1) * s
        frac = L / 1280.0
        if frac < 0.30:
            bad("entry line is long enough",
                f"{L:.0f}px = {frac*100:.0f}% of frame width",
                "people walk around a short line; span the doorway wall to wall")
        else:
            ok("entry line length", f"{L:.0f}px = {frac*100:.0f}% of frame width")
        # both endpoints should sit inside the entry polygon
        for z in entry:
            poly = [(px * s, py * s) for px, py in polys.get(z, [])]
            if len(poly) < 3:
                continue
            inside = sum(_in_poly(poly, (x1 * s, y1 * s)) +
                         _in_poly(poly, (x2 * s, y2 * s)) for _ in [0])
            if inside == 2:
                ok(f"line endpoints inside '{z}'")
            else:
                warn(f"line endpoints inside '{z}'", f"{inside} of 2",
                     "a line outside its own zone is how it fires zero times")

    masks = [z for z, r in roles.items() if "mask" in (r or [])]
    if masks:
        ok("dead-area masks", ", ".join(masks))
    else:
        warn("dead-area masks", "none",
             "the plant and mirror will keep producing person detections")

    if "ground_points" in cfg:
        n = len(cfg["ground_points"])
        (ok if n >= 4 else bad)("ground_points", f"{n} point(s)")
    else:
        warn("ground_points", "absent (template only)",
             "the size filter loosens itself to 5x when the auto fit is bad — "
             "this is why giant boxes survive on empty wall")


def _in_poly(poly, pt):
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1
            if x < xin:
                inside = not inside
    return inside


def check_gallery():
    section("3. STAFF GALLERY")
    from kevacv.pipeline import staff_gallery_findings
    for lvl, msg in staff_gallery_findings(ROOT / "staff_gallery"):
        (ok if lvl == "INFO" else bad)("staff photos", msg[:90])


def check_env():
    section("4. ENVIRONMENT")
    import importlib.util as u
    for m in ("numpy", "cv2", "torch", "supervision", "ultralytics",
              "boxmot", "insightface", "onnxruntime"):
        (ok if u.find_spec(m) else bad)(m)
    try:
        import torch
        if torch.cuda.is_available():
            ok("CUDA", torch.cuda.get_device_name(0))
        else:
            bad("CUDA", "not available", "a full hour on CPU is not viable")
    except Exception as e:
        bad("CUDA", str(e))
    try:
        import onnxruntime as ort
        provs = ort.get_available_providers()
        if any("CUDA" in p for p in provs):
            ok("face model on GPU", "CUDAExecutionProvider present")
        else:
            warn("face model on GPU", f"CPU only: {provs}",
                 "pip install --force-reinstall onnxruntime-gpu  "
                 "(face path is ~20x slower without it)")
    except Exception as e:
        bad("onnxruntime", str(e),
            "pip install --force-reinstall onnxruntime-gpu")
    weights = ROOT / "models" / "best.pt"
    if weights.exists():
        try:
            from ultralytics import YOLO
            names = YOLO(str(weights)).names
            ok("detector weights", f"{weights.name}  classes={names}")
            if 1 not in names:
                warn("head class", "absent",
                     "occlusion recovery and merged-box splitting need it")
        except Exception as e:
            warn("detector weights", f"present but not loadable: {e}")
    else:
        bad("detector weights", f"missing {weights}")


def check_disk(out_dir):
    section("5. DISK — a full volume killed a 28-minute run at 95%")
    from kevacv import config as C
    from kevacv.capabilities import disk_findings
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    for lvl, msg in disk_findings(
            target, 3600, C.FPS_TARGET, want_render=C.RENDER_VIDEO,
            want_eval=False, want_dataset=False,
            frame_w=C.ANALYSIS_MAX_W):
        {"INFO": ok, "WARN": warn}.get(lvl, bad)("space", msg[:100])
    if target.exists() and any(target.iterdir()):
        warn("output dir is not empty", str(target),
             f"an old video can survive a failed run and look like a result — "
             f"rm -rf {target}")
    else:
        ok("output dir is clean", str(target))


def check_settings():
    section("6. THE SETTINGS THIS RUN WILL USE")
    from kevacv import config as C
    for n in ("FPS_TARGET", "ANALYSIS_MAX_W", "YOLO_IMGSZ", "REID_RATIO",
              "LIVE_REID_ABS_FLOOR", "ENABLE_HEAD_RECOVERY",
              "ENABLE_MERGED_SPLIT", "ENABLE_LIVE_PHANTOM_SUPPRESS",
              "ENABLE_TILED_DETECT", "RENDER_VIDEO"):
        print(f"           {n:<30} {getattr(C, n, '### MISSING ###')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zones", default=str(ROOT / "zones" / "CAM.112_zone.json"))
    ap.add_argument("--out", default=str(ROOT / "output" / "run2"))
    a = ap.parse_args()

    print("=" * 74)
    print("  PREFLIGHT — nothing here is a claim, it reads the actual files")
    print("=" * 74)
    bid = check_build()
    check_zones(a.zones)
    check_gallery()
    check_env()
    check_disk(a.out)
    check_settings()

    print("\n" + "=" * 74)
    if FAILS:
        print(f"  {len(FAILS)} FAILURE(S) — do not run yet:")
        for f in FAILS:
            print(f"    - {f}")
        print("=" * 74)
        return 1
    print(f"  READY.  {len(WARNS)} warning(s).")
    print(f"  The video HUD must show build {bid}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
