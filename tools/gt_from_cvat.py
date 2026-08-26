#!/usr/bin/env python3
"""gt_from_cvat.py — pull labelled tracks out of CVAT and write MOT ground truth.

WHY THIS EXISTS
    Every accuracy number this project has ever produced was self-graded.
    `HOTA NOT MEASURED` printed on every run because gt.txt was a static file
    tied to no particular chunk, and on 2026-08-25 it was found to be scoring
    one camera against another's footage ("HOTA 0.0057" on unrelated videos).

    This is the last link in the chain that fixes that:

        CVAT (label / correct)  ->  gt_<CAMERA>.txt  ->  eval_harness  ->  HOTA

WHY IT WRITES THE FILE THE PIPELINE ALREADY LOOKS FOR
    pipeline.py searches, in order, for a CAMERA-SCOPED ground truth:
        <out>/gt.txt, <out>/gt_<cam>.txt,
        <zones dir>/gt_<cam>.txt, ./gt_<cam>.txt
    and REFUSES a bare gt.txt that is not scoped to this camera. So writing
    gt_CAM_112.txt means the next run scores itself with no extra flag.

THE FRAME-ALIGNMENT TRAP, STATED
    MOT ground truth is frame-INDEXED, so it is only valid for a run whose
    frame sampling matches the labelled clip. The clip for this task was cut at
    1920x1080 @ 15 fps precisely because that is what the pipeline analyses at
    (`analysis_max_w: 1920`, `fps: 15`). Label a 30 fps clip, or a 4K one, and
    the numbers will be confidently wrong. `tools/score_by_time.py` exists for
    when they cannot be made to match.

Usage:
    python tools/gt_from_cvat.py --task 1 --camera CAM.112
    python tools/gt_from_cvat.py --task 1 --camera CAM.112 --out eval/
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def api(base, path, token=None, data=None, method=None):
    url = base.rstrip("/") + path
    body = json.dumps(data).encode() if data is not None else None
    r = urllib.request.Request(url, data=body,
                               method=method or ("POST" if body else "GET"))
    r.add_header("Content-Type", "application/json")
    if token:
        r.add_header("Authorization", "Token " + token)
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            txt = resp.read().decode()
            return json.loads(txt) if txt.strip()[:1] in "{[" else txt
    except urllib.error.HTTPError as e:
        raise SystemExit(f"CVAT {path} -> HTTP {e.code}: {e.read().decode()[:300]}")


def aabb(points, rotation):
    """CVAT rectangle -> axis-aligned x,y,w,h.

    WHY THIS IS NOT JUST points[:4]
        CVAT rectangles carry a separate `rotation` in degrees about the box
        CENTRE, and the stored points stay UNROTATED. Reading points[:4] and
        ignoring rotation therefore emits a box in the wrong orientation with
        no warning -- and MOT 1.1 has no rotation field to put it in anyway.

        A tilted box is easier for a human to draw around a diagonal person,
        but the detector can only ever emit axis-aligned boxes, so a rotated
        ground truth can never be matched against a prediction. The enclosing
        axis-aligned box is the only comparable thing, so that is what we
        write: the labeller's intent is preserved, the metric stays valid.
    """
    x1, y1, x2, y2 = (float(v) for v in points[:4])
    if not rotation or abs(float(rotation)) < 0.5:
        return x1, y1, x2 - x1, y2 - y1
    import math
    a = math.radians(float(rotation))
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    ca, sa = math.cos(a), math.sin(a)
    xs, ys = [], []
    for px, py in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
        dx, dy = px - cx, py - cy
        xs.append(cx + dx * ca - dy * sa)
        ys.append(cy + dx * sa + dy * ca)
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--camera", required=True, help="e.g. CAM.112")
    p.add_argument("--base", default="http://localhost:8080/api")
    p.add_argument("--user", default="prabh")
    p.add_argument("--password", default=os.environ.get("CVAT_PASSWORD", ""))
    p.add_argument("--out", default=".", help="directory for gt_<camera>.txt")
    a = p.parse_args(argv)
    # PREFER THE STORED TOKEN.
    #
    # These tools used to demand CVAT_PASSWORD and SystemExit without it. Piped
    # through grep in a shell one-liner that exit is INVISIBLE, and the previous
    # gt_*.txt stays on disk looking current -- which is how a stale roles file
    # was read as fresh on 2026-08-26. cvat_snapshot.py already mints a token to
    # ~/.cvat_token for cron; reuse it so the password is optional, not required.
    tok = None
    _tf = Path.home() / ".cvat_token"
    if _tf.exists():
        _t = _tf.read_text(encoding="utf-8").strip()
        try:
            api(a.base, "/users/self", token=_t)
            tok = _t
        except Exception:
            tok = None
    if tok is None:
        if not a.password:
            raise SystemExit(
                "no valid ~/.cvat_token and no --password/CVAT_PASSWORD.\n"
                "  mint one:  CVAT_PASSWORD=... python tools/cvat_snapshot.py")
        tok = api(a.base, "/auth/login",
                  data={"username": a.user, "password": a.password})["key"]
    ann = api(a.base, f"/tasks/{a.task}/annotations", token=tok)

    rows = []          # (frame, track_id, x, y, w, h, role)
    # TRACKS are the interesting half: a CVAT track carries one id across
    # frames, which is exactly what an identity metric needs. Loose SHAPES have
    # no id, so they can only ever support DETECTION scoring -- they are emitted
    # with id -1 and reported separately rather than silently mixed in.
    next_id = 1
    rotated = 0
    for tr in ann.get("tracks") or []:
        tid = next_id
        next_id += 1
        # ROLE LIVES ON THE SHAPES, NOT THE TRACK.
        #
        # `role` was declared mutable, so CVAT stores a value per SHAPE and
        # leaves track.attributes empty. Reading the track (as this did) means
        # every role exports as "unknown" -- silently, with no error, which is
        # the worst possible failure for the one piece of truth in this file
        # that nothing else in the project has. Caught 2026-08-26 when a task
        # with 775 shapes marked `staff` reported 0 staff.
        #
        # A person does not change role mid-clip, so the track's role is the
        # majority of its shapes, ignoring "unknown" -- a labeller who set the
        # role on some frames and not others meant it for the whole track.
        from collections import Counter as _C
        _votes = _C()
        for _sh in tr.get("shapes") or []:
            for _at in _sh.get("attributes") or []:
                _v = _at.get("value")
                if _v in ("staff", "guest", "unknown"):
                    _votes[_v] += 1
        _named = [(v, n) for v, n in _votes.items() if v != "unknown"]
        role = max(_named, key=lambda kv: kv[1])[0] if _named else "unknown"
        for sh in tr.get("shapes") or []:
            if sh.get("outside"):
                continue
            pts = sh.get("points") or []
            if len(pts) < 4:
                continue
            x, y, w, h = aabb(pts, sh.get("rotation"))
            rotated += 1 if abs(float(sh.get("rotation") or 0)) >= 0.5 else 0
            rows.append((int(sh["frame"]) + 1, tid, x, y, w, h, role))

    loose = 0
    for sh in ann.get("shapes") or []:
        pts = sh.get("points") or []
        if len(pts) < 4:
            continue
        x, y, w, h = aabb(pts, sh.get("rotation"))
        rotated += 1 if abs(float(sh.get("rotation") or 0)) >= 0.5 else 0
        rows.append((int(sh["frame"]) + 1, -1, x, y, w, h, "unknown"))
        loose += 1

    if not rows:
        raise SystemExit("no annotations in that task yet — label some frames first")

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = a.camera.replace(".", "_")
    gt = outdir / f"gt_{tag}.txt"
    rows.sort()
    with gt.open("w", encoding="utf-8") as f:
        for fr, tid, x, y, w, h, _role in rows:
            # MOT 1.1: frame,id,x,y,w,h,conf,-1,-1,-1
            f.write(f"{fr},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")

    roles = outdir / f"gt_{tag}_roles.json"
    roles.write_text(json.dumps(
        {str(t): r for _, t, _, _, _, _, r in rows if t != -1}, indent=1),
        encoding="utf-8")

    n_tracks = len({t for _, t, *_ in rows if t != -1})
    print(f"  wrote {gt}  ({len(rows)} rows, {n_tracks} track(s))")
    print(f"  wrote {roles}  (the staff/guest truth, which nothing else has)")
    if rotated:
        print(f"  NOTE: {rotated} shape(s) were ROTATED in CVAT. MOT has no "
              f"rotation field and the detector only emits axis-aligned boxes, "
              f"so each was written as its ENCLOSING axis-aligned box. Prefer "
              f"un-rotated boxes: the enclosing box of a tilted rectangle is "
              f"slightly looser than one drawn axis-aligned, which costs IoU.")
    if loose:
        print(f"  NOTE: {loose} row(s) came from loose SHAPES with no track id "
              f"(written as id -1). Those can score DETECTION but not identity "
              f"-- convert them to tracks in CVAT to get HOTA's association half.")
    if n_tracks == 0:
        print("  WARNING: no CVAT *tracks* at all, so HOTA's association "
              "component is unmeasurable from this export.")
    print()
    print("  Next: drop it where the pipeline looks, then just run normally --")
    print(f"    cp {gt} eval/            # or the run's --out directory")
    print("    the run scores itself; look for the HOTA line in the log.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
