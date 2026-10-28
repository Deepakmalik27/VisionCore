#!/usr/bin/env python3
"""cvat_to_calibration.py — draw the venue in CVAT, get a working config out.

WHY THIS EXISTS
    Every geometric input this pipeline needs was authored by clicking on a
    still frame in tools/zone_mapper_v2.html, and every one of them has been
    wrong at least once:

      entry line    drawn ~1050 px too short. Fired ZERO times in 20 minutes
                    while dining/staff doors fired 55 and 71. Eight GM-facing
                    numbers silently went to zero.
      ground plane  still reports horizon row -303, which is geometrically
                    impossible for a downward-looking camera.
      dead areas    the plant mask removes 13.8% of ALL detections -- the
                    single largest filter loss in the funnel.
      mirrors       55 track pairs move in lockstep and are flagged as possible
                    reflections every run, and suppressed never, because
                    deciding WHICH half is the reflection needs geometry the
                    pipeline does not have.

    The common cause is that all of it is authored blind, on one still, and
    only checked much later against a run. CVAT lets you draw the SAME shapes
    while scrubbing the actual footage -- you see people walk around a short
    entry line immediately instead of discovering it in a scorecard.

WHAT IT READS  (CVAT label name -> what it becomes)
    tile_corner   POINTS    -> ground_points, for tools/calibrate_plane.py.
                              Needs >=5 ADJACENT floor-tile corners, no skips
                              and no duplicates: four points fit ANY labelling
                              with zero error, which is why calibrate_plane's
                              enough_evidence() refuses them. Delilah's
                              checkerboard marble floor is close to ideal.
    entry_line    POLYLINE  -> entry_lines["entry line"]
    door:<name>   POLYLINE  -> entry_lines[<name>]
    zone:<name>   POLYGON   -> polygons[<name>]
    mirror        POLYGON   -> mirror_regions[]  (reflection suppression)
    dead_area     POLYGON   -> polygons["<name> mask"]

COORDINATES
    Everything is written in SOURCE pixels, because that is what zones/*.json
    holds. Pass --frame-size to say what the labelled media was, and this
    scales. Getting that wrong is a landmine that has already fired once in
    this project: a 1080p entry line applied to a 4K source put the counting
    line across the reception desk.

IT NEVER OVERWRITES BY DEFAULT
    Writes <zones>.PROPOSED.json and prints a diff. Moving a counting line
    changes every arrival number in the report, so that is an operator
    decision, not a script's.

Usage:
    python tools/cvat_to_calibration.py --task 2 --zones zones/CAM.112_zone.json \\
        --frame-size 1920 1080 --tile-cm 30
    python tools/cvat_to_calibration.py --task 2 --zones ... --apply
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def api(base, path, token=None, data=None):
    url = base.rstrip("/") + path
    body = json.dumps(data).encode() if data is not None else None
    r = urllib.request.Request(url, data=body,
                               method="POST" if body else "GET")
    r.add_header("Content-Type", "application/json")
    if token:
        r.add_header("Authorization", "Token " + token)
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            txt = resp.read().decode()
            return json.loads(txt) if txt.strip()[:1] in "{[" else txt
    except urllib.error.HTTPError as e:
        raise SystemExit(f"CVAT {path} -> HTTP {e.code}: {e.read().decode()[:300]}")


def pairs(points):
    """CVAT flat [x,y,x,y,...] -> [[x,y], ...]"""
    return [[float(points[i]), float(points[i + 1])]
            for i in range(0, len(points) - 1, 2)]


def scale(pts, src_wh, dst_wh):
    if not src_wh or not dst_wh or tuple(src_wh) == tuple(dst_wh):
        return pts, 1.0
    sx, sy = dst_wh[0] / src_wh[0], dst_wh[1] / src_wh[1]
    return [[round(x * sx), round(y * sy)] for x, y in pts], sx


def lattice_world(points, tile_cm, k=None, frame_size=None):
    """Bare tile corners -> the [{"image":..,"world":..}] GroundPlane wants.

    WHY THIS IS NOT A PASS-THROUGH
        GroundPlane.from_zone_config expects a LIST of correspondences with
        world coordinates in METRES:

            "ground_points": [{"image": [x, y], "world": [X, Z]}, ...]

        This tool used to write {"tile_cm": .., "points": [[x, y], ..]}, which
        from_zone_config reads, raises KeyError on, CATCHES, and silently falls
        back to the automatic fit. So a venue that measured its floor got the
        guess anyway and nothing said so -- exactly the failure the engine's own
        comment warns about ("the one failure mode that makes an operator stop
        measuring"). Caught 2026-08-26: 57 hand-clicked corners had no effect.

    HOW THE WORLD COORDS ARE RECOVERED
        A clicked corner does not say WHICH tile it is, and the caller cannot
        reasonably be asked to type a grid index 57 times. But the corners of a
        flat tiled floor are a projective image of an integer lattice, so the
        grid is recoverable: guess that four mutually-near corners are one tile,
        solve the homography sending them to the unit square, and count how many
        DISTINCT lattice cells the rest land on. The true grid wins outright.

        Scored on distinct cells, not inliers: the degenerate homography that
        squashes all 57 points into a 2x2 block scores a perfect 57/57 on
        inliers and means nothing.

    THE LENS
        A homography assumes a pinhole camera. With fisheye k set, the outer
        corners bend off the lattice -- measured here as 21/57 raw vs 36/57
        rectified. Points are undistorted first when k is known.
    """
    import itertools
    import numpy as np

    P = np.asarray(points, dtype=float)
    if len(P) < 5:
        return None, "fewer than 5 corners"
    Q = P
    if k and frame_size:
        try:
            import sys as _s
            _s.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from kevacv.fisheye import to_rectified
            Q = np.array([to_rectified(x, y, float(k), tuple(frame_size))
                          for x, y in P], dtype=float)
        except Exception:
            Q = P

    def homog(src, dst):
        A = []
        for (x, y), (u, v) in zip(src, dst):
            A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
            A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
        _, _, V = np.linalg.svd(np.array(A))
        H = V[-1].reshape(3, 3)
        return H / H[2, 2]

    def apply(H, pts):
        q = np.c_[pts, np.ones(len(pts))] @ H.T
        return q[:, :2] / q[:, 2:3]

    UNIT = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    best = (0, None, None)
    n = len(Q)
    for i in range(n):
        d = np.linalg.norm(Q - Q[i], axis=1)
        near = np.argsort(d)[1:min(10, n)]
        for combo in itertools.combinations(near, 3):
            quad = np.array([Q[i], *(Q[c] for c in combo)])
            c = quad.mean(0)
            quad = quad[np.argsort(np.arctan2(quad[:, 1] - c[1], quad[:, 0] - c[0]))]
            try:
                H = homog(quad, UNIT)
            except Exception:
                continue
            G = apply(H, Q)
            if not np.isfinite(G).all() or np.abs(G).max() > 40:
                continue
            ok = np.abs(G - np.round(G)).max(1) < 0.25
            cells = {tuple(v) for v in np.round(G[ok]).astype(int)}
            if len(cells) > best[0]:
                best = (len(cells), H, ok)
    ncell, H, ok = best
    if H is None or ncell < 5:
        return None, f"no consistent lattice ({ncell} cells)"
    G = np.round(apply(H, Q)).astype(int)
    T = float(tile_cm) / 100.0                      # cm -> METRES, what world wants
    out = [{"image": [float(px), float(py)],
            "world": [float(gx * T), float(gy * T)]}
           for (px, py), (gx, gy), good in zip(P, G, ok) if good]
    return out, f"{len(out)}/{len(P)} corners on {ncell} distinct lattice cells"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--zones", required=True)
    p.add_argument("--base", default="http://localhost:8080/api")
    p.add_argument("--user", default="prabh")
    p.add_argument("--password", default=os.environ.get("CVAT_PASSWORD", ""))
    p.add_argument("--frame-size", nargs=2, type=int, metavar=("W", "H"),
                   help="resolution of the media you labelled in CVAT")
    p.add_argument("--tile-cm", type=float,
                   help="floor tile edge in cm (required for tile_corner points)")
    p.add_argument("--fisheye-k", type=float, default=None,
                   help="lens k (e.g. -0.083); undistorts before the grid fit")
    p.add_argument("--apply", action="store_true")
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
    task = api(a.base, f"/tasks/{a.task}", token=tok)
    ann = api(a.base, f"/tasks/{a.task}/annotations", token=tok)
    # ALWAYS the /labels endpoint.
    #
    # task["labels"] is not a list on CVAT >= 2.x -- it is a paginated stub, so
    # iterating it yields strings and `l["id"]` raises TypeError. The dedicated
    # endpoint is the only shape-stable source. (Same trap already hit once in
    # gt_from_cvat.py on 2026-08-26.)
    names = {}
    for l in api(a.base, f"/labels?task_id={a.task}", token=tok).get("results", []):
        names[l["id"]] = l["name"]
    if not names and isinstance(task.get("labels"), list):
        names = {l["id"]: l["name"] for l in task["labels"] if isinstance(l, dict)}

    zpath = Path(a.zones)
    cfg = json.loads(zpath.read_text(encoding="utf-8"))
    dst = tuple(cfg.get("frame_size") or ())
    src = tuple(a.frame_size) if a.frame_size else dst
    print(f"  labelled at {src[0]}x{src[1]}  ->  zones file is {dst[0]}x{dst[1]}"
          if src and dst else "  (frame sizes unknown — no scaling applied)")

    shapes = list(ann.get("shapes") or [])
    for tr in ann.get("tracks") or []:
        for sh in tr.get("shapes") or []:
            if not sh.get("outside"):
                shapes.append({**sh, "label_id": tr["label_id"]})
    if not shapes:
        raise SystemExit("no shapes in that task — draw something first")

    tiles, lines, zones, mirrors, deads = [], {}, {}, [], {}
    bad_lines = []
    for sh in shapes:
        nm = names.get(sh.get("label_id"), "")
        pts, _ = scale(pairs(sh.get("points") or []), src, dst)
        if not pts:
            continue
        t = sh.get("type")
        if nm == "tile_corner":
            tiles.extend(pts)
        elif nm in ("entry_line",) or nm.startswith("door:"):
            if t != "polyline":
                print(f"  !! {nm} must be a POLYLINE, got {t} — skipped")
                continue
            # A COUNTING LINE IS EXACTLY TWO POINTS.
            #
            # supervision's LineZone is a single segment: it decides in/out by
            # which SIDE of that segment a track's anchor moved across. There is
            # no multi-segment form. This used to silently do pts[:2], so a
            # 9-point polyline swept across the room became the tiny first
            # segment of it -- a counting line in the wrong place, with no
            # warning, feeding eight GM-facing numbers. Refusing is the only
            # safe behaviour: a wrong line reads exactly like a quiet room.
            if len(pts) != 2:
                bad_lines.append((nm, len(pts)))
                continue
            key = "entry line" if nm == "entry_line" else nm.split(":", 1)[1]
            lines[key] = pts
        elif nm.startswith("zone:") and t in ("polygon", "mask"):
            zones[nm.split(":", 1)[1]] = pts
        elif nm == "mirror" and t == "polygon":
            mirrors.append(pts)
        elif nm == "dead_area" and t == "polygon":
            deads[f"dead area {len(deads) + 1} mask"] = pts

    if bad_lines:
        print()
        for nm, n in bad_lines:
            print(f"  !! REFUSED {nm}: {n} points. A counting line is a segment,")
            print(f"     so it must be EXACTLY 2 points -- one each side of the")
            print(f"     opening people walk through. Redraw it with two clicks.")
    print("\nFOUND")
    print(f"  tile_corner points : {len(tiles)}")
    print(f"  entry/door lines   : {sorted(lines) or 'none'}")
    print(f"  zones              : {sorted(zones) or 'none'}")
    print(f"  mirror regions     : {len(mirrors)}")
    print(f"  dead areas         : {len(deads)}")

    print("\nDIFF")
    old_lines = cfg.get("entry_lines") or {}
    for k, v in lines.items():
        was = old_lines.get(k)
        print(f"  {'~' if was else '+'} line  {k:<16} {was} -> {v}")
    old_poly = cfg.get("polygons") or {}
    for k, v in zones.items():
        n = len(old_poly.get(k) or [])
        print(f"  {'~' if k in old_poly else '+'} zone  {k:<16} {n} pts -> {len(v)} pts")
    for k, v in deads.items():
        print(f"  + dead  {k:<16} {len(v)} pts")
    if mirrors:
        print(f"  + mirror_regions      {len(mirrors)} polygon(s)  "
              f"(55 lockstep pairs were flagged and never suppressed)")

    if tiles:
        if not a.tile_cm:
            print("\n  !! tile_corner points found but no --tile-cm. The plane needs a "
                  "REAL-WORLD edge length or it can fit anything; refusing to write "
                  "ground_points without it.")
        elif len(tiles) < 5:
            print(f"\n  !! only {len(tiles)} tile corner(s). calibrate_plane requires "
                  f">=5 ADJACENT corners: four points fit ANY labelling with zero "
                  f"error, so four is not evidence.")
        else:
            gp, why = lattice_world(tiles, a.tile_cm, k=a.fisheye_k,
                                    frame_size=dst or src)
            if gp is None:
                print(f"\n  !! could not recover the tile grid: {why}. "
                      f"ground_points NOT written -- a format the engine cannot "
                      f"read is worse than none, because it fails silently.")
            else:
                cfg["ground_points"] = gp
                cfg.pop("_ground_points_TEMPLATE", None)
                print(f"\n  ground_points: {why}, tile {a.tile_cm} cm")
                print(f"  written as [{{'image':..,'world':..}}] in METRES, which "
                      f"is what GroundPlane.from_zone_config actually reads.")

    if lines:
        cfg.setdefault("entry_lines", {}).update(lines)
    if zones:
        cfg.setdefault("polygons", {}).update(zones)
    if deads:
        cfg.setdefault("polygons", {}).update(deads)
    if mirrors:
        cfg["mirror_regions"] = mirrors
    cfg["_calibration_note"] = (
        f"lines/zones/points imported from CVAT task {a.task} by "
        f"tools/cvat_to_calibration.py, drawn against the real footage rather "
        f"than a single still.")

    out = zpath if a.apply else zpath.with_suffix(".PROPOSED.json")
    out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    if not a.apply:
        print("  read the diff above, then re-run with --apply.")
        print("  moving a counting line changes every arrival number in the report.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
