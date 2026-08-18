"""calibrate_plane.py — turn four floor tiles into a real ground plane.

WHY THIS EXISTS
    zones/CAM.112_zone.json has carried "_ground_points_TEMPLATE" — four
    [0,0] placeholders — since it was written, so every run falls back to the
    automatic perspective fit. That fit guessed camera heights of 1.12m, then
    1.25, 1.48, 2.87, 3.08, 2.72, 3.26 and finally 2.42m IN A SINGLE HOUR, all
    from the same room.

    It is not a cosmetic number. The implausible-size filter predicts how tall
    a person at a given footline can be. With the plane wrong the filter
    over-fired, the D1 guard measured a 12% drop rate and RELAXED its own
    tolerance 2.5x -> 5.0x for the rest of the video — which is why the giant
    box on the right-hand wall got WORSE, not better. Every metre gate in
    identity matching (walk speed, hand-off distance) rides on the same plane.

    GroundPlane.from_correspondences has always been able to do this exactly.
    Nobody could supply the four numbers, because there was no way to know
    whether the four you supplied were RIGHT.

WHAT MAKES THIS SAFE
    Four points always produce a homography. Four BAD points produce a
    confident, wrong one — the same class of failure as the entry line that
    fired zero times. So this refuses to write unless the plane reproduces
    distances it was not fitted to explain:

      * every pair of your points is measured through the plane and compared
        against the distance the tile grid says it must be
      * --check takes EXTRA points that are held out of the fit entirely

    A guess that cannot be corrected is worse than no guess.

HOW TO USE IT — tile size, and FIVE OR MORE tile corners
    The checkered floor is the ruler. Pick corners you can point at in the
    image, and say which tile corner each one is as (col,row) on the grid.
    They must NOT be in a line, and the wider they spread, the better.

    FIVE, not four. Four correspondences fit a homography EXACTLY — the error
    is 0.00m whether your points are right or wrong, so four points cannot be
    checked at all. See enough_evidence().

    python tools/calibrate_plane.py --tile-cm 30 \\
        --pt 812,1490@0,0  --pt 1180,1470@3,0 \\
        --pt 900,1900@0,3  --pt 1330,1860@3,3 \\
        --check 1050,1680@2,2 \\
        --frame output/zone_preview_v2.png

    Add --write once it says PASS. A --check point is held out of the fit
    entirely and is the strongest test available.

    Image coordinates are in the ZONE FILE's space (3840x2160 for CAM.112) —
    the same space every polygon uses. Use --image-size if you read the pixels
    off a 1920x1080 render instead; they are scaled for you.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kevacv.ground_plane import GroundPlane  # noqa: E402

# A plane good enough to gate metres must reproduce a distance it was fitted
# on to well inside a floor tile; a held-out point is allowed a little more.
FIT_TOL_M = 0.15
HOLDOUT_TOL_M = 0.30


def parse_point(s):
    """'812,1490@0,3' -> ((812.0, 1490.0), (0.0, 3.0))  image px @ tile col,row"""
    try:
        img, tile = s.split("@")
        x, y = (float(v) for v in img.split(","))
        c, r = (float(v) for v in tile.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{s!r}: expected X,Y@col,row  (e.g. 812,1490@0,3)")
    return (x, y), (c, r)


def world_of(tile, tile_m):
    """Tile (col,row) -> floor (X, Z) in metres. Origin is wherever you put
    tile (0,0); only relative distances matter to the homography."""
    return (tile[0] * tile_m, tile[1] * tile_m)


def _dist(p, q):
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


def residuals(plane, pts, tile_m):
    """-> [(label_a, label_b, expected_m, measured_m, error_m)] for every pair.

    Measures through the PLANE and compares against what the tile grid says.
    Pure, so it is testable without a camera, a frame or a file.
    """
    out = []
    for (i, (ia, ta)), (j, (ib, tb)) in combinations(list(enumerate(pts)), 2):
        want = _dist(world_of(ta, tile_m), world_of(tb, tile_m))
        ga, gb = plane.to_ground(*ia), plane.to_ground(*ib)
        if ga is None or gb is None:
            out.append((f"p{i+1}", f"p{j+1}", want, None, None))
            continue
        got = _dist(ga, gb)
        out.append((f"p{i+1}", f"p{j+1}", want, got, abs(got - want)))
    return out


def verdict(rows, tol_m):
    """-> (ok, worst_error_or_None). A pair the plane could not map at all is
    a hard failure, not a missing row."""
    errs = [r[4] for r in rows]
    if any(e is None for e in errs):
        return False, None
    worst = max(errs) if errs else None
    return (worst is not None and worst <= tol_m), worst


def enough_evidence(n_fit, n_check):
    """-> (ok, why). Does this measurement set CONTAIN a real test?

    THE TRAP, found by this module's own test: a homography has 8 degrees of
    freedom and 4 point correspondences give exactly 8 equations, so it
    reproduces ANY four points with zero residual — including four points where
    you named the wrong tile corner. At n=4 the error is 0.00m whether you are
    right or wrong, so the check carries no information at all.

    Evidence therefore means one of:
        * a 5th fitted point — the fit becomes over-determined and has to
          compromise, so a bad point finally shows up as error, or
        * a held-out --check point, which the fit never saw.

    This is the difference between a verification and a ritual.
    """
    if n_check > 0:
        return True, f"{n_check} held-out point(s)"
    if n_fit >= 5:
        return True, f"{n_fit} fitted points — over-determined, so error is real"
    return False, (
        f"only {n_fit} fitted points and no --check. A homography fits ANY 4 "
        f"points with zero error, so the numbers below prove nothing. Add a "
        f"5th --pt, or a --check point held out of the fit.")


def render(frame_path, out_path, pts, image_size, zone_size):
    """Draw the clicked points on a frame. Seeing that you pointed at the tile
    corner you meant is the check no arithmetic can do for you."""
    try:
        import cv2
    except ImportError:
        print("  (cv2 unavailable — skipping the overlay)")
        return None
    img = cv2.imread(str(frame_path))
    if img is None:
        print(f"  (could not read {frame_path} — skipping the overlay)")
        return None
    h, w = img.shape[:2]
    sx, sy = w / float(zone_size[0]), h / float(zone_size[1])
    for i, ((x, y), tile) in enumerate(pts, 1):
        px, py = int(round(x * sx)), int(round(y * sy))
        cv2.drawMarker(img, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 26, 3)
        cv2.circle(img, (px, py), 13, (0, 0, 255), 2)
        cv2.putText(img, f"p{i} tile({tile[0]:g},{tile[1]:g})", (px + 18, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "ground-plane points — is each cross ON the tile corner?",
                (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                cv2.LINE_AA)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Fit and VERIFY a ground plane from floor tiles.")
    ap.add_argument("--zones", default=str(ROOT / "zones" / "CAM.112_zone.json"))
    ap.add_argument("--tile-cm", type=float, required=True,
                    help="floor tile edge in centimetres (measure one tile)")
    ap.add_argument("--pt", type=parse_point, action="append", required=True,
                    metavar="X,Y@COL,ROW", help="repeat at least 4 times")
    ap.add_argument("--check", type=parse_point, action="append", default=[],
                    metavar="X,Y@COL,ROW",
                    help="held-out point(s): fitted on nothing, must still land")
    ap.add_argument("--image-size", default=None, metavar="W,H",
                    help="size of the image you read pixels off, if it is not "
                         "the zone file's frame_size (they get scaled)")
    ap.add_argument("--frame", default=None, help="render the points onto this")
    ap.add_argument("--out", default=str(ROOT / "output" / "plane_points.png"))
    ap.add_argument("--write", action="store_true",
                    help="write ground_points into the zones file (only on PASS)")
    a = ap.parse_args(argv)

    cfg = json.loads(Path(a.zones).read_text(encoding="utf-8"))
    zw, zh = cfg.get("frame_size") or [3840, 2160]
    tile_m = a.tile_cm / 100.0

    pts, checks = list(a.pt), list(a.check)
    if a.image_size:
        iw, ih = (float(v) for v in a.image_size.split(","))
        sx, sy = zw / iw, zh / ih
        pts = [(((x * sx), (y * sy)), t) for (x, y), t in pts]
        checks = [(((x * sx), (y * sy)), t) for (x, y), t in checks]
        print(f"  scaled your pixels {iw:g}x{ih:g} -> zone space {zw}x{zh}")

    if len(pts) < 4:
        ap.error(f"need at least 4 --pt, got {len(pts)}")

    evid_ok, evid_why = enough_evidence(len(pts), len(checks))
    if not evid_ok:
        print(f"\n  NO REAL CHECK POSSIBLE: {evid_why}\n")

    plane = GroundPlane.from_correspondences(
        [p for p, _ in pts], [world_of(t, tile_m) for _, t in pts], (zw, zh))
    print(f"\n  tile {a.tile_cm:g}cm · {len(pts)} point(s)"
          f"{f' + {len(checks)} held out' if checks else ''}")
    print(f"  plane: mode={plane.mode} — {plane.note}\n")
    if not plane.ok:
        print("  FAIL — no plane. Are your four points in a straight line?")
        return 1

    rows = residuals(plane, pts, tile_m)
    print("  pair    expected   measured   error")
    for la, lb, want, got, err in rows:
        g = "  n/a  " if got is None else f"{got:6.2f}m"
        e = "   -   " if err is None else f"{err:6.2f}m"
        flag = "" if (err is not None and err <= FIT_TOL_M) else "   <-- off"
        print(f"  {la}-{lb}  {want:6.2f}m   {g}   {e}{flag}")
    ok, worst = verdict(rows, FIT_TOL_M)

    hold_ok = True
    if checks:
        print("\n  HELD-OUT (fitted on nothing — the only honest test):")
        hrows = residuals(plane, pts + checks, tile_m)
        n = len(pts)
        hrows = [r for r in hrows
                 if int(r[1][1:]) > n or int(r[0][1:]) > n]
        for la, lb, want, got, err in hrows:
            g = "  n/a  " if got is None else f"{got:6.2f}m"
            e = "   -   " if err is None else f"{err:6.2f}m"
            flag = "" if (err is not None and err <= HOLDOUT_TOL_M) else "   <-- off"
            print(f"  {la}-{lb}  {want:6.2f}m   {g}   {e}{flag}")
        hold_ok, hworst = verdict(hrows, HOLDOUT_TOL_M)
        print(f"  worst held-out error: "
              f"{'n/a' if hworst is None else f'{hworst:.2f}m'}")

    print(f"\n  worst fit error: {'n/a' if worst is None else f'{worst:.2f}m'} "
          f"(tolerance {FIT_TOL_M}m)")

    if a.frame:
        got = render(a.frame, a.out, pts, a.image_size, (zw, zh))
        if got:
            print(f"  overlay -> {got}  — CHECK EACH CROSS IS ON ITS TILE CORNER")

    passed = ok and hold_ok and evid_ok
    if not evid_ok:
        print(f"\n  UNVERIFIED — {evid_why}")
    print("\n  " + ("PASS — this plane reproduces distances it was checked against."
                    if passed else
                    "FAIL — do NOT write this. A wrong plane is worse than none:\n"
                    "         it makes the size filter relax itself and the giant\n"
                    "         boxes get bigger. Re-check which tile corner each\n"
                    "         point is, and that col,row match the same grid."))

    if a.write and not passed:
        print("  --write IGNORED because the check failed.")
    elif a.write:
        cfg["ground_points"] = [
            {"image": [round(x, 1), round(y, 1)],
             "world": [round(w0, 3), round(w1, 3)]}
            for (x, y), t in pts for w0, w1 in [world_of(t, tile_m)]]
        cfg.pop("_ground_points_TEMPLATE", None)
        cfg["_ground_points_note"] = (
            f"MEASURED, not fitted. {len(pts)} floor-tile corners at "
            f"{a.tile_cm:g}cm, verified by tools/calibrate_plane.py: worst "
            f"pairwise error {worst:.2f}m against the tile grid"
            + (f", worst held-out {len(checks)}-point error within "
               f"{HOLDOUT_TOL_M}m" if checks else "")
            + ". Replaces the automatic perspective fit, which produced eight "
              "different camera heights (1.12m to 3.26m) in one hour and made "
              "the implausible-size filter relax its own tolerance 2.5x->5.0x.")
        Path(a.zones).write_text(json.dumps(cfg, indent=2) + "\n",
                                 encoding="utf-8")
        print(f"  WROTE ground_points -> {a.zones}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
