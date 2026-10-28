"""cvat_mask_to_zones.py — draw a zone like a mask, run it like a polygon.

WHY THIS EXISTS
    Zones are drawn today in tools/zone_mapper_v2.html by clicking polygon
    vertices on a still frame. That is how the entry line ended up ~200 px too
    short, fired ZERO times on the first real CAM.112 hour, and silently
    collapsed eight GM-facing numbers to 0 while 46 people waited.

    The 2026-08-20 design call asked for the other way round: paint the region
    the way Photoshop does -- brush, or one SAM click -- instead of placing
    vertices by hand.

WHY IT CONVERTS BACK TO A POLYGON INSTEAD OF KEEPING THE MASK
    Every zone test in the running pipeline is a polygon test:

        engine._in_poly(poly, anchor)                        the counter
        observe.zone_and_confidence -> cv2.pointPolygonTest   the obs rows
        helpers.classify_zones / load_zone_config             role assignment
        pipeline._mask_swallows_zone                          the preflight guard

    Storing masks instead would mean changing the geometry under all four --
    i.e. editing the counting path -- to gain sub-pixel edges on a camera whose
    error budget is dominated by occlusion and IR flips, not by whether a zone
    border is 3 px off. So: paint a mask, ship a polygon. Drawing gets easier
    and the counting code does not move.

    The contour is the FAITHFUL outline, never a convex hull. observe.py's
    zone_conf is a signed distance to the polygon edge, so a hull would push
    the boundary outward across a concave reception desk and quietly inflate
    confidence for bodies that are actually outside it.

USAGE
    # In CVAT: one LABEL PER ZONE, named exactly as the zone should be named.
    # Draw each with the brush or a SAM click. Export "Segmentation mask 1.1".
    python tools/cvat_mask_to_zones.py export.zip --zones zones/CAM.112_zone.json
    #   -> writes zones/CAM.112_zone.PROPOSED.json and prints a diff. Nothing
    #      is overwritten until you have read that diff and passed --apply.
    python tools/cvat_mask_to_zones.py export.zip --zones zones/... --apply
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np

# A zone smaller than this is a stray brush dab, not a region. Rejected loudly:
# a 40-px blob silently becoming "reception" is exactly the class of bug this
# tool exists to end.
MIN_AREA_PX = 2000
# Contour simplification, as a fraction of the shape's own perimeter. 0.2% keeps
# a doorway's corners while dropping the pixel staircase a brush leaves behind.
SIMPLIFY_FRAC = 0.002


def parse_labelmap(text):
    """CVAT labelmap.txt -> {label: (r, g, b)}.

    Format is `name:R,G,B:...`, with a leading comment line. `background` is
    dropped -- it is the absence of a zone, not a zone.
    """
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(":")
        if len(parts) < 2 or not parts[1]:
            continue
        name = parts[0].strip()
        try:
            r, g, b = (int(v) for v in parts[1].split(",")[:3])
        except ValueError:
            continue
        if name.lower() == "background" or (r, g, b) == (0, 0, 0):
            continue
        out[name] = (r, g, b)
    return out


def mask_to_polygon(mask_rgb, colour, simplify_frac=SIMPLIFY_FRAC,
                    min_area=MIN_AREA_PX):
    """One label's colour in an RGB mask -> [[x, y], ...] or None.

    Returns the LARGEST external contour. A zone is one connected region; if a
    label paints two blobs that is a mistake worth surfacing, not silently
    unioning into a polygon that spans the gap between them.
    """
    r, g, b = colour
    sel = np.all(mask_rgb == np.array([r, g, b], dtype=mask_rgb.dtype), axis=-1)
    if not sel.any():
        return None, "colour not present in the mask"
    binary = (sel.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "no contour"
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    big = contours[0]
    area = cv2.contourArea(big)
    if area < min_area:
        return None, f"largest blob is only {int(area)} px^2 (< {min_area})"
    note = ""
    if len(contours) > 1:
        second = cv2.contourArea(contours[1])
        if second > min_area:
            note = (f"WARNING: {len(contours)} separate blobs; kept the largest "
                    f"({int(area)} px^2), DISCARDED one of {int(second)} px^2")
    eps = simplify_frac * cv2.arcLength(big, True)
    poly = cv2.approxPolyDP(big, eps, True).reshape(-1, 2)
    if len(poly) < 3:
        return None, "simplified to fewer than 3 points"
    return [[int(x), int(y)] for x, y in poly], note


def scale_polygon(poly, src_wh, dst_wh):
    """Mask pixels -> zone-file pixels.

    CVAT exports at the resolution of the media you uploaded. The zones file is
    in SOURCE pixels (3840x2160 on CAM.112). Labelling a downscaled proxy and
    writing those coordinates straight in would halve every zone -- the exact
    landmine that already fired once in this project when a 1080p entry line
    was applied to a 4K source.
    """
    if not src_wh or not dst_wh or tuple(src_wh) == tuple(dst_wh):
        return poly, 1.0, 1.0
    sx, sy = dst_wh[0] / src_wh[0], dst_wh[1] / src_wh[1]
    return [[int(round(x * sx)), int(round(y * sy))] for x, y in poly], sx, sy


def polygons_from_zip(zip_path):
    """CVAT 'Segmentation mask 1.1' zip -> ({name: poly}, (w, h), [notes])."""
    notes, polys, size = [], {}, None
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        lm = next((n for n in names if n.endswith("labelmap.txt")), None)
        if lm is None:
            raise SystemExit(
                "no labelmap.txt in the zip -- export format must be "
                "'Segmentation mask 1.1' (not COCO, not Datumaro).")
        labelmap = parse_labelmap(z.read(lm).decode("utf-8", "replace"))
        if not labelmap:
            raise SystemExit("labelmap.txt has no labels besides background.")
        pngs = [n for n in names
                if n.lower().endswith(".png") and "SegmentationClass" in n]
        if not pngs:
            pngs = [n for n in names if n.lower().endswith(".png")]
        if not pngs:
            raise SystemExit("no mask PNG in the zip.")
        if len(pngs) > 1:
            notes.append(f"{len(pngs)} mask frames in the zip; using {pngs[0]}. "
                         f"Zones are static -- label ONE frame.")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "m.png"
            p.write_bytes(z.read(pngs[0]))
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"could not decode {pngs[0]}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        size = (rgb.shape[1], rgb.shape[0])
        for name, colour in labelmap.items():
            poly, note = mask_to_polygon(rgb, colour)
            if poly is None:
                notes.append(f"SKIPPED {name!r}: {note}")
                continue
            polys[name] = poly
            if note:
                notes.append(f"{name!r}: {note}")
    return polys, size, notes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip", help="CVAT 'Segmentation mask 1.1' export")
    ap.add_argument("--zones", required=True, help="the zones/*.json to update")
    ap.add_argument("--apply", action="store_true",
                    help="overwrite --zones instead of writing a .PROPOSED.json")
    a = ap.parse_args(argv)

    zpath = Path(a.zones)
    cfg = json.loads(zpath.read_text(encoding="utf-8"))
    dst_wh = tuple(cfg.get("frame_size") or ())
    new, src_wh, notes = polygons_from_zip(a.zip)

    print(f"mask resolution : {src_wh[0]}x{src_wh[1]}")
    print(f"zones file      : {dst_wh[0]}x{dst_wh[1]}" if dst_wh else
          "zones file      : frame_size MISSING -- cannot verify scale")
    scaled = {}
    for name, poly in new.items():
        sp, sx, sy = scale_polygon(poly, src_wh, dst_wh)
        scaled[name] = sp
        if sx != 1.0 or sy != 1.0:
            print(f"  scaled {name!r} by {sx:.3f}x{sy:.3f}")

    print("\nDIFF")
    old = cfg.get("polygons") or {}
    for name, poly in scaled.items():
        if name in old:
            print(f"  ~ {name:<20} {len(old[name]):>3} pts -> {len(poly):>3} pts")
        else:
            print(f"  + {name:<20} NEW, {len(poly)} pts")
    for name in old:
        if name not in scaled:
            print(f"  = {name:<20} untouched (not in this export)")
    for n in notes:
        print(f"  ! {n}")

    # roles are NOT invented here. helpers.classify_zones falls back to guessing
    # a role from the name, and a silently-guessed role is how a customer area
    # becomes a staff zone.
    roles = cfg.get("roles") or {}
    missing = [n for n in scaled if n not in roles]
    if missing:
        print(f"\n  ROLES MISSING for {missing} -- add them to 'roles' in the "
              f"zones file by hand. Not guessed here on purpose.")

    cfg.setdefault("polygons", {}).update(scaled)
    cfg["_mask_import_note"] = (
        f"polygons for {sorted(scaled)} imported from a CVAT segmentation mask "
        f"({src_wh[0]}x{src_wh[1]}) by tools/cvat_mask_to_zones.py")
    out = zpath if a.apply else zpath.with_suffix(".PROPOSED.json")
    out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    if not a.apply:
        print("  (read the diff, then re-run with --apply to overwrite the real file)")
    return 0


def demo():
    """One runnable check: a painted mask survives the round trip."""
    import io

    W, H = 800, 600
    m = np.zeros((H, W, 3), dtype=np.uint8)
    cv2.rectangle(m, (100, 100), (300, 400), (220, 20, 60), -1)     # reception
    cv2.circle(m, (600, 300), 120, (0, 128, 255), -1)               # dining
    cv2.rectangle(m, (10, 10), (24, 24), (7, 7, 7), -1)             # a stray dab

    lm = ("# label:color\nbackground:0,0,0::\nreception:220,20,60::\n"
          "dining:0,128,255::\nsmudge:7,7,7::\n")
    assert parse_labelmap(lm) == {"reception": (220, 20, 60),
                                  "dining": (0, 128, 255),
                                  "smudge": (7, 7, 7)}, "labelmap parse"

    rgb = m  # already RGB here
    poly, note = mask_to_polygon(rgb, (220, 20, 60))
    assert poly is not None and len(poly) == 4, (poly, note)
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    assert 95 <= min(xs) <= 105 and 295 <= max(xs) <= 305, poly
    assert 95 <= min(ys) <= 105 and 395 <= max(ys) <= 405, poly

    # the stray dab must be REJECTED, not silently shipped as a zone
    bad, why = mask_to_polygon(rgb, (7, 7, 7))
    assert bad is None and "px^2" in why, (bad, why)

    # a circle must keep its shape, not collapse to a box
    circ, _ = mask_to_polygon(rgb, (0, 128, 255))
    assert len(circ) >= 8, f"circle simplified to {len(circ)} pts"

    # CONCAVITY: an L-shape must stay concave. A convex hull here would cover
    # floor that is not in the zone, and observe.zone_conf would report a body
    # standing in that notch as confidently inside.
    lm2 = np.zeros((H, W, 3), dtype=np.uint8)
    cv2.rectangle(lm2, (100, 100), (400, 200), (10, 200, 10), -1)
    cv2.rectangle(lm2, (100, 200), (200, 400), (10, 200, 10), -1)
    lpoly, _ = mask_to_polygon(lm2, (10, 200, 10))
    notch = (350, 350)   # inside the hull, OUTSIDE the real L
    arr = np.asarray(lpoly, dtype=np.int32).reshape(-1, 1, 2)
    assert cv2.pointPolygonTest(arr, notch, False) < 0, \
        "L-shape lost its concavity — a hull would mis-count this point"

    # scaling: a 1080p-labelled zone applied to a 4K source must double
    sp, sx, sy = scale_polygon([[100, 100], [200, 200]], (1920, 1080), (3840, 2160))
    assert sp == [[200, 200], [400, 400]] and sx == 2.0 and sy == 2.0, sp
    same, sx2, _ = scale_polygon([[1, 2]], (100, 100), (100, 100))
    assert same == [[1, 2]] and sx2 == 1.0

    # end to end through a real zip
    buf = io.BytesIO()
    ok, png = cv2.imencode(".png", cv2.cvtColor(m, cv2.COLOR_RGB2BGR))
    assert ok
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("labelmap.txt", lm)
        z.writestr("SegmentationClass/frame_000000.png", png.tobytes())
    buf.seek(0)
    polys, size, notes = polygons_from_zip(buf)
    assert size == (W, H), size
    assert set(polys) == {"reception", "dining"}, set(polys)
    assert any("smudge" in n for n in notes), notes
    print(f"cvat_mask_to_zones: OK  ({len(polys)} zones, "
          f"{len(notes)} note(s), concavity preserved)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo()
    else:
        sys.exit(main())
