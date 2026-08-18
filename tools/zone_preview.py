"""zone_preview.py — see what the PIPELINE sees, in two seconds.

WHY THIS EXISTS
    Every zone mistake on this camera was obvious once you looked at it, and
    none of them was findable any other way than a 30-minute run:

      * main_entrance covered the right-hand WALL, the porthole and a plant.
        The entry line fired zero times in an hour.
      * the entry_line was a 654px diagonal in a corner, with both endpoints
        outside every polygon. People walked around it.
      * reception straddled BOTH sides of the counter, so a guest at the front
        triggered the same zone as the staff behind it -- 19 of 28 identities
        came back "staff".
      * reception's top edge sat at y135 while a staff member standing behind
        the counter has their box CENTRE at y137. They flickered in and out
        all hour, and nobody exceeded 16% desk time.

    Drawing the zones is not the hard part -- tools/zone_mapper_v2.html does
    that fine. Knowing whether what you drew is RIGHT is the hard part, and
    the only feedback available was to run the pipeline and read a number.

    This closes that loop. It renders the zone file onto a real frame exactly
    as the engine reads it: same scaling, same roles, and -- the one that
    mattered most -- the same ANCHOR RULE.

THE ANCHOR RULE, MADE VISIBLE
    A staff zone claims a person by their box CENTRE; every other zone claims
    them by their FEET. That single asymmetry caused two of the four failures
    above and is invisible in any drawing tool. Here it is drawn: a dot where
    the zone would test, for a person standing at that spot.

Run:
    python tools/zone_preview.py
    python tools/zone_preview.py --frame output/run3/viewref_CAM.112.png
    python tools/zone_preview.py --person 640,400   # test one spot
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kevacv.helpers import (anchor_point, classify_zones, load_zone_config,
                            uses_centre_anchor)

ROLE_BGR = {
    "entry":   (80, 200, 80),
    "staff":   (60, 140, 255),
    "wait":    (230, 180, 60),
    "seating": (200, 120, 220),
    "mask":    (60, 60, 220),
    "walkway": (170, 170, 170),
    "other":   (150, 150, 150),
}


def _colour(roles):
    for r in ("mask", "entry", "staff", "wait", "seating", "walkway"):
        if r in roles:
            return ROLE_BGR[r]
    return ROLE_BGR["other"]


def draw(frame, zones_path, person=None):
    h, w = frame.shape[:2]
    # EXACTLY how the engine loads them: scaled to the frame it analyses.
    polys, lines = load_zone_config(zones_path, frame_size=(w, h))
    roles = classify_zones(list(polys))
    staff_zones = {z for z, r in roles.items() if "staff" in r}

    overlay = frame.copy()
    for name, poly in polys.items():
        col = _colour(roles.get(name, []))
        pts = poly.astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [pts], col)
    frame = cv2.addWeighted(overlay, 0.22, frame, 0.78, 0)

    for name, poly in polys.items():
        rs = roles.get(name, [])
        col = _colour(rs)
        pts = poly.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], True, col, 2, cv2.LINE_AA)
        top = min(poly.tolist(), key=lambda p: (p[1], p[0]))
        anchor = "CENTRE" if uses_centre_anchor(name, staff_zones) else "feet"
        label = f"{name} [{'/'.join(rs)}] <-{anchor}"
        x = int(np.clip(top[0] + 4, 2, w - 8 * len(label) - 4))
        y = int(np.clip(top[1] - 6, 14, h - 4))
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    col, 1, cv2.LINE_AA)

    for lname, pts in lines.items():
        (x1, y1), (x2, y2) = pts
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3,
                 cv2.LINE_AA)
        for px, py in ((x1, y1), (x2, y2)):
            cv2.circle(frame, (int(px), int(py)), 7, (0, 0, 255), -1)
        length = float(np.hypot(x2 - x1, y2 - y1))
        cv2.putText(frame, f"{lname}: {length:.0f}px = {100*length/w:.0f}% of width",
                    (int(min(x1, x2)), int(min(y1, y2)) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    # A TEST PERSON. The box is drawn at a plausible size for that footline and
    # both anchor points are marked, so you can see which zones would claim
    # them -- the question every one of the past failures actually turned on.
    if person:
        fx, fy = person
        bh = max(40.0, 0.55 * fy)          # rough perspective: taller near cam
        bw = bh / 3.2
        box = (fx - bw / 2, fy - bh, fx + bw / 2, fy)
        cv2.rectangle(frame, (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])), (255, 255, 255), 2)
        feet = anchor_point(box, False)
        ctr = anchor_point(box, True)
        cv2.circle(frame, (int(feet[0]), int(feet[1])), 6, (0, 255, 255), -1)
        cv2.circle(frame, (int(ctr[0]), int(ctr[1])), 6, (255, 0, 255), -1)
        claimed = []
        for name, poly in polys.items():
            c = uses_centre_anchor(name, staff_zones)
            pt = ctr if c else feet
            if cv2.pointPolygonTest(poly.astype(np.float32),
                                    (float(pt[0]), float(pt[1])), False) >= 0:
                claimed.append(f"{name}({'centre' if c else 'feet'})")
        print(f"  a person standing at {person} would be claimed by: "
              f"{', '.join(claimed) if claimed else 'NO ZONE'}")
        cv2.putText(frame, "feet", (int(feet[0]) + 9, int(feet[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)
        cv2.putText(frame, "centre", (int(ctr[0]) + 9, int(ctr[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)

    cv2.putText(frame, f"as the engine reads it @ {w}x{h}", (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return frame, polys, roles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zones", default=str(ROOT / "zones" / "CAM.112_zone.json"))
    ap.add_argument("--frame", default=None,
                    help="a reference frame; defaults to the newest viewref_*.png")
    ap.add_argument("--out", default=str(ROOT / "output" / "zone_preview.png"))
    ap.add_argument("--width", type=int, default=1920,
                    help="render at the width the engine ANALYSES at")
    ap.add_argument("--person", default=None, metavar="X,Y",
                    help="foot position of a test person, in rendered px")
    a = ap.parse_args()

    frame_path = a.frame
    if not frame_path:
        cands = sorted(ROOT.glob("output/**/viewref_*.png"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            print("  no frame given and no output/**/viewref_*.png found.\n"
                  "  pass --frame <a still from this camera>")
            return 2
        frame_path = cands[0]
        print(f"  using {frame_path}")

    frame = cv2.imread(str(frame_path))
    if frame is None:
        print(f"  cannot read {frame_path}")
        return 2
    if frame.shape[1] != a.width:
        frame = cv2.resize(frame, (a.width,
                                   int(round(frame.shape[0] * a.width / frame.shape[1]))))

    person = None
    if a.person:
        px, py = a.person.split(",")
        person = (float(px), float(py))

    out, polys, roles = draw(frame, a.zones, person)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out), out)
    print(f"\n  {len(polys)} zone(s) drawn -> {a.out}")
    for n, r in sorted(roles.items()):
        print(f"    {n:<16} {r}")
    print("\n  yellow dot = feet (what most zones test)")
    print("  magenta dot = box centre (what STAFF zones test)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
