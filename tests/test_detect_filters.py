"""Tests for kevacv/detect_filters.py — the two phantoms the real run produced.

Cases are built from what was actually observed on CAM.112, not invented:

  D1  P3 was a box spanning nearly the whole frame height with its foot near the
      bottom. A real person there is ~450 px. It must be dropped, and a genuinely
      tall person must NOT be.
  D2  P8 sat on the potted plant for the whole chunk at identical pixels. It must
      be dropped. The receptionist stood at the desk for the same duration and
      must SURVIVE, because she moves.

The asymmetry is deliberate throughout: when the filter cannot tell, it keeps
the detection. Deleting a real person costs more than keeping a phantom.

Run: python test_detect_filters.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kevacv.detect_filters import (BODY_ASPECT, describe, drop_tracks,
                                   implausible_size_mask, protected_ids,
                                   static_track_ids)

FAILED = []
rng = random.Random(11)


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


# the fit the real run produced: h = 0.508 * foot_y + 87
def expected_h(foot_y):
    h = 0.508 * float(foot_y) + 87.0
    return h if h > 1.0 else None


def no_fit(_foot_y):
    return None


print("=" * 74)
print("  D1 — a box too BIG to be a person standing there")
print("=" * 74)
# P3 as observed: top just under the HUD, bottom at the frame edge.
P3 = (855, 90, 1280, 800)
person_near = (300, 350, 380, 800)      # 80 x 450 at foot_y 800
person_far = (600, 300, 640, 420)       # 40 x 120 at foot_y 420
tall_person = (300, 250, 400, 800)      # 100 x 550, a genuinely tall person
two_merged = (300, 300, 480, 800)       # 180 x 500, two people in one box
boxes = [P3, person_near, person_far, tall_person, two_merged]
mask = implausible_size_mask(boxes, expected_h)
names = ["P3 half-frame", "person near", "person far", "TALL person", "2 people merged"]
print(f"    {'case':18s}{'w x h':>12s}{'area':>10s}{'expected':>10s}{'ratio':>7s}{'':>7s}")
for nm, b, m in zip(names, boxes, mask):
    w, h = b[2]-b[0], b[3]-b[1]
    exp = expected_h(b[3])**2 / BODY_ASPECT
    print(f"    {nm:18s}{f'{w}x{h}':>12s}{w*h:>10.0f}{exp:>10.0f}{w*h/exp:>7.2f}"
          f"{'  DROP' if m else '  keep':>7s}")
check(mask[0] is True, "P3 (the half-frame box) is dropped",
      f"{(P3[2]-P3[0])*(P3[3]-P3[1])/(expected_h(P3[3])**2/BODY_ASPECT):.2f}x too big")
check(mask[1] is False, "a normal person near the camera is kept")
check(mask[2] is False, "a small person far away is kept")
check(mask[3] is False, "a genuinely TALL person is kept",
      f"{(tall_person[2]-tall_person[0])*(tall_person[3]-tall_person[1])/(expected_h(tall_person[3])**2/BODY_ASPECT):.2f}x")
check(mask[4] is False, "two people merged into one box are kept (not a phantom)")
print("    -> height alone gave 1.44 vs 1.11 and could not separate these")

check(all(m is False for m in implausible_size_mask(boxes, no_fit)),
      "no scene-geometry fit -> NOTHING is dropped")
check(implausible_size_mask([], expected_h) == [], "empty input -> empty, no crash")
above_horizon = [(600, 10, 640, 40)]
check(implausible_size_mask(above_horizon, lambda y: None) == [False],
      "a point above the horizon -> kept, not fabricated")
strict = implausible_size_mask([tall_person], expected_h, tol=0.5)
check(strict[0] is True, "the tolerance is real (tol=0.5 would drop the tall person)")

print()
print("=" * 74)
print("  D2 — furniture does not fidget")
print("=" * 74)


def track(tid, n, t0=0.0, dt=1.0, cx=700, cy=600, w=90, h=250, jitter=0.0):
    """jitter is a FRACTION of body height applied to centre and size."""
    rows = []
    for i in range(n):
        j = h * jitter
        rows.append((tid,
                     cx - w / 2 + rng.gauss(0, j), cy - h / 2 + rng.gauss(0, j),
                     cx + w / 2 + rng.gauss(0, j), cy + h / 2 + rng.gauss(0, j)))
    return rows


def build(*tracks_with_len):
    """-> frame_log with each track present for its own number of frames."""
    per_frame = {}
    for rows in tracks_with_len:
        for i, r in enumerate(rows):
            per_frame.setdefault(i, []).append(r)
    return [(i, float(i), v) for i, v in sorted(per_frame.items())]


PLANT = track("plant", 600, cx=760, cy=560, w=190, h=430, jitter=0.0005)
RECEPTIONIST = track("sarah", 600, cx=390, cy=300, w=120, h=300, jitter=0.05)
PASSERBY = track(7, 40, cx=200, cy=500, w=80, h=260, jitter=0.05)
flog = build(PLANT, RECEPTIONIST, PASSERBY)

flagged = static_track_ids(flog)
for cid, d in flagged.items():
    print(f"    flagged {cid!r}: {d['seconds']:.0f}s, centre jitter "
          f"{d['centre_jitter']:.4f} of body height")
check("plant" in flagged, "the potted plant is flagged as furniture")
check("sarah" not in flagged,
      "the receptionist standing at the desk for the SAME duration SURVIVES",
      "she moves; the plant does not")
check(7 not in flagged, "a passer-by is never flagged (too short a life)")

# the guard that matters most
prot = protected_ids(crossings=[{"track_id": "plant"}])
check("plant" in static_track_ids(flog) and "plant" not in
      static_track_ids(flog, protected=prot),
      "anything that CROSSED THE DOOR is never dropped, whatever the geometry says")
prot2 = protected_ids(face_ids=["plant"])
check("plant" not in static_track_ids(flog, protected=prot2),
      "anything with a RECOGNISED FACE is never dropped either")

# sensitivity: how still must something be before it is called furniture?
print(f"    {'jitter (frac of body height)':34s}{'flagged?':>10s}")
for j in (0.0005, 0.005, 0.015, 0.02, 0.03, 0.05):
    fl = static_track_ids(build(track(f"t{j}", 400, jitter=j)))
    print(f"    {j:<34.4f}{'FURNITURE' if fl else 'person':>10s}")
check(not static_track_ids(build(track("moving", 400, jitter=0.05))),
      "5% jitter (a still-standing human) is NOT furniture")
check(static_track_ids(build(track("rigid", 400, jitter=0.0005))),
      "0.05% jitter (a fixed object) IS furniture")

check(not static_track_ids(build(track("short", 60, jitter=0.0))),
      "a perfectly rigid but SHORT track is not flagged (needs 120 s)")
check(static_track_ids([]) == {}, "empty frame_log -> nothing, no crash")
check(static_track_ids(flog, canon={"plant": "P9"}).get("P9") is not None,
      "canonical mapping is honoured (fragments judged as one identity)")

print()
print("=" * 74)
print("  removal is consistent across every structure")
print("=" * 74)
events = [{"track_id": "plant", "zone": "reception"}, {"track_id": "sarah", "zone": "reception"}]
crossings = [{"track_id": "plant", "direction": "in"}, {"track_id": 7, "direction": "in"}]
e2, c2, f2 = drop_tracks(events, crossings, flog, {"plant"})
check(all(e["track_id"] != "plant" for e in e2), "events purged")
check(all(c["track_id"] != "plant" for c in c2), "crossings purged")
check(all(b[0] != "plant" for _fi, _t, bx in f2 for b in bx), "frame_log purged")
check(len(e2) == 1 and len(c2) == 1, "nothing else was touched")
check(sum(len(bx) for _f, _t, bx in f2) ==
      sum(len(bx) for _f, _t, bx in flog) - len(PLANT), "exactly the plant's boxes went")

print()
print("=" * 74)
print("  it says what it did")
print("=" * 74)
txt = describe(17, flagged)
print("   " + txt.replace("\n", "\n   "))
check("D1 dropped 17" in txt and "D2 dropped" in txt, "both filters report")
check("no phantoms" in describe(0, {}), "a clean chunk says so explicitly")

print()
print("=" * 74)
# PYTEST VISIBILITY. This file's checks run at IMPORT and reported only via
# an exit code. A module-level sys.exit aborts pytest COLLECTION, which hid
# 59 of 74 test files -- and simply guarding the exit would have hidden the
# FAILURES instead. So the same condition is also asserted as a real test.
def test_script_level_checks_passed():
    assert not (FAILED), "module-level checks in this file failed"


if __name__ == "__main__":
    if FAILED:
        print(f"  {len(FAILED)} FAILED:")
        for f in FAILED:
            print(f"    - {f}")
        sys.exit(1)
print("  ALL PASS")
print("=" * 74)
