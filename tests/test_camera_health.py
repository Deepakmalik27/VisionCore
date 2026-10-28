"""Tests for camera_health.py — "is this still the same camera view?"

Built with a synthetic room: a textured background with fixed furniture, then
known transforms applied to it. Every expected answer is known because we
created the movement ourselves.

The cases that matter:
  * a knocked camera must be CAUGHT and must invalidate the run
  * people walking through must NOT be mistaken for the camera moving
    (this is the whole reason it is RANSAC on background features and not a
     frame difference)
  * the tolerance must be measured on the ZONE vertices, because a small
    rotation is nothing at the image centre and a lot at the doorway
  * out of focus / blinded / blocked / frozen must each be caught

Run: python test_camera_health.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kevacv.camera_health import CameraHealth, verdict_line

FAILED = []
rng = np.random.default_rng(7)


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


W, H = 1280, 720


def room(seed=1):
    """A textured static scene with plenty of corners to match on."""
    r = np.random.default_rng(seed)
    img = np.full((H, W, 3), 120, np.uint8)
    img += r.integers(-18, 18, (H, W, 3), dtype=np.int16).astype(np.uint8)
    for _ in range(90):                       # furniture / wall features
        x, y = int(r.integers(0, W - 90)), int(r.integers(0, H - 90))
        w, h = int(r.integers(25, 90)), int(r.integers(25, 90))
        c = tuple(int(v) for v in r.integers(0, 255, 3))
        cv2.rectangle(img, (x, y), (x + w, y + h), c, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (10, 10, 10), 2)
    for _ in range(40):
        p1 = (int(r.integers(0, W)), int(r.integers(0, H)))
        p2 = (int(r.integers(0, W)), int(r.integers(0, H)))
        cv2.line(img, p1, p2, tuple(int(v) for v in r.integers(0, 255, 3)), 2)
    return img


def shift(img, dx=0, dy=0, deg=0.0, scale=1.0):
    M = cv2.getRotationMatrix2D((W / 2, H / 2), deg, scale)
    M[0, 2] += dx
    M[1, 2] += dy
    return cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)


def add_people(img, n=6, seed=3):
    """Big moving blobs — the outliers RANSAC has to ignore."""
    r = np.random.default_rng(seed)
    out = img.copy()
    for _ in range(n):
        x, y = int(r.integers(80, W - 80)), int(r.integers(200, H - 60))
        cv2.rectangle(out, (x - 30, y - 150), (x + 30, y), (40, 40, 40), -1)
        cv2.circle(out, (x, y - 168), 22, (60, 55, 50), -1)
    return out


REF = room()
ZONES = {"reception": np.array([[900, 380], [1180, 380], [1180, 600], [900, 600]]),
         "waiting": np.array([[120, 400], [520, 400], [520, 660], [120, 660]])}
ch = CameraHealth.from_frame(REF)

print("=" * 74)
print("  the camera has NOT moved")
print("=" * 74)
res = ch.check(REF, ZONES)
check(res["valid"], "identical frame -> valid", verdict_line(res)[:70])
check(res["zone_shift_px"] < 2.0, "zone shift ~0 px", f"{res['zone_shift_px']:.2f}")

res = ch.check(add_people(REF, 8), ZONES)
check(res["valid"], "EIGHT people walking through -> still valid",
      f"zone shift {res['zone_shift_px']:.1f} px, {res['inliers']} background features")
print("    -> this is why it is RANSAC on background features, not a frame diff")

lit = np.clip(REF.astype(np.int16) * 0.55 + 30, 0, 255).astype(np.uint8)
res = ch.check(lit, ZONES)
check(res["valid"], "lights dimmed (day -> evening) -> still valid",
      f"zone shift {res['zone_shift_px']:.1f} px")

print()
print("=" * 74)
print("  the camera HAS moved — must be caught, must invalidate")
print("=" * 74)
# Expectations follow the DERIVED tolerance (0.8% of diagonal = 11.7 px here),
# not a feeling. Below ~10 px a zone-edge answer cannot flip; above it, it can.
for dx, dy, deg, want in [(3, 2, 0.0, False), (6, 4, 0.0, False),
                          (40, 0, 0.0, True), (0, 35, 0.0, True),
                          (0, 0, 1.5, True), (0, 0, 0.4, None), (60, 40, 2.0, True)]:
    res = ch.check(shift(REF, dx, dy, deg), ZONES)
    s = res["zone_shift_px"]
    tag = f"dx={dx:>3} dy={dy:>3} rot={deg:>4}deg"
    if want is None:
        print(f"    {tag}  zone shift {s:6.1f} px  -> {'INVALID' if not res['valid'] else 'valid'} (borderline, informational)")
        continue
    check(res["moved"] == want,
          f"{tag} -> {'CAUGHT as moved' if want else 'tolerated'}",
          f"zone shift {s:.1f} px vs tol {res['zone_tol_px']:.0f} px")

res = ch.check(shift(REF, 60, 40), ZONES)
check(not res["valid"], "a moved camera makes the run INVALID (does not just warn)")
check("CAMERA MOVED" in " ".join(res["reasons"]), "the reason names the cause")
check("Re-draw the zones" in " ".join(res["reasons"]), "the reason says what to do")

# tolerance must follow the ZONES, not the frame corners
far_zones = {"door": np.array([[20, 40], [180, 40], [180, 200], [20, 200]])}
mid_zones = {"middle": np.array([[600, 330], [700, 330], [700, 400], [600, 400]])}
r_far = ch.check(shift(REF, 0, 0, 1.2), far_zones)
r_mid = ch.check(shift(REF, 0, 0, 1.2), mid_zones)
check(r_far["zone_shift_px"] > r_mid["zone_shift_px"] * 2,
      "the SAME rotation hurts a corner zone far more than a centre zone",
      f"corner {r_far['zone_shift_px']:.1f} px vs centre {r_mid['zone_shift_px']:.1f} px")
print("    -> tolerance on frame corners would have flagged the wrong things")

print()
print("=" * 74)
print("  the tolerance sits above the noise and below the answer-flip point")
print("=" * 74)
jpg = lambda im, q: cv2.imdecode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])[1], 1)
still_cases = {
    "identical": REF,
    "14 people walking": add_people(REF, 14, seed=12),
    "lights dimmed 45%": np.clip(REF.astype(np.int16) * 0.55 + 30, 0, 255).astype(np.uint8),
    "jpeg q=30": jpg(REF, 30),
    "sensor noise": np.clip(REF.astype(np.int16)
                            + rng.integers(-25, 25, REF.shape), 0, 255).astype(np.uint8),
    "all combined": np.clip(add_people(jpg(REF, 35), 10, seed=5).astype(np.int16)
                            * 0.6 + 25, 0, 255).astype(np.uint8),
}
worst = 0.0
for name, img in still_cases.items():
    s_px = ch.check(img, ZONES)["zone_shift_px"] or 0.0
    worst = max(worst, s_px)
    print(f"    camera STILL, {name:20s} -> reported shift {s_px:5.2f} px")
tol_px = ch.zone_tol_frac * ((W ** 2 + H ** 2) ** 0.5)
person_w_third = (0.15 * H / 3.5) / 3
check(worst < 1.0, "estimator noise floor is sub-pixel even under abuse",
      f"worst {worst:.2f} px")
check(tol_px > worst * 5, "tolerance is far above the noise floor",
      f"tol {tol_px:.1f} px vs noise {worst:.2f} px")
check(tol_px <= person_w_third * 1.3,
      "tolerance is at or below the point a zone-edge answer can flip",
      f"tol {tol_px:.1f} px vs 1/3 shoulder width {person_w_third:.1f} px")
print("    -> both ends measured, so the number is derived rather than guessed")

print()
print("=" * 74)
print("  the video is not what you think it is")
print("=" * 74)
res = ch.check(cv2.GaussianBlur(REF, (0, 0), 6), ZONES)
check(not res["valid"] and any("FOCUS" in p for p in res["quality"]["problems"]),
      "out of focus -> INVALID", res["quality"]["problems"][:1])
res = ch.check(np.full_like(REF, 250), ZONES)
check(not res["valid"], "blinded by light -> INVALID",
      str(res["reasons"])[:64])
res = ch.check(np.full_like(REF, 4), ZONES)
check(not res["valid"], "lens blocked / pitch black -> INVALID",
      str(res["reasons"])[:64])
res = ch.check(REF, ZONES, prev_frame=REF)
check(any("FROZEN" in p for p in res["quality"]["problems"]),
      "identical consecutive frames -> FROZEN STREAM caught")
res = ch.check(add_people(REF, 4), ZONES, prev_frame=REF)
check(not any("FROZEN" in p for p in res["quality"]["problems"]),
      "a genuinely different frame is not called frozen")

print()
print("=" * 74)
print("  resolution change (U2) and un-verifiable views")
print("=" * 74)
small = cv2.resize(REF, (640, 360))
res = ch.check(small, ZONES)
check(not res["valid"] and "RESOLUTION CHANGED" in " ".join(res["reasons"]),
      "resolution change -> INVALID, zones no longer mean the same thing",
      str(res["reasons"])[:70])
res = ch.check(room(seed=99), ZONES)
check(not res["valid"], "a completely different room -> INVALID",
      f"{res['inliers']} inliers")

print()
print("=" * 74)
print("  persistence across chunks")
print("=" * 74)
import tempfile
with tempfile.TemporaryDirectory() as td:
    p = ch.save(Path(td) / "ref")
    ch2 = CameraHealth.load(p)
    check(ch2 is not None, "reference saves and loads")
    r1, r2 = ch.check(shift(REF, 45, 20), ZONES), ch2.check(shift(REF, 45, 20), ZONES)
    check(r1["moved"] == r2["moved"] is True,
          "a reloaded reference gives the same verdict",
          "chunk 7 is compared against chunk 1's view, not its own")

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
