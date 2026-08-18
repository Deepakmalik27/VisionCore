"""test_validity.py — "I did not observe" must never read as "I observed nothing".

Four shipped bugs share this shape: a broken entry line reported as 0 arrivals,
a valid 58 MB video reported as "nobody appeared", a camera-moved guard
comparing against a mangled filename, and a config audit printing values
something later overrode. This module is the one field that prevents the class.

Run: python tests/test_validity.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kevacv.detect_filters import rigid_track_ids  # noqa: E402
from kevacv.validity import (BAD_FRAME, BLIND_CAMERA, DETECTOR_BLIND,
                             GEOMETRY_CHANGED, OK, SKIPPED_IDLE,
                             TIME_WENT_BACKWARDS, DetectorCanary,
                             ValidityLedger, frame_validity)  # noqa: E402

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


def frame(mean=120, std=40, shape=(720, 1280)):
    rng = np.random.default_rng(0)
    f = rng.normal(mean, std, shape).clip(0, 255).astype("uint8")
    return f


print("=" * 74)
print("  a blind camera is not an empty room")
print("=" * 74)
v, d = frame_validity(np.zeros((720, 1280), dtype="uint8"))
check(v == BLIND_CAMERA, "an all-black frame is BLIND_CAMERA", v)
check("not an empty room" in d, "and says so in words")

# the trap: infrared footage is DARK but textured. Condemning it would delete
# 58% of this venue's night.
ir = frame(mean=28, std=18)
check(frame_validity(ir)[0] == OK,
      "a dark BUT TEXTURED infrared frame is OK", "58% of CAM.112 is infrared")
check(frame_validity(frame(mean=9, std=1))[0] == BLIND_CAMERA,
      "dark AND flat is blind — variance is what separates them")

check(frame_validity(None)[0] == BAD_FRAME, "a None frame is BAD_FRAME")
check(frame_validity(np.zeros((0, 10), dtype="uint8"))[0] == BAD_FRAME,
      "a zero-size frame is BAD_FRAME")
v, d = frame_validity(frame(), expect_shape=(1080, 1920))
check(v == GEOMETRY_CHANGED, "a mid-stream resolution change is caught", v)
check("zones were scaled" in d, "and names the consequence for zones")

print()
print("=" * 74)
print("  the ledger divides by OBSERVED time, not elapsed time")
print("=" * 74)
led = ValidityLedger(step_s=1.0)
for t in range(100):
    led.record(t, BLIND_CAMERA if 40 <= t < 60 else OK)
s = led.summary()
check(s["frames"] == 100, "every frame is recorded", str(s["frames"]))
check(s["observed_s"] == 80.0, "20 blind seconds are NOT observed",
      f"{s['observed_s']}s of {s['elapsed_s']}s")
check(round(s["observed_share"], 2) == 0.81 or s["observed_share"] < 1.0,
      "observed share is below 1", str(s["observed_share"]))
win = led.observed_windows()
check(len(win) == 2, "observed time splits into two windows around the blindness",
      str(len(win)))

led2 = ValidityLedger(step_s=1.0)
for t in range(100):
    led2.record(t, SKIPPED_IDLE if t < 50 else OK)
check(led2.summary()["observed_s"] == 100.0,
      "the motion gate declining to look IS an observation",
      "it looked and decided nothing moved — that is data")

print()
print("=" * 74)
print("  non-monotonic time is caught, not silently averaged")
print("=" * 74)
led3 = ValidityLedger(step_s=1.0)
led3.record(10.0)
led3.record(20.0)
v = led3.record(15.0)
check(v == TIME_WENT_BACKWARDS, "a timestamp before the previous one is flagged", v)
check(any("earlier than the one before" in m for _, m in led3.findings()),
      "and reported — every dwell calculation assumes time moves forward")

print()
print("=" * 74)
print("  findings escalate by how much was lost")
print("=" * 74)
led4 = ValidityLedger(step_s=1.0)
for t in range(1000):
    led4.record(t, BLIND_CAMERA if t < 5 else OK)
lv = [l for l, _ in led4.findings()]
check("WARN" in lv and "ERROR" not in lv, "0.5% blind -> WARN", str(lv))
led5 = ValidityLedger(step_s=1.0)
for t in range(1000):
    led5.record(t, BLIND_CAMERA if t < 300 else OK)
check(any(l == "ERROR" for l, _ in led5.findings()), "30% blind -> ERROR")
check(not ValidityLedger().findings(), "a clean run produces no findings")

print()
print("=" * 74)
print("  a dead detector does not get to look like a quiet venue")
print("=" * 74)
can = DetectorCanary(run_length=10)
for t in range(100):
    can.observe(t, motion=True, n_detections=0)
can.close(100)
f = can.findings()
check(f and f[0][0] == "ERROR", "sustained motion with zero detections -> ERROR")
check("not that the venue was empty" in f[0][1], "and refuses the empty reading")

quiet = DetectorCanary(run_length=10)
for t in range(100):
    quiet.observe(t, motion=False, n_detections=0)
quiet.close(100)
check(not quiet.findings(), "no motion + no detections is a genuinely quiet venue",
      "the two signals AGREE, so nothing is wrong")

busy = DetectorCanary(run_length=10)
for t in range(100):
    busy.observe(t, motion=True, n_detections=3)
busy.close(100)
check(not busy.findings(), "motion + detections is healthy")

flick = DetectorCanary(run_length=30)
for t in range(100):
    flick.observe(t, motion=True, n_detections=0 if t % 3 else 2)
flick.close(100)
check(not flick.findings(), "brief gaps do not trip it — only SUSTAINED runs",
      "people genuinely leave frame")

print()
print("=" * 74)
print("  a plant is rigid; a person cannot hold one shape")
print("=" * 74)


def log_for(aspects, tid="x", t0=0.0, step=1.0, jitter_px=0):
    rows = []
    for i, a in enumerate(aspects):
        w, h = 40.0, 40.0 * a
        x = 100 + (i % 3) * jitter_px
        rows.append((i, t0 + i * step, [(tid, x, 200, x + w, 200 + h)]))
    return rows


# 75 samples at 1 s = a 74 s life, comfortably past the 60 s minimum
plant = log_for([3.00, 3.01, 2.99, 3.00, 3.01] * 15, tid="plant", jitter_px=4)
person = log_for([3.5, 2.2, 3.4, 1.6, 3.6, 2.0, 3.3, 1.8] * 10, tid="p1")
r = rigid_track_ids(plant, frame_wh=(1280, 720))
check("plant" in r, "a swaying plant with a jittering box is RIGID",
      f"cv={r.get('plant', {}).get('aspect_cv')}")
check("cannot hold one shape" in r["plant"]["why"], "and the reason is stated")
check("p1" not in rigid_track_ids(person, frame_wh=(1280, 720)),
      "a walking, turning, sitting person is NOT rigid")

# A demo caught this filter flagging a WALKING person whose detector box kept a
# steady aspect. Rigidity alone is not enough: furniture also does not travel.
walker = [(i, float(i), [("w1", 100 + i * 8, 300, 140 + i * 8, 440)])
          for i in range(75)]
check("w1" not in rigid_track_ids(walker, frame_wh=(1280, 720)),
      "a constant-shape box that TRAVELS is not furniture",
      "it is usually a reflection — mirrored_pair_ids' job, not this one")
check("w1" in rigid_track_ids(walker, frame_wh=(1280, 720), max_travel_frac=1.0),
      "and the travel limit is what excludes it, not the aspect test")

check("plant" not in rigid_track_ids(plant, protected={"plant"}),
      "protected ids are never rigid, whatever the geometry says",
      "someone who crossed the door is a human, full stop")
check(rigid_track_ids(log_for([3.0] * 5, tid="brief")) == {},
      "too few sightings -> no verdict, rather than a confident wrong one")
check(rigid_track_ids(log_for([3.0] * 40, tid="short", step=0.5),
                      min_life_s=60.0) == {},
      "a short life -> no verdict either")

print()
print("=" * 74)
if FAILED:
    print(f"  {len(FAILED)} FAILED:")
    for f in FAILED:
        print(f"    - {f}")
    sys.exit(1)
print("  ALL PASS")
print("=" * 74)
