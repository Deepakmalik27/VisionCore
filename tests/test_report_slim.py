"""Tests for kevacv/report_slim.py — the three-file deliverable.

Built from the 68b97311f9 run, where nine output files between them still
managed to publish "2 people came through the door" as EXACT*. Every test
below asks: would this report have let that through?

Run: python test_report_slim.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kevacv.report_slim import (PEOPLE_COLUMNS, coverage_strip, people_csv,
                                people_rows, summary_txt)

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


META = {"camera": "CAM.112", "date": "Tue 28 Jul 2026", "start": "16:30",
        "end": "17:30", "footage_h": 1.0, "missing_h": 0.0, "t_end_s": 3600,
        "source": "CAM.112_...4.30pm.mp4", "provenance_ok": True,
        "hota": {"day": 0.52, "ir": 0.44, "verdict": "PASS"}}

ANSWERS = [
    {"label": "Was the desk covered?", "value": "68.9%", "tier": "EXACT",
     "extra": [("longest gap", "15.6 min @ 16:47", "EXACT")]},
    {"label": "How many guests?", "value": "18  (range 14-22)",
     "tier": "ESTIMATE", "extra": [("came back later", 2, "WEAK")]},
]
STAFF = [{"name": "receptionist_sarah", "minutes": 38.0, "pct": 63,
          "source": "face", "confidence": 95},
         {"name": "staff_2", "minutes": 21.0, "pct": 35,
          "source": "zone", "confidence": 44}]
ANOM = [{"time": "16:47", "what": "desk empty 15.6 min, 2 guests waiting",
         "clip": "clip_01.mp4", "severity": "HIGH"}]

print("=" * 74)
print("  every number carries the tier it earned")
print("=" * 74)
txt = summary_txt(META, ANSWERS, STAFF, ANOM,
                  notes=["58% infrared -> identity signals reduced"],
                  covered_windows=[(0, 1000), (2000, 3600)])
check("[EXACT]" in txt and "[ESTIMATE]" in txt and "[WEAK]" in txt,
      "the tiers actually appear next to the values")
check(txt.count("[UNKNOWN]") == 0, "nothing is UNKNOWN when tiers are supplied")

untagged = summary_txt(META, [{"label": "mystery", "value": 7}], [], [])
check("[UNKNOWN]" in untagged,
      "a number with NO tier prints UNKNOWN, never bare",
      "an untiered number is how 'EXACT*' happened")
bogus = summary_txt(META, [{"label": "x", "value": 1, "tier": "DEFINITELY"}], [], [])
check("[UNKNOWN]" in bogus and "DEFINITELY" not in bogus,
      "an invented tier is rejected, not printed")

print()
print("=" * 74)
print("  the report refuses to look confident when it isn't")
print("=" * 74)
noprov = dict(META); noprov["provenance_ok"] = False
t = summary_txt(noprov, ANSWERS, STAFF, ANOM)
check("PROVENANCE UNVERIFIED" in t, "unverified provenance is shouted, not hidden")
check("do not trust the times" in t, "and says what it means for the reader")

nohota = dict(META); nohota["hota"] = None
t = summary_txt(nohota, ANSWERS, STAFF, ANOM)
check("NOT MEASURED" in t, "no HOTA -> says so instead of implying quality")

print()
print("=" * 74)
print("  the annotated video is part of the deliverable, honestly reported")
print("=" * 74)
vmeta = dict(META, video={"annotated": "output/CAM112_annotated_h264.mp4",
                          "codec": "h264", "size_mb": 58.2, "speedup": 4.0,
                          "clips": ["clip_01.mp4", "clip_02.mp4"]})
t = summary_txt(vmeta, ANSWERS, STAFF, ANOM)
check("WATCH" in t, "a WATCH section appears when there is a video")
check("_annotated_h264.mp4" in t, "and names the file")
check("58 MB" in t, "with its size")
check("plays 4.0x faster" in t, "and how long it takes to watch")
check("Windows Media Player" in t, "and confirms h264 actually plays anywhere")
check("2 clip(s)" in t and "clip_01.mp4" in t, "moment clips are listed too")

missing = dict(META, video={"annotated": None, "why": "renderer was not run"})
t = summary_txt(missing, ANSWERS, STAFF, ANOM)
check("NOT FOUND" in t, "a missing video says NOT FOUND")
check("NOT evidence the venue was empty" in t,
      "and refuses to imply an empty venue",
      "the 68b97311f9 run said 'nobody appeared' about a valid 58 MB file")
check("_annotated_h264.mp4" in t, "and points at the name that was really used")
check("WATCH" not in summary_txt(META, ANSWERS, STAFF, ANOM),
      "no video key -> no WATCH section, rather than an empty one")

import tempfile  # noqa: E402
from kevacv.report_slim import describe_video  # noqa: E402

with tempfile.TemporaryDirectory() as td:
    base = Path(td) / "CAM112_annotated.mp4"
    real = Path(td) / "CAM112_annotated_h264.mp4"
    real.write_bytes(b"x" * 2_097_152)
    v = describe_video(str(base), clips=["c1.mp4"], analysed_fps=7.5,
                       playback_fps=30.0)
    check(v["annotated"] == str(real),
          "describe_video finds the _h264 name when asked for the plain one",
          "exactly the lookup that failed in production")
    check(v["codec"] == "h264", "and identifies the codec")
    check(round(v["size_mb"]) == 2, "and the size", f"{v['size_mb']:.1f} MB")
    check(v["speedup"] == 4.0, "and the playback speed-up")
    v2 = describe_video(str(Path(td) / "nope.mp4"))
    check(v2["annotated"] is None and "exist on disk" in v2["why"],
          "and reports WHICH names it looked for when nothing is there")
check(describe_video(None)["annotated"] is None, "no path -> no crash")

print()
print("=" * 74)
print("  the coverage strip shows WHEN, not just how much")
print("=" * 74)
s = coverage_strip([(0, 1800)], 3600, width=20)
check(s == "#" * 10 + "." * 10, "first half covered, second half not", s)
check(coverage_strip([], 3600, width=8) == "." * 8, "no coverage -> all dots")
check(coverage_strip([(0, 3600)], 3600, width=8) == "#" * 8, "full coverage")
check(coverage_strip([], 0, width=8) == "?" * 8,
      "no footage -> '?', which is NOT the same as 'not covered'")

print()
print("=" * 74)
print("  people.csv: a missing measurement is never a zero")
print("=" * 74)
rows = people_rows([{"person": "P1", "minutes": 17.2, "waited_s": None,
                     "confidence": 82}])
check(rows[0]["waited_s"] == "", "an unmeasured value is blank, not 0",
      repr(rows[0]["waited_s"]))
check(list(rows[0].keys()) == PEOPLE_COLUMNS, "column order is fixed")
check(rows[0]["greeted"] == "", "absent keys fill blank, no KeyError")

csv_text = people_csv([
    {"person": "P1", "snap": "P1.jpg", "role": "guest", "minutes": 17.2,
     "greeted": "yes", "confidence": 82},
    {"person": "P3", "snap": "P3.jpg", "role": "staff",
     "role_from": "face:sarah", "confidence": 95, "flags": ""}])
lines = csv_text.strip().split("\n")
check(lines[0] == ",".join(PEOPLE_COLUMNS), "header is the contract", lines[0][:40])
check(len(lines) == 3, "one header + two people", str(len(lines)))
check(csv_text.count("\r") == 0, "unix line endings, no stray CR")

print()
print("=" * 74)
print("  the empty night must still produce a readable file")
print("=" * 74)
t = summary_txt({"camera": "CAM.112"}, [], [], [])
check("nobody was identified at the desk" in t, "empty staff list says so")
check("nothing flagged" in t, "empty anomaly list says so")
check("n/a" in t, "missing footage hours print n/a, not 0.0")
check(people_csv([]).strip() == ",".join(PEOPLE_COLUMNS),
      "no people -> header only, still a valid csv")

print()
print("=" * 74)
if FAILED:
    print(f"  {len(FAILED)} FAILED:")
    for f in FAILED:
        print(f"    - {f}")
    sys.exit(1)
print("  ALL PASS")
print("=" * 74)
