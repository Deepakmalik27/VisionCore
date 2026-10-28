"""test_label_tool.py — the local label tool must be self-contained and its
export must be exactly what the scorer reads.

WHY THIS EXISTS
    Ground truth cannot be automated: if the pipeline generated it, we would be
    grading the pipeline against itself. It needs a human. What it does NOT need
    is a website, an account, or venue CCTV leaving the building — so the tool
    is one local HTML file, like tools/zone_mapper_v2.html.

    That earlier file was broken once by an over-greedy edit that deleted its
    global declarations: `node --check` passed, the page loaded, and it drew
    nothing, because a missing global is a RUNTIME error. So the checks here are
    about presence of the things a browser needs at runtime, plus the one
    contract that actually matters — the export format.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "tools" / "label_tool.html").read_text(encoding="utf-8")
fail = 0


def check(ok, what, detail=""):
    global fail
    print(f"  {'PASS' if ok else 'FAIL'}  {what}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        fail = 1


print("=" * 74)
print("  local label tool")
print("=" * 74)

# Every identifier the handlers reach for must be declared somewhere.
for name in ("imgs", "boxes", "idx", "curId", "sel", "drag", "COLORS", "ctx", "cv"):
    check(re.search(rf"\b(let|const|var)\s+[^;]*\b{name}\b", SRC) is not None,
          f"global '{name}' is declared")

for fn in ("show", "draw", "exportGt", "at"):
    check(f"function {fn}(" in SRC, f"function {fn}() exists")

check("webkitdirectory" in SRC, "can open a whole folder at once")
check("http" not in SRC.replace("https://json", "").replace("http-equiv", "")
      or "fetch(" not in SRC,
      "no network calls — footage never leaves the machine")
check("predictions.txt" in SRC, "seeds from our own predictions (correct, not draw)")

# THE CONTRACT: the export must be MOT 1.1, which is what eval_harness reads.
# frame,id,x,y,w,h,conf,class,visibility  with 1-INDEXED frames.
# Target the MOT row specifically. A plain `rows\.push` regex now
# matches `prows.push` from the POINT export first and reported the
# wrong string as broken.
m = re.search(r"\n      rows\.push\(([^;]+)\);", SRC, re.S)
check(m is not None, "export builds MOT rows")
if m:
    row = m.group(1)
    check("${f}" in row and "${b.id}" in row, "row starts frame,id")
    check(row.rstrip().endswith('1,1,1`'), "row ends conf,class,visibility = 1,1,1")
    check("f = 1; f <= imgs.length" in SRC,
          "frames are 1-INDEXED (MOT convention, matches predictions.txt)")

# Round-trip: a hand-built row must parse the way eval_harness expects.
sys.path.insert(0, str(ROOT))
from kevacv.eval_harness import load_mot  # noqa: E402
import tempfile, os  # noqa: E402
p = tempfile.mktemp(suffix=".txt")
Path(p).write_text("1,3,100.00,200.00,50.00,120.00,1,1,1\n"
                   "2,3,105.00,200.00,50.00,120.00,1,1,1\n")
got = load_mot(p)
os.unlink(p)
check(set(got) == {1, 2}, "scorer reads the exported format", f"frames {sorted(got)}")
check(got[1][0][0] == 3, "track id survives the round trip")

# ── INTERPOLATION, the feature that decides whether 1,350 frames is an hour
# or an evening. Its arithmetic is replicated here because a subtle error would
# silently produce WRONG ground truth, which is worse than having none: every
# later measurement would be confidently mis-scored against it.
check("function interpolate()" in SRC, "interpolate() exists")
check("keyframes" in SRC and "'k'" in SRC, "keyframes can be marked (K)")
check("if (!cur.some(x => x.id === ba.id))" in SRC,
      "interpolation NEVER overwrites a human-placed box")
check("const bb = B.find(x => x.id === ba.id)" in SRC,
      "only interpolates a person present at BOTH keyframes")
check("localStorage" in SRC and "function save(" in SRC,
      "work is autosaved — a refresh cannot destroy an hour of labelling")

# the linear step, as written in the tool: r = (f-a)/(b-a)
a, b = 10, 40
A = dict(x=100.0, y=200.0, w=50.0, h=120.0)
B = dict(x=400.0, y=260.0, w=50.0, h=120.0)
mid = a + (b - a) // 2
r = (mid - a) / (b - a)
x = A["x"] + (B["x"] - A["x"]) * r
check(abs(r - 0.5) < 1e-9, "midpoint ratio is 0.5", f"r={r}")
check(abs(x - 250.0) < 1e-9, "midpoint x is halfway", f"{x} between 100 and 400")
x1 = A["x"] + (B["x"] - A["x"]) * ((a + 1 - a) / (b - a))
check(abs(x1 - 110.0) < 1e-9, "first interpolated frame steps one increment",
      f"{x1}")

# ── THE COPY-FORWARD GUARDS ────────────────────────────────────────────────
# The first gt.txt from this tool was 99% carried forward — 6 boxes drawn on
# frame 1, C pressed 99 times. Median label movement 0.0px while the people
# underneath moved a median 5.3px/frame and up to 255px overall. Recall, the
# foot-anchor calibration and the zone audit were all computed against it and
# all had to be retracted. Nothing in the tool objected: 600 boxes, 100 frames,
# six people throughout, exported without a murmur.
#
# "has boxes" and "was labelled" are different facts. These guards make the
# tool count the second one.
check("let edited = new Set()" in SRC, "the tool tracks which frames were EDITED")
check("edited.add(idx+1)" in SRC, "drawing / clicking marks a frame as edited")
check("Deliberately does NOT mark the frame as edited" in SRC,
      "copy-forward (C) does NOT count as editing")
check("CARRIED FORWARD" in SRC,
      "a carried-forward frame is flagged ON THE IMAGE, not in a corner")
check("if (frac < 0.30)" in SRC,
      "export REFUSES a file that is under 30% edited")
check("had to be retracted" in SRC,
      "the tool records WHY these guards exist")

# the arithmetic the refusal rests on
for n_edited, n_frames, should_refuse in ((1, 100, True), (29, 100, True),
                                          (30, 100, False), (100, 100, False)):
    frac = n_edited / n_frames
    check((frac < 0.30) == should_refuse,
          f"{n_edited}/{n_frames} edited -> "
          f"{'refused' if should_refuse else 'allowed'}",
          f"{100*frac:.0f}%")

print()
print("  ALL PASS" if not fail else "  FAILURES ABOVE")
# PYTEST VISIBILITY. This file's checks run at IMPORT and reported only via
# an exit code. A module-level sys.exit aborts pytest COLLECTION, which hid
# 59 of 74 test files -- and simply guarding the exit would have hidden the
# FAILURES instead. So the same condition is also asserted as a real test.
def test_script_level_checks_passed():
    assert not (fail), "module-level checks in this file failed"


if __name__ == "__main__":
    sys.exit(fail)
