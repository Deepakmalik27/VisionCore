"""test_venue_provenance.py — the venue in the config must match the camera in
the footage.

WHY THIS EXISTS
    config/cam112.yaml said `venue: delilah_la` for weeks. The footage is
    Delilah DALLAS. The NVR burns an overlay into every exported clip:

        2026-07-28 06:34:58 PM | CAM.112 (PP.09:12)

    and "PP.09:12" appears exactly once across all 17 H.WOOD consoles — DLH
    DAL, a G6 Turret. DLH LA also has a CAM.112, "(G3) (PP.02:08)", a G4 Pro.
    Two different cameras, two different cities, the same short name. That
    collision is what hid it, and nothing in the pipeline ever cross-checked
    the name against the footage.

    The cost was not cosmetic. The staff gallery is enrolled per venue, so LA
    photos against Dallas footage can never match — which is exactly what the
    run logged ("enrolled but NEVER matched: staff2, staff4") and which was
    being blamed on the match threshold.

WHAT IS ENFORCED
    The three independent signals must agree with each other. If someone
    points this config at a new chunk from a different venue, this fails
    before a 35-minute run does.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = (ROOT / "config" / "cam112.yaml").read_text(encoding="utf-8")
fail = 0


def check(ok, what, detail=""):
    global fail
    print(f"  {'PASS' if ok else 'FAIL'}  {what}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        fail = 1


print("=" * 74)
print("  venue provenance — config vs the footage it analyses")
print("=" * 74)

venue = re.search(r"^\s*venue:\s*(\S+)", CFG, re.M)
venue = venue.group(1) if venue else ""
check(venue == "delilah_dallas",
      "venue matches the camera burned into the source overlay (PP.09:12)",
      venue)

# The chunk filter and filenames carry the venue's timezone. Dallas is Central.
TZ_BY_VENUE = {"delilah_dallas": ("CDT", "CST"),
               "delilah_la": ("PDT", "PST"),
               "delilah_mia": ("EDT", "EST"),
               "delilah_nyc": ("EDT", "EST")}
check(venue in TZ_BY_VENUE, "venue is one we know the timezone for", venue)
if venue in TZ_BY_VENUE:
    # data/ filenames are the ground truth for which clock the export used
    data = ROOT / "data"
    names = [p.name for p in data.glob("*.mp4")] if data.is_dir() else []
    if names:
        want = TZ_BY_VENUE[venue]
        ok = all(any(w in n for w in want) for n in names)
        check(ok, f"every chunk filename carries {'/'.join(want)}",
              names[0][:60] if names else "")
    else:
        print("  SKIP  no local chunks to check (they live on the GPU box)")

gen = re.search(r"^\s*generation:\s*(\S+)", CFG, re.M)
gen = gen.group(1) if gen else ""
check(gen == "G6", "camera generation matches the fleet inventory (G6 Turret)", gen)
check("PP.09:12" in CFG,
      "the config records the overlay string that proves the venue")

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
