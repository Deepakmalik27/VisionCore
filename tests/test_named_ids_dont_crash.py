"""test_named_ids_dont_crash.py — a matched face must not kill the run.

THE BUG, THREE TIMES
    When the staff gallery matches somebody, that track's id becomes a STRING
    ("staff2", "sarah") so the report can name them. Three separate places then
    assumed the id was an int:

      2026-08-14  analytics.events() -- a bare tuple sort raised TypeError.
                  Killed a run 1h58m in, at the last step before the report.
      (also)      derive.py carries the same lesson in a comment.
      2026-08-25  sv.LineZone.trigger() does `int(tracker_id)` internally.
                  Killed the first Delilah chunk at 18%.

    Every time, the run had already burned an hour of GPU. Every time, short
    smoke tests missed it -- because the gallery never matches in a smoke test.
    camera1.mp4 (a UK corner shop scored against Delilah faces) matched nobody
    and sailed straight through.

WHY THIS TEST IS CHEAP AND THE BUG WAS EXPENSIVE
    The condition is "a face matched". That needs no video, no GPU and no
    model -- just a Detections object with one string id. The reason this went
    undetected for months is not difficulty; it is that nothing ever
    constructed that object on purpose.

Run: python tests/test_named_ids_dont_crash.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


import supervision as sv                      # noqa: E402
from kevacv.engine import _numeric_tracker_ids  # noqa: E402


def _dets(ids):
    n = len(ids)
    return sv.Detections(
        xyxy=np.array([[100.0 + 10 * i, 200.0, 160.0 + 10 * i, 400.0]
                       for i in range(n)], dtype=np.float32),
        confidence=np.full(n, 0.9, dtype=np.float32),
        class_id=np.zeros(n, dtype=int),
        tracker_id=np.array(ids, dtype=object),
    )


print()
print("=" * 74)
print("  the aliaser itself")
print("=" * 74)

alias, nxt = {}, [-1]

# all-numeric must pass through UNTOUCHED -- the common path must cost nothing
d_num = _dets([1, 2, 3])
v = _numeric_tracker_ids(d_num, alias, nxt)
check(v is d_num, "an all-numeric Detections is returned unchanged",
      "the common path must not copy")

# mixed ids: names become negative ints, real ids keep their value
d_mix = _dets([7, "staff2", 9, "sarah"])
v = _numeric_tracker_ids(d_mix, alias, nxt)
check(v is not d_mix, "a mixed Detections is COPIED, not mutated")
check(list(d_mix.tracker_id) == [7, "staff2", 9, "sarah"],
      "the ORIGINAL still holds the real names",
      "downstream code needs the names; mutating would silently rename people")
check(v.tracker_id.dtype.kind in "iu", "the view's ids are a true integer array",
      str(v.tracker_id.dtype))
check(v.tracker_id[0] == 7 and v.tracker_id[2] == 9,
      "real numeric ids are preserved exactly")
check(v.tracker_id[1] < 0 and v.tracker_id[3] < 0,
      "named ids alias to NEGATIVE ints", str(list(v.tracker_id)))
check(len(set(v.tracker_id)) == 4, "no two ids collide after aliasing")

# STABILITY across frames is the whole point: LineZone keeps per-id state, so a
# per-frame alias would make one person look like a new crosser every frame.
v2 = _numeric_tracker_ids(_dets(["sarah", "staff2"]), alias, nxt)
check(v2.tracker_id[0] == v.tracker_id[3], "'sarah' keeps the SAME alias next frame",
      f"{v.tracker_id[3]} -> {v2.tracker_id[0]}")
check(v2.tracker_id[1] == v.tracker_id[1], "'staff2' keeps the SAME alias next frame")

# a negative real id must never be handed out as an alias to something else
check(all(a < 0 for a in alias.values()), "every alias is negative")

print()
print("=" * 74)
print("  THE ACTUAL CRASH: sv.LineZone.trigger on a named id")
print("=" * 74)

line = sv.LineZone(start=sv.Point(0, 300), end=sv.Point(1000, 300))
named = _dets([1, "staff2"])

# WHETHER THE RAW PATH CRASHES IS VERSION-DEPENDENT, so it is reported, not
# asserted. supervision 0.26.1 does not int() the tracker_id; 0.30.0 does:
#
#     local dev box : supervision 0.26.1  -> no crash
#     the GPU box   : supervision 0.30.0  -> ValueError, killed the run at 18%
#
# Asserting "the third-party library is buggy" would make this suite fail the
# day they fix it. What matters is that OUR path is safe on both. The version
# split is itself worth knowing -- a bug that cannot reproduce locally is the
# most expensive kind, so the version is printed on every run.
raised = None
try:
    line.trigger(named)
except Exception as e:                        # noqa: BLE001
    raised = e
print(f"       supervision {sv.__version__}: raw named ids -> "
      f"{type(raised).__name__ if raised else 'no crash on this version'}")
check(True, f"raw-path behaviour recorded for supervision {sv.__version__}",
      "informational — the assertion below is the one that matters")

line2 = sv.LineZone(start=sv.Point(0, 300), end=sv.Point(1000, 300))
ok = True
try:
    view = _numeric_tracker_ids(named, alias, nxt)
    ci, co = line2.trigger(view)
    # masks are POSITIONAL, so indexing the ORIGINAL gives the real ids back
    _ = named.tracker_id[np.asarray(ci, dtype=bool)]
except Exception as e:                        # noqa: BLE001
    ok = False
    print(f"       {type(e).__name__}: {e}")
check(ok, "through the aliaser it does NOT crash",
      "and the mask still indexes the original names")

print()
print("=" * 74)
print("  the engine actually USES it at both supervision call sites")
print("=" * 74)
ENG = (ROOT / "kevacv" / "engine.py").read_text(encoding="utf-8")
check("_sv_dets = _numeric_tracker_ids(dets, _id_alias, _id_alias_next)" in ENG,
      "one aliased view is built per frame")
check("line_zone.trigger(_sv_dets)" in ENG, "LineZone gets the aliased view")
check("zone.trigger(_sv_dets)" in ENG, "PolygonZone gets it too")
check("dets.tracker_id[zone.trigger(_sv_dets)]" in ENG,
      "but the REAL ids are what get recorded",
      "recording the alias would put '-1' in the report instead of 'sarah'")

print()
print("=" * 74)


def test_script_level_checks_passed():
    assert not FAILED, f"named-id checks failed: {FAILED}"


if __name__ == "__main__":
    if FAILED:
        print(f"  {len(FAILED)} FAILED:")
        for x in FAILED:
            print(f"    - {x}")
        sys.exit(1)
    print("  ALL PASS")
    print("=" * 74)
