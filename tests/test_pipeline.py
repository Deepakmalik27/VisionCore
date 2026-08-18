"""test_pipeline.py — the orchestrator, exercised without a GPU.

The whole point of pipeline.py is that a run can be driven end to end with a
stub engine. A pipeline that only runs on a GPU box is a pipeline nobody tests,
and this project already learned what untested orchestration costs: a fallback
arrival count was published as EXACT* for a full hour of footage.

Run: python tests/test_pipeline.py
"""
import logging
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kevacv.log import close, get_logger  # noqa: E402
from kevacv.pipeline import preflight, resolve_identities, run_camera  # noqa: E402

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.rows = []

    def emit(self, r):
        self.rows.append((r.levelname, getattr(r, "stagepath", "-"), r.getMessage()))


cap = Capture()
get_logger().addHandler(cap)
get_logger().setLevel(logging.DEBUG)

ZONES = {"main_entrance": [[0, 0]], "waiting_area": [[1, 1]]}
ROLES = {"main_entrance": ["entry"], "waiting_area": ["wait"]}


def make_analyse(entry_hits=1, movers=22):
    def _f(video_path, zones_path, **kw):
        ev = [{"track_id": f"e{i}", "zone": "main_entrance", "t_in": 10 + i,
               "t_out": 15 + i, "duration": 5} for i in range(entry_hits)]
        ev += [{"track_id": f"e{i}", "zone": "waiting_area", "t_in": 30 + i,
                "t_out": 200, "duration": 170} for i in range(entry_hits)]
        ev += [{"track_id": f"m{i}", "zone": "waiting_area", "t_in": 100 + i * 30,
                "t_out": 220 + i * 30, "duration": 120}
               for i in range(movers - entry_hits)]
        return {"events": ev, "crossings": [], "roles": {},
                "zone_roles": ROLES, "duration_s": 3600.0,
                "annotated_video": None, "provenance_ok": True}
    return _f


print("=" * 74)
print("  preflight refuses a zone set that cannot answer the questions")
print("=" * 74)
ok, f = preflight(ZONES, ROLES)
check(ok and not f, "a sane zone set passes cleanly")
ok, f = preflight(ZONES, {"waiting_area": ["wait"]})
check(not ok and any("ENTRY role" in m for _, m in f), "no entry zone -> ERROR")
ok, f = preflight(ZONES, {"main_entrance": ["entry"]})
check(not ok and any("arrive INTO" in m for _, m in f), "no interior zone -> ERROR")
ok, f = preflight({}, ROLES)
check(not ok, "no polygons at all -> ERROR")
try:
    preflight(ZONES, {}, strict=True)
    check(False, "strict=True raises")
except Exception as e:
    check("Preflight" in type(e).__name__, "strict=True raises rather than returns",
          type(e).__name__)

print()
print("=" * 74)
print("  the CAM.112 failure is caught and shouted, not published")
print("=" * 74)
cap.rows.clear()
with tempfile.TemporaryDirectory() as td:
    r = run_camera("chunk.mp4", "zones.json", td, camera_id="CAM.112",
                   analyse_fn=make_analyse(entry_hits=1, movers=22),
                   zones=ZONES, zone_roles=ROLES)
    xc = r["arrivals"]["cross_check"]
    check(xc["trust"] == "neither", "trusts NEITHER arrival count", xc["trust"])
    check("MISPLACED" in xc["verdict"], "and names the fault")
    check(any(lvl == "ERROR" and "ENTRY ZONE MISPLACED" in m
              for lvl, _, m in cap.rows), "the banner fires at ERROR")
    names = {Path(p).name for p in r["written"]}
    check("SUMMARY.txt" in names and "people.csv" in names,
          "and the three-file deliverable is still written", str(sorted(names)))
    check((Path(td) / "debug").is_dir(), "with a debug/ folder for everything else")

print()
print("=" * 74)
print("  a healthy venue is NOT flagged")
print("=" * 74)
cap.rows.clear()
with tempfile.TemporaryDirectory() as td:
    r = run_camera("chunk.mp4", "zones.json", td, camera_id="CAM.112",
                   analyse_fn=make_analyse(entry_hits=20, movers=22),
                   zones=ZONES, zone_roles=ROLES)
    check(r["arrivals"]["cross_check"]["trust"] != "neither",
          "everyone came through the door -> the count is usable",
          r["arrivals"]["cross_check"]["trust"])
    check(not any("ENTRY ZONE MISPLACED" in m for _, _, m in cap.rows),
          "and no false alarm")

print()
print("=" * 74)
print("  the timeline reads root to leaf")
print("=" * 74)
cap.rows.clear()
with tempfile.TemporaryDirectory() as td:
    run_camera("c.mp4", "z.json", td, analyse_fn=make_analyse(20, 22),
               zones=ZONES, zone_roles=ROLES)
paths = [p for _, p, _ in cap.rows]
for phase in ("run", "run > preflight", "run > analyse", "run > identity",
              "run > answers", "run > report"):
    check(phase in paths, f"stage {phase!r} appears in the log")
check(any("events=" in m for _, _, m in cap.rows), "counters reach the log")
check(paths[0] == "run" and paths[-1] == "run", "opens and closes at the root")

print()
print("=" * 74)
print("  a failing stage names itself, and the run stops there")
print("=" * 74)
cap.rows.clear()


def boom(video_path, zones_path, **kw):
    raise RuntimeError("ffmpeg pipe died")


with tempfile.TemporaryDirectory() as td:
    try:
        run_camera("c.mp4", "z.json", td, analyse_fn=boom,
                   zones=ZONES, zone_roles=ROLES)
        check(False, "the exception propagates")
    except RuntimeError:
        check(True, "the exception propagates rather than being swallowed")
errs = [(p, m) for lvl, p, m in cap.rows if lvl == "ERROR"]
check(any(p == "run > analyse" for p, _ in errs), "the failing stage is named",
      "run > analyse")
check(any("ffmpeg pipe died" in m for _, m in errs), "with the real reason")

print()
print("=" * 74)
print("  identity resolution applies the topology veto BEFORE the union")
print("=" * 74)
seen = {}


def spy_merge(tw, emb, **kw):
    seen.update(kw)
    return {t: t for t in tw}, [], {"tier_counts": {}}


W = {"a": (0.0, 10.0), "b": (300.0, 310.0)}
POS = {"a": ((400, 400), (1150, 600)), "b": ((400, 400), (400, 400))}
mapping, edges, diag = resolve_identities(
    W, {}, merge_fn=spy_merge, positions=POS, zones=ZONES, zone_roles=ROLES,
    frame_wh=(1280, 720))
check("topology_vetoed" in diag, "the veto count is reported in diagnostics",
      str(diag.get("topology_vetoed")))
check(diag["doors_used"] >= 1, "doors were derived from the entry zone")
d2 = resolve_identities(W, {}, merge_fn=spy_merge, positions=POS, zones=ZONES,
                        zone_roles=ROLES, frame_wh=(1280, 720),
                        use_topology=False)[2]
check(d2["doors_used"] == 0 and d2["topology_vetoed"] == 0,
      "and use_topology=False turns it off completely")

close()
print()
print("=" * 74)
if FAILED:
    print(f"  {len(FAILED)} FAILED:")
    for f in FAILED:
        print(f"    - {f}")
    sys.exit(1)
print("  ALL PASS")
print("=" * 74)
