"""Tests for kevacv/learn_zones.py — find the door without drawing it.

The failure this module exists to prevent: a hand-drawn entry line 200px too
short fired ZERO times over a full hour. The failure this module could ITSELF
cause is worse — proposing a "door" in the middle of the lobby because the
tracker fragments there. That case is tested hardest.

Run: python test_learn_zones.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kevacv.learn_zones import (describe, learn_dwell_zones, learn_entry_zones,
                                to_zone_config, track_endpoints)

FAILED = []
W, H = 1280, 808


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


def walk(tid, x0, y0, x1, y1, t0, dur, fps=7.5):
    """A person walking from (x0,y0) to (x1,y1). -> [(t, tid, box)]"""
    n = max(2, int(dur * fps))
    out = []
    for i in range(n):
        f = i / (n - 1)
        cx, cy = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
        out.append((t0 + i / fps, tid, (cx - 40, cy - 220, cx + 40, cy)))
    return out


def to_log(rows, fps=7.5):
    by = {}
    for t, tid, b in rows:
        by.setdefault(round(t * fps), []).append((tid, *b))
    return [(i, i / fps, by[i]) for i in sorted(by)]


def in_box(pt, poly, pad=90):
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    return (min(xs) - pad <= pt[0] <= max(xs) + pad
            and min(ys) - pad <= pt[1] <= max(ys) + pad)


# ── the real geometry: door at bottom-right, desk at upper-left ─────────────
DOOR = (1120, 760)
DESK = (400, 300)

print("=" * 74)
print("  the door is found from the tracks alone — nothing is drawn")
print("=" * 74)
rows = []
for k in range(14):                       # guests walk in the door, to the desk
    rows += walk(f"g{k}", *DOOR, *DESK, t0=k * 40, dur=9)
    rows += walk(f"x{k}", *DESK, *DOOR, t0=k * 40 + 20, dur=9)   # and back out
log = to_log(rows)
ent, stats = learn_entry_zones(log, frame_wh=(W, H))
check(len(ent) >= 1, "an entrance is proposed", f"{len(ent)}")
check(ent and in_box(DOOR, ent[0]["polygon"]),
      "and it is AT THE REAL DOOR", f"{ent[0]['centre'] if ent else None} vs {DOOR}")
check(ent and not in_box(DESK, ent[0]["polygon"]),
      "not at the desk, where people also stop")
check(ent and ent[0]["at_frame_edge"], "it notices the door is at the frame edge")
check(ent and ent[0]["births"] > 0 and ent[0]["deaths"] > 0,
      "and that it is used in BOTH directions",
      f"births {ent[0]['births']} deaths {ent[0]['deaths']}")
print()
print("   " + describe(ent, stats=stats).replace("\n", "\n   "))

print()
print("=" * 74)
print("  THE DANGEROUS CASE: fragmentation must not invent a door mid-room")
print("=" * 74)
frag = []
for k in range(60):                       # one person, re-born every 2s at the desk
    frag += walk(f"frag{k}", DESK[0], DESK[1], DESK[0] + 6, DESK[1] + 4,
                 t0=k * 2, dur=1.9)
ent2, st2 = learn_entry_zones(to_log(frag), frame_wh=(W, H))
check(not any(in_box(DESK, p["polygon"]) for p in ent2),
      "60 fragments at the desk propose NO entrance there", f"{len(ent2)} proposal(s)")
check(st2["dropped_still"] > 0, "they are rejected for never travelling",
      f"{st2['dropped_still']} dropped as stationary")
print("    -> a fragment that appears and dies on the spot is not a journey")

frag_far = []
for k in range(40):                       # fragments that DO move a little
    frag_far += walk(f"ff{k}", 600, 400, 640, 430, t0=k * 3, dur=2.5)
ent3, _ = learn_entry_zones(to_log(frag_far), frame_wh=(W, H))
check(not any(in_box((620, 415), p["polygon"], pad=10) for p in ent3),
      "short drifting fragments mid-room also propose no door",
      f"{len(ent3)} proposal(s)")

print()
print("=" * 74)
print("  it must say 'cannot tell' rather than invent something")
print("=" * 74)
e, s = learn_entry_zones([], frame_wh=(W, H))
check(e == [], "no frames -> no proposal, no crash")
check("no entrance proposed" in describe(e, stats=s),
      "and describe() says so plainly")
e2, s2 = learn_entry_zones(to_log(walk("solo", *DOOR, *DESK, t0=0, dur=9)),
                           frame_wh=(W, H))
check(len(e2) == 0, "ONE track is not enough evidence for a door", f"{len(e2)}")
txt = describe([], stats={"used": 2, "tracks": 100, "dropped_short": 50,
                          "dropped_still": 48})
check("fragments badly" in txt,
      "it warns when most tracks were unusable",
      "low usable-track ratio is itself the finding")

print()
print("=" * 74)
print("  dwell zones: where people STOP")
print("=" * 74)
stand = []
for k in range(12):                       # queue standing near the desk
    stand += [(k * 3 + i / 7.5, f"w{k}", (DESK[0] - 40, DESK[1] - 220,
                                          DESK[0] + 40, DESK[1]))
              for i in range(40)]
dw, dst = learn_dwell_zones(to_log(stand + rows), frame_wh=(W, H))
check(len(dw) >= 1, "a dwell zone is found", f"{len(dw)}")
check(any(in_box(DESK, p["polygon"]) for p in dw),
      "at the desk, where people actually stand", str([p["centre"] for p in dw]))

print()
print("=" * 74)
print("  the payoff: a zones file you can actually load")
print("=" * 74)
cfg = to_zone_config(ent, dw, frame_wh=(W, H))
check("main_entrance" in cfg["polygons"], "names the busiest door main_entrance")
check(cfg["roles"]["main_entrance"] == ["entry"],
      "with the role the classifier already understands")
check("entry_line" in cfg, "and proposes an entry_line")
(lx1, ly1), (lx2, ly2) = cfg["entry_line"]
ex = [p[0] for p in cfg["polygons"]["main_entrance"]]
ey = [p[1] for p in cfg["polygons"]["main_entrance"]]
span = max(abs(lx2 - lx1), abs(ly2 - ly1))
want = max(max(ex) - min(ex), max(ey) - min(ey))
check(span >= want * 1.5,
      "that OVERSHOOTS the entrance rather than stopping inside it",
      f"line {span}px vs entrance cluster {want}px")
check("_entry_line_note" in cfg, "and says why it overshoots")
print("    -> a too-short line is the exact failure this module exists to stop")
check(cfg["frame_size"] == [W, H], "and records the frame it was learned at",
      "so it rescales correctly on a 4K source")
check("PROPOSALS" in cfg["_generated"],
      "the file says out loud that it is a proposal, not authority")

print()
print("=" * 74)
if FAILED:
    print(f"  {len(FAILED)} FAILED:")
    for f in FAILED:
        print(f"    - {f}")
    sys.exit(1)
print("  ALL PASS")
print("=" * 74)
