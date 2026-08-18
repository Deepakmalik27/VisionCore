"""Tests for kevacv/merge_ab.py — measure the topology veto before shipping it.

Built from run 68b97311f9: 69 merges accepted, 356 starved by window overlap.
The hypothesis under test is that removing IMPOSSIBLE pairs early reduces
window pollution and frees genuinely-good merges. These tests check the
harness measures that faithfully — including reproducing the greedy flaw it
exists to expose.

Run: python test_merge_ab.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kevacv.merge_ab import (ab_topology, apply_topology_veto, assert_matches,
                             describe, greedy_union, windows_overlap)

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


WH = (1280, 720)
DOOR = (1150, 600)
DOORS = [DOOR]
MID = (400, 400)

print("=" * 74)
print("  windows_overlap matches the notebook exactly")
print("=" * 74)
check(windows_overlap([(0, 100)], [(50, 150)]), "plainly overlapping windows")
check(not windows_overlap([(0, 100)], [(100, 200)]), "touching is not overlapping")
check(not windows_overlap([(0, 100)], [(99, 200)]),
      "within the 2s tolerance is not overlapping", "tolerance_s=2")
check(windows_overlap([(0, 100)], [(90, 200)]), "3s of real overlap is")
check(not windows_overlap([], [(0, 10)]), "empty window set never overlaps")

print()
print("=" * 74)
print("  the greedy union is reproduced, flaws included")
print("=" * 74)
W = {1: (0, 10), 2: (20, 30), 3: (40, 50)}
r = greedy_union([(0.9, 1, 2, "hand-off"), (0.8, 2, 3, "gallery")], W)
check(r["n_accepted"] == 2 and r["n_identities"] == 1,
      "a transitive chain collapses to one identity", str(r["n_identities"]))
check(r["tier_counts"] == {"hand-off": 1, "gallery": 1}, "tiers are counted")

# the poisoning: a strong late pair starved because an early union widened
# the group's window across it
# 1+2 merge legitimately (no overlap), which widens group 1's window to span
# 40-50; the later 1<->3 pair then collides with that WIDENED window even
# though tracks 1 and 3 never overlapped each other.
W2 = {1: (0, 10), 2: (40, 50), 3: (45, 55)}
r = greedy_union([(0.95, 1, 2, "gallery"), (0.92, 1, 3, "stationary")], W2)
check(r["n_accepted"] == 1 and r["n_overlap_blocked"] == 1,
      "an early union starves a strong later pair — the real failure mode",
      f"{r['n_accepted']} accepted / {r['n_overlap_blocked']} starved")
check(r["overlap_blocked"][0][3] == "stationary",
      "and the starved one is the physical-evidence pair, as observed in the run")

check(greedy_union([], W)["n_identities"] == 3, "no pairs -> nothing merges")
check(greedy_union([(0.9, 99, 98, "x")], W)["n_accepted"] == 0,
      "pairs referencing unknown tracks are skipped, not crashed")
r = greedy_union([(0.9, 1, 2, "a"), (0.8, 2, 1, "b")], W)
check(r["n_accepted"] == 1, "a duplicate pair in the other order is a no-op")

print()
print("=" * 74)
print("  the veto removes impossible pairs and nothing else")
print("=" * 74)
W3 = {1: (0, 10), 2: (300, 310), 3: (600, 610)}
POS = {1: (MID, DOOR),        # ends at the door
       2: (MID, MID),         # appears mid-room  -> impossible after door exit
       3: (DOOR, DOOR)}       # appears at the door -> plausible return
pairs = [(0.9, 1, 2, "gallery"), (0.9, 1, 3, "gallery")]
possible, vetoed = apply_topology_veto(pairs, W3, POS, DOORS, WH)
check(len(vetoed) == 1 and vetoed[0][2] == 2,
      "door-exit -> mid-room birth is vetoed", str(len(vetoed)))
check(len(possible) == 1 and possible[0][2] == 3,
      "door-exit -> door birth survives")

_, v_nodoors = apply_topology_veto(pairs, W3, POS, [], WH)
check(len(v_nodoors) == 0, "no doors -> nothing vetoed (abstain)")
_, v_nopos = apply_topology_veto(pairs, W3, {}, DOORS, WH)
check(len(v_nopos) == 0, "no positions -> nothing vetoed (cannot judge)")

print()
print("=" * 74)
print("  A/B reports the delta the run actually needs")
print("=" * 74)
# an impossible pair wins on evidence, unions first, and its widened window
# then starves a legitimate later merge. Exactly the 356-vs-69 shape.
W4 = {1: (0, 10), 2: (100, 400), 3: (200, 210)}
POS4 = {1: (MID, DOOR), 2: (MID, MID), 3: (DOOR, DOOR)}
pairs4 = [(0.95, 1, 2, "gallery"),      # impossible: door exit -> mid-room
          (0.90, 1, 3, "hand-off")]     # legitimate: door -> door
ab = ab_topology(pairs4, W4, POS4, DOORS, WH)
check(ab["without"]["n_accepted"] == 1 and ab["without"]["n_overlap_blocked"] == 1,
      "WITHOUT the veto the impossible pair wins and starves the good one")
check(ab["with"]["n_accepted"] == 1 and ab["with"]["n_overlap_blocked"] == 0,
      "WITH the veto the good merge lands instead")
check(ab["n_vetoed"] == 1, "one pair vetoed")
check(ab["delta"]["overlap_blocked"] == -1,
      "the delta shows starvation falling", str(ab["delta"]["overlap_blocked"]))
check(ab["with"]["accepted"][0][3] == "hand-off",
      "and the surviving merge is the physically-evidenced one")
txt = describe(ab)
check("freed starved merges" in txt, "describe() reads the delta correctly")
print("\n".join("   " + l for l in txt.split("\n")))

print()
print("=" * 74)
print("  it warns when the veto HURTS, instead of claiming success")
print("=" * 74)
# A genuine door-to-door return, but the door map points at the wrong corner.
# Note the drift must exceed the still-radius: a person who reappears in
# almost the same spot is covered by occlusion_recovery and survives even a
# wrong door map, which is a real robustness property of the gate.
W5 = {1: (0, 10), 3: (200, 210)}
POS5 = {1: (MID, (1150, 600)), 3: ((1250, 650), MID)}   # door -> door, 112px
BADDOOR = [(10, 10)]
ab_ok = ab_topology([(0.9, 1, 3, "hand-off")], W5, POS5, DOORS, WH)
check(ab_ok["n_vetoed"] == 0, "with the RIGHT doors the return is allowed")

ab_bad = ab_topology([(0.9, 1, 3, "hand-off")], W5, POS5, BADDOOR, WH)
check(ab_bad["delta"]["accepted"] < 0, "a bad door map removes real merges",
      str(ab_bad["delta"]["accepted"]))
check("Check the door positions" in describe(ab_bad),
      "and the summary says to check the doors, not to ship it")

# the robustness property, stated as its own claim
POS6 = {1: (MID, (1150, 600)), 3: ((1155, 605), MID)}   # same spot, 7px
ab_still = ab_topology([(0.9, 1, 3, "hand-off")], W5, POS6, BADDOOR, WH)
check(ab_still["n_vetoed"] == 0,
      "a reappearance in the SAME SPOT survives even a wrong door map",
      "occlusion_recovery does not depend on knowing where the doors are")

print()
print("=" * 74)
print("  the copied loop cannot silently drift from the notebook")
print("=" * 74)
nb = Path("notebooks/pipeline.ipynb")
if nb.exists():
    import json
    src = "".join("".join(c["source"]) for c in json.loads(
        nb.read_text(encoding="utf-8"))["cells"])
    ok, notes = assert_matches(src)
    check(ok, "the notebook's union loop still matches this copy",
          "; ".join(notes) if notes else "4/4 anchors present")
else:
    print("  SKIP  notebook not found — staleness guard not checked")

ok, notes = assert_matches("nothing like the real loop")
check(not ok and len(notes) == 4, "and the guard fails loudly when it drifts",
      f"{len(notes)} anchors missing")

print()
print("=" * 74)
if FAILED:
    print(f"  {len(FAILED)} FAILED:")
    for f in FAILED:
        print(f"    - {f}")
    sys.exit(1)
print("  ALL PASS")
print("=" * 74)
