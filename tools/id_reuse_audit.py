#!/usr/bin/env python3
"""Does this pipeline REUSE a tracker id for a different person? Measure it.

WHY THIS EXISTS
    supervision 0.26.1 (what we pin) never prunes LineZone.crossing_state_history
    -- it is a defaultdict that grows for the life of the run. supervision 0.31
    adds _evict_stale_crossing_history, which drops a tracker's history after
    crossing_history_length consecutive absent frames.

    That difference only matters if ids are actually reused. If they are:

        id 30 dies. Its crossing_state_history deque survives.
        id 30 is reused for a DIFFERENT person somewhere else in frame.
        Their first state appends onto the dead person's history, the states
        differ, the unique-oldest rule passes -> a crossing that never happened.

    tools/slit_count.py's docstring claims this happens here ("id 30 sweeps
    x 448..1754 inside one 13s window"). This turns that claim into a number,
    so the 0.31 upgrade is decided by evidence rather than by whoever argues
    harder. A zero here KILLS the upgrade's stated payoff.

WHAT COUNTS AS EVIDENCE
    A gap alone is not reuse -- a tracker legitimately coasts through an
    occlusion and returns where it left off. Reuse is a gap PLUS a jump: the
    id reappears somewhere a person could not have walked to. Both are
    reported, because only the second is a false-crossing risk.

USAGE
    python tools/id_reuse_audit.py output/<run>/debug/CAM.112_frames.json.gz
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict

# supervision's LineZone keeps max(2, minimum_crossing_threshold + 1) states.
# Below that many absent frames 0.31 would ALSO have kept the history, so the
# two versions agree and the gap is not evidence of anything.
DEFAULT_HISTORY_FRAMES = 2
# A person cannot teleport. Anything past this between a death and a rebirth
# of the same id is a different body wearing the same number.
JUMP_PX = 200.0


def load(path):
    """-> {track_id: [(frame_index, centre_x, centre_y), ...]} sorted by frame."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        rows = json.load(fh)
    tracks = defaultdict(list)
    for frame_index, _t, dets in rows:
        for tid, x1, y1, x2, y2 in dets:
            tracks[tid].append((frame_index, (x1 + x2) / 2.0, (y1 + y2) / 2.0))
    for v in tracks.values():
        v.sort()
    return tracks, len(rows)


def audit(tracks, history_frames=DEFAULT_HISTORY_FRAMES, jump_px=JUMP_PX):
    """-> (summary dict, [worst offenders]).

    A "stale gap" is one long enough that supervision 0.31 would have evicted
    the crossing history while 0.26.1 keeps it. A "reuse" is a stale gap whose
    reappearance is further than jump_px from the disappearance.
    """
    stale_gaps = reuses = 0
    ids_with_stale_gap, ids_with_reuse = set(), set()
    offenders = []
    for tid, pts in tracks.items():
        for (fa, xa, ya), (fb, xb, yb) in zip(pts, pts[1:]):
            gap = fb - fa - 1
            if gap < history_frames:
                continue
            stale_gaps += 1
            ids_with_stale_gap.add(tid)
            dist = ((xb - xa) ** 2 + (yb - ya) ** 2) ** 0.5
            if dist > jump_px:
                reuses += 1
                ids_with_reuse.add(tid)
                offenders.append((dist, tid, fa, fb, gap))
    offenders.sort(reverse=True)
    return {
        "tracks": len(tracks),
        "stale_gaps": stale_gaps,
        "ids_with_stale_gap": len(ids_with_stale_gap),
        "reuses": reuses,
        "ids_with_reuse": len(ids_with_reuse),
    }, offenders[:10]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    path = sys.argv[1]
    tracks, n_frames = load(path)
    summary, offenders = audit(tracks)

    print(f"source  {path}")
    print(f"frames  {n_frames}   tracks {summary['tracks']}")
    print(f"history window used: {DEFAULT_HISTORY_FRAMES} frames "
          f"(supervision 0.31 evicts past this; 0.26.1 never does)")
    print()
    print(f"  gaps >= history window          {summary['stale_gaps']:>6}"
          f"   across {summary['ids_with_stale_gap']} ids")
    print(f"  of those, REAPPEAR > {JUMP_PX:.0f}px away  {summary['reuses']:>6}"
          f"   across {summary['ids_with_reuse']} ids   <-- false-crossing risk")
    print()
    if offenders:
        print("  worst offenders (a person cannot walk this far while absent):")
        print(f"    {'jump px':>8} {'id':>6} {'died@f':>8} {'reborn@f':>9} {'gap':>5}")
        for dist, tid, fa, fb, gap in offenders:
            print(f"    {dist:>8.0f} {tid:>6} {fa:>8} {fb:>9} {gap:>5}")
    print()
    if summary["reuses"] == 0:
        print("VERDICT: no id reuse. supervision 0.31's eviction fix buys NOTHING")
        print("         here -- drop it from the upgrade's justification.")
    else:
        print("VERDICT: ids ARE reused for different bodies. In supervision 0.26.1")
        print("         each reuse inherits the dead track's crossing history and")
        print("         can emit a crossing that never happened.")
    return 0


def _selftest():
    """One runnable check: a coast-and-return is not reuse; a teleport is."""
    coast = {1: [(0, 100.0, 100.0), (10, 105.0, 100.0)]}
    s, _ = audit(coast)
    assert s["stale_gaps"] == 1 and s["reuses"] == 0, s

    teleport = {2: [(0, 100.0, 100.0), (10, 900.0, 100.0)]}
    s, off = audit(teleport)
    assert s["reuses"] == 1 and off[0][1] == 2, (s, off)

    adjacent = {3: [(0, 100.0, 100.0), (1, 900.0, 100.0)]}
    s, _ = audit(adjacent)
    assert s["stale_gaps"] == 0, "no gap -> both versions agree -> not evidence"
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        raise SystemExit(main())
