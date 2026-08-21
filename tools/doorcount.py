"""Door-local counting: count the way an eye does.

WHY THIS EXISTS
    The pipeline counts an arrival only when ONE identity survives from the
    entry zone all the way to an interior zone. On CAM.112 identity fragments
    (69 fragments for 18 people, and the camera flips colour<->IR every few
    seconds, which wrecks appearance ReID), so the person is seen perfectly
    and counted zero times.

    An eye does not do that. It watches the door for ~3 seconds and says
    "someone came in". Counting at a door is a LOCAL, SHORT-HORIZON task; it
    does not need to know who the person is or follow them across the room.

    Measured at the door (x>=1150, y>=380), 300-345s:
        by eye              ~9-12 people
        distinct track ids   6
        pipeline arrivals    4   -- for the whole 600s, not 45s

THE RULE
    A track that appears in the door band and whose x DECREASES by more than
    `min_travel` px (door is on the right, lobby on the left) has come in.
    Increasing x has gone out. Short tracks are fine: 3 seconds is enough,
    which is the point.

    Dedupe merges arrivals within `dedupe_s` that start within `dedupe_px`,
    so one person fragmenting into two ids at the threshold counts once.
"""
import json, gzip, collections, sys

DOOR_X, DOOR_Y = 1150, 380
MIN_TRAVEL = 120.0
DEDUPE_S, DEDUPE_PX = 2.5, 180.0


def _spans(frames):
    """track_id -> (t_first, t_last), for the co-visibility test."""
    sp = {}
    for f in frames:
        t = f[1]
        for b in f[2]:
            a = sp.get(b[0])
            sp[b[0]] = (t, t) if a is None else (a[0], t)
    return sp


def door_events(frames, door_x=DOOR_X, door_y=DOOR_Y, min_travel=MIN_TRAVEL,
                gap_s=3.0):
    """Every time ANY track crosses INTO the door band moving inward.

    The first version fired once per track, on its first-ever band sighting.
    That assumes one track == one person == one arrival, and track ids on this
    camera are reused: measured on the held-out window 221-234s, three real
    guests walked in and were absorbed into long-lived ids 3 and 30 (whose
    door events sat at t=10.7 and t=137.8, over three minutes earlier). Id 30
    spans x 448..1754 inside that window -- one id sweeping the whole frame.
    Recall on that window was 0.

    So an event is a TRANSITION: outside the band -> inside it, with inward
    travel afterwards. A track that leaves and re-enters produces two, which
    is correct -- it either really did come back, or the id was recycled onto
    somebody else. Re-entries closer than gap_s are treated as box jitter on
    the band edge, not a new arrival.
    """
    tr = collections.defaultdict(list)
    for f in frames:
        t = f[1]
        for b in f[2]:
            tr[b[0]].append((t, (b[1]+b[3])/2.0, float(b[4])))
    evs = []
    for tid, pts in tr.items():
        pts.sort()
        inside = [p[1] >= door_x and p[2] >= door_y for p in pts]
        for i, isin in enumerate(inside):
            if not isin or (i and inside[i - 1]):
                continue                       # not an entry transition
            t_in, x_in = pts[i][0], pts[i][1]
            if evs and evs[-1]["tid"] == tid and t_in - evs[-1]["t"] < gap_s:
                continue                       # jitter on the band edge
            # where did this track go over the next stretch?
            after = [p for p in pts[i:]]
            x_last = after[-1][1]
            travel = x_in - x_last
            if abs(travel) < min_travel:
                continue
            evs.append({"t": t_in, "tid": tid, "x": x_in, "y": pts[i][2],
                        "dir": "IN" if travel > 0 else "OUT"})
    return sorted(evs, key=lambda e: e["t"])


def dedupe(evs, dedupe_s=DEDUPE_S, dedupe_px=DEDUPE_PX, spans=None,
           overlap_s=0.4):
    """Fold ONE person's fragments into one count -- without folding a GROUP.

    CO-VISIBILITY: two tracks that are visible AT THE SAME MOMENT cannot be
    the same person, so they must never be deduped together however close in
    time and space they are. The codebase already relies on this principle
    for Re-ID (ENABLE_COVISIBILITY_BLOCK); without it here, a party walking
    in abreast is indistinguishable from one person's id churn.

    Measured on eval/gt_entries_305_318.json: six guests entered as a group,
    ids 135/138/144 starting within seconds of each other at x=1739/1785/1696.
    Time-and-distance dedupe alone collapsed them to one and the window
    counted 2 of 6.
    """
    kept = []
    for e in evs:
        dup = False
        for k in reversed(kept[-6:]):
            if e["dir"] != k["dir"] or e["t"] - k["t"] > dedupe_s:
                continue
            if abs(e["x"] - k["x"]) > dedupe_px:
                continue
            if spans is not None:
                a, b = spans.get(e["tid"]), spans.get(k["tid"])
                if a and b and min(a[1], b[1]) - max(a[0], b[0]) > overlap_s:
                    continue          # co-visible -> two different people
            dup = True
            break
        if not dup:
            kept.append(e)
    return kept


if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else "deskE"
    frames = json.load(gzip.open(f"output/{run}/debug/CAM.112_frames.json.gz", "rt"))
    raw = door_events(frames)
    kept = dedupe(raw, spans=_spans(frames))
    ins = [e for e in kept if e["dir"] == "IN"]
    outs = [e for e in kept if e["dir"] == "OUT"]
    print(f"run={run}   raw door events {len(raw)} -> deduped {len(kept)}")
    print(f"   IN  {len(ins)}")
    print(f"   OUT {len(outs)}")
    w = [e for e in kept if 300 <= e['t'] <= 345]
    print(f"\nin the hand-checked window 300-345s: {len(w)} event(s) "
          f"(IN {sum(1 for e in w if e['dir']=='IN')}, "
          f"OUT {sum(1 for e in w if e['dir']=='OUT')})")
    for e in w:
        print(f"   t={e['t']:6.1f}  id={e['tid']:>5}  {e['dir']}")

    # self-check
    assert len(kept) <= len(raw)
    fake = [{"t": 10.0, "tid": 1, "x": 1600, "y": 900, "dir": "IN"},
            {"t": 10.5, "tid": 2, "x": 1620, "y": 905, "dir": "IN"}]
    # sequential ids (one person, track re-minted) -> one count
    seq = {1: (8.0, 10.2), 2: (10.4, 12.0)}
    assert len(dedupe(fake, spans=seq)) == 1, "one person as two ids must count once"
    # OVERLAPPING ids (two people side by side) -> two counts
    both = {1: (8.0, 12.0), 2: (9.0, 13.0)}
    assert len(dedupe(fake, spans=both)) == 2, "co-visible tracks are two people"
    assert len(dedupe(fake)) == 1, "without spans, old behaviour is unchanged"
    print("\nself-check ok: fragments merge, co-visible people do not")
