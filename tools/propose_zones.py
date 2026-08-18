"""propose_zones.py — find the door without drawing it. Runs LOCALLY on the
chunk artifacts a night run leaves behind; no Kaggle time, no notebook change.

The failure this closes: a hand-drawn entry line 200px too short fired ZERO
times over a full real hour and every guest metric silently read 0. The
kevacv.learn_zones module (built + tested, never wired anywhere) proposes
entry and dwell polygons from where tracks are BORN, DIE and STOP — evidence
attached, nothing overwritten. This script runs it against your hand-drawn
zones so a bad line is a five-second visual check instead of a wasted hour.

Usage:
    python propose_zones.py <chunk_NN_frames.json.gz> [zones_<stem>.json]
    python propose_zones.py poc_output/chunk_events/          # every chunk
"""
import gzip
import json
import sys
from pathlib import Path

# the package lives one level UP from tools/, so inserting HERE put
# tools/ on the path and `import kevacv` failed — this script has
# never been runnable from a clean checkout.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kevacv.learn_zones import (describe, learn_dwell_zones,
                                learn_entry_zones, to_zone_config)


def _load_frame_log(path):
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt") as fh:
        return [(fi, t, [tuple(b) for b in bx]) for fi, t, bx in json.load(fh)]


def _frame_wh(frame_log):
    xs = [b[3] for _f, _t, bs in frame_log for b in bs]
    ys = [b[4] for _f, _t, bs in frame_log for b in bs]
    return (max(xs) if xs else 1280, max(ys) if ys else 720)


def propose(flog_path, zones_path=None):
    print("=" * 72)
    print(f"  {flog_path.name}")
    print("=" * 72)
    flog = _load_frame_log(flog_path)
    n_boxes = sum(len(bs) for _f, _t, bs in flog)
    if n_boxes < 200:
        print(f"  only {n_boxes} boxes logged — too little traffic to learn "
              f"zones from this chunk. Try a busier one.")
        return
    wh = _frame_wh(flog)
    entries, e_stats = learn_entry_zones(flog, frame_wh=wh)
    dwells, _ = learn_dwell_zones(flog, frame_wh=wh)
    print(describe(entries, dwells, e_stats))

    if zones_path and Path(zones_path).exists():
        hand = json.loads(Path(zones_path).read_text())
        print(f"\n  YOUR HAND-DRAWN ZONES ({Path(zones_path).name}):")
        for z in hand.get("zones", []):
            xs = [p[0] for p in z.get("polygon", [])]
            ys = [p[1] for p in z.get("polygon", [])]
            if xs:
                print(f"    {z.get('name', '?'):16s} bbox "
                      f"x {min(xs):.0f}-{max(xs):.0f}  y {min(ys):.0f}-{max(ys):.0f}")
        el = hand.get("entry_line") or hand.get("entry_lines")
        print(f"    entry line: {el if el else 'MISSING'}")
        print("\n  COMPARE: if the top proposed entry region does not overlap "
              "your entry line's\n  span, the line is in the wrong place — "
              "that is the exact failure that cost a full hour.")

    out = flog_path.with_name(flog_path.stem.replace(".json", "")
                              + "_proposed_zones.json")
    out.write_text(json.dumps(to_zone_config(entries, dwells, frame_wh=wh),
                              indent=2))
    print(f"\n  editable proposal -> {out.name}  (load it in zone_mapper.html "
          f"to accept/adjust — nothing overwrites your zones file)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    target = Path(sys.argv[1])
    zones = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    files = (sorted(target.glob("*_frames.json.gz")) if target.is_dir()
             else [target])
    if not files:
        sys.exit(f"no *_frames.json.gz under {target}")
    for f in files:
        propose(f, zones)
