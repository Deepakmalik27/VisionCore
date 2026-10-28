#!/usr/bin/env python3
"""Refuse to score against ground truth that isn't ground truth.

WHY THIS EXISTS — twice in one day, on the same file
    gt.txt has 600 rows across 100 frames and only 14 UNIQUE BOXES. Three of
    its six people never move a single pixel in 13 seconds. It is a
    copy-forward artefact of the labelling tool, not a labelled sequence.

    It was caught once, a recall figure and a foot-anchor change were
    retracted over it -- and then it was used again hours later to grade a
    detector A/B, producing "HOTA 0.4762, +107%, clears the target floor".
    All of that was withdrawn. An IoU sweep showed the real gap was ~10 points
    of coverage, not 6x.

    A bad reference does not look like an error. It looks like a result. So
    the check has to be automatic and it has to run BEFORE the score, not
    after somebody notices.

WHAT IT CHECKS
    copy-forward    unique boxes vs rows, and per-track motion. A person
                    standing still still SWAYS -- a box that is identical to
                    the pixel across seconds was propagated, not observed.
    frozen tracks   any track whose box never changes at all.
    box shape       a standing adult is roughly 2-3x taller than wide. Median
                    h/w far outside that means the boxes describe something
                    other than upright people (gt.txt reports 1.14 -- nearly
                    square -- which is why it cannot adjudicate box shape
                    either).
    sample size     scoring a handful of frames yields numbers that move on
                    noise.
    entry events    a sequence with no track entering or leaving cannot score
                    IN/OUT, however good it is at detection. quick100 has
                    ZERO, which is why counting has never been graded.

EXIT CODE
    0 usable, 1 usable with warnings, 2 REFUSED.
"""
from __future__ import annotations
import sys, collections
from pathlib import Path

MIN_FRAMES = 30
MIN_UNIQUE_FRAC = 0.30      # unique boxes / rows
MAX_FROZEN_FRAC = 0.25      # share of tracks that never move
HW_LO, HW_HI = 1.5, 4.5     # plausible h/w for an upright person


def load_rows(path):
    out = []
    for ln in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        p = ln.strip().split(",")
        if len(p) >= 6:
            try:
                out.append((int(float(p[0])), str(p[1]),
                            float(p[2]), float(p[3]), float(p[4]), float(p[5])))
            except ValueError:
                continue
    return out


def validate(path, need_entries=False):
    rows = load_rows(path)
    fail, warn, info = [], [], []
    if not rows:
        return 2, [f"{path}: no parseable MOT rows"], [], []

    frames = sorted({r[0] for r in rows})
    tracks = collections.defaultdict(list)
    for r in rows:
        tracks[r[1]].append(r)
    uniq = {(r[2], r[3], r[4], r[5]) for r in rows}
    info.append(f"{len(rows)} rows · {len(frames)} frames · {len(tracks)} tracks "
                f"· {len(uniq)} unique boxes")

    # --- copy-forward -----------------------------------------------------
    frac = len(uniq) / len(rows)
    if frac < MIN_UNIQUE_FRAC:
        fail.append(
            f"COPY-FORWARD: only {len(uniq)} unique boxes across {len(rows)} "
            f"rows ({frac:.0%}). Labels were propagated, not observed. "
            f"Scores from this file grade a frozen scene.")

    frozen = []
    for tid, rs in tracks.items():
        rs.sort()
        moved = sum(1 for a, b in zip(rs, rs[1:])
                    if (a[2], a[3], a[4], a[5]) != (b[2], b[3], b[4], b[5]))
        if len(rs) >= 5 and moved == 0:
            frozen.append(tid)
    if frozen:
        ffrac = len(frozen) / len(tracks)
        msg = (f"FROZEN TRACKS: {len(frozen)} of {len(tracks)} never move at "
               f"all ({', '.join(map(str, frozen[:6]))}). A real person sways.")
        (fail if ffrac > MAX_FROZEN_FRAC else warn).append(msg)

    # --- box shape --------------------------------------------------------
    hw = sorted(r[5] / max(r[4], 1e-6) for r in rows if r[5] > 0)
    if hw:
        med = hw[len(hw) // 2]
        info.append(f"median box h/w {med:.2f}")
        if not (HW_LO <= med <= HW_HI):
            warn.append(
                f"BOX SHAPE: median h/w {med:.2f} is outside {HW_LO}-{HW_HI}, "
                f"the plausible range for upright people. This reference "
                f"cannot adjudicate box shape -- do not use it to argue that "
                f"predictions are 'too tall' or 'too wide'.")

    # --- sample size ------------------------------------------------------
    if len(frames) < MIN_FRAMES:
        warn.append(f"SMALL: {len(frames)} frames. Differences of a few "
                    f"percent will be noise.")

    # --- entry events -----------------------------------------------------
    first_last = {tid: (min(r[0] for r in rs), max(r[0] for r in rs))
                  for tid, rs in tracks.items()}
    lo, hi = frames[0], frames[-1]
    entering = [t for t, (a, _b) in first_last.items() if a > lo]
    leaving = [t for t, (_a, b) in first_last.items() if b < hi]
    info.append(f"tracks appearing after the first frame: {len(entering)} · "
                f"disappearing before the last: {len(leaving)}")
    if not entering and not leaving:
        msg = ("NO ENTRY/EXIT EVENTS: every track spans the whole clip. This "
               "file can score detection and association, and CANNOT score "
               "IN/OUT counting at all.")
        (fail if need_entries else warn).append(msg)

    code = 2 if fail else (1 if warn else 0)
    return code, fail, warn, info


def main(argv):
    if not argv:
        print(__doc__.strip().splitlines()[0])
        print("usage: gt_validate.py <gt.txt> [--need-entries]")
        return 2
    need = "--need-entries" in argv
    path = argv[0]
    code, fail, warn, info = validate(path, need_entries=need)
    print(f"── ground truth check: {path}")
    for i in info:
        print(f"   {i}")
    for f in fail:
        print(f"   ✗ {f}")
    for w in warn:
        print(f"   ! {w}")
    print("   " + {0: "USABLE", 1: "USABLE WITH WARNINGS",
                   2: "REFUSED — do not score against this"}[code])
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
