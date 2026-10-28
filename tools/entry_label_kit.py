#!/usr/bin/env python3
"""Build, validate and hold out ENTRY-EVENT ground truth.

WHY THIS EXISTS
    Everything in this repo scores DETECTION and ASSOCIATION. Nothing scores
    the number the venue actually wants: how many people came in and out.
    label_pkg/quick100 contains ZERO entry events -- every track spans the
    whole clip -- so IN/OUT has never been graded even once.

    The three windows that do exist were read by eye in one sitting, and they
    hid a regression: the ONLY window containing a GROUP arrival was the one
    used for tuning, so held-out recall read a clean 100%/0FP while four real
    guests were being deleted from the group. A blind spot in the reference is
    indistinguishable from success.

THE RULES THIS ENFORCES
    1. Windows are CHOSEN BY DETECTOR ACTIVITY, before anyone looks at the
       footage. You cannot cherry-pick a window you have already read.
    2. Every batch must contain quiet windows (truth 0). Without them a
       counter that over-fires scores perfectly.
    3. At least one GROUP window must be HELD OUT, not spent on tuning.
    4. Held-out windows are sealed: `score` refuses to report them until
       `unseal` is called, and unsealing is recorded in the file.

USAGE
    plan    <run_dir> --n 8            pick windows by activity, write a plan
    sheets  <plan.json> --video V      render a contact sheet per window
    score   <plan.json> <events.json>  score a counter against the labels
"""
from __future__ import annotations
import argparse, json, collections, gzip, sys
from pathlib import Path

WINDOW_S = 13.0
DOOR_X, DOOR_Y = 1150, 380


def _load_frames(run_dir, camera="CAM.112"):
    p = next(Path(run_dir).rglob(f"{camera}_frames.json.gz"), None)
    if p is None:
        p = next(Path(run_dir).rglob("*_frames.json.gz"), None)
    if p is None:
        sys.exit(f"no *_frames.json.gz under {run_dir}")
    return json.load(gzip.open(p, "rt"))


def plan(run_dir, n=8, window_s=WINDOW_S, door_x=DOOR_X, door_y=DOOR_Y,
         out="eval/entry_plan.json", camera="CAM.112"):
    """Pick windows by DOOR ACTIVITY ONLY -- never by looking at the video."""
    frames = _load_frames(run_dir, camera)
    per_s = collections.defaultdict(set)
    tmax = 0.0
    for f in frames:
        t = f[1]; tmax = max(tmax, t)
        for b in f[2]:
            if (b[1] + b[3]) / 2.0 >= door_x and float(b[4]) >= door_y:
                per_s[int(t)].add(b[0])
    step = int(window_s)
    scored = []
    for s in range(0, int(tmax) - step, step):
        ids = set()
        for k in range(s, s + step):
            ids |= per_s.get(k, set())
        scored.append((len(ids), float(s), float(s + step)))
    scored.sort()
    quiet = [w for w in scored if w[0] == 0][:max(2, n // 4)]
    # busy must EXCLUDE what quiet already took. It used to slice the tail of
    # `scored` blindly, so on a short clip -- where the tail still contains the
    # zero-activity windows -- the same window landed in both lists and the
    # plan emitted it twice. A duplicated window is not cosmetic: it is counted
    # twice in the quiet/held-out tallies that `check` uses to decide whether
    # the reference is usable, and twice again in `score`.
    _taken = {w[1:] for w in quiet}
    busy = [w for w in scored if w[1:] not in _taken][-(n - len(quiet)):] \
        if n > len(quiet) else []
    chosen = sorted(quiet + busy, key=lambda w: w[1])

    # HELD-OUT SPLIT -- the part that actually decides whether a score means
    # anything. Roughly half is sealed, and the split is deliberate, not
    # random:
    #   * the BUSIEST window is always sealed. It is the one most likely to
    #     contain a group arrival, and the group case is exactly what hid a
    #     regression last time: the only group window had been tuned on, so
    #     held-out read 100% while four real guests were being deleted.
    #   * at least one QUIET window is sealed. Without a truth-0 window in the
    #     held-out set, a counter that over-fires scores perfectly.
    #   * the rest alternate, so held-out is not just "the busy ones".
    _quiet = [w for w in chosen if w[0] == 0]
    _loud = [w for w in chosen if w[0] > 0]
    _held = set()
    if _loud:
        _held.add(max(_loud, key=lambda w: w[0])[1:])        # busiest
    if _quiet:
        _held.add(_quiet[0][1:])                             # one quiet
    for i, w in enumerate(sorted(_loud, key=lambda w: -w[0])):
        if len(_held) >= max(2, len(chosen) // 2):
            break
        if i % 2 == 1:
            _held.add(w[1:])

    windows = []
    for act, a, b in chosen:
        _is_held = (a, b) in _held
        windows.append({
            "t0": a, "t1": b, "door_track_ids": act,
            "kind": "quiet" if act == 0 else ("busy" if act >= 3 else "light"),
            "held_out": _is_held,
            "sealed": _is_held,
            "truth": None, "entries": [], "note": ""})
    doc = {"_comment": ("Entry-event ground truth. Windows chosen by DETECTOR "
                        "ACTIVITY at the door BEFORE anyone watched the "
                        "footage, so they cannot be cherry-picked. Fill "
                        "`truth` and `entries` by reading the contact sheet."),
           "camera": camera, "source_run": str(run_dir),
           "window_s": window_s, "door_band": [door_x, door_y],
           "windows": sorted(windows, key=lambda w: w["t0"])}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    nq = sum(1 for w in windows if w["kind"] == "quiet")
    nh = sum(1 for w in windows if w["held_out"])
    print(f"  {len(windows)} windows -> {out}")
    print(f"     quiet (truth must be 0): {nq}   held-out/sealed: {nh}")
    for w in doc["windows"]:
        print(f"     {w['t0']:7.0f}-{w['t1']:<7.0f} {w['kind']:>5} "
              f"door_ids={w['door_track_ids']:<3}"
              f"{'  [SEALED]' if w['sealed'] else ''}")
    return doc


def check(plan_path):
    """Refuse a label set that cannot detect the failures we have already had."""
    doc = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    ws = doc["windows"]
    labelled = [w for w in ws if w.get("truth") is not None]
    held = [w for w in labelled if w.get("held_out")]
    fail, warn = [], []
    if not labelled:
        fail.append("nothing is labelled yet — fill `truth` per window")
    if not any(w.get("truth") == 0 for w in labelled):
        fail.append("NO QUIET WINDOW: without a window whose truth is 0, a "
                    "counter that over-fires scores perfectly.")
    if not held:
        fail.append("NOTHING HELD OUT: tuning on everything means the score "
                    "measures the tuning, not the counter.")
    if not any((w.get("truth") or 0) >= 3 for w in held):
        fail.append("NO GROUP WINDOW HELD OUT: the regression that halved "
                    "recall on a 6-person party was invisible precisely "
                    "because the only group window had been tuned on.")
    if any(w.get("sealed") for w in held):
        warn.append(f"{sum(1 for w in held if w.get('sealed'))} held-out "
                    f"window(s) still SEALED — `unseal` before scoring.")
    return (2 if fail else (1 if warn else 0)), fail, warn, doc


def score(plan_path, events_path, tol_s=1.5, allow_sealed=False):
    code, fail, warn, doc = check(plan_path)
    for f in fail:
        print(f"   ✗ {f}")
    for w in warn:
        print(f"   ! {w}")
    if code == 2:
        print("   REFUSED: this label set cannot detect the failures we have "
              "already had.")
        return 2
    evs = json.loads(Path(events_path).read_text(encoding="utf-8"))
    ins = [e for e in evs if str(e.get("dir", "IN")).upper() == "IN"]
    print(f"\n  {'window':>16} {'kind':>6} {'truth':>6} {'counted':>8} "
          f"{'err':>5}   set")
    rows = []
    for w in sorted(doc["windows"], key=lambda x: x["t0"]):
        if w.get("truth") is None:
            continue
        if w.get("sealed") and not allow_sealed:
            print(f"  {w['t0']:6.0f}-{w['t1']:<8.0f} {w['kind']:>6} "
                  f"{'':>6} {'':>8} {'':>5}   SEALED (not scored)")
            continue
        got = sum(1 for e in ins if w["t0"] <= float(e["t"]) <= w["t1"])
        rows.append((w, got))
        print(f"  {w['t0']:6.0f}-{w['t1']:<8.0f} {w['kind']:>6} "
              f"{w['truth']:>6} {got:>8} {got - w['truth']:>+5}   "
              f"{'held-out' if w.get('held_out') else 'TUNED (not a score)'}")
    ho = [(w, g) for w, g in rows if w.get("held_out")]
    tuned = [(w, g) for w, g in rows if not w.get("held_out")]
    if ho:
        hits = sum(min(g, w["truth"]) for w, g in ho)
        truth = sum(w["truth"] for w, g in ho)
        fp = sum(max(0, g - w["truth"]) for w, g in ho)
        print(f"\n  HELD-OUT (per window, never pooled)")
        print(f"     truth {truth}  hits {hits}  misses {truth - hits}  "
              f"false-pos {fp}"
              + (f"  recall {hits/truth:.0%}" if truth else ""))
    for w, g in tuned:
        if g < w["truth"]:
            print(f"  !! CANARY: tuned window {w['t0']:.0f}-{w['t1']:.0f} lost "
                  f"{w['truth'] - g} real arrival(s). Held-out cannot see this.")
    return 0


def unseal(plan_path, t0):
    doc = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    n = 0
    for w in doc["windows"]:
        if abs(w["t0"] - float(t0)) < 0.5 and w.get("sealed"):
            w["sealed"] = False
            w["unsealed_note"] = ("opened deliberately; from here on this "
                                  "window can no longer prove generalisation")
            n += 1
    Path(plan_path).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"  unsealed {n} window(s) — recorded in the file")


def sheets(plan_path, video, out_dir="eval/sheets", step_s=1.0, cols=5):
    """Render one contact sheet per planned window, so the windows can be READ.

    THE GAP THIS FILLS
        This file's own docstring has always advertised

            sheets  <plan.json> --video V   render a contact sheet per window

        and the subcommand did not exist. So the documented workflow stopped
        dead after `plan`: you could choose windows by detector activity, and
        then had no way to put frames in front of a human to label them. That
        is why 10 of 13 regression cases are still NO-DATA and held-out recall
        rests on 3 truth entries -- the labelling step was unreachable, not
        skipped.

    SEALED WINDOWS ARE STILL RENDERED. Sealing governs SCORING, not labelling:
    a held-out window is worthless unless somebody labels it. What must not
    happen is tuning against it, and `score` enforces that separately.
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    import cv2
    from contact_sheet import frame_at

    doc = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    made = []
    for w in doc["windows"]:
        t0, t1 = float(w["t0"]), float(w["t1"])
        times = [t0 + i * step_s for i in range(int((t1 - t0) / step_s) + 1)]
        tiles = []
        for t in times:
            f = frame_at(video, t)
            if f is None:
                continue
            f = cv2.resize(f, (480, int(480 * f.shape[0] / f.shape[1])))
            # the timestamp is the whole point: the labeller writes down WHEN
            cv2.rectangle(f, (0, 0), (150, 26), (0, 0, 0), -1)
            cv2.putText(f, f"t={t:.1f}s", (6, 19), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(f)
        if not tiles:
            print(f"  !! no frames decoded for {t0:.0f}-{t1:.0f}s "
                  f"— is {video} the chunk this plan was made from?")
            continue
        th, tw = tiles[0].shape[:2]
        rows = (len(tiles) + cols - 1) // cols
        import numpy as _np
        sheet = _np.zeros((rows * th, cols * tw, 3), _np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = tile
        name = (f"win_{int(t0):05d}_{int(t1):05d}_{w['kind']}"
                f"{'_HELDOUT' if w.get('held_out') else ''}.jpg")
        cv2.imwrite(str(out / name), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        made.append(name)
        print(f"  {name}  ({len(tiles)} frames)")

    (out / "HOW_TO_LABEL.txt").write_text(
        "ENTRY-EVENT LABELLING\n"
        "=====================\n\n"
        f"plan file: {plan_path}\n"
        f"sheets:    {out}\n\n"
        "For each win_*.jpg, watch the frames left-to-right, top-to-bottom.\n"
        "Count people who CROSS THE THRESHOLD INTO the venue in that window.\n\n"
        "Then in the plan file, for the window with the matching t0, set:\n\n"
        '  \"truth\":   <integer>            how many came IN\n'
        '  \"entries\": [{\"t\": 12.4, \"note\": \"group of 3, one holds door\"}]\n'
        '  \"note\":    \"\"                  anything odd about the window\n\n'
        "RULES\n"
        "  * A quiet window with truth 0 is REAL DATA, not a skipped window.\n"
        "    Without them, a counter that over-fires scores perfectly.\n"
        "  * Label the _HELDOUT windows too. Sealing stops them being SCORED\n"
        "    early; it does not excuse them from being labelled.\n"
        "  * Someone walking past the door without entering is NOT an entry.\n"
        "  * If you cannot tell, write it in note and leave truth null. An\n"
        "    honest gap beats a guess -- a guess becomes the reference.\n\n"
        "When done:  python tools/entry_label_kit.py check " + str(plan_path)
        + "\n", encoding="utf-8")
    print(f"\n  {len(made)} sheet(s) -> {out}")
    print(f"  read {out / 'HOW_TO_LABEL.txt'} and fill in {plan_path}")
    return made


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("plan"); p1.add_argument("run_dir")
    p1.add_argument("--n", type=int, default=8)
    p1.add_argument("--out", default="eval/entry_plan.json")
    p1.add_argument("--camera", default="CAM.112")
    p2 = sub.add_parser("check"); p2.add_argument("plan")
    p3 = sub.add_parser("score"); p3.add_argument("plan"); p3.add_argument("events")
    p3.add_argument("--allow-sealed", action="store_true")
    p4 = sub.add_parser("unseal"); p4.add_argument("plan"); p4.add_argument("t0")
    p5 = sub.add_parser("sheets"); p5.add_argument("plan")
    p5.add_argument("--video", required=True)
    p5.add_argument("--out", default="eval/sheets")
    p5.add_argument("--step", type=float, default=1.0)
    a = ap.parse_args()
    if a.cmd == "plan":
        plan(a.run_dir, n=a.n, out=a.out, camera=a.camera); return 0
    if a.cmd == "check":
        code, fail, warn, _ = check(a.plan)
        for f in fail: print(f"   ✗ {f}")
        for w in warn: print(f"   ! {w}")
        print("   " + {0: "USABLE", 1: "USABLE WITH WARNINGS",
                       2: "REFUSED"}[code])
        return code
    if a.cmd == "score":
        return score(a.plan, a.events, allow_sealed=a.allow_sealed)
    if a.cmd == "unseal":
        unseal(a.plan, a.t0); return 0
    if a.cmd == "sheets":
        sheets(a.plan, a.video, out_dir=a.out, step_s=a.step); return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
