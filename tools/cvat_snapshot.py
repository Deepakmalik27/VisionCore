#!/usr/bin/env python3
"""cvat_snapshot.py — never label the same frame twice.

WHY THIS EXISTS
    Labelling is the only work in this project that cannot be recomputed. A
    run can be re-run, a model can be re-trained, a config can be re-derived --
    but 3,511 hand-drawn boxes and the staff/guest truth behind them exist in
    exactly one place: CVAT's Postgres. And CVAT's Postgres has already been
    corrupted once in this project, on 2026-08-25, when a restore script ran
    `~` under sudo, resolved it to /root, skipped `docker compose stop` and
    wrote into a live database.

    There is no undo for that. So this takes a copy, on a timer, forever.

WHAT IT DOES
    Every run, for every task: pull the annotations, hash them, and write a
    timestamped copy ONLY if the content changed. Unchanged polls cost one API
    call and no disk. A ledger line records what each snapshot contained, so
    "when did I lose track 9's role?" is answerable without opening files.

WHY CONTENT-HASHED AND NOT EVERY-N-MINUTES
    A snapshot every 5 minutes for a week is 2,016 identical files and no
    signal. Hashing means the snapshot count IS the number of times the work
    actually changed, and every file in the directory is a distinct state worth
    going back to.

RESTORING
    --restore <file> pushes a snapshot back. This is the half that makes the
    other half worth having, and it is deliberately NOT automatic: overwriting
    good work with an old snapshot is the same loss this file exists to
    prevent, so it asks.

Usage:
    python tools/cvat_snapshot.py                     # snapshot 1,2,3
    python tools/cvat_snapshot.py --tasks 1 3
    python tools/cvat_snapshot.py --list              # what do I have?
    python tools/cvat_snapshot.py --restore <file>    # put it back
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAP = ROOT / "eval" / "cvat_snapshots"
LEDGER = SNAP / "ledger.tsv"
TOKEN_FILE = Path.home() / ".cvat_token"


def api(base, path, token=None, data=None, method=None):
    url = base.rstrip("/") + path
    body = json.dumps(data).encode() if data is not None else None
    r = urllib.request.Request(url, data=body,
                               method=method or ("POST" if body else "GET"))
    r.add_header("Content-Type", "application/json")
    if token:
        r.add_header("Authorization", "Token " + token)
    with urllib.request.urlopen(r, timeout=180) as resp:
        txt = resp.read().decode()
        return json.loads(txt) if txt.strip()[:1] in "{[" else txt


def get_token(base, user, password):
    """A stored TOKEN, not a stored password.

    Cron needs a credential on disk; there is no way around that. A token is
    the better one to leave there: it can be revoked in CVAT without changing
    the password that also unlocks everything else.
    """
    if TOKEN_FILE.exists():
        t = TOKEN_FILE.read_text(encoding="utf-8").strip()
        try:
            api(base, "/users/self", token=t)
            return t
        except urllib.error.HTTPError:
            pass                       # expired or revoked -> fall through
    if not password:
        raise SystemExit(
            f"no valid token in {TOKEN_FILE} and no CVAT_PASSWORD set.\n"
            f"  run once with CVAT_PASSWORD=... to mint one.")
    t = api(base, "/auth/login", data={"username": user, "password": password})["key"]
    TOKEN_FILE.write_text(t, encoding="utf-8")
    TOKEN_FILE.chmod(0o600)
    print(f"  minted a token -> {TOKEN_FILE} (mode 600)")
    return t


def summarise(ann):
    """tracks / boxes / how many carry a role -- the numbers worth a ledger."""
    tracks = ann.get("tracks") or []
    boxes = labelled = 0
    roles = {}
    for i, tr in enumerate(tracks, 1):
        fr = [s for s in tr.get("shapes") or [] if not s.get("outside")]
        boxes += len(fr)
        vals = [a.get("value") for s in tr.get("shapes") or []
                for a in s.get("attributes") or []
                if a.get("value") in ("staff", "guest")]
        if vals:
            labelled += 1
            roles[i] = max(set(vals), key=vals.count)
    return {"tracks": len(tracks), "boxes": boxes,
            "shapes_loose": len(ann.get("shapes") or []),
            "roles_set": labelled, "roles": roles}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--base", default="http://localhost:8080/api")
    p.add_argument("--user", default="prabh")
    p.add_argument("--password", default=os.environ.get("CVAT_PASSWORD", ""))
    p.add_argument("--list", action="store_true")
    p.add_argument("--restore", metavar="FILE")
    p.add_argument("--quiet", action="store_true", help="for cron: print only on change")
    a = p.parse_args(argv)

    SNAP.mkdir(parents=True, exist_ok=True)

    if a.list:
        rows = LEDGER.read_text(encoding="utf-8").splitlines() if LEDGER.exists() else []
        if not rows:
            print("  no snapshots yet")
            return 0
        print(f"  {'when':<20} {'task':>4} {'tracks':>7} {'boxes':>7} {'roles':>6}  file")
        for r in rows[-40:]:
            f = r.split("\t")
            if len(f) >= 6:
                print(f"  {f[0]:<20} {f[1]:>4} {f[2]:>7} {f[3]:>7} {f[4]:>6}  {f[5]}")
        print(f"\n  {len(rows)} snapshot(s) in {SNAP}")
        return 0

    tok = get_token(a.base, a.user, a.password)

    if a.restore:
        src = Path(a.restore)
        if not src.exists():
            src = SNAP / a.restore
        blob = json.loads(src.read_text(encoding="utf-8"))
        tid, ann = blob["task"], blob["annotations"]
        s = summarise(ann)
        cur = summarise(api(a.base, f"/tasks/{tid}/annotations", token=tok))
        print(f"  RESTORE task {tid} from {src.name}")
        print(f"    snapshot : {s['tracks']:>4} tracks  {s['boxes']:>6} boxes  {s['roles_set']:>3} roles")
        print(f"    CVAT now : {cur['tracks']:>4} tracks  {cur['boxes']:>6} boxes  {cur['roles_set']:>3} roles")
        if cur["boxes"] > s["boxes"] or cur["roles_set"] > s["roles_set"]:
            print("\n  !! CVAT currently holds MORE work than this snapshot.")
            print("     Restoring would DELETE the difference. That is the exact")
            print("     loss this script exists to prevent, so it is not automatic.")
        if input("  type RESTORE to proceed: ").strip() != "RESTORE":
            print("  aborted, nothing written")
            return 1
        api(a.base, f"/tasks/{tid}/annotations", token=tok, data=ann, method="PUT")
        print(f"  restored task {tid}")
        return 0

    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    changed = 0
    for tid in a.tasks:
        try:
            ann = api(a.base, f"/tasks/{tid}/annotations", token=tok)
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            print(f"  task {tid}: UNREACHABLE ({e}) — no snapshot taken")
            continue
        body = json.dumps(ann, sort_keys=True)
        h = hashlib.sha256(body.encode()).hexdigest()[:12]
        d = SNAP / f"task{tid}"
        d.mkdir(exist_ok=True)
        if any(f.name.endswith(f"_{h}.json") for f in d.glob("*.json")):
            if not a.quiet:
                s = summarise(ann)
                print(f"  task {tid}: unchanged ({s['tracks']} tracks, "
                      f"{s['boxes']} boxes, {s['roles_set']} roles) [{h}]")
            continue
        s = summarise(ann)
        out = d / f"{now.replace(':', '')}_{h}.json"
        out.write_text(json.dumps(
            {"task": tid, "taken_utc": now, "hash": h,
             "summary": s, "annotations": ann}, indent=1), encoding="utf-8")
        with LEDGER.open("a", encoding="utf-8") as f:
            f.write(f"{now}\t{tid}\t{s['tracks']}\t{s['boxes']}\t"
                    f"{s['roles_set']}\t{out.relative_to(SNAP)}\n")
        print(f"  task {tid}: SAVED  {s['tracks']} tracks  {s['boxes']} boxes  "
              f"{s['roles_set']} roles  -> {out.relative_to(SNAP)}")
        changed += 1
    if changed and not a.quiet:
        print(f"\n  {changed} task(s) changed. --list shows the history, "
              f"--restore <file> puts one back.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
