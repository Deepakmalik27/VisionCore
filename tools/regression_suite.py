#!/usr/bin/env python3
"""The 13-case regression matrix. Run every pipeline change against it.

follow_up.txt calls this the backbone, and it is the only thing that stops
"I fixed one thing and silently broke three others". It is also the honest
place to record what CANNOT yet be tested: 11 of the 13 cases have no labelled
window, and a suite that quietly skips them looks identical to a suite that
passes them.

THE RULE, same as everywhere else in this project:
    a case with no data reports NO-DATA and NEVER passes.
Two accuracy claims were withdrawn on 2026-08-19 for exactly this reason.

Usage:  python3 tools/regression_suite.py <run> [<run> ...]
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kevacv.scorecard import score_windows          # noqa: E402
from kevacv.visits import build_visits              # noqa: E402

PASS, FAIL, NODATA = "PASS", "FAIL", "NO-DATA"
CAM = "CAM.112"


def _load(run):
    cp = f"output/{run}/debug/{CAM}_crossings.json"
    if not os.path.exists(cp):
        return None, None
    d = json.load(open(cp))
    cross = d if isinstance(d, list) else d.get("crossings", d)
    video = None
    sp = f"output/{run}/SUMMARY.txt"
    if os.path.exists(sp):
        import re
        m = re.search(r"^\s*source\s+(.+?\.mp4)\s*$",
                      open(sp, errors="ignore").read(), re.M)
        video = m.group(1).strip() if m else None
    return cross, video


def evaluate(run):
    cases = json.load(open("eval/cases.json"))["cases"]
    cross, video = _load(run)
    if cross is None:
        return [{"id": c["id"], "name": c["name"], "state": NODATA,
                 "detail": "no crossings file for this run"} for c in cases]

    scored = score_windows(cross, video=video)
    by_window = {w["window"]: w for w in scored.get("windows", [])}
    visits = build_visits(cross, line_name="entry line")

    out = []
    for c in cases:
        w = c.get("window")
        if not w:
            out.append({"id": c["id"], "name": c["name"], "state": NODATA,
                        "detail": f"no labelled window — needs a clip showing "
                                  f"'{c['name']}'"})
            continue
        row = by_window.get(w)
        if row is None:
            out.append({"id": c["id"], "name": c["name"], "state": NODATA,
                        "detail": f"{w} not scored (labels are from another "
                                  f"chunk than this run)"})
            continue
        got, truth = row["got"], row["truth"]
        ok = (got == truth)
        note = " [TUNED — not evidence]" if row.get("tuned") else ""
        out.append({"id": c["id"], "name": c["name"],
                    "state": PASS if ok else FAIL,
                    "detail": f"{w}: got {got}, truth {truth}{note}"})
    # C05 needs no labels: it is a property of the visit model
    for r in out:
        if r["id"] == "C05":
            r["state"] = NODATA
            r["detail"] = (f"no re-entry window labelled; this run observed "
                           f"{visits['repeat_visitors']} repeat visitor(s) "
                           f"across {visits['n_visits']} visit(s) — "
                           f"unverified without truth")
    return out


def main(argv):
    runs = argv or ["m_v5"]
    rc = 0
    for run in runs:
        rows = evaluate(run)
        n_pass = sum(1 for r in rows if r["state"] == PASS)
        n_fail = sum(1 for r in rows if r["state"] == FAIL)
        n_nod = sum(1 for r in rows if r["state"] == NODATA)
        print(f"\n=== {run} ===  {n_pass} pass · {n_fail} fail · "
              f"{n_nod} NO-DATA of {len(rows)}")
        for r in rows:
            print(f"  [{r['state']:>8s}] {r['id']} {r['name']:<34s} {r['detail']}")
        if n_fail:
            rc = 1
        print(f"  coverage: {100.0*(n_pass+n_fail)/len(rows):.0f}% of cases "
              f"are testable at all")
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
