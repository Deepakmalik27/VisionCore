#!/usr/bin/env python3
"""A tuned constant is only as good as the code it was measured against.

WHY THIS EXISTS
    LINE_MAX_W = 120 was measured correctly. Its docstring still records the
    evidence:

        quiet window, both events false : widths 220, 7
        every real-window event         : widths 8 .. 104
        "A cut at 120 keeps 15/15 real events"

    Then foreground() gained a per-lighting-regime background, which merges
    neighbouring people into wider blobs. The evidence did not change. The
    number did not change. The MEANING of the number changed, and nothing said
    so. Three runs later the six-person group window read 4/6, because two real
    guests measured w=121 and w=126 -- over a cut whose justification had
    quietly expired.

    That is not a slit-counter bug. It is what 201 tuned constants look like
    when the code beneath them moves. This turns "the evidence for this number
    was collected under conditions that no longer hold" into a check.

WHY NOT build_id.py
    build_id hashes every file in kevacv/ into one id. That answers "is this
    the same code?" and it is the right tool for that. It is useless here:
    editing an unrelated module would expire every constant in the repo. Expiry
    needs FUNCTION granularity -- LINE_MAX_W depends on foreground() and
    split_blobs(), and on nothing else.

WHAT IT DOES NOT DO
    It cannot tell you the new correct value. It tells you the old one is no
    longer evidence-backed, which is the part nobody noticed for three runs.
    Re-measuring is still work a human authorises.

USAGE
    python tools/threshold_expiry.py check
    python tools/threshold_expiry.py stamp <file:CONST> --run <run-id> [--note ...]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "eval" / "threshold_provenance.json"


def function_hash(spec: str) -> str | None:
    """Hash one function's source. `spec` is "path/to/file.py:func_name".

    Hashing the function rather than the file keeps an unrelated edit in the
    same module from expiring a constant that does not depend on it. Comments
    and docstrings are INCLUDED deliberately: a comment that changes what a
    stage means is exactly the kind of change worth re-checking, and stripping
    them would need a normaliser whose bugs would be invisible.
    """
    path_str, _, name = spec.partition(":")
    path = ROOT / path_str
    if not path.exists():
        return None
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            segment = ast.get_source_segment(source, node)
            if segment is None:
                return None
            return hashlib.sha1(segment.encode("utf-8")).hexdigest()[:12]
    return None


def current_value(spec: str):
    """Read a module-level constant's literal value without importing anything.

    Importing tools/slit_count.py would pull in cv2 and could execute code; the
    checker must run on a laptop with no CV stack.
    """
    path_str, _, name = spec.partition(":")
    path = ROOT / path_str
    if not path.exists():
        return None
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    try:
                        return ast.literal_eval(node.value)
                    except (ValueError, SyntaxError):
                        return None
    return None


def load_registry() -> dict:
    """-> the provenance registry, or an empty one on first use."""
    if not REGISTRY.exists():
        return {}
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def check(registry: dict) -> list[dict]:
    """-> one verdict per registered constant, worst first.

    EXPIRED  a dependency's source changed since the constant was measured.
    DRIFTED  the constant itself was edited without re-stamping, so the
             recorded evidence describes a value that is no longer in the file.
    MISSING  a dependency or the constant no longer exists.
    OK       the evidence still describes the code that is running.
    """
    rank = {"EXPIRED": 0, "DRIFTED": 1, "MISSING": 2, "OK": 3}
    out = []
    for spec, meta in registry.items():
        value_now = current_value(spec)
        reasons, status = [], "OK"

        if value_now is None:
            status, reasons = "MISSING", [f"{spec} not found"]
        elif value_now != meta.get("value_when_measured"):
            status = "DRIFTED"
            reasons.append(
                f"value is {value_now!r}, evidence was collected at "
                f"{meta.get('value_when_measured')!r}")

        for dep, recorded in (meta.get("dep_hashes") or {}).items():
            now = function_hash(dep)
            if now is None:
                status = "MISSING" if status == "OK" else status
                reasons.append(f"dependency {dep} no longer exists")
            elif now != recorded:
                status = "EXPIRED"
                reasons.append(f"{dep} changed ({recorded} -> {now})")

        out.append({"spec": spec, "status": status, "value": value_now,
                    "measured_on": meta.get("measured_on", "?"),
                    "measured_by": meta.get("measured_by", "?"),
                    "evidence": meta.get("evidence", ""),
                    "reasons": reasons})
    out.sort(key=lambda r: (rank.get(r["status"], 9), r["spec"]))
    return out


def stamp(spec: str, run: str, note: str = "") -> None:
    """Record that `spec` was (re-)measured now, against today's dependencies.

    Refuses an unknown spec: the dependency list is a judgement about which
    code the measurement actually rests on, and inventing one automatically
    would produce a confident registry entry nobody thought about.
    """
    registry = load_registry()
    if spec not in registry:
        raise SystemExit(
            f"{spec} is not registered. Add it to {REGISTRY.name} with a "
            f"depends_on list first -- naming its dependencies is the part that "
            f"needs a human.")
    meta = registry[spec]
    meta["value_when_measured"] = current_value(spec)
    meta["measured_by"] = run
    meta["measured_on"] = __import__("datetime").date.today().isoformat()
    if note:
        meta["evidence"] = note
    meta["dep_hashes"] = {d: function_hash(d) for d in meta.get("depends_on", [])}
    REGISTRY.write_text(json.dumps(registry, indent=1) + "\n", encoding="utf-8")
    print(f"stamped {spec} = {meta['value_when_measured']!r} against {run}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="report expired / drifted constants")
    sp = sub.add_parser("stamp", help="record a fresh measurement")
    sp.add_argument("spec")
    sp.add_argument("--run", required=True, help="run id or artefact that measured it")
    sp.add_argument("--note", default="", help="the evidence, in one line")
    args = ap.parse_args()

    if args.cmd == "stamp":
        stamp(args.spec, args.run, args.note)
        return 0

    registry = load_registry()
    if not registry:
        print(f"no registry at {REGISTRY.relative_to(ROOT)} -- nothing to check")
        return 0
    rows = check(registry)
    bad = [r for r in rows if r["status"] != "OK"]
    print(f"THRESHOLD PROVENANCE   {len(rows)} registered, {len(bad)} need attention\n")
    for r in rows:
        mark = {"OK": "  ok  ", "EXPIRED": " EXPIRED", "DRIFTED": " DRIFTED",
                "MISSING": " MISSING"}[r["status"]]
        print(f"{mark}  {r['spec']} = {r['value']!r}")
        print(f"          measured {r['measured_on']} on {r['measured_by']}")
        if r["evidence"]:
            print(f"          evidence: {r['evidence']}")
        for reason in r["reasons"]:
            print(f"          ! {reason}")
        print()
    if bad:
        print("EXPIRED means the evidence describes code that no longer runs.")
        print("It does NOT mean the value is wrong -- it means nothing currently")
        print("shows it is right. Re-measure, then: threshold_expiry.py stamp <spec>")
    return 1 if bad else 0


def _selftest():
    """One runnable check: a changed dependency must expire its constant."""
    reg = {
        "tools/slit_count.py:LINE_MAX_W": {
            "value_when_measured": current_value("tools/slit_count.py:LINE_MAX_W"),
            "depends_on": ["tools/slit_count.py:foreground"],
            "dep_hashes": {"tools/slit_count.py:foreground": "deadbeef0000"},
        }
    }
    verdict = check(reg)[0]
    assert verdict["status"] == "EXPIRED", verdict
    assert "foreground changed" in verdict["reasons"][0], verdict

    live = function_hash("tools/slit_count.py:foreground")
    assert live is not None, "foreground() should be hashable"
    reg["tools/slit_count.py:LINE_MAX_W"]["dep_hashes"] = {
        "tools/slit_count.py:foreground": live}
    assert check(reg)[0]["status"] == "OK", check(reg)[0]

    reg["tools/slit_count.py:LINE_MAX_W"]["value_when_measured"] = -999
    assert check(reg)[0]["status"] == "DRIFTED"
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        raise SystemExit(main())
