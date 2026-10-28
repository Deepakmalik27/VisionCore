"""embed_kevacv.py — re-embed the kevacv package into the notebook bootstrap.

WHY THIS EXISTS
    Kaggle has no kevacv on disk, so Cell 2f carries the whole package as a
    dict of source strings and writes it out at runtime. That means the
    package exists TWICE: the real files in kevacv/, and the frozen copy in
    the notebook.

    They drift silently. Adding topology.py, threshold.py, calibration.py,
    merge_ab.py, report_slim.py and tiled.py to kevacv/ did nothing for a
    Kaggle run — the notebook still materialised the old ten modules. Worse,
    arrivals.py gained entry_zone_coverage() on disk while the embedded copy
    kept the old version, and patch_v74 wired a CALL to that function into the
    analytics. On Kaggle that is a NameError at runtime, an hour into a run.

    test_v56_phase6 catches exactly this. Run this tool whenever kevacv/
    changes, then run the test.

USAGE
    python tools/embed_kevacv.py            # rewrite the bootstrap cell
    python tools/embed_kevacv.py --check    # exit 1 if stale, change nothing
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "kevacv"
NB = ROOT / "notebooks" / "pipeline.ipynb"
START = "_KEVACV_SRC = {"


# Modules whose code ALSO still exists as a notebook cell. Embedding them
# would define every one of their functions twice in the same notebook — once
# in the cell, once in the bootstrap — and the cell runs later, so it silently
# shadows the package version. test_v56_phase3 catches exactly that
# ("tier_a_crossings defined once").
#
# Remove an entry here in the SAME change that deletes its cell, never before.
#   analytics.py <- Cell 5
#   engine.py    <- Cell 7
SHADOWED_BY_A_CELL = {"analytics.py", "engine.py"}


def collect():
    """-> {filename: source}. __init__ first so the dict reads sensibly."""
    files = sorted((p for p in PKG.glob("*.py")
                    if p.name not in SHADOWED_BY_A_CELL),
                   key=lambda p: (p.name != "__init__.py", p.name))
    out = {}
    for p in files:
        text = p.read_text(encoding="utf-8")
        # r'''...''' cannot survive these. Fail loudly here rather than emit a
        # cell that dies with 'unterminated triple-quoted string' thousands of
        # lines from the cause.
        if "'''" in text:
            raise SystemExit(f"{p.name} contains ''' — use \"\"\" docstrings")
        if text.rstrip("\n").endswith("'") or text.rstrip("\n").endswith("\\"):
            raise SystemExit(f"{p.name} ends with a quote or backslash")
        out[p.name] = text
    return out


def render(src):
    """Emit each module as r'''...''' — the format test_v56_phase6 parses.

    json.dumps would escape more safely, but the existing test reads the cell
    back by looking for r''' blocks, and a private format that its verifier
    cannot parse is worse than a slightly fragile one it can. The fragility is
    bounded by the guard in collect().
    """
    lines = [START]
    for name, text in src.items():
        lines.append(f"    {name!r}: r'''{text}''',")
    lines.append("}")
    return "\n".join(lines)


# The writer loop that immediately follows the dict. Anchoring on this rather
# than on a bare '}' matters: embedded module sources contain lines that are
# exactly '}' at column 0, so scanning for the first one cuts the dict short
# and leaves orphaned string content behind — which surfaces thousands of
# lines later as 'unterminated triple-quoted string literal'.
TAIL_ANCHOR = "_pkg = _Path("


def splice(cell_src, new_dict):
    """Replace the _KEVACV_SRC literal, leaving header and tail untouched."""
    i = cell_src.index(START)
    j = cell_src.find(TAIL_ANCHOR, i)
    if j < 0:
        raise SystemExit(f"tail anchor {TAIL_ANCHOR!r} not found after the dict "
                         f"— the bootstrap cell's shape has changed")
    return cell_src[:i] + new_dict + "\n\n" + cell_src[j:]


def main():
    check_only = "--check" in sys.argv
    if not NB.exists():
        raise SystemExit(f"notebook not found: {NB}")
    src = collect()
    nb = json.loads(NB.read_text(encoding="utf-8"))

    target = None
    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and START in "".join(cell["source"]):
            target = cell
            break
    if target is None:
        raise SystemExit("no cell contains _KEVACV_SRC — is this the right notebook?")

    old = "".join(target["source"])
    new = splice(old, render(src))

    if old == new:
        print(f"kevacv bootstrap already current ({len(src)} modules)")
        return 0
    if check_only:
        print(f"STALE: notebook bootstrap does not match kevacv/ "
              f"({len(src)} modules on disk)")
        print("       run: python tools/embed_kevacv.py")
        return 1

    shutil.copy(NB, NB.with_suffix(".BACKUP-preembed.ipynb"))
    target["source"] = new.splitlines(keepends=True)
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"embedded {len(src)} module(s), cell {len(old)} -> {len(new)} chars")
    for n in src:
        print(f"   {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
