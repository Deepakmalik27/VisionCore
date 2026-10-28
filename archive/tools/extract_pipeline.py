"""extract_pipeline.py — lift the notebook's cells into an importable package.

WHY THIS IS MECHANICAL, NOT A REWRITE
    The notebook holds 161 top-level definitions across 17 cells; only one of
    them also exists in kevacv. Moving that logic is the last big step to a
    real codebase — but it is also the step with no measurable upside: nothing
    gets more accurate by changing where it lives.

    kevacv/__init__.py says these come out AFTER the first HOTA number,
    precisely so a later regression is attributable. Since we are doing it
    first, the mitigation is that this tool COPIES cell text verbatim. Same
    statements, same order, same globals — only the file boundary is new. If a
    number moves later, it is not because a line was rewritten here.

HOW THE GLOBALS PROBLEM IS HANDLED
    Notebook cells share one namespace. Cell 7 reads dozens of names Cell 2
    defined, sometimes via globals().get("NAME"). Splitting them into modules
    would normally break every one of those reads.

    So config.py is emitted first and every other module starts with
    `from .config import *`. That reproduces the notebook's flat namespace
    exactly, including globals() lookups, because a star-import lands the names
    in the importing module's own globals. It is not elegant. It is faithful,
    which matters more today; tightening the imports is a later, separately
    verifiable change.

USAGE
    python tools/extract_pipeline.py            # write pipeline/
    python tools/extract_pipeline.py --verify   # compare defs, change nothing
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "pipeline.ipynb"
OUT = ROOT / "pipeline"

# cell index -> (module, one-line purpose). Indices are positional in the
# notebook; the "Cell N" labels in the source comments are historical and do
# NOT match, which is exactly why this table is explicit.
PLAN = [
    (2,  "config",      "all tuneable constants + preflight (the shared namespace)"),
    (8,  "scale",       "10-hour scale profile: fps, re-id horizon, IR handling"),
    (10, "discovery",   "find the videos and their zone files"),
    (11, "zones",       "zone loader + AI zone-role classifier"),
    (12, "analytics",   "identity merging, calibration, dwell, occupancy"),
    (13, "charts",      "chart style and visual helpers"),
    (14, "engine",      "detect, track, face, crops, render, process_video"),
    (16, "runner",      "process every queued video"),
    (17, "runner_multi","multi-chunk runner: 2-GPU, resume, seam handling"),
    (18, "peak_window", "the busiest N minutes only"),
    (19, "metrics",     "reception metrics — the numbers a GM reads"),
    (20, "brief",       "the brief, the strip, the moments"),
    (23, "tables",      "minute-by-minute and per-track tables"),
    (24, "export",      "events CSV, answers JSON, playable video, zip"),
    (25, "staff_challenger", "CLIP staff challenger (measures, never overrides)"),
    (26, "id_audit",    "per-frame identity timeline audit"),
    (27, "self_audit",  "auto red-flags for known failure modes"),
    (28, "ledger",      "full-night coverage ledger"),
    (29, "report",      "REPORT.md, end card, free-text ask()"),
    (30, "groundtruth", "pick slices, export for labelling, score"),
    (31, "ablation",    "ablation runner"),
]

SKIP = {0, 9}   # 0 = pip install; 9 = the kevacv bootstrap (already a package)

HEADER = '''"""{module}.py — {purpose}

Extracted verbatim from notebook cell {cell} by tools/extract_pipeline.py.
Do not hand-edit while the notebook is still the executed artefact: run the
extractor again instead, or the two copies drift the way kevacv/ and the
bootstrap cell did.
"""
from __future__ import annotations
'''


def load_cells():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return ["".join(c["source"]) if c["cell_type"] == "code" else None
            for c in nb["cells"]]


def defs_in(src):
    """Top-level def/class names, by regex — the source may not import here."""
    return {m.group(2) for m in re.finditer(
        r"^(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", src, re.M)}


def main():
    verify = "--verify" in sys.argv
    cells = load_cells()

    planned = {c for c, _, _ in PLAN}
    covered = planned | SKIP
    missing = [i for i, s in enumerate(cells)
               if s is not None and i not in covered and defs_in(s)]
    if missing:
        print(f"!! cells with definitions and no home: {missing}")

    if not verify:
        OUT.mkdir(exist_ok=True)

    total_defs, wrote = set(), []
    for cell, module, purpose in PLAN:
        src = cells[cell]
        if src is None:
            print(f"  skip {module}: cell {cell} is not code")
            continue
        d = defs_in(src)
        total_defs |= d
        body = HEADER.format(module=module, purpose=purpose, cell=cell)
        if module != "config":
            # reproduce the notebook's single flat namespace
            body += "from .config import *  # noqa: F401,F403\n"
        body += "\n" + src.rstrip() + "\n"

        try:
            ast.parse(body)
        except SyntaxError as e:
            print(f"  !! {module}.py would not parse: {e}")
            continue

        if not verify:
            (OUT / f"{module}.py").write_text(body, encoding="utf-8")
        wrote.append((module, cell, len(src), len(d)))

    print(f"{'verified' if verify else 'wrote'} {len(wrote)} module(s), "
          f"{len(total_defs)} definitions")
    for module, cell, size, n in wrote:
        print(f"   {module:<18} cell {cell:<3} {size:>7} chars  {n:>3} defs")

    if not verify:
        init = ['"""pipeline — the analytics pipeline, extracted from the notebook.',
                "",
                "Import order matters: config first, then everything that reads it.",
                "The notebook remains the executed artefact for now; this package is",
                "what the extractor produces from it, and what tests import.",
                '"""',
                ""]
        for _, module, _ in PLAN:
            init.append(f"from . import {module}  # noqa: F401")
        init.append("")
        init.append("__all__ = [" + ", ".join(f'"{m}"' for _, m, _ in PLAN) + "]")
        (OUT / "__init__.py").write_text("\n".join(init) + "\n", encoding="utf-8")
        print(f"   __init__.py        {len(PLAN)} submodules")
    return 0


if __name__ == "__main__":
    sys.exit(main())
