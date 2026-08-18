"""test_engine_compiles.py — engine.py must at least COMPILE.

WHY THIS EXISTS
    On 2026-08-15 engine.py sat broken with an IndentationError while EVERY
    test suite passed. No test imports it — it pulls torch/ultralytics/boxmot,
    which is exactly why the package is arranged so a laptop can run the tests
    at all. So the one file that does the work was the one file nothing checked.

    It shipped to the GPU box. It would have died on the next run, after the
    3 GB download and the model load, with a syntax error a compiler catches in
    40 milliseconds.

    compile() needs no torch. That is the whole point.
"""
import py_compile
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_fail = False


def check(ok, what, detail=""):
    global _fail
    print(f"  {'PASS' if ok else 'FAIL'}  {what}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        _fail = True


print("=" * 74)
print("  every module compiles — including the ones tests cannot import")
print("=" * 74)
with tempfile.TemporaryDirectory() as td:
    for src in sorted((ROOT / "kevacv").glob("*.py")) + sorted((ROOT / "tools").glob("*.py")):
        rel = f"{src.parent.name}/{src.name}"
        try:
            py_compile.compile(str(src), cfile=str(Path(td) / (src.stem + ".pyc")),
                               doraise=True)
            ok, detail = True, ""
        except py_compile.PyCompileError as exc:
            ok, detail = False, str(exc).splitlines()[-1][:90]
        if not ok or rel.endswith(("engine.py", "analytics.py", "pipeline.py")):
            check(ok, rel, detail)
    check(not _fail, "ALL modules compile")

# ---------------------------------------------------------------------------
# compile() is necessary but NOT sufficient.
#
# On 2026-08-15 engine.py compiled clean and died on the GPU box at
#     zcfg_flip = (_zone_cfg_raw or {}).get(...)   # line 2062
#     _zone_cfg_raw = json.loads(...)              # line 2309
# — a local read 240 lines before its only assignment. UnboundLocalError is a
# RUNTIME error, so the compiler is happy and no test that cannot import torch
# will ever reach the line.
#
# pyflakes resolves scopes statically and reports exactly this. No torch needed.
# ---------------------------------------------------------------------------
#
# pyflakes was tried first and is NO USE here: engine.py does `from .config
# import *`, and under a star import pyflakes downgrades the message to
# "'_zone_cfg_raw' may be undefined, or defined from star imports" — the same
# thing it says about every legitimate config global in the file. Matching
# that string flags hundreds of correct lines; not matching it misses the bug.
#
# So check the one thing that is unambiguous: inside ONE function body, a name
# READ at a top-level statement that appears BEFORE that name's first
# top-level assignment in the same function. Reads inside loops are skipped —
# there the previous iteration may legitimately have assigned it.
print()
print("=" * 74)
print("  no local is read above its own assignment  (AST order check)")
print("=" * 74)
import ast


def _first_lines(fn):
    """(first assignment line, first read line) per name, top-level stmts only."""
    assigned, read, declared = {}, {}, set()
    # Own-scope, deferred-execution, or later-line-is-legal constructs. A read
    # inside a nested def/lambda runs whenever that closure is CALLED, which is
    # after the enclosing body finished assigning — flagging those was 10 more
    # false alarms.
    # Own-scope, deferred-execution, or later-line-is-legal constructs, skipped
    # wherever they appear — not merely at the top level:
    #   comprehension  — own scope; in a multi-line one the `for` target is
    #                    written BELOW the element expression that uses it.
    #   def / lambda   — runs when CALLED, i.e. after the body finished.
    #   for / while    — the previous iteration may have done the assigning.
    # Each of these cost false alarms before it was added.
    _skip = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
             ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef,
             ast.For, ast.AsyncFor, ast.While)
    for stmt in fn.body:
        if isinstance(stmt, _skip):
            continue
        skip = {id(n) for c in ast.walk(stmt) if isinstance(c, _skip)
                for n in ast.walk(c)}
        for node in ast.walk(stmt):
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                declared.update(node.names)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                # `except E as exc` binds exc, but stores no ast.Name node, so
                # a later `exc = other` looked like the only assignment and the
                # handler body looked like a read above it.
                prev = assigned.get(node.name)
                assigned[node.name] = (node.lineno if prev is None
                                       else min(prev, node.lineno))
            elif isinstance(node, ast.Name) and id(node) not in skip:
                bucket = assigned if isinstance(node.ctx, ast.Store) else read
                # MIN, not setdefault: ast.walk is breadth-first, so the first
                # node it yields for a name is not the lowest line number. With
                # setdefault this reported "assigned at 598" for a name plainly
                # assigned at 566, and called correct code broken.
                prev = bucket.get(node.id)
                bucket[node.id] = node.lineno if prev is None else min(prev, node.lineno)
    return assigned, read, declared


_order_bad = 0
for src in sorted((ROOT / "kevacv").glob("*.py")) + sorted((ROOT / "tools").glob("*.py")):
    rel = f"{src.parent.name}/{src.name}"
    tree = ast.parse(src.read_text(encoding="utf-8"), rel)
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = {a.arg for a in
                fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs}
        if fn.args.vararg:
            args.add(fn.args.vararg.arg)
        if fn.args.kwarg:
            args.add(fn.args.kwarg.arg)
        assigned, read, declared = _first_lines(fn)
        for name, a_line in assigned.items():
            if name in args or name in declared:
                continue
            r_line = read.get(name)
            if r_line is not None and r_line < a_line:
                _order_bad += 1
                check(False, rel,
                      f"{fn.name}(): '{name}' read at line {r_line}, "
                      f"assigned at line {a_line}")
check(_order_bad == 0, "no local read above its own assignment")

print()
print("=" * 74)
print("  ALL PASS" if not _fail else "  FAILURES ABOVE")
print("=" * 74)
sys.exit(1 if _fail else 0)
