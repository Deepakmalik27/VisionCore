"""test_no_missing_names.py — nothing may reference a name that does not exist.

WHY THIS EXISTS
    Extracting notebook cells into modules loses the shared namespace. A cell
    could use ZONE_HEXES because Cell 6 happened to define it; a module cannot.
    Python resolves globals at CALL time, so the import succeeds and the
    NameError waits — in the worst case until after a 3.4 GB download, the
    model load and the first frame.

    That happened three times in one afternoon (ZONE_HEXES, _safe_id, cv2 in
    detect_filters), each costing a full download-and-retry cycle. Finding them
    one per run is the wrong method. This finds all of them at once, statically,
    in under a second, and fails the suite if a new one appears.

WHAT IT DOES NOT CATCH
    Names created dynamically (globals()[...] = x) or supplied by a caller at
    runtime. engine.py deliberately reads late-bound globals, so its runtime
    placeholders are declared at module level and this check sees them.

Run: python tests/test_no_missing_names.py
"""
import ast
import builtins
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


def star_imported(tree, pkg="kevacv"):
    """Names a `from .x import *` brings in, resolved by importing x."""
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names):
            mod = n.module or ""
            try:
                m = importlib.import_module(f"{pkg}.{mod}" if n.level else mod)
            except Exception:
                continue
            out |= {k for k in dir(m) if not k.startswith("_")}
            out |= {k for k in dir(m) if k.isupper()}
    return out


# Always present in any module, so not "missing" however they are used.
MODULE_DUNDERS = {"__file__", "__name__", "__doc__", "__package__", "__spec__",
                  "__loader__", "__builtins__", "__debug__", "__path__"}


def unresolved(path):
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    bound = set(MODULE_DUNDERS)
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            bound.add(n.id)
        elif isinstance(n, ast.arg):
            bound.add(n.arg)
        elif isinstance(n, ast.Import):
            for a in n.names:
                bound.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                if a.name != "*":
                    bound.add(a.asname or a.name)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            bound |= set(n.names)
        elif isinstance(n, ast.comprehension):
            for t in ast.walk(n.target):
                if isinstance(t, ast.Name):
                    bound.add(t.id)
    bound |= star_imported(tree)
    used = {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return sorted(u for u in used - bound if not hasattr(builtins, u))


print("=" * 74)
print("  every module resolves every name it uses")
print("=" * 74)
mods = sorted(p for p in (ROOT / "kevacv").glob("*.py"))
total = 0
for p in mods:
    miss = unresolved(p)
    total += len(miss)
    if miss:
        check(False, f"{p.name}", ", ".join(miss[:8]))
if total == 0:
    check(True, f"all {len(mods)} modules clean", "no NameError can hide here")

print()
print("=" * 74)
print("  and the tools scripts too")
print("=" * 74)
tmiss = 0
for p in sorted((ROOT / "tools").glob("*.py")):
    miss = unresolved(p)
    tmiss += len(miss)
    if miss:
        check(False, f"tools/{p.name}", ", ".join(miss[:8]))
if tmiss == 0:
    check(True, "tools are clean")

print()
print("=" * 74)
# PYTEST VISIBILITY. This file's checks run at IMPORT and reported only via
# an exit code. A module-level sys.exit aborts pytest COLLECTION, which hid
# 59 of 74 test files -- and simply guarding the exit would have hidden the
# FAILURES instead. So the same condition is also asserted as a real test.
def test_script_level_checks_passed():
    assert not (FAILED), "module-level checks in this file failed"


if __name__ == "__main__":
    if FAILED:
        print(f"  {len(FAILED)} FAILED:")
        for f in FAILED:
            print(f"    - {f}")
        print()
        print("  A missing name does not fail at import — it fails at CALL time,")
        print("  which on this pipeline means after the download and the model load.")
        sys.exit(1)
print("  ALL PASS")
print("=" * 74)
