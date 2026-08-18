"""test_config_completeness.py — no setting may exist only as an inline default.

WHY THIS EXISTS
    engine.py reads 29 settings through `globals().get(NAME, <literal>)`. That
    is deliberate late binding. The hazard is that a NAME nobody ever defines
    still WORKS — it silently returns the literal, or None — so the feature is
    off, or pinned, and nothing anywhere says so.

    This has now cost real time three separate ways:

      RENDER_DIRECT_H264   never defined on the package path, so every run
                           silently took the mp4v branch and produced 4.2 GB
                           files browsers refuse to play.
      EVAL_EXPORT          never defined ANYWHERE, no default. The labelling
                           export therefore never ran once — which is why
                           there is still no ground truth, no HOTA, and no
                           training set.
      _BUILD_ID            never set on the package path, so every annotated
                           video said "build ?" while a stale copy on the GPU
                           box produced five days of identical output.

    A missing global is indistinguishable from a disabled feature. This test
    makes that class of bug fail in a second, locally, instead of an hour into
    a run on a rented GPU.

Run: python tests/test_config_completeness.py
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kevacv import config as CFG

FAILED = []

# Names that are RUNTIME STATE, not configuration: set during a run, or
# injected by the caller. Each needs a reason, so the list cannot quietly
# become a dumping ground for "tests were annoying".
RUNTIME_INJECTED = {
    "_BUILD_ID": "set by tools/run_pipeline.py from the package content hash",
    "_EMB_WARNED": "one-shot flag so the embedder warning prints once per run",
    "_RENDER_INDEX_SHIFT": "computed during PASS 2 frame alignment",
    "VIDEO_START_CLOCK": "wall-clock burn-in, defined in kevacv/helpers.py",
}


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


def _engine_tree():
    return ast.parse((ROOT / "kevacv" / "engine.py").read_text(encoding="utf-8"))


def late_bound_names(tree):
    """Every NAME in a globals().get("NAME", ...) call. -> {name: has_default}"""
    out = {}
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "get"
                and isinstance(n.func.value, ast.Call)
                and getattr(n.func.value.func, "id", None) == "globals"
                and n.args and isinstance(n.args[0], ast.Constant)):
            out[n.args[0].value] = len(n.args) > 1
    return out


def module_level_names(tree):
    out = set()
    for n in tree.body:
        if isinstance(n, ast.Assign):
            out |= {t.id for t in n.targets if isinstance(t, ast.Name)}
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            out.add(n.name)
    return out


def test_every_late_bound_name_is_defined_somewhere():
    tree = _engine_tree()
    names = late_bound_names(tree)
    check(len(names) > 20, "the sweep actually found the lookups", str(len(names)))
    defined = module_level_names(tree)
    cfg = {k for k in dir(CFG) if not k.startswith("__")}
    orphans = sorted(n for n in names
                     if n not in cfg and n not in defined
                     and n not in RUNTIME_INJECTED)
    check(orphans == [],
          "no setting exists only as an inline default",
          ", ".join(orphans) if orphans else "none")


def test_runtime_allowlist_is_not_stale():
    """An allowlist entry for a name that no longer exists is a lie about the
    code, and hides the next real orphan behind it."""
    names = late_bound_names(_engine_tree())
    stale = sorted(set(RUNTIME_INJECTED) - set(names))
    check(stale == [], "every allowlisted name is still read by engine.py",
          ", ".join(stale) if stale else "none")


def test_allowlist_entries_have_reasons():
    blank = sorted(k for k, v in RUNTIME_INJECTED.items() if not str(v).strip())
    check(blank == [], "every allowlist entry says why it is exempt",
          ", ".join(blank) if blank else "none")


def test_the_three_that_actually_bit_are_defined():
    """Named individually so a regression on any of them fails with a message
    that says which one, not just 'an orphan appeared'."""
    cfg = {k for k in dir(CFG) if not k.startswith("__")}
    for name in ("RENDER_DIRECT_H264", "EVAL_EXPORT", "ENABLE_PHANTOM_FILTER"):
        check(name in cfg, f"{name} is defined in config.py")


def test_eval_export_defaults_to_off():
    """It writes thousands of JPEGs. Defaulting it ON would surprise a
    production run; defaulting it MISSING is what caused the original bug."""
    check(CFG.EVAL_EXPORT is False, "EVAL_EXPORT default is explicitly False",
          repr(CFG.EVAL_EXPORT))
    check(CFG.EVAL_WINDOW is None, "EVAL_WINDOW default is None (whole chunk)",
          repr(CFG.EVAL_WINDOW))


def main():
    print(__doc__.strip().splitlines()[0])
    for fn in (test_every_late_bound_name_is_defined_somewhere,
               test_runtime_allowlist_is_not_stale,
               test_allowlist_entries_have_reasons,
               test_the_three_that_actually_bit_are_defined,
               test_eval_export_defaults_to_off):
        print(f"\n{fn.__name__}")
        fn()
    print()
    if FAILED:
        print(f"FAILED ({len(FAILED)}): " + ", ".join(FAILED))
        return 1
    print("all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
