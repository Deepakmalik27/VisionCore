#!/usr/bin/env python3
"""Is the pipeline actually CONNECTED? Seven ways it silently is not.

WHY THIS EXISTS
    Every bug found on 2026-08-20 was the same shape -- code that exists, looks
    right, has tests, and never runs:

      * 10 modules (1,682 LOC) imported only by __init__.py, called by nothing.
        graph_fusion and reid_engine among them -- the two things most likely to
        move Re-ID separability off 0.658, neither ever executed.
      * ground_plane.py REDEFINED DEFAULT_HFOV_DEG as 82.0 against config.py's
        90.0. focal_px = (frame_w/2)/tan(hfov/2), so one camera had two focal
        lengths 15% apart and every metre threshold inherited whichever module
        did the arithmetic.
      * those same constants were used as SIGNATURE DEFAULTS, which bind once at
        import, so apply_run_config could set the module global and change
        nothing at the call site.
      * apply_run_config propagates to a HARDCODED list of modules; anything
        holding a config value outside it keeps its import-time default forever.
        phantoms was missed once already and a "MEASURED: NO EFFECT" verdict was
        recorded on a knob that never reached the code.
      * the slit counter's reversal guard ran per 15s chunk, so every reversal
        straddling a seam escaped it for a whole 20-minute run.
      * config/cam112.yaml ships fps: 15 while tests assert FPS_TARGET == 8.

    Each was found by hand, one at a time, over a day. Each is a mechanical
    property of the source. This is that hunt, as one command.

WHAT IT CANNOT SEE
    Whether a wired thing is CORRECT. It reports connection, not truth. A knob
    that reaches its module and holds a bad value passes every check here.

USAGE
    python tools/wiring_audit.py            # full report, exit 1 if anything found
    python tools/wiring_audit.py --only C   # one check
"""
from __future__ import annotations

import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG = ROOT / "kevacv"
TOOLS = ROOT / "tools"


def _parse(path):
    try:
        return ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return None


def _mod_stems():
    return sorted(p.stem for p in PKG.glob("*.py") if p.stem != "__init__")


def _prod_files():
    """Files that constitute a real run. __init__.py is excluded on purpose:
    importing a module there proves packaging, not that anything calls it."""
    return [p for p in PKG.glob("*.py") if p.stem != "__init__"] + list(TOOLS.glob("*.py"))


def _imports_of(path, stems):
    """-> {module_stem: [bound names]}, including imports inside functions.

    Alias-aware. A plain name match reports `from .decision_log import Ledger as
    _Ledger` as dead, which is how two earlier counts of this came out wrong.
    """
    out, tree = {}, _parse(path)
    if tree is None:
        return out
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module:
            base = n.module.split(".")[-1]
            if base in stems:
                out.setdefault(base, []).extend(a.asname or a.name for a in n.names)
        elif isinstance(n, ast.Import):
            for a in n.names:
                base = a.name.split(".")[-1]
                if base in stems:
                    out.setdefault(base, []).append(a.asname or base)
    return out


def _names_used(path):
    s, tree = set(), _parse(path)
    if tree is None:
        return s
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            s.add(n.id)
        elif isinstance(n, ast.Attribute):
            s.add(n.attr)
    return s


def _config_constants():
    src = (PKG / "config.py").read_text(encoding="utf-8")
    return set(re.findall(r"^([A-Z][A-Z_0-9]+)\s*=", src, re.M))


def _propagation_list():
    """The hardcoded module tuple apply_run_config writes into.

    Read via AST, not regex. A `for _modname in \\(([^)]*)\\)` pattern truncates
    at the first ')' inside the tuple's own comments -- and one of those
    comments contains `globals()`, so the regex silently returned a SHORT list
    and reported correctly-wired modules as missing.
    """
    tree = _parse(PKG / "config.py")
    if tree is None:
        return set()
    for n in ast.walk(tree):
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name) \
                and n.target.id == "_modname":
            try:
                return set(ast.literal_eval(n.iter))
            except (ValueError, SyntaxError):
                return set()
    return set()


def _run_config_keys():
    src = (PKG / "config.py").read_text(encoding="utf-8")
    block = re.search(r"RUN_CONFIG_KEYS\s*=\s*\{(.*?)\n\}", src, re.S)
    return dict(re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', block.group(1))) if block else {}


# ── the seven checks ────────────────────────────────────────────────────────

# NOT every uncalled module is a wiring mistake. Each entry here was READ and
# the claim in its docstring CHECKED against the pipeline -- because four of
# these were mis-ranked as "wire this next" purely from their prose, and two of
# those would have caused regressions. Removing an entry requires the same.
NOT_PIPELINE = {
    "merge_ab": "BY DESIGN — measurement harness; its own docstring forbids "
                "becoming the production path. tests/test_merge_ab.py holds the "
                "staleness guard against notebooks/pipeline.ipynb.",
    "seams": "BLOCKED — nothing to bridge. No caller ever passes chunk_tag or "
             "start_seconds (always \"\" and 0.0), and run_pipeline.py calls "
             "run_camera exactly once. A multi-chunk driver must exist first.",
    "triage": "BLOCKED — plan_segments needs a cheap `scan` pass; the only "
              "producer of one is tests/.",
    "anomaly_baseline": "BLOCKED — needs {zone_id, hour, count} records; the "
                        "only producer is tests/test_phase_c_d.py:32.",
    "reid_engine": "DO NOT WIRE — scaffold. self.model is set to None and never "
                   "assigned; no loader exists. Its fallback returns an "
                   "average-pooled thumbnail, or on exception a constant vector "
                   "that makes every person cosine-identical.",
    "tracker_wrapper": "DO NOT WIRE AS-IS — SUPPORTED_TRACKERS advertises "
                       "strongsort and ocsort, which have no branch in "
                       "_init_tracker and fall through to a fallback that uses "
                       "the detector's row index as the track id.",
}


def check_A_dead_modules():
    """Modules no production file imports AND uses.

    Modules in NOT_PIPELINE are reported separately: they are uncalled for a
    reason that was verified, not for want of an import line.
    """
    stems = _mod_stems()
    live = set()
    for p in _prod_files():
        used = _names_used(p)
        for m, binds in _imports_of(p, stems).items():
            if m != p.stem and any(b in used for b in binds):
                live.add(m)
    rows = []
    for m in stems:
        if m in live or m == "__main__":
            continue
        if m in NOT_PIPELINE:
            continue
        loc = len((PKG / f"{m}.py").read_text(errors="ignore").splitlines())
        rows.append((f"kevacv/{m}.py", f"{loc} LOC, never called in production"))
    return rows


def check_A2_uncalled_by_design():
    """Uncalled on purpose. Reported so the reason stays visible, not silent."""
    stems = set(_mod_stems())
    rows = []
    for m, why in sorted(NOT_PIPELINE.items()):
        if m not in stems:
            rows.append((f"kevacv/{m}.py", "listed in NOT_PIPELINE but the file "
                                           "is gone — drop the entry"))
        else:
            rows.append((f"kevacv/{m}.py", why))
    return rows


def check_B_shadowed_constants():
    """A config constant REDEFINED in another module = a second source of truth.

    This is the ground_plane DEFAULT_HFOV_DEG bug. No config change can reach a
    shadow, and the two values drift apart silently.

    Reports whether the two values currently AGREE or have DIVERGED. Both are
    findings -- a shadow that agrees today is what the hfov shadow was before it
    drifted to 82.0 against 90.0 -- but only DIVERGED is wrong right now, and
    conflating them buries the live bug in a list of latent ones.
    """
    consts, rows = _config_constants(), []
    cfg_tree = _parse(PKG / "config.py")
    cfg_vals = {}
    for n in (cfg_tree.body if cfg_tree else []):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id in consts:
                    try:
                        cfg_vals[t.id] = ast.literal_eval(n.value)
                    except (ValueError, SyntaxError):
                        pass
    for p in PKG.glob("*.py"):
        if p.stem in ("config", "__init__"):
            continue
        tree = _parse(p)
        if tree is None:
            continue
        for n in tree.body:
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if not (isinstance(t, ast.Name) and t.id in consts):
                        continue
                    try:
                        here = ast.literal_eval(n.value)
                    except (ValueError, SyntaxError):
                        here = "<expr>"
                    there = cfg_vals.get(t.id, "<expr>")
                    same = here == there
                    verdict = (f"AGREES ({here!r}) — latent" if same
                               else f"DIVERGED: config={there!r} here={here!r}")
                    rows.append((f"kevacv/{p.stem}.py:{n.lineno}",
                                 f"shadows {t.id} — {verdict}"))
    return rows


def check_C_signature_defaults():
    """Config constants used as default ARGUMENT values.

    A default binds once, when the def executes. setattr on the module global
    afterwards updates the global and changes nothing at the call site, so the
    knob is dead while looking wired.
    """
    consts, rows = _config_constants(), []
    for p in list(PKG.glob("*.py")) + list(TOOLS.glob("*.py")):
        if p.stem == "config":
            continue
        tree = _parse(p)
        if tree is None:
            continue
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for d in list(n.args.defaults) + list(n.args.kw_defaults):
                    if isinstance(d, ast.Name) and d.id in consts:
                        rows.append((f"{p.parent.name}/{p.name}:{n.lineno}",
                                     f"{n.name}() default binds {d.id} at import"))
    return rows


def check_D_stale_modules():
    """Modules holding a config value but outside the propagation list."""
    consts, prop, rows = _config_constants(), _propagation_list(), []
    for p in PKG.glob("*.py"):
        if p.stem in ("config", "__init__") or p.stem in prop:
            continue
        binds = _imports_of(p, {"config"}).get("config", [])
        held = [b for b in binds if b in consts] or (
            [c for c in consts if re.search(rf"\b{c}\b", p.read_text(errors='ignore'))]
            if "from .config import *" in p.read_text(errors="ignore") else [])
        if held:
            rows.append((f"kevacv/{p.stem}.py",
                         f"holds {len(held)} config value(s), not in propagation list"))
    return rows


def check_E_unsettable_flags():
    """Flags reachable from NEITHER the explicit table NOR the name convention.

    apply_run_config gained `_implicit_key`, which resolves `analysis.foo` to a
    scalar constant FOO, so a flag no longer needs a hand-written table entry to
    be settable. Only a flag that fails BOTH routes is a finding -- checking the
    explicit table alone would report 25 flags that are now perfectly tunable.
    """
    keys, rows = set(_run_config_keys().values()), []
    src = (PKG / "config.py").read_text(encoding="utf-8")
    has_fallback = "_implicit_key" in src
    for m in re.finditer(r"^(ENABLE_[A-Z_0-9]+)\s*=\s*([^\n#]+)", src, re.M):
        name, literal = m.group(1), m.group(2).strip()
        if name in keys:
            continue
        # The convention only resolves scalars; a non-scalar default means even
        # the fallback cannot reach it.
        scalar = literal in ("True", "False") or re.fullmatch(r"-?[\d.]+|'[^']*'|\"[^\"]*\"", literal)
        if not (has_fallback and scalar):
            rows.append((f"kevacv/config.py:{src[:m.start()].count(chr(10))+1}",
                         f"{name} reachable by neither table nor convention"))
    return rows


def check_F_broken_key_mapping():
    """RUN_CONFIG_KEYS pointing at a constant config.py does not define."""
    consts, rows = _config_constants(), []
    for key, name in _run_config_keys().items():
        if name not in consts:
            rows.append((f"RUN_CONFIG_KEYS[{key!r}]",
                         f"maps to {name}, which config.py does not define"))
    return rows


def check_G_dead_tools():
    """Scripts in tools/ that nothing references and that have no CLI."""
    rows = []
    referenced = " ".join(
        p.read_text(errors="ignore")
        for p in list(PKG.glob("*.py")) + list(TOOLS.glob("*.py")) + [ROOT / "run.sh"]
        if p.exists())
    for p in sorted(TOOLS.glob("*.py")):
        if p.stem in ("__init__",):
            continue
        src = p.read_text(errors="ignore")
        has_cli = "__main__" in src
        hits = referenced.count(p.name) + referenced.count(f"tools.{p.stem}")
        if not has_cli and hits <= 1:
            rows.append((f"tools/{p.name}", "no CLI entry and nothing imports it"))
    return rows


def check_H_dead_functions():
    """Public functions/classes inside LIVE modules that nothing calls.

    WHY THIS EXISTS SEPARATELY FROM [A]
        Check [A] is module-granular, so wiring ONE function retires a whole
        module from the dead list. That happened immediately: resilience.py
        left [A] when run_batched was wired, while Checkpoint -- the half that
        makes a crash cost one chunk instead of a night -- stayed uncalled and
        invisible. A module-level count that says "connected" when half of it
        is unreachable is the same kind of comfortable lie as a green test over
        code that never runs.

    Only reports modules where MOST of the API is unused, because a utility
    module legitimately exposes helpers the pipeline does not need today.
    """
    stems, rows = _mod_stems(), []
    used = set()
    for p in _prod_files():
        names = _names_used(p)
        used |= names
        # Count the ORIGINAL name when a symbol is imported under an alias.
        # pipeline.py does `from .provenance import build_stamp as _stamp`, so
        # only `_stamp` appears at the call site and build_stamp read as dead.
        # This is the third time aliasing has produced a wrong count in this
        # file; resolve it everywhere rather than per-check.
        tree = _parse(p)
        for n in ast.walk(tree) if tree else []:
            if isinstance(n, ast.ImportFrom):
                for a in n.names:
                    if a.asname and a.asname in names:
                        used.add(a.name)
    for m in stems:
        path = PKG / f"{m}.py"
        tree = _parse(path)
        if tree is None:
            continue
        public = [n.name for n in tree.body
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                  and not n.name.startswith("_")]
        if len(public) < 2:
            continue
        # `used` already spans every production file INCLUDING this one, and a
        # def's own name is not a Name/Attribute node, so it only ever counts
        # real call sites. Subtracting own-module usage (an earlier attempt)
        # reported pipeline.preflight and engine.apply_clahe as dead -- both are
        # called, just from inside the module that defines them, which is
        # reachable the moment its entry point runs.
        dead = [n for n in public if n not in used]
        if dead and len(dead) < len(public):        # partly wired: the risk case
            rows.append((f"kevacv/{m}.py",
                         f"{len(public) - len(dead)}/{len(public)} of its API is "
                         f"called; unused: {', '.join(sorted(dead)[:4])}"))
    return rows


CHECKS = {
    "A": ("DEAD MODULES — written, tested, never called", check_A_dead_modules),
    "A2": ("UNCALLED BY DESIGN — verified reason, not a wiring gap",
           check_A2_uncalled_by_design),
    "H": ("HALF-WIRED MODULES — live, but most of the API never called",
          check_H_dead_functions),
    "B": ("SHADOWED CONSTANTS — a second source of truth config cannot reach",
          check_B_shadowed_constants),
    "C": ("IMPORT-TIME DEFAULTS — knob set, call site unchanged",
          check_C_signature_defaults),
    "D": ("STALE MODULES — outside apply_run_config's propagation list",
          check_D_stale_modules),
    "E": ("UNSETTABLE FLAGS — no yaml key, source edit only", check_E_unsettable_flags),
    "F": ("BROKEN KEY MAPPING — yaml key points at a missing constant",
          check_F_broken_key_mapping),
    "G": ("DEAD TOOLS — no CLI, nothing imports them", check_G_dead_tools),
}


def main() -> int:
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1].upper()
    total = 0
    print("=" * 78)
    print("  WIRING AUDIT — is the pipeline connected?")
    print("=" * 78)
    for code, (title, fn) in CHECKS.items():
        if only and code != only:
            continue
        rows = fn()
        total += len(rows)
        print(f"\n[{code}] {title}")
        if not rows:
            print("     clean")
            continue
        for where, what in rows:
            print(f"     {where:<44} {what}")
        print(f"     ---- {len(rows)} finding(s)")
    print("\n" + "=" * 78)
    print(f"  {total} finding(s). This reports CONNECTION, not correctness:")
    print("  a knob that reaches its module and holds a bad value passes here.")
    print("=" * 78)
    return 1 if total else 0


def _selftest():
    """One runnable check per detector that has a known-true answer today."""
    # B and C were both real bugs in ground_plane.py and were fixed on
    # 2026-08-20; if either regresses, this fires.
    gp = (PKG / "ground_plane.py").read_text()
    assert "from .config import DEFAULT_HFOV_DEG" in gp, "hfov shadow is back"
    # Use the AST detector, not a string match: `=PERSON_H_M` also appears in
    # the comment explaining why the default was removed, so a substring check
    # fires on its own documentation.
    assert not [r for r in check_C_signature_defaults() if "ground_plane" in r[0]], \
        "ground_plane signature default is back"
    assert not [r for r in check_B_shadowed_constants() if "ground_plane" in r[0]], \
        "ground_plane shadows a config constant again"
    assert {"ground_plane", "venue_profile"} <= _propagation_list(), \
        "ground_plane dropped out of the propagation list"
    keys = _run_config_keys()
    assert keys.get("analysis.hfov_deg") == "DEFAULT_HFOV_DEG"
    assert not check_F_broken_key_mapping(), "a yaml key points at a missing constant"
    assert _config_constants(), "config constants should not be empty"
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        raise SystemExit(main())
