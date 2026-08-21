"""A per-run ledger of WHAT happened, WHY, and WHAT IT COST.

WHY THIS EXISTS
    Today produced two confident headline numbers that were both withdrawn.
    Neither came from a bad model -- they came from not being able to see what
    the pipeline was actually doing:

      * "tier-A crossing dedupe: 52 -> 45 (id churn at the line)" -- the log
        said id churn. The real cause was a door filter deleting every
        interior crossing. The message named the wrong mechanism, so the
        funnel was unreadable.
      * a yaml knob reported "0.37 -> 0.60 applied" while the code that merges
        people kept using 0.37, because the value reached one module and not
        another.
      * a knob was A/B'd, recorded as "MEASURED: NO EFFECT", and never reached
        its module at all.
      * "MEASURED 36.4% -> 54.1%" for a parameter that only affects the
        rendered video and cannot change a count.

    Every one is the same failure: a number with no provenance. The fix is not
    more logging -- the run already prints thousands of lines -- it is
    STRUCTURED provenance: for each stage, what came in, what left, what was
    dropped and on whose authority, which parameters were in force, and where
    each parameter's value came from.

WHAT IT RECORDS PER STAGE
    module + the source location that owns the decision
    inputs / outputs / dropped, each with a reason string
    every parameter READ, tagged with its provenance:
        config   set from a venue yaml (and which key)
        default  the module default, nobody chose it for this venue
        fallback a value used because the preferred path failed
        derived  computed from other values (records the formula)
        hardcoded a literal in the code with no knob at all  <-- the ones
                 that cannot be A/B'd, which is how drift hides
    warnings, each with WHY IT MATTERS and the LIKELY CAUSE
    the files it read and wrote

OUTPUT
    <out>/LEDGER.json   machine-readable, diffable between two runs
    <out>/LEDGER.txt    the same thing a human can read top to bottom
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path

CONFIG, DEFAULT, FALLBACK, DERIVED, HARDCODED = (
    "config", "default", "fallback", "derived", "hardcoded")


class Ledger:
    def __init__(self, run_id="run", enabled=True):
        self.run_id = run_id
        self.enabled = enabled
        self.stages = []
        self._cur = None
        self._t0 = time.time()

    # ---------------------------------------------------------------- stages
    @contextmanager
    def stage(self, name, *, module=None, owns=None, does=None):
        """owns: file:line that makes the decision.  does: one plain sentence."""
        st = {"stage": name, "module": module, "owns": owns, "does": does,
              "t_start_s": round(time.time() - self._t0, 2),
              "params": [], "flow": [], "drops": [], "warnings": [],
              "files": [], "notes": []}
        prev, self._cur = self._cur, st
        try:
            yield self
        except Exception as exc:
            st["failed"] = f"{type(exc).__name__}: {exc}"
            st["warnings"].append({
                "what": f"stage raised {type(exc).__name__}",
                "why_it_matters": "a stage that fails silently leaves its "
                                  "inputs in the count as if it had passed",
                "likely_cause": str(exc)})
            raise
        finally:
            st["t_end_s"] = round(time.time() - self._t0, 2)
            self.stages.append(st)
            self._cur = prev

    def _s(self):
        if self._cur is None:
            self._cur = {"stage": "(outside any stage)", "params": [],
                         "flow": [], "drops": [], "warnings": [],
                         "files": [], "notes": []}
            self.stages.append(self._cur)
        return self._cur

    # ---------------------------------------------------------------- record
    def param(self, name, value, source=DEFAULT, *, why=None, key=None,
              formula=None, alternatives=None):
        """Record a parameter AND where its value came from."""
        self._s()["params"].append({
            "name": name, "value": _safe(value), "source": source,
            "yaml_key": key, "why": why, "formula": formula,
            "alternatives_considered": alternatives})
        return value

    def flow(self, what, n_in=None, n_out=None, *, unit="items", note=None):
        self._s()["flow"].append({"what": what, "in": n_in, "out": n_out,
                                  "unit": unit, "note": note})

    def drop(self, n, what, *, why, authority=None, recoverable=None):
        """authority: which rule/threshold decided. recoverable: how to get it back."""
        self._s()["drops"].append({"n": n, "what": what, "why": why,
                                   "authority": authority,
                                   "how_to_recover": recoverable})

    def warn(self, what, *, why_it_matters, likely_cause=None, evidence=None):
        self._s()["warnings"].append({"what": what,
                                      "why_it_matters": why_it_matters,
                                      "likely_cause": likely_cause,
                                      "evidence": _safe(evidence)})

    def file(self, path, mode="read", *, what=None):
        self._s()["files"].append({"path": str(path), "mode": mode,
                                   "what": what})

    def note(self, text):
        self._s()["notes"].append(text)

    # ---------------------------------------------------------------- output
    def summary(self):
        hard = [(s["stage"], p) for s in self.stages for p in s["params"]
                if p["source"] == HARDCODED]
        fall = [(s["stage"], p) for s in self.stages for p in s["params"]
                if p["source"] == FALLBACK]
        drops = [(s["stage"], d) for s in self.stages for d in s["drops"]]
        warns = [(s["stage"], w) for s in self.stages for w in s["warnings"]]
        return {"stages": len(self.stages), "hardcoded": hard,
                "fallbacks": fall, "drops": drops, "warnings": warns}

    def write(self, out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        doc = {"run_id": self.run_id,
               "duration_s": round(time.time() - self._t0, 2),
               "stages": self.stages}
        (out / "LEDGER.json").write_text(json.dumps(doc, indent=1, default=str),
                                         encoding="utf-8")
        (out / "LEDGER.txt").write_text(self.render(), encoding="utf-8")
        return [out / "LEDGER.json", out / "LEDGER.txt"]

    def render(self):
        L = ["=" * 78, f"  RUN LEDGER — {self.run_id}",
             "  what ran, what it consumed, what it dropped, and on whose authority",
             "=" * 78]
        for s in self.stages:
            L.append("")
            L.append(f"── {s['stage']}"
                     + (f"   [{s['module']}]" if s.get("module") else ""))
            if s.get("does"):
                L.append(f"     {s['does']}")
            if s.get("owns"):
                L.append(f"     decided at: {s['owns']}")
            if s.get("failed"):
                L.append(f"     !! FAILED: {s['failed']}")
            for f in s["flow"]:
                a, b = f.get("in"), f.get("out")
                arrow = (f"{a} -> {b}" if a is not None and b is not None
                         else f"{a if a is not None else b}")
                L.append(f"     flow   {f['what']}: {arrow} {f['unit']}"
                         + (f"   ({f['note']})" if f.get("note") else ""))
            for d in s["drops"]:
                L.append(f"     DROP   {d['n']} {d['what']}")
                L.append(f"            why: {d['why']}")
                if d.get("authority"):
                    L.append(f"            authority: {d['authority']}")
                if d.get("how_to_recover"):
                    L.append(f"            recover: {d['how_to_recover']}")
            for p in s["params"]:
                tag = p["source"].upper()
                line = f"     param  {p['name']} = {p['value']}   [{tag}"
                line += f" {p['yaml_key']}]" if p.get("yaml_key") else "]"
                L.append(line)
                if p.get("formula"):
                    L.append(f"            = {p['formula']}")
                if p.get("why"):
                    L.append(f"            {p['why']}")
                if p.get("alternatives_considered"):
                    L.append(f"            alternatives: "
                             f"{p['alternatives_considered']}")
            for w in s["warnings"]:
                L.append(f"     WARN   {w['what']}")
                L.append(f"            matters because: {w['why_it_matters']}")
                if w.get("likely_cause"):
                    L.append(f"            likely cause: {w['likely_cause']}")
                if w.get("evidence"):
                    L.append(f"            evidence: {w['evidence']}")
            for f in s["files"]:
                L.append(f"     file   {f['mode']:>5} {f['path']}"
                         + (f"   ({f['what']})" if f.get("what") else ""))
            for n in s["notes"]:
                L.append(f"     note   {n}")

        sm = self.summary()
        L += ["", "=" * 78, "  WHAT TO LOOK AT FIRST", "=" * 78]
        L.append(f"  HARDCODED values with no knob: {len(sm['hardcoded'])}"
                 + ("   <- these cannot be A/B'd, which is how drift hides"
                    if sm["hardcoded"] else ""))
        for stage, p in sm["hardcoded"]:
            L.append(f"     {stage}: {p['name']} = {p['value']}"
                     + (f"   ({p['why']})" if p.get("why") else ""))
        L.append(f"  FALLBACKS in force: {len(sm['fallbacks'])}"
                 + ("   <- the preferred path failed; the number is a "
                    "degraded answer" if sm["fallbacks"] else ""))
        for stage, p in sm["fallbacks"]:
            L.append(f"     {stage}: {p['name']} = {p['value']}"
                     + (f"   ({p['why']})" if p.get("why") else ""))
        tot = sum(d["n"] or 0 for _s, d in sm["drops"])
        L.append(f"  TOTAL DROPPED across all stages: {tot}")
        for stage, d in sorted(sm["drops"], key=lambda x: -(x[1]["n"] or 0))[:8]:
            L.append(f"     {d['n']:>7} {d['what']} ({stage}) — {d['why']}")
        L.append(f"  WARNINGS: {len(sm['warnings'])}")
        for stage, w in sm["warnings"]:
            L.append(f"     [{stage}] {w['what']}")
        L.append("=" * 78)
        return "\n".join(L)


def _safe(v):
    try:
        json.dumps(v)
        return v
    except Exception:
        return str(v)


NULL = Ledger(enabled=False)
