"""run_night.py — headless production runner for the video-surv notebook.

The notebook stays the engine (it is the validated, build-stamped asset);
this wrapper runs it unattended on a pod and parameterizes it per run.

    KV_CHUNK_FILTER="7-28-2026, 7.30.00pm" KV_USE_DRIVE=True python run_night.py
    python run_night.py --dry            # show planned overrides, touch nothing

How parameters work: every environment variable KV_<NAME>=<value> becomes a
config override injected as a new cell IMMEDIATELY AFTER the CONFIG cell —
after the defaults are defined, before anything reads them. Values are parsed
as Python literals when possible ("True", "600", "[1,2]"), else kept as
strings. So ANY Cell-2 knob is a CLI knob with zero notebook edits:

    KV_USE_DRIVE=True  KV_CHUNK_FILTER=""        # full night from Drive
    KV_PROVE_SECONDS=600                          # 10-min proof first
    KV_RUN_ABLATION=True                          # session-2 ablation
    KV_ENABLE_TENSORRT=True                       # flip the TRT engine on

Outputs land in poc_output/ as always, PLUS the fully-executed notebook is
saved next to it (executed_<stamp>.ipynb) — the run's own outputs become the
run's provenance record.

Exit code: 0 = notebook completed; 1 = a cell raised; the executed notebook
is saved EITHER WAY so a failure is always inspectable.
"""
import ast
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB_PATH = Path(os.environ.get("KV_NOTEBOOK",
                              HERE.parent / "video-surv (3).ipynb"))
OUT_DIR = Path(os.environ.get("KV_OUTDIR", Path.cwd()))


def collect_overrides():
    out = {}
    for k, v in sorted(os.environ.items()):
        if not k.startswith("KV_") or k in ("KV_NOTEBOOK", "KV_OUTDIR"):
            continue
        name = k[3:]
        try:
            val = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            val = v
        out[name] = val
    return out


def inject(nb, overrides):
    """Insert the override cell right after the CONFIG cell."""
    if not overrides:
        return nb
    lines = ["# === run_night.py overrides (from KV_* environment) ==="]
    lines += [f"{n} = {v!r}" for n, v in overrides.items()]
    summary = "run_night overrides: " + ", ".join(
        f"{n}={v!r}" for n, v in overrides.items())
    lines += [f"print({summary!r})"]
    cell = {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": [l + "\n" for l in lines]}
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code" and "# Cell 2 — CONFIG" in "".join(c["source"]):
            nb["cells"].insert(i + 1, cell)
            return nb
    raise SystemExit("CONFIG cell not found — wrong notebook?")


def main():
    dry = "--dry" in sys.argv
    overrides = collect_overrides()
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    src = "".join("".join(c["source"]) for c in nb["cells"])
    build = src.split('_BUILD_ID = "')[1].split('"')[0] if '_BUILD_ID = "' in src else "?"
    print(f"notebook : {NB_PATH.name}  (build {build}, {len(nb['cells'])} cells)")
    print(f"overrides: {overrides or '(none — notebook defaults)'}")
    nb = inject(nb, overrides)
    if dry:
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] == "code":
                ast.parse("".join(c["source"]))
        print("--dry: all cells parse, override cell injected cleanly. "
              "Nothing executed.")
        return 0

    from nbclient import NotebookClient          # pip install nbclient
    import nbformat
    nbf = nbformat.reads(json.dumps(nb), as_version=4)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    exec_path = OUT_DIR / f"executed_{build}_{stamp}.ipynb"
    client = NotebookClient(nbf, timeout=None,
                            kernel_name="python3",
                            resources={"metadata": {"path": str(OUT_DIR)}})
    t0 = time.time()
    rc = 0
    try:
        client.execute()
        print(f"✅ notebook completed in {(time.time() - t0) / 60:.1f} min")
    except Exception as e:
        rc = 1
        print(f"❌ a cell failed after {(time.time() - t0) / 60:.1f} min: "
              f"{type(e).__name__}: {str(e)[:400]}")
    finally:
        # provenance either way — a failure you can't inspect is two failures
        exec_path.write_text(nbformat.writes(nbf), encoding="utf-8")
        print(f"executed notebook (with outputs) -> {exec_path}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
