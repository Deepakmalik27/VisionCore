# archive/ — the notebook era

Nothing in here is on the run path. `tools/run_pipeline.py -> kevacv.pipeline.run_camera`
is the only entry point; see `docs/ARCHITECTURE_MAP.md`.

Archived 2026-08-24 (not deleted — reversible, and the notebook is still the
provenance record for every result produced before the package existed).

    notebooks/pipeline.ipynb                 the 34-cell original engine
    notebooks/pipeline.BACKUP-preembed.ipynb its pre-embed backup
    tools/embed_kevacv.py                    froze kevacv/ INTO the notebook
    tools/extract_pipeline.py                lifted cells OUT into kevacv/
    tools/build_clean_from_source.py         rebuilt the .ipynb from kevacv/
    tools/verify_notebook_health.py          notebook AST/runtime audit
    tools/audit_notebook_edge_cases.py       already dead (wiring_audit [G])
    tools/run_night.py                       papermill runner for the notebook

WHAT WAS DELIBERATELY *NOT* ARCHIVED
    The "EXTRACTED FROM notebook Cell N" docstrings in kevacv/*.py stay. They
    are not notebook debt, they are the reason each module is shaped the way it
    is -- config.py exists because Cell 2e silently overrode three Cell 2
    constants and the audit printed the wrong ones. Delete the history and the
    bug comes back.
