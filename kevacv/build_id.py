"""build_id.py — which code produced this output?

WHY THIS EXISTS
    Five days of runs produced byte-identical results while the source on a
    laptop changed every day. The cause was not a bad fix: the GPU box was
    running a COPY of kevacv/ made once by setup_pod.sh, and bootstrap.sh
    skips setup entirely whenever the venv already exists. The pod re-ran
    frozen code forever and nothing in the output said so.

    tools/stamp_build.py already solved this for the notebook, and its header
    names the lesson exactly: "That is why fixes appeared not to work. Not
    because they were wrong — because there was no way to tell a fresh import
    from a stale one." The codebase path had no equivalent. engine.py even
    burns _BUILD_ID onto the annotated video's HUD, but nothing ever set it,
    so every video ever produced said "build ?".

    This computes the id from the ACTUAL CONTENT of kevacv/*.py. It cannot go
    stale, because there is nothing to remember to update.

HOW TO USE IT
    The id appears in three places, and all three must agree:
      * the first line of the run log
      * the HUD band of the annotated video
      * `python -m kevacv.build_id` on either machine

    If the video says a different id than your laptop, you are looking at
    output from different code. That is now a five-second check instead of a
    five-day mystery.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

PKG = Path(__file__).resolve().parent


def _source_files(pkg_dir=None):
    """Every .py in the package, sorted, excluding caches. Sorted because the
    id must not depend on filesystem enumeration order."""
    d = Path(pkg_dir) if pkg_dir else PKG
    return sorted(p for p in d.glob("*.py") if p.name != "__pycache__")


def compute(pkg_dir=None, length=12):
    """Short content hash of the package source. -> str

    Bytes, not text: a file that differs only by line endings IS a different
    file to Python on the other machine, and pretending otherwise would hide
    exactly the class of drift this exists to catch.
    """
    h = hashlib.sha256()
    for p in _source_files(pkg_dir):
        h.update(p.name.encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:length]


def manifest(pkg_dir=None, length=8):
    """Per-file hashes, so a mismatch can be localised to the file that
    differs instead of only reported as 'something differs'."""
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:length]
            for p in _source_files(pkg_dir)}


def describe(pkg_dir=None, width=78):
    files = _source_files(pkg_dir)
    bid = compute(pkg_dir)
    return "\n".join([
        "=" * width,
        f"  BUILD {bid}   ({len(files)} modules in {Path(pkg_dir or PKG)})",
        "  If the annotated video's HUD shows a different id, the run that",
        "  produced it used different code. Sync before believing the output.",
        "=" * width,
    ])


def diff(other_manifest, pkg_dir=None):
    """What differs between this checkout and another machine's manifest.
    -> {"only_here": [...], "only_there": [...], "changed": [...]}"""
    mine = manifest(pkg_dir)
    theirs = dict(other_manifest or {})
    return {
        "only_here": sorted(set(mine) - set(theirs)),
        "only_there": sorted(set(theirs) - set(mine)),
        "changed": sorted(k for k in set(mine) & set(theirs)
                          if mine[k] != theirs[k]),
    }


if __name__ == "__main__":
    # Run this FILE directly, never `python -m kevacv.build_id`: `-m` imports
    # the package __init__, which pulls in numpy, cv2, matplotlib and the rest.
    # The build id must be computable on any machine with a bare Python —
    # including the deploy script's side of an ssh connection, before anything
    # is installed. Nothing above this line imports more than hashlib+pathlib.
    import json
    import sys
    if "--manifest" in sys.argv:
        print(json.dumps(manifest(), indent=2, sort_keys=True))
    elif "--id" in sys.argv:
        print(compute())
    else:
        print(describe())
