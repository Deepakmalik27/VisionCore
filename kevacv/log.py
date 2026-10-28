"""log.py — one timeline for the whole run, root to leaf.

WHY THIS EXISTS
    The pipeline reports itself with bare print(). That produced a 191,977
    character wall of `'half' is deprecated` in one cell output, buried the
    line that mattered, and gave no way to answer the only question you ever
    ask afterwards: WHERE did the time go, and WHAT was true at that moment.

    Worse, prints have no severity. "🚨 ENTRY LINE NEVER TRIGGERED" and
    "loading model" arrive looking exactly alike, so the loud failure scrolls
    past with the noise.

WHAT THIS GIVES
    A nested stage timeline. Every stage logs when it starts, when it ends, how
    long it took, and what it counted — with the full path from the root, so a
    line is readable on its own:

        12:04:07 INFO  run > chunk1 > pass1 > detect      | 27060 frames, 8 fps
        12:19:44 INFO  run > chunk1 > pass1               | done in 15m37s
        12:19:44 WARN  run > chunk1 > identity            | 356 merges starved
        12:19:45 ERROR run > chunk1 > zones               | entry line fired 0x

    Counters are first-class. `stage.count("frames", 27060)` is recorded
    against the stage and printed in its summary, so the numbers that matter
    end up next to the time they cost — instead of scattered across prints.

DESIGN NOTES
    * stdout AND a file, always. The file is the run's provenance record.
    * A stage that raises still logs its end, with the exception, so a crash
      leaves a complete timeline rather than a truncated one.
    * No global mutable logger config beyond setup(); import order cannot
      change behaviour.
    * Zero dependencies. This must import on a laptop with no GPU.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

_LOCAL = threading.local()          # stage stack is per-thread (2-GPU runner)
_ROOT_NAME = "kevacv"

# THE RUN LEDGER — every stage counter, kept instead of thrown away.
#
# Stage.count() was already first-class, and then the numbers died: they were
# formatted into one log line and dropped. That is why "did my change help?"
# has never been answerable on this pipeline. The measured_baseline block in
# config/cam112.yaml is a HUMAN hand-copying numbers out of a scrolled-back
# log, which is why it says entry_line_crossings: 0 long after that stopped
# being the current truth.
#
# Same counters, same call sites, now written to a JSON next to the .log and
# diffed against the previous run automatically. A run answers "what changed
# since last time" by itself, at the moment the evidence exists.
_RUN = {"path": None, "stages": []}
_COUNTERS_SUFFIX = "_counters.json"

LEVELS = {"DEBUG": logging.DEBUG, "INFO": logging.INFO,
          "WARN": logging.WARNING, "WARNING": logging.WARNING,
          "ERROR": logging.ERROR}


def _stack():
    if not hasattr(_LOCAL, "stack"):
        _LOCAL.stack = []
    return _LOCAL.stack


class _PathFilter(logging.Filter):
    """Kept for handlers that want it explicitly; the record factory below is
    what actually guarantees the field exists."""

    def filter(self, record):
        if not hasattr(record, "stagepath"):
            record.stagepath = " > ".join(_stack()) or "-"
        return True


def _install_record_factory():
    """Stamp `stagepath` onto EVERY record at creation.

    A logging.Filter attached to a logger does not run for records that
    propagate up from a child logger — so `get_logger("pipeline")` produced
    records with no stagepath, and any handler formatting "%(stagepath)s"
    raised or printed nothing. Setting it at the factory means the field
    exists no matter which logger created the record or which handler
    formats it, including handlers a caller attaches itself.
    """
    old = logging.getLogRecordFactory()
    if getattr(old, "_kevacv", False):
        return                      # idempotent: never wrap our own wrapper
    def factory(*args, **kwargs):
        rec = old(*args, **kwargs)
        if not hasattr(rec, "stagepath"):
            rec.stagepath = " > ".join(_stack()) or "-"
        return rec
    factory._kevacv = True
    logging.setLogRecordFactory(factory)


_install_record_factory()


def setup(log_dir="logs", level=None, name=None, stream=True):
    """Configure the run's logger. Safe to call twice; the second call is a
    no-op rather than a duplicated handler (double-printed logs are how people
    stop reading them)."""
    logger = logging.getLogger(_ROOT_NAME)
    if getattr(logger, "_kevacv_configured", False):
        return logger

    level = LEVELS.get(str(level or os.environ.get("KV_LOG_LEVEL", "INFO")).upper(),
                       logging.INFO)
    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-5s %(stagepath)-46s | %(message)s",
        datefmt="%H:%M:%S")
    flt = _PathFilter()

    if stream:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(fmt)
        h.addFilter(flt)
        logger.addHandler(h)

    if log_dir:
        d = Path(log_dir)
        d.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        # append, never with_suffix: run ids and camera names carry dots (V73)
        stem = f"{name or 'run'}_{stamp}"
        fh = logging.FileHandler(str(d / f"{stem}.log"), encoding="utf-8")
        fh.setFormatter(fmt)
        fh.addFilter(flt)
        logger.addHandler(fh)
        _RUN["path"] = d / f"{stem}{_COUNTERS_SUFFIX}"
        _RUN["stages"] = []

    logger._kevacv_configured = True
    return logger


def _record(st, failed=None):
    """Append a finished stage to the run ledger. The stage is still on the
    stack here, so the path is the full root-to-leaf one."""
    row = {"stage": " > ".join(_stack()) or st.name,
           "elapsed_s": round(st.elapsed, 3),
           "counts": dict(st.counts)}
    if failed:
        row["failed"] = failed
    _RUN["stages"].append(row)


def flatten(stages):
    """Ledger rows -> {"stage.counter": value} for the numeric counters.

    Flat keys are what makes two runs comparable: nesting differs between runs
    (a chunk that crashed has fewer levels), a flat key does not.
    """
    out = {}
    for row in stages or ():
        for k, v in (row.get("counts") or {}).items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue        # display strings are not comparable
            out[f"{row['stage']}.{k}"] = v
    return out


def compare(before, after):
    """-> [(key, before, after)] for every counter that moved, appeared or
    vanished. Pure and file-free, so it is testable without a run.

    `None` on either side means the counter did not exist in that run — a
    stage that stopped running at all is exactly as important as a number that
    changed, and the old text logs made it invisible.
    """
    a, b = flatten_or_dict(before), flatten_or_dict(after)
    rows = []
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k), b.get(k)
        if va != vb:
            rows.append((k, va, vb))
    return rows


def flatten_or_dict(x):
    """Accept either a ledger (list of stage rows) or an already-flat dict."""
    return flatten(x) if isinstance(x, list) else dict(x or {})


def _previous_ledger(path):
    """The most recent counters file in this directory that is not `path`."""
    try:
        sibs = sorted(p for p in Path(path).parent.glob("*" + _COUNTERS_SUFFIX)
                      if p != Path(path))
    except Exception:
        return None, None
    if not sibs:
        return None, None
    prev = sibs[-1]
    try:
        return prev, json.loads(prev.read_text(encoding="utf-8")).get("stages")
    except Exception:
        return prev, None


def write_ledger(path=None):
    """Write the run ledger and log a diff against the previous run.

    This is the loop that was missing. Every counter the run already produced
    is kept, and the NEXT run says what moved — so "did that change help?" is
    answered by the run itself instead of by scrolling two logs side by side.
    """
    path = path or _RUN["path"]
    if not path or not _RUN["stages"]:
        return None
    log = get_logger("log")
    prev_path, prev = _previous_ledger(path)
    try:
        Path(path).write_text(json.dumps(
            {"stages": _RUN["stages"]}, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning(f"could not write the run ledger to {path}: {exc}")
        return None

    if not prev:
        log.info(f"run ledger -> {path} (no previous run here to compare)")
        return path
    rows = compare(prev, _RUN["stages"])
    if not rows:
        banner("RUN LEDGER · nothing moved since the previous run",
               [f"previous: {Path(prev_path).name}",
                "Same inputs and same code produce the same counters. If you "
                "changed something and this says nothing moved, the change "
                "did not reach the run."], level="WARN", module="log")
        return path
    # A counter that fell to zero is the failure mode this pipeline keeps
    # hitting: the entry line fired 0x and eight GM-facing numbers silently
    # collapsed with it. Zero is never just another value here.
    zeroed = [r for r in rows if r[2] == 0 and (r[1] or 0) > 0]
    gone = [r for r in rows if r[2] is None]
    lines = [f"previous: {Path(prev_path).name}", ""]
    lines += [f"  {k:<52} {_fmt(a)} -> {_fmt(b)}" for k, a, b in rows[:40]]
    if len(rows) > 40:
        lines.append(f"  ... and {len(rows) - 40} more (see {Path(path).name})")
    if zeroed or gone:
        lines.append("")
        for k, a, b in zeroed:
            lines.append(f"  !! {k} FELL TO ZERO (was {_fmt(a)})")
        for k, a, _ in gone:
            lines.append(f"  !! {k} STOPPED BEING RECORDED (was {_fmt(a)})")
    banner(f"RUN LEDGER · {len(rows)} counter(s) moved since the previous run",
           lines, level="ERROR" if (zeroed or gone) else "INFO", module="log")
    return path


def _fmt(v):
    return "-" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))


def close():
    """Detach and close every handler, releasing the log file.

    Needed for real reasons, not just tests: on Windows an open FileHandler
    keeps a lock on the file, so a run that wants to move, zip or delete its
    own log directory cannot until this is called. It also lets setup() be
    called again with a different destination — one log file per chunk, say.
    """
    logger = logging.getLogger(_ROOT_NAME)
    # Ledger first: it logs its own diff, and that has to land in the file
    # handler that is about to be detached.
    try:
        write_ledger()
    except Exception as exc:                     # never let bookkeeping kill a run
        logger.warning(f"run ledger skipped: {type(exc).__name__}: {exc}")
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    logger._kevacv_configured = False
    _RUN["path"], _RUN["stages"] = None, []
    return logger


def get_logger(module=None):
    """A logger for one module. setup() need not have run — an unconfigured
    logger simply produces nothing, which is correct for `import kevacv` in a
    test."""
    base = logging.getLogger(_ROOT_NAME)
    if not base.handlers:
        base.addHandler(logging.NullHandler())
        base.addFilter(_PathFilter())
    return base.getChild(module) if module else base


class Stage:
    """A named span of work. Returned by stage(); counters attach to it."""

    def __init__(self, name, log):
        self.name = name
        self.log = log
        self.counts = {}
        self.t0 = time.time()

    def count(self, key, value=1):
        """Record a number against this stage. Adds if numeric, else sets."""
        cur = self.counts.get(key)
        self.counts[key] = (cur + value) if isinstance(cur, (int, float)) \
            and isinstance(value, (int, float)) else value
        return self

    def note(self, msg, level="INFO"):
        self.log.log(LEVELS.get(level.upper(), logging.INFO), msg)
        return self

    @property
    def elapsed(self):
        return time.time() - self.t0


def human(seconds):
    s = float(seconds)
    if s < 1:
        return f"{s*1000:.0f}ms"
    if s < 60:
        return f"{s:.1f}s"
    m, s = divmod(int(s), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


@contextmanager
def stage(name, module=None, quiet_under_s=0.0):
    """Time a stage and log its start, end and counters.

        with stage("detect") as st:
            st.count("frames", n)

    An exception is logged at ERROR with the elapsed time and re-raised, so a
    crash still leaves a complete timeline instead of stopping mid-sentence.
    """
    log = get_logger(module)
    st = Stage(name, log)
    _stack().append(name)
    log.info("start")
    try:
        yield st
    except Exception as exc:
        # Record BEFORE re-raising. A crashed run's counters are the most
        # valuable ones there are — they say how far it got.
        _record(st, failed=f"{type(exc).__name__}: {exc}")
        log.error(f"FAILED after {human(st.elapsed)} — "
                  f"{type(exc).__name__}: {exc}")
        raise
    else:
        _record(st)
        if st.elapsed >= quiet_under_s:
            bits = "  ".join(f"{k}={v}" for k, v in st.counts.items())
            log.info(f"done in {human(st.elapsed)}" + (f"   {bits}" if bits else ""))
    finally:
        _stack().pop()


def banner(title, lines=(), level="INFO", module=None):
    """A finding a human must not scroll past. Use for the things that
    invalidate a run — a misplaced entry zone, an unverified clock — not for
    progress."""
    log = get_logger(module)
    lvl = LEVELS.get(level.upper(), logging.INFO)
    log.log(lvl, "=" * 70)
    log.log(lvl, title)
    for ln in lines:
        log.log(lvl, f"  {ln}")
    log.log(lvl, "=" * 70)
