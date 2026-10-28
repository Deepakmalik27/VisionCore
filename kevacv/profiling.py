"""profiling.py — where the time actually goes, per stage, every run.

WHY THIS EXISTS
    The bottleneck on this pipeline has been misdiagnosed three times:

        "it's the detector"   -> wrong, DET_BATCH=24 already batches frames
        "it's the tiling"     -> wrong, 9 calls/frame was not the dominant cost
        "it's the proxy"      -> right, but only found by noticing GPU at 14%

    Each wrong guess cost a run. A cascade — spend compute in proportion to
    information — is only worth building once you know which tier is expensive.
    Selective ReID is a 3-5x win if ReID is 60% of the run and a waste of a week
    if it is 15%, and nothing in the logs could tell those apart.

    So: measure, then decide. This is the gate on Phase C.

WHAT IT COSTS
    One perf_counter pair per stage per frame. At 8 fps over an hour that is
    ~28k timer reads, which is microseconds against a 20-minute run. It is on
    by default because a profiler you have to remember to enable is a profiler
    that is off when you need it.

WHAT IT REPORTS
    Wall time and share per stage, plus calls and per-call cost, so a stage that
    is slow because it is CALLED A LOT reads differently from one that is slow
    per call. Those need opposite fixes: the first wants a cascade gate, the
    second wants a faster kernel.

    Every stage total also lands in the run ledger, so the NEXT run diffs its
    profile automatically and "did that make it faster?" stops being a stopwatch
    and a memory.
"""
from __future__ import annotations

import time
from contextlib import contextmanager


class Profile:
    """Accumulates wall time per named stage. Not thread-safe by design: the
    analysis loop is single-threaded and a lock would cost more than it measures.
    """

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.total = {}      # stage -> seconds
        self.calls = {}      # stage -> count
        self._t0 = None

    @contextmanager
    def stage(self, name):
        if not self.enabled:
            yield
            return
        t = time.perf_counter()
        try:
            yield
        finally:
            self.total[name] = self.total.get(name, 0.0) + (time.perf_counter() - t)
            self.calls[name] = self.calls.get(name, 0) + 1

    def add(self, name, seconds, calls=1):
        """For a stage timed elsewhere (a subprocess, a library callback)."""
        if not self.enabled:
            return
        self.total[name] = self.total.get(name, 0.0) + float(seconds)
        self.calls[name] = self.calls.get(name, 0) + int(calls)

    def start_wall(self):
        self._t0 = time.perf_counter()

    @property
    def wall(self):
        return (time.perf_counter() - self._t0) if self._t0 else 0.0

    def counters(self):
        """{stage_ms: int} for the run ledger — ints so the diff is readable."""
        return {f"t_{k}_ms": int(v * 1000) for k, v in sorted(self.total.items())}

    def describe(self, width=78):
        """The table. Ordered by cost, because the first line is the only one
        most people read."""
        if not self.enabled or not self.total:
            return "  (profiling disabled or nothing recorded)"
        measured = sum(self.total.values())
        wall = self.wall or measured
        L = ["=" * width,
             "  WHERE THE TIME WENT — measured, not assumed",
             "=" * width,
             f"  {'stage':<22}{'seconds':>10}{'% wall':>9}{'calls':>10}{'ms/call':>10}",
             "  " + "-" * (width - 4)]
        for k, v in sorted(self.total.items(), key=lambda kv: -kv[1]):
            n = self.calls.get(k, 0)
            L.append(f"  {k:<22}{v:>10.1f}{100.0*v/max(wall,1e-9):>8.1f}%"
                     f"{n:>10}{1000.0*v/max(n,1):>10.2f}")
        L.append("  " + "-" * (width - 4))
        L.append(f"  {'MEASURED':<22}{measured:>10.1f}{100.0*measured/max(wall,1e-9):>8.1f}%")
        L.append(f"  {'wall clock':<22}{wall:>10.1f}")
        unacc = wall - measured
        if unacc > 0.05 * wall:
            L.append(f"  {'UNACCOUNTED':<22}{unacc:>10.1f}{100.0*unacc/max(wall,1e-9):>8.1f}%"
                     "   <- decode, I/O, or a stage nobody timed")
        L.append("")
        L.append("  A stage slow because it is CALLED A LOT wants a cascade gate.")
        L.append("  A stage slow PER CALL wants a faster kernel (TensorRT, batch).")
        L.append("  Read ms/call before choosing which.")
        L.append("=" * width)
        return "\n".join(L)

    def verdict(self, reid_key="reid", gate_frac=0.30):
        """The Phase C decision, stated by the data rather than by argument."""
        if not self.total:
            return None
        wall = self.wall or sum(self.total.values())
        reid = sum(v for k, v in self.total.items() if reid_key in k)
        frac = reid / max(wall, 1e-9)
        if frac >= gate_frac:
            return (f"ReID is {frac:.0%} of the run — SELECTIVE ReID IS WORTH "
                    f"BUILDING (gate: {gate_frac:.0%}). Embedding only on "
                    f"ambiguous association could return most of this.")
        return (f"ReID is {frac:.0%} of the run — BELOW the {gate_frac:.0%} gate. "
                f"Selective ReID would not pay for itself; the time is elsewhere "
                f"in this table.")
