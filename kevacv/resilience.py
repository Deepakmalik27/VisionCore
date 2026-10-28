"""resilience.py — an 8-hour run must not die at hour 7.

WHY THIS EXISTS
    DET_BATCH is 12 at imgsz 1280. Batch size is chosen for the average frame,
    but VRAM is consumed by the worst one — a crowded doorway with twenty
    bodies costs far more than an empty corridor. So the run survives seven
    hours of quiet footage and dies on the busiest minute of the night, which
    is the minute the report exists to describe.

    There is no OOM handling anywhere in the pipeline, no empty_cache, and no
    per-chunk checkpoint. A crash at hour 7 currently costs all seven hours.

TWO GUARANTEES
    1. An OOM is survivable. Halve the batch, empty the cache, retry. A frame
       processed slowly is worth infinitely more than a frame not processed.
    2. A crash costs ONE chunk, not the night. Checkpoint after each chunk so a
       resumed run skips what is already done.

WHY NOT JUST USE A SMALLER BATCH
    Because then every quiet hour pays the cost of the busy minute. Adaptive
    beats conservative: start fast, degrade only where the footage demands it,
    and record where that happened — a batch that had to shrink is itself a
    finding about crowd density.
"""
from __future__ import annotations

import json
from pathlib import Path

from .log import get_logger

_log = get_logger("resilience")

MIN_BATCH = 1


def _is_oom(exc):
    """True for a CUDA out-of-memory, without importing torch to find out."""
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return "out of memory" in text or "cuda oom" in text


def _empty_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            return True
    except Exception:
        pass
    return False


def run_batched(items, fn, batch=12, min_batch=MIN_BATCH, on_shrink=None):
    """Apply `fn(chunk_of_items)` in batches, surviving out-of-memory.

    Yields results in order. On OOM the batch is halved and retried; the
    smaller size STICKS for the rest of the run, because a scene that OOMed
    once will do it again a second later — retrying at the original size every
    time turns one crowded minute into thousands of failed attempts.
    """
    i, n = 0, len(items)
    shrinks = 0
    while i < n:
        take = min(batch, n - i)
        try:
            out = fn(items[i:i + take])
        except Exception as exc:
            if not _is_oom(exc) or batch <= min_batch:
                raise
            _empty_cache()
            batch = max(min_batch, batch // 2)
            shrinks += 1
            _log.warning(f"CUDA out of memory — batch reduced to {batch} and "
                         f"retrying (shrink #{shrinks}). A slow frame beats a "
                         f"missing one.")
            if on_shrink:
                on_shrink(batch, i)
            continue
        for r in (out or []):
            yield r
        i += take
    if shrinks:
        _log.warning(f"batch was reduced {shrinks} time(s) this run; final "
                     f"batch {batch}. Crowded stretches cost more VRAM than "
                     f"the average frame — that is itself a density signal.")


class Checkpoint:
    """Which chunks are already done, so a crash costs one chunk not a night."""

    def __init__(self, path):
        self.path = Path(path)
        self.done = {}
        if self.path.exists():
            try:
                self.done = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception as e:
                _log.warning(f"checkpoint unreadable ({e}); starting fresh")
                self.done = {}

    def is_done(self, key):
        return str(key) in self.done

    def mark(self, key, info=None):
        self.done[str(key)] = info or True
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # append, never with_suffix — chunk keys carry dots (V73)
            tmp = Path(str(self.path) + ".tmp")
            tmp.write_text(json.dumps(self.done, indent=1), encoding="utf-8")
            tmp.replace(self.path)      # atomic: a killed process cannot
        except Exception as e:          # leave a half-written checkpoint
            _log.warning(f"could not write checkpoint ({e})")
        return self

    def pending(self, keys):
        return [k for k in keys if not self.is_done(k)]

    def summary(self):
        return {"done": len(self.done), "keys": sorted(self.done)}
