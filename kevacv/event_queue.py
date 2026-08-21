"""Events leave the CV thread through a queue, not a database call.

WHY
---
follow_up.txt, and it is a real production risk rather than a style point:

    Bad:   Frame -> inference -> database INSERT -> next frame
    Better: CV pipeline -> Event Queue -> Database Worker
    "The video pipeline should never wait for PostgreSQL."

Right now nothing writes per-event during a run -- everything is buffered in
memory and dumped at the end. That is fine for the 600s chunks this pipeline
has lived on and wrong for the 24h runs it is meant for: a crash at hour 23
loses the whole night, and there is no backpressure story at all.

DESIGN, and it is deliberately small
------------------------------------
A bounded queue plus a worker thread. Two properties matter more than
throughput:

  NEVER BLOCK THE PRODUCER.
      If the sink stalls, the CV loop must keep running. A dropped event that
      is COUNTED as dropped is recoverable; a stalled decoder is not. So the
      queue is bounded and overflow is recorded, never awaited.

  NEVER LOSE AN EVENT SILENTLY.
      `dropped` is part of the result, and the pipeline reports it. This
      codebase already has a habit of stages that "removed nothing all chunk"
      because they were misconfigured rather than unnecessary; a queue that
      quietly discards would be the same failure with worse consequences.

stdlib only -- queue.Queue and threading. No broker, no dependency. The sink is
a callable, so the same class serves a JSONL file today and a DB worker later
without the pipeline knowing which.
"""
from __future__ import annotations

import json
import queue
import threading
from typing import Any, Callable, Dict, List, Optional


class EventQueue:
    def __init__(self, sink: Callable[[List[Dict[str, Any]]], None],
                 maxsize: int = 10000, batch: int = 200,
                 flush_s: float = 2.0):
        self.sink = sink
        self.batch = max(1, int(batch))
        self.flush_s = float(flush_s)
        self._q: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize)
        self._t: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.submitted = 0
        self.written = 0
        self.dropped = 0
        self.sink_errors = 0

    def start(self) -> "EventQueue":
        if self._t is None:
            self._t = threading.Thread(target=self._run, name="event-queue",
                                       daemon=True)
            self._t.start()
        return self

    def put(self, event: Dict[str, Any]) -> bool:
        """Never blocks. Returns False if the event was dropped."""
        self.submitted += 1
        try:
            self._q.put_nowait(event)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _drain(self, buf: List[Dict[str, Any]]) -> None:
        if not buf:
            return
        try:
            self.sink(buf)
            self.written += len(buf)
        except Exception:
            # A failing sink must not kill the run OR pretend it wrote.
            self.sink_errors += 1

    def _run(self) -> None:
        buf: List[Dict[str, Any]] = []
        while True:
            try:
                item = self._q.get(timeout=self.flush_s)
            except queue.Empty:
                self._drain(buf)
                buf = []
                if self._stop.is_set():
                    return
                continue
            if item is None:
                self._drain(buf)
                return
            buf.append(item)
            if len(buf) >= self.batch:
                self._drain(buf)
                buf = []

    def close(self, timeout: float = 10.0) -> Dict[str, Any]:
        self._stop.set()
        if self._t is not None:
            try:
                self._q.put_nowait(None)
            except queue.Full:
                pass
            self._t.join(timeout)
            self._t = None
        return self.stats()

    def stats(self) -> Dict[str, Any]:
        return {"submitted": self.submitted, "written": self.written,
                "dropped": self.dropped, "sink_errors": self.sink_errors,
                "lost": self.submitted - self.written}


def jsonl_sink(path: str) -> Callable[[List[Dict[str, Any]]], None]:
    """Append-only JSONL. A crash at hour 23 keeps hours 1-22."""
    def _sink(rows: List[Dict[str, Any]]) -> None:
        with open(path, "a") as fh:
            for r in rows:
                fh.write(json.dumps(r, default=str) + "\n")
            fh.flush()
    return _sink
