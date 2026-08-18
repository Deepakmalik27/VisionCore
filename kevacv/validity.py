"""validity.py — "I did not observe" is not "I observed nothing".

THE PRINCIPLE
    arrivals.py already states it: *0 and "cannot tell" must never look the
    same in a report.* That is not a rule about arrivals. It is the shape of
    almost every silent failure this pipeline has produced:

        camera went black        vs  the room was empty
        detector stopped working vs  nobody was there
        motion gate skipped it   vs  nothing happened
        the clock was wrong      vs  that is the real duration
        the entry line is broken vs  no one arrived
        the render failed        vs  nobody appeared in any frame

    Four of those have already shipped as wrong numbers. They are not six bugs
    needing six patches; they are one missing field.

WHAT THIS MODULE IS
    The field. Every frame gets a verdict. The ledger accumulates OBSERVED
    time separately from ELAPSED time, so any metric can divide by the right
    denominator.

    The pipeline already does this correctly at the coarsest level —
    covered_windows() / missing_hours, and desk_covered_pct divides by footage
    we actually have. This is the same idea one level down, where the failures
    actually happen.

WHAT IT DOES NOT DO
    It never repairs anything and never drops a frame on its own. It records a
    verdict and lets the caller decide. A module that silently discards data is
    indistinguishable from the bugs it exists to catch.
"""
from __future__ import annotations

from collections import Counter

from .log import get_logger

_log = get_logger("validity")

# ── verdicts ────────────────────────────────────────────────────────────────
OK = "ok"
BLIND_CAMERA = "blind_camera"        # black / near-zero variance: nothing to see
SKIPPED_IDLE = "skipped_idle"        # motion gate declined to look
DETECTOR_BLIND = "detector_blind"    # motion present, detector returned nothing
BAD_FRAME = "bad_frame"              # decode failure / None
GEOMETRY_CHANGED = "geometry_changed"  # resolution changed mid-stream
TIME_WENT_BACKWARDS = "time_backwards"  # non-monotonic timestamp

# A frame with any of these was NOT observed. Time under them must never be
# counted as "we watched and saw nobody".
NOT_OBSERVED = {BLIND_CAMERA, BAD_FRAME, GEOMETRY_CHANGED, TIME_WENT_BACKWARDS}

# Default: a frame darker and flatter than this cannot contain a person.
# Chosen against IR footage, which is legitimately dark but never FLAT — a live
# IR frame still has texture. A blanked sensor has neither.
BLACK_MEAN = 12.0
BLACK_STD = 3.0


def frame_validity(frame, *, black_mean=BLACK_MEAN, black_std=BLACK_STD,
                   expect_shape=None):
    """-> (verdict, detail). Cheap enough to run on every sampled frame.

    Deliberately checks variance as well as brightness. Infrared footage is
    dark but textured; a dead sensor, a covered lens or a dropped stream is
    dark AND flat. Testing brightness alone would condemn every IR frame.
    """
    if frame is None:
        return BAD_FRAME, "decoder returned None"
    try:
        h, w = frame.shape[0], frame.shape[1]
    except Exception:
        return BAD_FRAME, "frame has no shape"
    if h == 0 or w == 0:
        return BAD_FRAME, f"zero-size frame {w}x{h}"
    if expect_shape and (h, w) != tuple(expect_shape):
        return (GEOMETRY_CHANGED,
                f"resolution changed {tuple(expect_shape)} -> {(h, w)}; "
                f"zones were scaled for the old size")
    try:
        m = float(frame.mean())
        s = float(frame.std())
    except Exception:
        return OK, ""                      # not an array we can measure; trust it
    if m <= black_mean and s <= black_std:
        return BLIND_CAMERA, f"mean={m:.1f} std={s:.1f} — no signal, not an empty room"
    return OK, ""


class ValidityLedger:
    """Observed time vs elapsed time, and why the difference exists.

    Every metric that says "X% of the night" should divide by
    `observed_seconds`, not by the wall-clock span. Dividing by elapsed time
    turns a blind camera into a well-behaved quiet venue.
    """

    def __init__(self, step_s=None):
        self.rows = []               # (t, verdict)
        self.step_s = step_s
        self.counts = Counter()
        self._last_t = None
        self._t0 = None
        self._t1 = None

    def record(self, t, verdict=OK, detail=""):
        t = float(t)
        if self._last_t is not None and t < self._last_t:
            verdict = TIME_WENT_BACKWARDS
            detail = detail or f"t={t:.3f} after t={self._last_t:.3f}"
        self._last_t = max(t, self._last_t) if self._last_t is not None else t
        self._t0 = t if self._t0 is None else min(self._t0, t)
        self._t1 = t if self._t1 is None else max(self._t1, t)
        self.rows.append((t, verdict))
        self.counts[verdict] += 1
        if verdict in NOT_OBSERVED and detail:
            _log.warning(f"{verdict} at t={t:.1f}s — {detail}")
        return verdict

    # ── derived ─────────────────────────────────────────────────────────────
    @property
    def elapsed_seconds(self):
        if self._t0 is None:
            return 0.0
        return max(0.0, self._t1 - self._t0)

    def _step(self):
        if self.step_s:
            return float(self.step_s)
        ts = sorted(t for t, _ in self.rows)
        gaps = [b - a for a, b in zip(ts, ts[1:]) if b > a]
        return sorted(gaps)[len(gaps) // 2] if gaps else 0.0

    @property
    def observed_seconds(self):
        """Time we actually looked at usable pixels.

        SKIPPED_IDLE counts as observed: the motion gate looked, decided
        nothing was moving, and that IS an observation. BLIND_CAMERA does not.
        """
        step = self._step()
        n = sum(1 for _, v in self.rows if v not in NOT_OBSERVED)
        return n * step

    def observed_windows(self, join_gap_s=None):
        """Contiguous stretches actually observed, for clipping any metric."""
        step = self._step()
        join = join_gap_s if join_gap_s is not None else step * 1.5
        out = []
        for t, v in sorted(self.rows):
            if v in NOT_OBSERVED:
                continue
            if out and t - out[-1][1] <= join:
                out[-1][1] = t + step
            else:
                out.append([t, t + step])
        return [tuple(w) for w in out]

    def summary(self):
        el = self.elapsed_seconds
        ob = self.observed_seconds
        return {"frames": len(self.rows),
                "elapsed_s": round(el, 1),
                "observed_s": round(ob, 1),
                "unobserved_s": round(max(0.0, el - ob), 1),
                "observed_share": round(ob / el, 4) if el > 0 else None,
                "by_verdict": dict(self.counts)}

    def findings(self, blind_share_error=0.02):
        """-> [(level, message)]. What a human must be told about this run."""
        out = []
        s = self.summary()
        blind = self.counts.get(BLIND_CAMERA, 0)
        if blind and s["frames"]:
            share = blind / s["frames"]
            lvl = "ERROR" if share >= blind_share_error else "WARN"
            out.append((lvl, f"camera was BLIND for {share*100:.1f}% of sampled "
                             f"frames ({blind}) — that time is not an empty "
                             f"room and is excluded from every denominator"))
        if self.counts.get(TIME_WENT_BACKWARDS):
            out.append(("ERROR", f"{self.counts[TIME_WENT_BACKWARDS]} frame(s) "
                                 f"arrived with a timestamp earlier than the "
                                 f"one before — durations cannot be trusted"))
        if self.counts.get(GEOMETRY_CHANGED):
            out.append(("ERROR", f"{self.counts[GEOMETRY_CHANGED]} resolution "
                                 f"change(s) mid-stream — zone polygons were "
                                 f"scaled for the original size only"))
        if self.counts.get(BAD_FRAME):
            out.append(("WARN", f"{self.counts[BAD_FRAME]} undecodable frame(s)"))
        if self.counts.get(DETECTOR_BLIND):
            out.append(("ERROR", f"{self.counts[DETECTOR_BLIND]} frame(s) had "
                                 f"motion but ZERO detections — a detector that "
                                 f"has stopped working looks exactly like an "
                                 f"empty venue"))
        return out


class DetectorCanary:
    """Motion happened and the detector saw nobody. How often, and in a row?

    A model that fails to load its weights, or whose confidence collapses when
    the scene goes infrared, produces a perfectly quiet report. The motion gate
    makes that look deliberate. This is the only cheap way to tell the
    difference without ground truth: the frame differencer and the detector are
    independent, so sustained disagreement means one of them is broken.
    """

    def __init__(self, run_length=30):
        self.run_length = run_length
        self.current = 0
        self.longest = 0
        self.total = 0
        self.episodes = []
        self._start_t = None

    def observe(self, t, motion, n_detections):
        if motion and n_detections == 0:
            self.total += 1
            if self.current == 0:
                self._start_t = t
            self.current += 1
            self.longest = max(self.longest, self.current)
        else:
            if self.current >= self.run_length:
                self.episodes.append((self._start_t, t, self.current))
            self.current = 0
        return self

    def close(self, t=None):
        if self.current >= self.run_length:
            self.episodes.append((self._start_t, t, self.current))
        self.current = 0
        return self

    def findings(self):
        if not self.episodes:
            return []
        worst = max(e[2] for e in self.episodes)
        return [("ERROR",
                 f"detector returned NOTHING across {len(self.episodes)} "
                 f"stretch(es) of moving frames (longest {worst} frames) — "
                 f"motion and detection are independent, so sustained "
                 f"disagreement means one of them is broken, not that the "
                 f"venue was empty")]
