"""phantoms.py — PHASE 10. Kill static false positives that CHANGE ID.


WHY THIS EXISTS
    D2 (detect_filters.static_track_ids) asks "did this TRACK ID sit still for
    a long time?". On the real CAM.112 hour it caught 3 ids and reported them
    honestly. It also missed the two phantoms that dominate every annotated
    frame:

        P3  a large box over the curved mirror on the right wall, present in
            essentially every frame, and the id-timeline audit's single worst
            offender: 2,319+ "REAPPEARED" jumps of 300-900px.
        P8  the potted plant, labelled "staff" because its box centre happens
            to fall in the reception zone.

    Both survived D2 for the same reason: THEIR IDS CHURN. The detector fires
    on the same pixels, the tracker keeps minting a fresh id, each individual
    id is short-lived, and a per-id lifetime test can never see the pattern.
    The thing that is static is the LOCATION, not the id.

THE PRINCIPLE
    Ask the question of the pixels instead of the ids. Aggregate every
    detection by where it landed, across the whole chunk. A location that
    keeps producing a box of near-identical geometry over many minutes is
    furniture, a reflection, a poster or a TV — no matter how many ids passed
    through it.

    This is the same idea as a hand-drawn mask zone, derived from the data
    instead of from a person with a mouse. That matters for universality: the
    next camera gets the same protection without anyone drawing anything.

WHAT KEEPS IT FROM EATING REAL PEOPLE
    A receptionist stands at her post for the whole hour, so "static for a
    long time" alone is NOT enough — it would delete the most important person
    in the venue. Three independent conditions must hold together:

      1. tiny centre jitter, as a FRACTION OF BODY HEIGHT (scale-free, so it
         means the same near and far). Real D2 furniture measured 0.008-0.012;
         a standing person shifts, leans and steps an order of magnitude more.
      2. near-constant box SIZE (low coefficient of variation). A person's box
         breathes as they turn and move; a mirror artefact does not.
      3. a long span AND many hits, so a brief coincidence cannot qualify.

    Plus an explicit veto: any location a PROTECTED id ever occupied (someone
    who crossed the entry line, or matched an enrolled face) is never flagged.

    ponytail: fixed grid, no clustering library. A phantom straddling a cell
    boundary splits into two weaker cells and may be missed — raise cell_frac
    or switch to real clustering if that shows up in practice.

HOW STRONG IS THE SEPARATION, HONESTLY
    Not as strong as it looks. A person standing at a desk and a plant are only
    a few thousandths of body height apart on jitter alone, and the margin is
    thin enough that a very still person could cross it. Two things carry the
    real weight:

      SIZE   the physical reason the filter works. The detector sees IDENTICAL
             pixels for a plant every frame, so it returns an almost identical
             box (size cv ~0.004 measured). A person's pixels genuinely change
             — arms, turning, leaning — and their box breathes by an order of
             magnitude more. This is the discriminator that does not depend on
             how still someone chooses to stand.
      VETO   any location a protected id occupied (crossed the entry line, or
             matched an enrolled face) is never flagged, whatever the geometry.

    So: never rely on jitter alone, and always enroll staff faces. If a real
    person is ever flagged, that is a bug worth a test, not a tuning exercise.
"""


from __future__ import annotations

import math
from collections import defaultdict

# Declared at module level so apply_run_config can REACH them.
#
# These were read only via globals().get(...), and config propagation uses
# `if hasattr(module, name)` -- which is False for a name that does not exist,
# so the setattr was skipped and the fast path was permanently off at 0.0.
# A yaml A/B on this knob was run and recorded in config/cam112.yaml as
# "MEASURED 2026-08-15 (profG, ratio 0.30): NO EFFECT". That conclusion is
# INVALID: the value never arrived. A dead knob produced a written finding
# that then guided later decisions.
PHANTOM_FAST_CV_RATIO = 0.0
PHANTOM_FAST_MIN_S = 30.0

# Defaults derived from the measured CAM.112 numbers, not guessed:
#   D2's confirmed furniture      centre jitter 0.008 - 0.012 of body height
#   -> 0.02 leaves ~2x headroom above the worst real phantom while staying
#      far below where a standing person lands.
MAX_CENTRE_JITTER = 0.02
# Measured on the synthetic rebuild of the real CAM.112 phantoms:
#     plant   size cv 0.004     mirror  size cv 0.003
#     a receptionist at her post 0.043  — an order of magnitude clear
# 0.015 sits in that gap: ~4x above the phantoms, ~3x below the person. This is
# the condition doing the real work, so it is set from measurement, not taste.
MAX_SIZE_CV = 0.015
MIN_SPAN_S = 240.0
MIN_HITS = 40
CELL_FRAC = 0.04


def _median(v):
    s = sorted(v)
    n = len(s)
    if not n:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _std(v):
    if len(v) < 2:
        return 0.0
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))


def phantom_regions(frame_log, frame_wh=None, protected=None,
                    min_span_s=MIN_SPAN_S, min_hits=MIN_HITS,
                    max_centre_jitter=MAX_CENTRE_JITTER,
                    max_size_cv=MAX_SIZE_CV, cell_frac=CELL_FRAC):
    """Locations that keep emitting the same box -> phantom regions.

    frame_log: [(idx, t, [(tid, x1, y1, x2, y2), ...]), ...]
    -> [ {box, centre, hits, span_s, centre_jitter, size_cv, ids, why}, ... ]

    Returns [] rather than guessing when there is nothing to go on, and never
    flags a location a protected id occupied.
    """
    protected = set(protected or ())
    if not frame_log:
        return []

    if frame_wh:
        fw, fh = float(frame_wh[0]), float(frame_wh[1])
    else:                                   # infer from the boxes themselves
        fw = max((b[3] for _i, _t, bs in frame_log for b in bs), default=1280.0)
        fh = max((b[4] for _i, _t, bs in frame_log for b in bs), default=720.0)
    cell = max(8.0, cell_frac * max(fw, fh))

    buckets = defaultdict(list)
    for _idx, t, boxes in frame_log:
        for tid, x1, y1, x2, y2 in boxes:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            buckets[(int(cx // cell), int(cy // cell))].append(
                (t, cx, cy, float(x2) - float(x1), float(y2) - float(y1), tid))

    out = []
    for _key, hits in buckets.items():
        if len(hits) < min_hits:
            continue
        ids = {h[5] for h in hits}
        if ids & protected:
            continue                        # a real, verified person was here
        ts = [h[0] for h in hits]
        span = max(ts) - min(ts)
        if span < min_span_s:
            continue
        hgt = _median([h[4] for h in hits])
        if hgt <= 1:
            continue
        # jitter as a fraction of body height: scale-free, so the same number
        # means the same thing at the door and at the back of the room.
        # max(), not hypot() — this is deliberately the SAME predicate D2 uses,
        # because D2's 0.02 is the one threshold validated against real
        # detections (it kept the receptionist and caught three real phantoms).
        # Combining the axes would silently make this filter stricter than the
        # one we have evidence for.
        jit = max(_std([h[1] for h in hits]), _std([h[2] for h in hits])) / hgt
        if jit > max_centre_jitter:
            continue
        wds = [h[3] for h in hits]
        hts = [h[4] for h in hits]
        mw, mh = _median(wds), _median(hts)
        size_cv = max(_std(wds) / mw if mw else 9, _std(hts) / mh if mh else 9)
        if size_cv > max_size_cv:
            continue
        cx, cy = _median([h[1] for h in hits]), _median([h[2] for h in hits])
        out.append({
            "box": (cx - mw / 2.0, cy - mh / 2.0, cx + mw / 2.0, cy + mh / 2.0),
            "centre": (round(cx), round(cy)),
            "hits": len(hits), "span_s": round(span, 1), "ids": len(ids),
            "centre_jitter": round(jit, 4), "size_cv": round(size_cv, 3),
            "why": (f"{len(hits)} detections over {span/60:.0f} min from "
                    f"{len(ids)} different id(s), centre jitter {jit:.4f} of "
                    f"body height, size cv {size_cv:.3f} — furniture, a "
                    f"reflection or a poster, not a person"),
        })
    # A phantom sitting on a cell boundary lands in two cells and gets reported
    # twice (seen immediately in testing: the mirror came back as 2401 hits and
    # 299 hits at the same spot). Merge overlapping CANDIDATES — merging raw
    # cells instead would let a busy neighbouring cell pull a real phantom's
    # jitter up and hide it.
    out.sort(key=lambda r: -r["hits"])
    merged = []
    for r in out:
        for m in merged:
            if _iou(r["box"], m["box"]) >= 0.3:
                m["hits"] += r["hits"]
                m["ids"] = max(m["ids"], r["ids"])
                m["span_s"] = max(m["span_s"], r["span_s"])
                break
        else:
            merged.append(r)
    return merged


class OnlineStaticSuppressor:
    """phantom_regions', asked once per frame instead of once per chunk.

    WHY BOTH EXIST
        phantom_regions() runs at the END, over the whole frame_log, and
        drop_tracks() then removes what it found from events, crossings and
        the video together. The final numbers are therefore already clean —
        so why suppress live at all?

        Because the phantom is not inert while it waits to be deleted. For its
        entire life it:
          * consumes tracker attention and Re-ID embeddings every frame,
          * occupies a canonical id, so co-visibility BLOCKS a real person
            from resolving to it — pushing a genuine track to mint a new id,
          * feeds the identity memory an anchor built from a plant.

        Deleting it afterwards cannot undo an id break it caused at 19:14. The
        end-of-chunk pass stays (it sees the whole picture and catches what a
        live view cannot); this stops the bleeding while the chunk is running.

    ASKED OF THE PIXELS, NOT THE IDS
        Same principle as phantom_regions: the mirror and the plant survive a
        per-id lifetime test precisely BECAUSE their ids churn. The thing that
        is static is the LOCATION.

    ZONE-AWARE PATIENCE
        min_life_for(x, y) -> seconds. A doorway needs ~30 s of rigidity to be
        furniture; the reception desk needs minutes, because people
        legitimately stand still there. Without this the suppressor deletes
        the receptionist, which is far worse than the plant it removes.
    """

    def __init__(self, frame_wh, min_life_for=None, default_life_s=120.0,
                 min_hits=15, max_centre_jitter=MAX_CENTRE_JITTER,
                 max_size_cv=MAX_SIZE_CV, cell_frac=CELL_FRAC,
                 forget_after_s=30.0):
        fw, fh = float(frame_wh[0]), float(frame_wh[1])
        self.cell = max(8.0, cell_frac * max(fw, fh))
        self.min_life_for = min_life_for
        self.default_life_s = float(default_life_s)
        self.min_hits = int(min_hits)
        self.max_centre_jitter = float(max_centre_jitter)
        self.max_size_cv = float(max_size_cv)
        self.forget_after_s = float(forget_after_s)
        self._cells = {}        # key -> {"t0","t1","cx","cy","w","h" lists}
        self.suppressed = {}    # key -> why
        self.n_suppressed = 0   # detections actually dropped
        self.n_fast_tracked = 0  # locations caught EARLY on rigidity alone

    def _key(self, cx, cy):
        return (int(cx // self.cell), int(cy // self.cell))

    def _patience(self, cx, cy):
        if self.min_life_for is None:
            return self.default_life_s
        try:
            return float(self.min_life_for(cx, cy))
        except Exception:
            return self.default_life_s

    def observe(self, t, box):
        """Record one detection. -> True if this location is now furniture."""
        x1, y1, x2, y2 = (float(v) for v in box)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        k = self._key(cx, cy)
        if k in self.suppressed:
            return True
        c = self._cells.get(k)
        # A cell that went quiet is forgotten. Otherwise a person who stands
        # still, leaves, and is replaced hours later by another person in the
        # same spot looks like one continuous rigid object.
        if c is not None and (t - c["t1"]) > self.forget_after_s:
            c = None
        if c is None:
            c = self._cells[k] = {"t0": t, "t1": t, "cx": [], "cy": [],
                                  "w": [], "h": []}
        c["t1"] = t
        c["cx"].append(cx); c["cy"].append(cy)
        c["w"].append(x2 - x1); c["h"].append(y2 - y1)
        # bound the memory: the statistics converge long before this
        if len(c["cx"]) > 400:
            for f in ("cx", "cy", "w", "h"):
                del c[f][:-400]

        if len(c["cx"]) < self.min_hits:
            return False

        mh = _median(c["h"])
        if mh <= 1:
            return False

        # ── EVIDENCE-SCALED PATIENCE ────────────────────────────────────────
        # Zone patience alone is why this stage reported "live phantom suppress
        # removed nothing all chunk" on a 600 s run: the plant and the mirror
        # sit inside the reception/seating polygons, which get 240 s BECAUSE
        # PEOPLE LEGITIMATELY STAND STILL THERE. Correct reasoning, wrong
        # consequence — the phantom survives the whole chunk, consuming a
        # canonical id and blocking real people from resolving to it, and only
        # the end-of-chunk pass removes it. That cannot undo an id break.
        #
        # The way out: "stands still" and "is furniture" are different claims,
        # and the gap between them is not subtle. Measured:
        #
        #     statue / mirror   size cv ~0.004   identical pixels in,
        #                                        identical box out
        #     person standing   size cv ~0.080   arms, turning, leaning
        #
        # 19x. Compare what else was tried on this problem: a flat height cap
        # separates a wall phantom from a near-field guest by 0.02 of frame,
        # and a top-anchor bound by 28 pixels — both fitted to two examples.
        # Rigidity-over-time is the only signal here with room to stand in.
        #
        # So a location may prove itself furniture EARLY, but only by being far
        # more rigid than a person can be. At or above the normal bar it waits
        # the full zone patience, unchanged.
        _wait = self._patience(cx, cy)
        _elapsed = c["t1"] - c["t0"]
        if _elapsed < _wait:
            _ratio = float(globals().get("PHANTOM_FAST_CV_RATIO", 0.0) or 0.0)
            if _ratio <= 0:
                return False                      # feature off: old behaviour
            _mw = _median(c["w"])
            _cv_now = max(_std(c["w"]) / _mw if _mw else 9.0,
                          _std(c["h"]) / mh if mh else 9.0)
            _floor = float(globals().get("PHANTOM_FAST_MIN_S", 30.0))
            if not (_cv_now <= _ratio * self.max_size_cv and _elapsed >= _floor):
                return False
            _fast_tracked = True
        else:
            _fast_tracked = False
        # Deliberately the SAME predicate as phantom_regions and D2 — max(),
        # not hypot(). Combining the axes would silently make this stricter
        # than the one threshold validated against real detections.
        jit = max(_std(c["cx"]), _std(c["cy"])) / mh
        if jit > self.max_centre_jitter:
            return False
        mw = _median(c["w"])
        size_cv = max(_std(c["w"]) / mw if mw else 9.0,
                      _std(c["h"]) / mh if mh else 9.0)
        if size_cv > self.max_size_cv:
            return False
        # Counted only once the EVIDENCE gates below have also passed. The
        # fast path skips the clock, never the proof.
        if _fast_tracked:
            self.n_fast_tracked += 1
        self.suppressed[k] = (
            f"location ({int(_median(c['cx']))},{int(_median(c['cy']))}) "
            f"emitted {len(c['cx'])} near-identical boxes over "
            f"{c['t1'] - c['t0']:.0f}s (jitter {jit:.4f} of body height, "
            f"size cv {size_cv:.3f}) — furniture, not a person"
            + (f" [FAST: cv {size_cv:.4f} is far below the {self.max_size_cv} "
               f"bar, so it did not wait the full "
               f"{self._patience(cx, cy):.0f}s zone patience]"
               if _fast_tracked else ""))
        return True

    def is_suppressed(self, box):
        x1, y1, x2, y2 = (float(v) for v in box)
        return self._key((x1 + x2) / 2.0, (y1 + y2) / 2.0) in self.suppressed

    def filter_boxes(self, t, boxes):
        """Observe this frame's boxes and say which to keep.

        -> boolean keep-mask, one entry per box.

        observe() returns True the moment a location crosses into "furniture",
        so the box that completes the evidence is itself dropped — the correct
        edge: it is the same static thing as the 200 before it.
        """
        keep = [not self.observe(t, b) for b in boxes]
        self.n_suppressed += sum(1 for k in keep if not k)
        return keep

    def describe(self):
        if not self.suppressed:
            return "live phantom suppression: nothing suppressed"
        L = [f"live phantom suppression: {len(self.suppressed)} location(s), "
             f"{self.n_suppressed} detection(s) dropped before they could "
             f"mint an id"]
        for why in list(self.suppressed.values())[:8]:
            L.append(f"      {why}")
        return "\n".join(L)


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def in_phantom(box, regions, min_iou=0.55):
    """Is this detection the phantom itself? Overlap, not centre-containment —
    a real person walking IN FRONT of the plant has a different box and must
    survive, which a centre-in-region test would not allow."""
    return any(_iou(box, r["box"]) >= min_iou for r in regions)


def drop_phantom_dets(boxes, regions, min_iou=0.55):
    """-> (kept, dropped) for a list of (tid, x1, y1, x2, y2)."""
    kept, dropped = [], []
    for b in boxes:
        (kept, dropped)[in_phantom(tuple(b[1:5]), regions, min_iou)].append(b)
    return kept, dropped


def describe(regions):
    if not regions:
        return "no static phantom regions found"
    L = [f"PHANTOM REGIONS — {len(regions)} location(s) emitting a person box "
         f"that never moves"]
    for r in regions:
        L.append(f"  at {r['centre']}  {r['why']}")
    L.append("  These are dropped BEFORE counting. If one of these is actually a "
             "person who stood still, raise min_span_s or protect their id.")
    return "\n".join(L)
