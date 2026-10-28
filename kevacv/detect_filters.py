"""detect_filters.py — PHASE 7. Two filters for two phantoms the real run produced.

WHAT THE FIRST REAL RUN SHOWED
    Five independent readers audited 35 annotated frames from CAM.112. Recall
    was excellent: 120 person-sightings, exactly ONE human missed. Precision was
    not: roughly 19% of the boxes were not people.

    They were not random. They were two specific, repeatable failures:

      P3   a box covering the ENTIRE right half of the frame, present in ~17 of
           35 frames, counted as a person every time.
      P8/  a box around the POTTED PLANT — and labelled "staff", because a plant
      P13  standing in a staff zone trivially satisfies the dwell rule.

    Swapping the detector does not fix this. Stock YOLO produces no giant boxes
    but finds only 32% of the people on this greyscale, steeply-tilted, wide
    footage. Our fine-tune has the recall; it needs its output filtered.

WHY THESE TWO FILTERS AND NOT A NEW MODEL
    Both phantoms are already detectable from data the pipeline collects:

      GIANT   a person's pixel height at a given footline is bounded — the
              scene-geometry fit (_PerspectiveModel) already predicts it. The
              carried-person rule uses that prediction to catch boxes that are
              too SHORT. This is the same test inverted.
      STATIC  a plant's box never moves and never changes size. A human's box
              always jitters — body sway, arm movement, detector noise. Track-
              level variance separates them with no new model at all.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict


# ---------------------------------------------------------------------------
# D1 — a person cannot be taller than the floor allows
# ---------------------------------------------------------------------------
BODY_ASPECT = 3.5      # a standing person is ~3.5x taller than wide


def implausible_size_mask(boxes, expected_h, tol=2.5, aspect=BODY_ASPECT):
    """-> list[bool], True where the box is too BIG to be a person standing there.

    boxes:      iterable of (x1, y1, x2, y2)
    expected_h: fn(foot_y) -> predicted pixel height of a standing person whose
                feet are at that row, or None when it cannot say (no fit yet, or
                the point is above the horizon).

    WHY AREA AND NOT HEIGHT. The first version of this tested height alone and
    its own test caught it failing: P3, the box covering half the frame, is only
    1.44x too TALL — barely separable from a genuinely tall person at 1.11x. But
    P3 is also enormously too WIDE, and area combines both:

        P3 (half the frame)   425 x 710 px  vs expected  69,500 px^2  ->  4.34x
        a very tall person    100 x 550 px  vs expected  69,500 px^2  ->  0.79x

    Four times versus four fifths. Height alone gave 1.44 versus 1.11, which no
    honest threshold separates.

    Expected area is expected_h^2 / aspect: a person's width is their height over
    the body aspect ratio, so the whole thing is still driven by the one number
    the scene-geometry fit actually predicts.

    tol=2.5 leaves room for two people merged into one box (~2x), an outstretched
    arm, or a coat — it is aimed at 4x+ absurdities, not at borderline calls.

    When expected_h returns None NOTHING is flagged. A filter that fires on
    missing information is worse than no filter, and deleting a real detection
    costs more than keeping a phantom.
    """
    out = []
    for x1, y1, x2, y2 in boxes:
        w, h = float(x2) - float(x1), float(y2) - float(y1)
        exp = expected_h(y2)
        if exp is None or w <= 0 or h <= 0:
            out.append(False)
            continue
        out.append(bool(w * h > tol * (exp * exp / aspect)))
    return out


# ---------------------------------------------------------------------------
# D2 — furniture does not fidget
# ---------------------------------------------------------------------------
def _point_in_poly(poly, x, y):
    """Ray casting. Pure Python on purpose: this module has no numpy/cv2 and
    is meant to stay testable on a laptop with neither."""
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = float(poly[i][0]), float(poly[i][1])
        x2, y2 = float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1])
        if (y1 > y) != (y2 > y):
            xin = (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1
            if x < xin:
                inside = not inside
    return inside


def static_min_life_by_id(frame_log, polygons, zone_roles, canon=None,
                          default_s=120.0, by_role=None):
    """How long each track must sit still before the static filter believes it
    is furniture — chosen by WHERE it sits. -> {canonical_id: seconds}

    WHY THIS IS NOT ONE NUMBER
        The threshold trades two opposite errors, and the right trade is
        different in different parts of the same frame. Confirmed with the
        operator 2026-08-12: people stand still at the reception desk and in
        the waiting area, but the corridors and the entrance are a
        thoroughfare where nothing human holds position.

        A single global 120 s therefore has to be conservative enough for the
        desk, which means a plant in the corridor mints ids and pollutes zone
        events for two full minutes before anything suppresses it. Per-zone,
        the corridor plant dies in 30 s and the person at the desk is still
        safe.

    by_role: {role: seconds}. The MOST CONSERVATIVE (largest) value wins when
        a track's position falls inside overlapping zones — an over-long wait
        leaves a phantom alive, an over-short one deletes a person, and only
        one of those is recoverable.
    """
    by_role = by_role or {}
    canon = canon or {}
    pts = {}
    for _fi, _t, boxes in frame_log:
        for tid, x1, y1, x2, y2 in boxes:
            cid = canon.get(tid, tid)
            # foot point: where the body meets the floor, and so which zone it
            # is standing in — the same anchor the zone triggers use.
            pts.setdefault(cid, []).append(
                ((float(x1) + float(x2)) / 2.0, float(y2)))
    out = {}
    for cid, ps in pts.items():
        x = statistics.median(p[0] for p in ps)
        y = statistics.median(p[1] for p in ps)
        best = None
        for name, poly in (polygons or {}).items():
            if len(poly) < 3 or not _point_in_poly(poly, x, y):
                continue
            for role in (zone_roles or {}).get(name, ()):
                v = by_role.get(role)
                if v is not None and (best is None or v > best):
                    best = v
        out[cid] = default_s if best is None else best
    return out


def static_track_ids(frame_log, canon=None, protected=(), min_life_s=120.0,
                     max_centre_jitter=0.02, max_size_jitter=0.03,
                     min_sightings=20, min_life_by_id=None):
    """Track ids whose box is so rigid over so long that it must be furniture.

    frame_log:  [(frame_idx, t, [(track_id, x1, y1, x2, y2), ...]), ...]
    canon:      optional {raw_id: canonical_id} so a stitched identity is judged
                as one thing rather than as its fragments.
    protected:  ids that must NEVER be dropped whatever the geometry says —
                anyone who crossed the entry line, or whose face was recognised.
                A human who was seen walking through the door is a human, and no
                statistic gets to overrule that.

    The thresholds are fractions of the box's own height, not pixels, so the
    same rule holds for someone near the camera and someone across the room.

    A person standing still at a desk still moves: they lean, turn, gesture, and
    the detector's own noise adds more. Measured on real detections, that is
    comfortably above 2% of body height. A plant sits at exactly the same pixels
    for twenty minutes.
    """
    canon = canon or {}
    seen = {}
    for _fi, t, boxes in frame_log:
        for tid, x1, y1, x2, y2 in boxes:
            cid = canon.get(tid, tid)
            seen.setdefault(cid, []).append(
                (float(t), (float(x1) + float(x2)) / 2.0,
                 (float(y1) + float(y2)) / 2.0,
                 float(x2) - float(x1), float(y2) - float(y1)))

    flagged = {}
    for cid, rows in seen.items():
        if cid in protected or len(rows) < min_sightings:
            continue
        life = rows[-1][0] - rows[0][0]
        # Per-zone patience (static_min_life_by_id) when the caller supplied
        # it, else the single global bar. A corridor plant and a person
        # standing at a desk need different answers to the same question.
        if life < (min_life_by_id or {}).get(cid, min_life_s):
            continue
        cx = [r[1] for r in rows]
        cy = [r[2] for r in rows]
        w = [r[3] for r in rows]
        h = [r[4] for r in rows]
        med_h = statistics.median(h) or 1.0
        jit = max(statistics.pstdev(cx), statistics.pstdev(cy)) / med_h
        size = max(statistics.pstdev(w), statistics.pstdev(h)) / med_h
        if jit <= max_centre_jitter and size <= max_size_jitter:
            flagged[cid] = {"seconds": round(life, 1), "sightings": len(rows),
                            "centre_jitter": round(jit, 4),
                            "size_jitter": round(size, 4),
                            "at": (round(statistics.median(cx)),
                                   round(statistics.median(cy)))}
    return flagged


def rigid_track_ids(frame_log, canon=None, protected=(), min_life_s=60.0,
                    max_aspect_cv=0.06, min_sightings=25,
                    max_travel_frac=0.10, frame_wh=None):
    """Track ids whose SHAPE never changes — a plant, mannequin, poster, sign.

    WHY THIS EXISTS, AND WHY static_track_ids IS NOT ENOUGH
        static_track_ids catches things that do not MOVE. It misses the whole
        family that moves a little and is still not a person: a plant swaying
        in the aircon, a mannequin whose box jitters as the detector re-fits it
        each frame, a poster whose box drifts with exposure changes.

        The detector cannot help here. best.pt was fine-tuned on CrowdHuman —
        humans only — so it has never been shown a mannequin and told "not a
        person". Its confidence on a coat stand is honest and useless. No
        threshold repairs an error that is not in the score.

    THE INDEPENDENT SIGNAL
        A person is DEFORMABLE. Walking, turning, reaching, sitting — the box
        aspect ratio moves constantly. A rigid object's aspect is near
        constant for its entire life, however much the box wanders.

        So we measure the coefficient of variation of h/w over the track. Low
        CV over a long life = rigid = not a person. This is geometry the
        detector cannot influence, which is the point: a category channel that
        is independent of the thing that got the category wrong.

    WHY TRAVEL IS ALSO REQUIRED
        Rigidity ALONE is not enough, and a synthetic test caught this filter
        flagging a walking person: any track whose detector box happens to keep
        a steady aspect would qualify — someone walking straight away from the
        camera, or a detector that boxes consistently.

        Furniture has a second property: it does not TRANSLATE. A plant, a
        mannequin and a poster stay put. So the verdict needs both — the shape
        never changes AND the thing never goes anywhere.

        A track that is rigid but DOES travel is a different animal: usually a
        reflection, which is mirrored_pair_ids' job. Keeping the two separate
        stops either from absorbing the other's false positives.

    WHAT IT DELIBERATELY DOES NOT CATCH
        A TV or monitor showing people. Those DO deform — they are people, just
        not present ones. They fail a different test: many identities born and
        dying inside one fixed rectangle, which is what phantom_regions looks
        for. Do not stretch this function to cover them.

    `protected` wins, always. Anyone who crossed the entry line or whose face
    was recognised is a human, and no statistic overrules that.
    """
    canon = canon or {}
    per = defaultdict(list)
    pos = defaultdict(list)
    for _idx, t, boxes in frame_log:
        for tid, x1, y1, x2, y2 in boxes:
            w = max(1e-6, float(x2) - float(x1))
            h = max(1e-6, float(y2) - float(y1))
            cid = canon.get(tid, tid)
            per[cid].append((float(t), h / w))
            pos[cid].append(((float(x1) + float(x2)) / 2.0, float(y2)))

    if frame_wh:
        diag = math.hypot(float(frame_wh[0]), float(frame_wh[1]))
    else:
        xs = [p[0] for v in pos.values() for p in v] or [1280.0]
        ys = [p[1] for v in pos.values() for p in v] or [720.0]
        diag = math.hypot(max(xs), max(ys))
    travel_limit = diag * max_travel_frac

    out = {}
    for tid, rows in per.items():
        if tid in protected or len(rows) < min_sightings:
            continue
        rows.sort()
        life = rows[-1][0] - rows[0][0]
        if life < min_life_s:
            continue
        vals = [a for _, a in rows]
        mean = sum(vals) / len(vals)
        if mean <= 0:
            continue
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        cv = (var ** 0.5) / mean
        if cv > max_aspect_cv:
            continue
        pts = pos.get(tid) or []
        travel = max((math.hypot(p[0] - q[0], p[1] - q[1])
                      for p in pts for q in pts[:1]), default=0.0) if pts else 0.0
        travel = max(travel, math.hypot(pts[-1][0] - pts[0][0],
                                        pts[-1][1] - pts[0][1])) if pts else 0.0
        if travel > travel_limit:
            continue          # rigid but mobile -> a reflection, not furniture
        out[tid] = {"aspect_cv": round(cv, 4), "aspect_mean": round(mean, 3),
                    "life_s": round(life, 1), "sightings": len(rows),
                    "travel_px": round(travel, 1),
                    "why": (f"aspect ratio varied by only {cv*100:.1f}% over "
                            f"{life:.0f}s and it moved {travel:.0f}px — a "
                            f"person walking, turning or sitting cannot hold "
                            f"one shape that long, and furniture does not "
                            f"wander")}
    return out


def _path_length(track_pts, times):
    """How far this track actually travelled over `times`, in pixels."""
    total = 0.0
    for i in range(1, len(times)):
        (x0, y0), (x1, y1) = track_pts[times[i - 1]], track_pts[times[i]]
        total += math.hypot(x1 - x0, y1 - y0)
    return total


def mirrored_pair_ids(frame_log, canon=None, protected=(), min_overlap_s=20.0,
                      max_offset_cv=0.10, min_samples=30, min_offset_px=15.0,
                      min_travel_px=80.0):
    """Track pairs that move in lockstep at a fixed offset — a reflection.

    WHY
        A reception with glass doors or a mirrored wall produces a second body
        walking in step with the first. It is not static, so static_track_ids
        misses it. It deforms exactly like a person, so rigid_track_ids misses
        it. It is person-shaped and person-sized, so the detector is right and
        the size filter passes it. Every existing guard is blind to it, and it
        inflates the headcount by up to 2x on the worst camera angle.

        The reflected-object literature says the giveaway is CONTEXT, not
        appearance — you cannot tell from the pixels. Here the context is
        motion: a real pair of people drift apart constantly, while an object
        and its reflection keep a near-constant separation for as long as both
        are visible, because one is a rigid transform of the other.

    HOW
        For every co-visible pair, measure the separation each frame and take
        its coefficient of variation. Low CV over a long co-visibility means
        the two never moved independently.

    WHAT IT RETURNS
        Candidates, NOT a deletion list — {(a, b): evidence}. Which of the two
        is the reflection needs the zone map (a reflection usually sits inside
        a glass/window dead area), and deleting the wrong one is worse than
        counting both. This flags; a human or a zone rule decides.
    """
    canon = canon or {}
    per = defaultdict(dict)                 # tid -> {t: (cx, foot_y)}
    for _idx, t, boxes in frame_log:
        for tid, x1, y1, x2, y2 in boxes:
            per[canon.get(tid, tid)][float(t)] = (
                (float(x1) + float(x2)) / 2.0, float(y2))

    ids = sorted(per, key=str)
    out = {}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if a in protected and b in protected:
                continue
            shared = sorted(set(per[a]) & set(per[b]))
            if len(shared) < min_samples:
                continue
            if shared[-1] - shared[0] < min_overlap_s:
                continue
            d = [math.hypot(per[a][t][0] - per[b][t][0],
                            per[a][t][1] - per[b][t][1]) for t in shared]
            mean = sum(d) / len(d)
            if mean < min_offset_px:
                continue                    # same body, duplicate box
            var = sum((x - mean) ** 2 for x in d) / len(d)
            cv = (var ** 0.5) / mean
            # THEY MUST HAVE MOVED. The evidence for a reflection is that two
            # bodies travelled and stayed rigidly separated the whole way —
            # "lockstep", as the docstring above promises. Constant separation
            # ALONE proves nothing: two people standing still at a counter are
            # a fixed distance apart with a coefficient of variation near zero,
            # and this flagged every one of them.
            #
            # That is not a small effect at a reception. The first real run
            # reported 105 lockstep pairs touching 92 of 231 identities — and
            # one track was "mirrored" with two different partners at two
            # different distances, which no actual reflection can be.
            travel_a = _path_length(per[a], shared)
            travel_b = _path_length(per[b], shared)
            if min(travel_a, travel_b) < min_travel_px:
                continue
            if cv <= max_offset_cv:
                out[(a, b)] = {
                    "offset_px": round(mean, 1), "offset_cv": round(cv, 4),
                    "co_visible_s": round(shared[-1] - shared[0], 1),
                    "samples": len(shared),
                    "travel_px": (round(travel_a, 1), round(travel_b, 1)),
                    "why": (f"stayed {mean:.0f}px apart (±{cv*100:.1f}%) for "
                            f"{shared[-1]-shared[0]:.0f}s — two independent "
                            f"people drift; an object and its reflection "
                            f"cannot")}
    return out


def protected_ids(crossings=(), face_ids=(), canon=None):
    """Ids the static filter may never touch: anyone who crossed the entry line,
    and anyone whose face was matched. Both are positive evidence of a human."""
    canon = canon or {}
    out = {canon.get(c.get("track_id"), c.get("track_id")) for c in crossings}
    out |= {canon.get(f, f) for f in face_ids}
    # a gallery-matched staff member carries a string name, never a number
    out |= {t for t in out if isinstance(t, str)}
    return {t for t in out if t is not None}


def drop_tracks(events, crossings, frame_log, drop, canon=None):
    """Remove flagged ids from every downstream structure at once, so a phantom
    cannot survive in one place after being removed from another.

    canon: {raw_id: canonical_id}. events/crossings are usually already
    remapped to canonical ids but frame_log never is — without canon a phantom
    whose fragments were Re-ID-merged is removed from every NUMBER and still
    DRAWN in the video (the bug the human review caught)."""
    drop = set(drop)
    canon = canon or {}
    return ([e for e in events if e.get("track_id") not in drop],
            [c for c in crossings if c.get("track_id") not in drop],
            [(fi, t, [b for b in boxes
                      if b[0] not in drop
                      and canon.get(b[0], b[0]) not in drop])
             for fi, t, boxes in frame_log])


def describe(giant_n, static_map):
    lines = []
    if giant_n:
        lines.append(f"D1 dropped {giant_n} detection(s) taller than a person "
                     f"could be at their own footline (the half-frame boxes)")
    if static_map:
        lines.append(f"D2 dropped {len(static_map)} track(s) that never moved "
                     f"and never changed size — furniture, not people:")
        for cid, d in sorted(static_map.items(),
                             key=lambda kv: -kv[1]["seconds"])[:6]:
            lines.append(f"     id {cid} at {d['at']} for {d['seconds']:.0f}s, "
                         f"centre jitter {d['centre_jitter']:.3f} of body height")
    if not lines:
        lines.append("D1/D2 found no phantoms this chunk")
    return "\n".join(lines)
