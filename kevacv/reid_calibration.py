"""calibration.py — measure the appearance signal WITHOUT using appearance.

WHY THIS EXISTS
    calibrate_appearance_threshold() (Cell 5) says of its two ground-truth
    sets: "BOTH independent of appearance (no circularity)". The same-person
    set is not. Its admission test ends with:

        and not _handoff_appearance_contradicts(a, b, anchor_embeddings)

    and that helper is `_cosine(va, vb) < HANDOFF_VETO_SIM`. So same-person
    pairs are admitted only if their appearance already agrees — the low tail
    of same_sims is cut off by the quantity being measured.

    The same set has a second problem pulling the other way. Its admission
    test is:

        (gap <= handoff_gap_s and dist <= handoff_px) or dist <= stationary_px

    The second clause has NO gap bound. Two tracks 30 px apart an hour apart
    qualify as "the same person". A reception has a queue spot, a chair, a
    counter edge; different customers stand in the same 30 px all evening.
    role_hint only excludes pairs with OPPOSITE earned roles, so
    customer-on-customer contamination passes straight through. On CAM.112,
    60 of ~182 candidate pairs were dropped as staff/customer role conflicts —
    a third of the raw set — which is how strong that effect is.

    Between them the measured 0.658 balanced accuracy has an unknown sign of
    bias. That number must not appear in a report until this is fixed.

THE FIX
    Exclude strangers on PHYSICS instead of on appearance:

      * bound the stationary clause with stationary_gap_s (default 30 s). A
        body that vanishes and returns to the same spot within 30 s is a
        tracker dropout. Forty minutes later it is the next customer.
      * remove the appearance veto from the MEASUREMENT. It belongs in the
        merge path, where a veto is the right tool; inside a calibration it is
        circular.

    Both remaining guards stay, because both are appearance-independent:
    role conflict (earned from zone dwell) and the duplicate-track guard
    (co-located at both ends of a shared lifetime).

    compare_to_legacy() runs the old admission rules alongside the new ones so
    the size of the correction is visible rather than asserted.
"""
from __future__ import annotations

import math

DEFAULT_HANDOFF_GAP_S = 2.5
DEFAULT_HANDOFF_PX = 90.0
DEFAULT_STATIONARY_PX = 30.0
# A tracker dropout, not the next person in the queue. This is the whole fix:
# a TIME bound where the original had none.
DEFAULT_STATIONARY_GAP_S = 30.0
DEFAULT_DUPLICATE_PX = 40.0


class LiveSeparability:
    """Measure the appearance signal from the run itself, with no labels.

    WHY THIS IS NOT CIRCULAR
        Both sets come from facts that hold regardless of what any embedding
        says:

        DIFFERENT people — two detections in the SAME FRAME. A person cannot
            be in two places at one instant. This is stronger than the
            window-overlap negatives calibrate() uses: overlapping lifetimes
            still admit a duplicated track of one body, whereas two boxes in
            one frame are two bodies by construction (and the co-visibility
            splitter now guarantees they carry different ids).

        SAME person — the same raw track id, a fraction of a second apart,
            while not occluded. Over a gap that short the association is
            motion, not appearance, so using it to judge appearance does not
            assume the answer.

    WHAT IT IS FOR
        config.py records a same-person p50 of 0.435 against accept bars of
        0.60/0.62/0.75 — i.e. by the project's own number the system rejects
        most true matches, which is symptoms 3, 4, 11 and 14. But that 0.435
        was measured circularly and the file says so. Nothing could be done
        about the thresholds while the only evidence for them was untrusted.

        This produces a trustworthy replacement on every run, for free: the
        embeddings are already computed for tracking, so this only compares
        vectors that exist.

    It RECOMMENDS. Applying a threshold automatically on a distribution the
    operator has not seen is how a system silently starts merging strangers.
    """

    def __init__(self, sim_fn=None, max_same_gap_s=0.5, max_pairs=20000):
        self.sim = sim_fn or cosine
        self.max_same_gap_s = float(max_same_gap_s)
        self.max_pairs = int(max_pairs)
        self.same, self.diff = [], []
        self._last = {}          # raw id -> (t, vec)
        self.n_frames = 0

    def observe(self, t, items):
        """items: [(raw_id, vec)] for ONE frame. vec may be None."""
        self.n_frames += 1
        live = [(i, v) for i, v in items if v is not None]
        # different-person pairs: everything co-visible this frame
        if len(self.diff) < self.max_pairs:
            for a in range(len(live)):
                for b in range(a + 1, len(live)):
                    if live[a][0] == live[b][0]:
                        continue          # same id twice = a duplicate bug
                    self.diff.append(self.sim(live[a][1], live[b][1]))
        # same-person pairs: this id, moments ago
        for rid, vec in live:
            prev = self._last.get(rid)
            if prev is not None:
                gap = t - prev[0]
                if 0 < gap <= self.max_same_gap_s and len(self.same) < self.max_pairs:
                    self.same.append(self.sim(prev[1], vec))
            self._last[rid] = (t, vec)

    def report(self, current_thresholds=None):
        if len(self.same) < 30 or len(self.diff) < 30:
            return {"ok": False,
                    "why": (f"not enough evidence (same={len(self.same)}, "
                            f"diff={len(self.diff)}); need 30 of each"),
                    "n_same": len(self.same), "n_diff": len(self.diff)}
        best_t, best_j = None, -1.0
        lo = min(min(self.same), min(self.diff))
        hi = max(max(self.same), max(self.diff))
        for k in range(201):
            th = lo + (hi - lo) * k / 200.0
            tpr = sum(1 for s in self.same if s >= th) / len(self.same)
            fpr = sum(1 for s in self.diff if s >= th) / len(self.diff)
            j = tpr - fpr                      # Youden's J
            if j > best_j:
                best_j, best_t = j, th
        out = {
            "ok": True,
            "n_same": len(self.same), "n_diff": len(self.diff),
            "same_p05": percentile(self.same, 0.05),
            "same_p50": percentile(self.same, 0.50),
            "diff_p50": percentile(self.diff, 0.50),
            "diff_p95": percentile(self.diff, 0.95),
            "recommended": best_t,
            "youden_j": best_j,
            "balanced_accuracy": 0.5 * (
                sum(1 for s in self.same if s >= best_t) / len(self.same)
                + sum(1 for s in self.diff if s < best_t) / len(self.diff)),
            "at_current": {},
        }
        for name, th in (current_thresholds or {}).items():
            out["at_current"][name] = {
                "threshold": th,
                "true_matches_accepted":
                    sum(1 for s in self.same if s >= th) / len(self.same),
                "strangers_accepted":
                    sum(1 for s in self.diff if s >= th) / len(self.diff),
            }
        return out


def describe_live(rep, width=78):
    L = ["=" * width,
         "  RE-ID SEPARABILITY — measured on THIS run, without labels",
         "=" * width]
    if not rep.get("ok"):
        L.append(f"  {rep.get('why')}")
        L.append("=" * width)
        return "\n".join(L)
    L.append(f"  same-person pairs {rep['n_same']:>6}   "
             f"p50 {rep['same_p50']:.3f}   p05 {rep['same_p05']:.3f}")
    L.append(f"  different-person  {rep['n_diff']:>6}   "
             f"p50 {rep['diff_p50']:.3f}   p95 {rep['diff_p95']:.3f}")
    L.append("  " + "-" * (width - 4))
    L.append(f"  recommended threshold {rep['recommended']:.3f}  "
             f"(balanced accuracy {rep['balanced_accuracy']:.3f})")
    for name, x in sorted(rep["at_current"].items()):
        L.append(f"  {name} = {x['threshold']}: accepts "
                 f"{100 * x['true_matches_accepted']:.0f}% of true matches, "
                 f"{100 * x['strangers_accepted']:.0f}% of strangers")
    L.append("  " + "-" * (width - 4))
    L.append("  A threshold accepting a small share of TRUE matches is why one")
    L.append("  person becomes many ids. Raising the accept rate trades that")
    L.append("  for merging strangers — this table is that trade, measured.")
    L.append("  Nothing is applied automatically: set REID_SIM_THRESHOLD /")
    L.append("  LIVE_REID_SIM_THRESHOLD yourself once you have read it.")
    L.append("=" * width)
    return "\n".join(L)


def cosine(a, b):
    if a is None or b is None:
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return num / (na * nb)


def percentile(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def _role_conflict(a, b, role_hint):
    if not role_hint:
        return False
    ra, rb = role_hint.get(a), role_hint.get(b)
    return bool(ra) and bool(rb) and ra != rb


def _looks_duplicate(pa, pb, duplicate_px):
    """Two ids co-located at BOTH ends of a shared lifetime are one body
    wearing two track ids, not two people who happened to stand close."""
    d_start = math.hypot(pa[0][0] - pb[0][0], pa[0][1] - pb[0][1])
    d_end = math.hypot(pa[1][0] - pb[1][0], pa[1][1] - pb[1][1])
    return d_start <= duplicate_px and d_end <= duplicate_px


def _ordered(a, b, windows, positions):
    """-> (gap, dist) between the earlier track's death and the later's birth."""
    wa, wb = windows[a], windows[b]
    pa, pb = positions[a], positions[b]
    if wa[0] <= wb[0]:
        first_end, first_pos, second_start, second_pos = wa[1], pa[1], wb[0], pb[0]
    else:
        first_end, first_pos, second_start, second_pos = wb[1], pb[1], wa[0], pa[0]
    gap = second_start - first_end
    dist = math.hypot(first_pos[0] - second_pos[0], first_pos[1] - second_pos[1])
    return gap, dist


def calibrate(windows, positions, anchor_embeddings, role_hint=None,
              handoff_gap_s=DEFAULT_HANDOFF_GAP_S,
              handoff_px=DEFAULT_HANDOFF_PX,
              stationary_px=DEFAULT_STATIONARY_PX,
              stationary_gap_s=DEFAULT_STATIONARY_GAP_S,
              duplicate_px=DEFAULT_DUPLICATE_PX,
              sim_fn=None, legacy_stationary_unbounded=False,
              appearance_veto_sim=None):
    """Measure same-person vs different-person appearance similarity.

    Set `legacy_stationary_unbounded=True` and `appearance_veto_sim=<float>` to
    reproduce the original admission rules — that is what compare_to_legacy()
    uses, and it is the only reason those switches exist.
    """
    sim = sim_fn or cosine
    ids = [t for t in windows if t in anchor_embeddings]

    diff_sims, diff_pairs = [], []
    n_duplicate = 0
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            wa, wb = windows[a], windows[b]
            if not (wa[0] < wb[1] and wb[0] < wa[1]):
                continue                      # not co-visible
            pa, pb = positions.get(a), positions.get(b)
            if pa and pb and _looks_duplicate(pa, pb, duplicate_px):
                n_duplicate += 1
                continue
            s = sim(anchor_embeddings[a], anchor_embeddings[b])
            diff_sims.append(s)
            diff_pairs.append((a, b, s))

    same_sims, same_pairs = [], []
    n_role, n_stale, n_veto = 0, 0, 0
    for i, a in enumerate(ids):
        if a not in positions:
            continue
        for b in ids[i + 1:]:
            if b not in positions:
                continue
            if _role_conflict(a, b, role_hint):
                n_role += 1
                continue
            gap, dist = _ordered(a, b, windows, positions)
            if gap < 0:
                continue
            handoff = gap <= handoff_gap_s and dist <= handoff_px
            if legacy_stationary_unbounded:
                stationary = dist <= stationary_px
            else:
                stationary = dist <= stationary_px and gap <= stationary_gap_s
            if not (handoff or stationary):
                # count the pairs the ORIGINAL rule would have admitted and
                # this one rejects: same spot, implausibly long gap
                if dist <= stationary_px:
                    n_stale += 1
                continue
            s = sim(anchor_embeddings[a], anchor_embeddings[b])
            if appearance_veto_sim is not None and s < appearance_veto_sim:
                n_veto += 1          # legacy behaviour: circular, measured only
                continue
            same_sims.append(s)
            same_pairs.append((a, b, s))

    same_p10 = percentile(same_sims, 0.10)
    diff_p90 = percentile(diff_sims, 0.90)
    return {
        "same_n": len(same_sims), "diff_n": len(diff_sims),
        "same_sims": same_sims, "diff_sims": diff_sims,
        "same_p10": same_p10, "same_p50": percentile(same_sims, 0.50),
        "diff_p50": percentile(diff_sims, 0.50), "diff_p90": diff_p90,
        "separable": bool(same_p10 is not None and diff_p90 is not None
                          and same_p10 > diff_p90),
        "appearance_independent": appearance_veto_sim is None,
        "excluded": {"role_conflict": n_role, "duplicate_track": n_duplicate,
                     "stale_stationary": n_stale, "appearance_veto": n_veto},
        "same_pairs_worst": sorted(same_pairs, key=lambda p: p[2])[:8],
        "diff_pairs_worst": sorted(diff_pairs, key=lambda p: -p[2])[:8],
        "params": {"handoff_gap_s": handoff_gap_s, "handoff_px": handoff_px,
                   "stationary_px": stationary_px,
                   "stationary_gap_s": (None if legacy_stationary_unbounded
                                        else stationary_gap_s)},
    }


def compare_to_legacy(windows, positions, anchor_embeddings,
                      appearance_veto_sim, **kw):
    """Old admission rules vs corrected ones, side by side.

    Returns both reports plus the deltas. The point is to make the size of the
    correction visible: if the two agree, the original number stands and this
    module has cost nothing; if they diverge, the report was quoting a biased
    figure and now we know by how much.
    """
    old = calibrate(windows, positions, anchor_embeddings,
                    legacy_stationary_unbounded=True,
                    appearance_veto_sim=appearance_veto_sim, **kw)
    new = calibrate(windows, positions, anchor_embeddings, **kw)

    def _d(k):
        a, b = old.get(k), new.get(k)
        if a is None or b is None:
            return None
        return round(b - a, 4)

    return {"legacy": old, "corrected": new,
            "delta": {"same_n": new["same_n"] - old["same_n"],
                      "same_p10": _d("same_p10"), "same_p50": _d("same_p50"),
                      "diff_p90": _d("diff_p90")}}


def describe(rep):
    L = ["CALIBRATION — appearance measured without using appearance"
         if rep.get("appearance_independent") else
         "CALIBRATION — LEGACY rules (circular: same-person set filtered by appearance)"]
    L.append(f"  same-person      n={rep['same_n']:<5} "
             f"p10={rep['same_p10']}  p50={rep['same_p50']}")
    L.append(f"  different-person n={rep['diff_n']:<5} "
             f"p50={rep['diff_p50']}  p90={rep['diff_p90']}")
    ex = rep["excluded"]
    L.append(f"  excluded: role_conflict={ex['role_conflict']} "
             f"duplicate_track={ex['duplicate_track']} "
             f"stale_stationary={ex['stale_stationary']} "
             f"appearance_veto={ex['appearance_veto']}")
    if ex["appearance_veto"]:
        L.append("  !! appearance_veto > 0 — this run is CIRCULAR, do not "
                 "quote its separability")
    if ex["stale_stationary"]:
        L.append(f"  {ex['stale_stationary']} same-spot pair(s) rejected for an "
                 f"implausible gap — these are the queue-spot strangers the "
                 f"original rule counted as one person")
    L.append(f"  separable: {rep['separable']}")
    return "\n".join(L)
