"""merge_ab.py — measure a merge-policy change before shipping it.

WHY THIS EXISTS
    Run 68b97311f9:

        merges accepted                  69
        blocked by window overlap       356      <- five times more
        blocked by role conflict         54

    The greedy union in merge_fragmented_tracks accumulates group_windows: each
    time a group absorbs a fragment its window set gets wider and gappier, so
    every later candidate is more likely to overlap SOMETHING already inside.
    Early merges progressively poison later ones, and the starved candidates
    were strong physical evidence:

        ID 47  <-> ID 345  tier=stationary  score=0.922
        ID 257 <-> ID 277  tier=stationary  score=0.918

    That is processing order deciding identity, not evidence. So any change
    that removes IMPOSSIBLE pairs early should show up twice: fewer wrong
    unions, and fewer good merges starved by the wrong unions' windows.

    "Should" is a hypothesis. This module measures it.

WHAT THIS IS NOT
    greedy_union() is a faithful re-implementation of the notebook's union
    loop (Cell 5, ~line 1519) for MEASUREMENT ONLY. It is not the production
    path and must never become it — two copies of a merge policy is exactly
    how they drift apart. If the notebook's loop changes, `assert_matches()`
    is here so a test can catch this copy going stale.

    It deliberately reproduces the greedy behaviour including its flaws. A
    harness that quietly fixes the thing it is measuring measures nothing.
"""
from __future__ import annotations

from collections import defaultdict

from .topology import reappearance_verdict


def windows_overlap(wins_a, wins_b, tolerance_s=2.0):
    """Exact copy of the notebook's _windows_overlap — same tolerance, same
    strict inequalities. Copied rather than imported because the original
    lives in a notebook cell; kept identical on purpose."""
    for a0, a1 in wins_a:
        for b0, b1 in wins_b:
            if a0 < b1 - tolerance_s and b0 < a1 - tolerance_s:
                return True
    return False


def greedy_union(pairs, track_windows, tolerance_s=2.0, sort_key=None):
    """Replay the greedy best-evidence-first union.

    pairs: [(sim, a, b, tier), ...] — the same tuples merge_fragmented_tracks
    builds. Returns the mapping plus the diagnostics that matter for A/B.

    `sort_key` mirrors the production `sorted(pairs, reverse=True)`. The
    default sorts on sim only, with a stable tiebreak on str(id) — the real
    loop has no key and would raise TypeError on an int/str tie, which is a
    latent bug there, not a behaviour worth copying.
    """
    parent = {t: t for t in track_windows}

    def find(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    group_windows = {t: [track_windows[t]] for t in track_windows}
    accepted, overlap_blocked = [], []
    tier_counts = defaultdict(int)

    key = sort_key or (lambda p: (p[0], str(p[1]), str(p[2])))
    for sim, a, b, tier in sorted(pairs, key=key, reverse=True):
        if a not in parent or b not in parent:
            continue
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if windows_overlap(group_windows[ra], group_windows[rb], tolerance_s):
            overlap_blocked.append((sim, a, b, tier))
            continue
        parent[rb] = ra
        group_windows[ra] = group_windows[ra] + group_windows[rb]
        accepted.append((sim, a, b, tier))
        tier_counts[tier] += 1

    mapping = {}
    canon = {}
    for t in sorted(track_windows, key=lambda x: (track_windows[x][0], str(x))):
        root = find(t)
        canon.setdefault(root, t)
        mapping[t] = canon[root]

    return {
        "mapping": mapping,
        "accepted": accepted,
        "overlap_blocked": overlap_blocked,
        "n_accepted": len(accepted),
        "n_overlap_blocked": len(overlap_blocked),
        "tier_counts": dict(tier_counts),
        "n_identities": len(set(mapping.values())),
        "n_tracks": len(track_windows),
    }


def apply_topology_veto(pairs, track_windows, positions, doors, frame_wh, **kw):
    """Split candidate pairs into (possible, vetoed) using the topology gate.

    A pair is (sim, a, b, tier). The earlier-starting track supplies the death
    position, the later one the birth position — the same convention
    merge_fragmented_tracks uses when it computes hand-off distance.
    """
    possible, vetoed = [], []
    for p in pairs:
        _sim, a, b, _tier = p
        if a not in track_windows or b not in track_windows:
            possible.append(p)
            continue
        wa, wb = track_windows[a], track_windows[b]
        early, late = (a, b) if wa[0] <= wb[0] else (b, a)
        pe, pl = positions.get(early), positions.get(late)
        if not pe or not pl:
            possible.append(p)          # no position -> cannot judge -> allow
            continue
        gap = track_windows[late][0] - track_windows[early][1]
        v = reappearance_verdict(pe[1], pl[0], gap, doors, frame_wh, **kw)
        (possible if v["allow"] else vetoed).append(p)
    return possible, vetoed


def ab_topology(pairs, track_windows, positions, doors, frame_wh,
                tolerance_s=2.0, **kw):
    """Run the union twice — as-is, and with impossible pairs removed first.

    -> {"without", "with", "vetoed", "delta"}. `delta` is what to read: if the
    veto helps, n_accepted goes UP (good merges stop being starved) while
    n_overlap_blocked goes DOWN, and identity count falls toward the truth.
    """
    without = greedy_union(pairs, track_windows, tolerance_s)
    possible, vetoed = apply_topology_veto(pairs, track_windows, positions,
                                           doors, frame_wh, **kw)
    with_ = greedy_union(possible, track_windows, tolerance_s)
    return {
        "without": without,
        "with": with_,
        "n_vetoed": len(vetoed),
        "vetoed": vetoed,
        "delta": {
            "accepted": with_["n_accepted"] - without["n_accepted"],
            "overlap_blocked": (with_["n_overlap_blocked"]
                                - without["n_overlap_blocked"]),
            "identities": with_["n_identities"] - without["n_identities"],
        },
    }


def describe(ab):
    w0, w1, d = ab["without"], ab["with"], ab["delta"]
    L = ["A/B — topology veto on the greedy merge",
         f"  {'':<22}{'without':>10}{'with':>10}{'delta':>10}",
         f"  {'candidate pairs vetoed':<22}{'-':>10}{ab['n_vetoed']:>10}{'':>10}",
         f"  {'merges accepted':<22}{w0['n_accepted']:>10}{w1['n_accepted']:>10}"
         f"{d['accepted']:>+10}",
         f"  {'starved by overlap':<22}{w0['n_overlap_blocked']:>10}"
         f"{w1['n_overlap_blocked']:>10}{d['overlap_blocked']:>+10}",
         f"  {'identities':<22}{w0['n_identities']:>10}{w1['n_identities']:>10}"
         f"{d['identities']:>+10}"]
    if d["overlap_blocked"] < 0 and d["accepted"] >= 0:
        L.append("  -> the veto freed starved merges: fewer wrong unions "
                 "polluting group windows")
    elif d["accepted"] < 0:
        L.append("  -> the veto is REMOVING merges the pipeline wanted. Check "
                 "the door positions before shipping this.")
    else:
        L.append("  -> no measurable effect on this data")
    return "\n".join(L)


def assert_matches(notebook_source):
    """Cheap staleness guard: does the notebook's union loop still look like
    the one copied here? Returns (ok, notes) — a test can fail on this so the
    copy cannot silently drift from the original."""
    notes = []
    for needle in ("_windows_overlap(group_windows[ra], group_windows[rb])",
                   "parent[rb] = ra",
                   "group_windows[ra] = group_windows[ra] + group_windows[rb]",
                   "for sim, a, b, tier in sorted(pairs, reverse=True)"):
        if needle not in notebook_source:
            notes.append(f"missing in notebook: {needle}")
    return (not notes), notes
