"""A veto must survive transitivity.

compute_pairwise_cost returns inf for pairs that are PHYSICALLY IMPOSSIBLE --
overlapping in time (two people at once) or carrying contradictory face ids.
Single-linkage union-find used to walk straight around that veto via a third
tracklet, merging the two people the veto exists to keep apart.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kevacv.graph_fusion import solve_graph_fusion, compute_pairwise_cost, FusionWeights


def _groups(canon_map):
    g = {}
    for raw, can in canon_map.items():
        g.setdefault(can, set()).add(raw)
    return sorted(g.values(), key=min)


def test_overlapping_tracklets_never_merge_through_a_third():
    w = FusionWeights()
    # A and C are on screen at the same time -> two different people.
    A = {"id": 1, "t_span": (0.0, 10.0),  "entry_pos": (100, 500), "exit_pos": (100, 500), "face_id": "alice"}
    C = {"id": 3, "t_span": (3.0, 20.0),  "entry_pos": (110, 500), "exit_pos": (110, 500), "face_id": "bob"}
    # B follows both and is a plausible continuation of either.
    B = {"id": 2, "t_span": (21.0, 30.0), "entry_pos": (105, 500), "exit_pos": (105, 500)}

    assert compute_pairwise_cost(A, C, w) == float("inf")
    assert compute_pairwise_cost(A, B, w) < 0.45
    assert compute_pairwise_cost(B, C, w) < 0.45

    m = solve_graph_fusion([A, B, C], weights=w, cost_cutoff=0.45)
    assert m[1] != m[3], f"vetoed pair merged transitively: {m}"
    # B may join either one, but all three must never be one identity.
    assert max(len(g) for g in _groups(m)) <= 2


def test_ordinary_chain_still_merges():
    # No veto anywhere -> the chain must still collapse to one identity, or the
    # guard above has been made too strict.
    w = FusionWeights()
    ts = [{"id": i, "t_span": (i * 10.0, i * 10.0 + 5.0),
           "entry_pos": (100, 500), "exit_pos": (100, 500)} for i in range(1, 4)]
    m = solve_graph_fusion(ts, weights=w, cost_cutoff=0.45)
    assert len(_groups(m)) == 1, m


if __name__ == "__main__":
    test_overlapping_tracklets_never_merge_through_a_third()
    test_ordinary_chain_still_merges()
    print("ok")
