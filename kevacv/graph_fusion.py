"""graph_fusion.py — P2 Graph-Based Identity Fusion (Min-Cost Flow / Global Optimization).

WHY THIS EXISTS
    The 6-tier Re-ID cascade (CLIP -> Anchor -> HSV -> Handoff -> Stationary -> Face) evaluates
    evidence in rigid sequential order ("first-tier-wins"). When CLIP and HSV disagree 
    (10-20% of cases), greedy heuristics pick the wrong tier and lock in false merges.

THE PRINCIPLE
    Joint Multi-Evidence Graph Optimization.
    1. Construct a global association graph where nodes are tracklets.
    2. Compute pairwise transition costs using ALL weighted evidence channels simultaneously:
       - Deep appearance distance (CLIP / FastReID)
       - Color histogram similarity (HSV)
       - Spatial & motion feasibility (GroundPlane floor distance & walk speed)
       - Face recognition biometrics
       - Temporal gap / overlap constraints
    3. Solve global minimum-cost assignment using SciPy Linear Sum Assignment (Hungarian / Min-Cost Flow).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


class FusionWeights:
    """Configurable weights for the multi-evidence identity graph edge costs."""
    def __init__(self,
                 w_reid: float = 0.45,
                 w_hsv: float = 0.15,
                 w_spatial: float = 0.25,
                 w_face: float = 0.15,
                 max_walk_speed_mps: float = 2.5,
                 max_temporal_gap_s: float = 300.0,
                 reid_threshold: float = 0.35):
        self.w_reid = w_reid
        self.w_hsv = w_hsv
        self.w_spatial = w_spatial
        self.w_face = w_face
        self.max_walk_speed_mps = max_walk_speed_mps
        self.max_temporal_gap_s = max_temporal_gap_s
        self.reid_threshold = reid_threshold


def compute_pairwise_cost(track_a: Dict[str, Any],
                          track_b: Dict[str, Any],
                          weights: FusionWeights,
                          ground_plane: Optional[Any] = None) -> float:
    """Compute total fusion edge cost between track_a and track_b (lower = more likely same person).
    Returns float('inf') if physically impossible (e.g. temporal overlap or unphysical speed).
    """
    # 1. Temporal feasibility check (cannot overlap in time)
    t_a_span = track_a.get("t_span")
    t_b_span = track_b.get("t_span")
    if t_a_span is None or t_b_span is None:
        return float('inf')
    t_a_start, t_a_end = t_a_span
    t_b_start, t_b_end = t_b_span

    # Overlap check with 1s tolerance
    if max(t_a_start, t_b_start) < min(t_a_end, t_b_end) - 1.0:
        return float('inf')

    # Determine direction (A before B or B before A)
    if t_a_end <= t_b_start:
        t_gap = t_b_start - t_a_end
        pos_exit = track_a.get("exit_pos", (0, 0))
        pos_entry = track_b.get("entry_pos", (0, 0))
    else:
        t_gap = t_a_start - t_b_end
        pos_exit = track_b.get("exit_pos", (0, 0))
        pos_entry = track_a.get("entry_pos", (0, 0))

    if t_gap > weights.max_temporal_gap_s:
        return float('inf')

    # 2. Motion / Speed Feasibility Check
    #
    # THE VETO IS ONLY AS GOOD AS THE PLANE. `speed > max_walk_speed -> inf` is
    # a HARD refusal: it forbids a merge outright, and nothing downstream can
    # appeal it. On CAM.112 the automatic perspective fit produced eight
    # different camera heights inside one hour (1.12m to 3.26m), so dist_m can
    # be wrong by ~2.9x -- and a veto wrong by 2.9x forbids real merges and
    # permits impossible ones, roughly at random. That is worse than no veto,
    # because it is confident.
    #
    # So the metric branch is taken ONLY for an EXACT plane -- a homography fit
    # to measured ground_points (see tools/calibrate_plane.py). An auto-fit
    # falls through to the pixel branch, which has no hard veto and can only
    # nudge the cost. Trust the geometry exactly as far as it was measured and
    # not one step further; this is what makes the module safe to wire in
    # BEFORE the room has been measured.
    _exact = getattr(ground_plane, "mode", None) == "exact"
    if ground_plane and getattr(ground_plane, "ok", False) and _exact:
        dist_m = ground_plane.dist_m(pos_exit, pos_entry)
        if dist_m is not None:
            speed = dist_m / max(t_gap, 0.5)
            if speed > weights.max_walk_speed_mps:
                return float('inf')  # Faster than humanly possible
            spatial_cost = min(1.0, dist_m / 10.0)
        else:
            spatial_cost = 0.5
    else:
        # Fallback to pixel distance normalized
        px_dist = math.hypot(pos_exit[0] - pos_entry[0], pos_exit[1] - pos_entry[1])
        spatial_cost = min(1.0, px_dist / 800.0)

    # 3. Deep Re-ID Embedding Distance (Cosine)
    emb_a = track_a.get("reid_emb")
    emb_b = track_b.get("reid_emb")
    if emb_a is not None and emb_b is not None:
        dot = np.dot(emb_a, emb_b)
        norm = (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
        sim = dot / max(norm, 1e-9)
        reid_dist = max(0.0, 1.0 - float(sim))
    else:
        reid_dist = 0.5

    # 4. Color HSV Histogram Distance
    hsv_a = track_a.get("hsv_hist")
    hsv_b = track_b.get("hsv_hist")
    if hsv_a is not None and hsv_b is not None:
        # Bhattacharyya / correlation distance
        hsv_dist = max(0.0, 1.0 - float(np.sum(np.minimum(hsv_a, hsv_b))))
    else:
        hsv_dist = 0.5

    # 5. Face Biometric Match
    face_a = track_a.get("face_id")
    face_b = track_b.get("face_id")
    if face_a and face_b:
        if face_a == face_b:
            face_dist = 0.0  # Strong positive
        else:
            return float('inf')  # Contradictory faces -> impossible merge
    else:
        face_dist = 0.5

    # Weighted composite cost calculation
    total_cost = (
        weights.w_reid * reid_dist +
        weights.w_hsv * hsv_dist +
        weights.w_spatial * spatial_cost +
        weights.w_face * face_dist
    )

    return total_cost


def solve_graph_fusion(tracklets: List[Dict[str, Any]],
                       weights: Optional[FusionWeights] = None,
                       ground_plane: Optional[Any] = None,
                       cost_cutoff: float = 0.45) -> Dict[int, int]:
    """Solve global tracklet identity association graph using Hungarian optimization.
    
    tracklets: List of dicts, each containing:
               - id: int
               - t_span: (t_start, t_end)
               - entry_pos: (x, y)
               - exit_pos: (x, y)
               - reid_emb: np.ndarray (optional)
               - hsv_hist: np.ndarray (optional)
               - face_id: str/int (optional)
    
    Returns: Mapping {raw_track_id: canonical_track_id}
    """
    if not tracklets:
        return {}

    if weights is None:
        weights = FusionWeights()

    n = len(tracklets)
    raw_ids = [t["id"] for t in tracklets]
    
    # Cost Matrix Construction
    cost_matrix = np.full((n, n), fill_value=np.inf)

    for i in range(n):
        for j in range(i + 1, n):
            c = compute_pairwise_cost(tracklets[i], tracklets[j], weights, ground_plane)
            if c <= cost_cutoff:
                cost_matrix[i, j] = c

    # Linear Sum Assignment / Hungarian matching
    canon_map: Dict[int, int] = {tid: tid for tid in raw_ids}

    # Find valid candidate edges
    valid_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if cost_matrix[i, j] < cost_cutoff:
                valid_pairs.append((cost_matrix[i, j], i, j))

    # Sort edges by cost and merge disjoint sets using Union-Find
    valid_pairs.sort(key=lambda x: x[0])

    parent = {tid: tid for tid in raw_ids}

    def find(i):
        """Iterative path-compression find — safe for >1000 tracklets."""
        root = i
        while parent[root] != root:
            root = parent[root]
        # Path compression
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_j] = root_i

    for cost, i, j in valid_pairs:
        id_i = raw_ids[i]
        id_j = raw_ids[j]
        if find(id_i) != find(id_j):
            union(id_i, id_j)

    for tid in raw_ids:
        canon_map[tid] = find(tid)

    return canon_map
