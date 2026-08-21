"""The stationary merge tier had no time bound.

Measured on output/p0classfix2 (CAM.112, 600s): stationary accounted for 33 of
54 merges, bridging pairs 14px/62.3s and 17px/125.1s apart, and 5 of 7
"customers" ended up spanning 7-10 minutes of a 10-minute chunk. Proximity at a
reception desk is evidence of a DESK, not of a person.
"""
import kevacv.config as C
from kevacv.analytics import merge_fragmented_tracks


def _case(gap_s):
    """Two fragments 14px apart, `gap_s` seconds apart — the measured shape of
    the 71->82 merge. Embeddings are deliberately inconclusive, which is what
    CLIP actually produces on this footage."""
    windows = {1: (0.0, 10.0), 2: (10.0 + gap_s, 20.0 + gap_s)}
    emb = {1: [[1.0, 0.0]], 2: [[0.0, 1.0]]}      # orthogonal = no support
    pos = {1: ((100.0, 100.0), (100.0, 100.0)),
           2: ((114.0, 100.0), (114.0, 100.0))}
    return windows, emb, pos


def _merged(gap_s, cap):
    w, e, p = _case(gap_s)
    mapping, edges, diag = merge_fragmented_tracks(
        w, e, sim_threshold=0.65, max_gap_s=480.0, positions=p,
        stationary_px=30.0, stationary_max_gap_s=cap)
    return mapping[1] == mapping[2]


def test_without_a_cap_a_two_minute_gap_still_merges():
    # Documents the pre-existing behaviour this cap exists to bound.
    assert _merged(125.0, None) is True


def test_cap_blocks_the_long_gap():
    assert _merged(125.0, 30.0) is False


def test_cap_still_allows_a_genuine_short_reappearance():
    assert _merged(5.0, 30.0) is True


def test_default_config_changes_nothing():
    # Shipping a typed constant here has been reverted three times; the default
    # must be inert so the value is chosen by A/B, not by argument.
    assert getattr(C, "REID_STATIONARY_MAX_GAP_S", "missing") is None


def test_knob_is_reachable_from_yaml():
    assert "analysis.reid_stationary_max_gap_s" in C.RUN_CONFIG_KEYS
