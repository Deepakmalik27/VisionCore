"""P0 wiring: knobs that were UNREACHABLE from yaml, and the GMC name split.

Both failures here were silent. A yaml key that maps to nothing does not error,
it just does nothing -- so an A/B "with GMC off" would have run with GMC on and
reported that GMC does not matter.
"""
import kevacv.config as C
from kevacv import engine as E


def test_gmc_and_identity_memory_are_reachable_from_yaml():
    for key in ("analysis.enable_gmc", "analysis.enable_live_identity_memory"):
        assert key in C.RUN_CONFIG_KEYS, f"{key} not mapped -- yaml would be ignored"


def test_every_mapped_key_points_at_a_real_constant():
    missing = [(k, v) for k, v in C.RUN_CONFIG_KEYS.items() if not hasattr(C, v)]
    assert not missing, f"mapped to non-existent constants: {missing}"


def test_ultralytics_gmc_translates_boxmot_spelling():
    # boxmot says 'sof'; ultralytics says 'sparseOptFlow'. Same algorithm.
    E.ENABLE_GMC, E.GMC_METHOD = True, "sof"
    assert E._ultralytics_gmc() == "sparseOptFlow"


def test_ultralytics_gmc_off_means_off():
    E.ENABLE_GMC, E.GMC_METHOD = False, "sof"
    assert E._ultralytics_gmc() == "none", (
        "the whole point of the fixed-camera A/B is being able to turn it OFF")


def test_gmc_method_is_actually_honoured_not_hardcoded():
    # The bug: this path returned 'sparseOptFlow' whatever GMC_METHOD said.
    E.ENABLE_GMC, E.GMC_METHOD = True, "ecc"
    assert E._ultralytics_gmc() == "ecc"
    E.ENABLE_GMC, E.GMC_METHOD = True, "none"
    assert E._ultralytics_gmc() == "none"
