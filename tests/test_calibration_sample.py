"""The 'CLIP cannot separate at any threshold' verdict may be an artefact.

calibrate_appearance_threshold builds its SAME-PERSON sample with
`dist <= stationary_px`, which until now had no time bound. Measured on
CAM.112 that rule bridged pairs 14px/62.3s and 17px/125.1s apart -- successive
guests at one reception desk. Pairs of DIFFERENT people were being labelled
same-person and then used to measure how well appearance separates the two.
"""
import numpy as np
from kevacv.analytics import calibrate_appearance_threshold


def _case(gap_s):
    """Two tracks 14px apart, `gap_s` seconds apart, with DIFFERENT
    appearance -- the measured shape of the 71->82 desk merge."""
    windows = {1: (0.0, 10.0), 2: (10.0 + gap_s, 20.0 + gap_s),
               3: (0.0, 10.0)}
    pos = {1: ((100.0, 100.0), (100.0, 100.0)),
           2: ((114.0, 100.0), (114.0, 100.0)),
           3: ((900.0, 900.0), (900.0, 900.0))}
    # SIMILAR appearance on purpose. calibrate_appearance_threshold admits a
    # same-person pair only if its appearance ALREADY agrees (a second, already
    # documented circularity -- see E2 in engine.py). Orthogonal vectors are
    # vetoed there, which would hide the thing under test. Two guests in
    # similar dark clothing at one desk is exactly the realistic failure.
    emb = {1: [1.0, 0.05], 2: [0.98, 0.10], 3: [0.0, 1.0]}
    return windows, pos, emb


def _same_n(gap_s, cap):
    w, p, e = _case(gap_s)
    c = calibrate_appearance_threshold(w, p, e, stationary_px=30.0,
                                       stationary_max_gap_s=cap)
    return c["same_n"]


def test_uncapped_admits_a_two_minute_desk_pair_as_same_person():
    """Documents the behaviour that made the verdict circular."""
    assert _same_n(125.0, None) >= 1


def test_capping_excludes_it():
    assert _same_n(125.0, 30.0) == 0, \
        "two people 125s apart at the same spot are not a same-person sample"


def test_a_genuine_short_reappearance_is_still_admitted():
    assert _same_n(5.0, 30.0) >= 1
