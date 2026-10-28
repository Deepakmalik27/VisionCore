"""The entry-line preflight must catch what actually broke, not pass by design.

The old "synthetic crossing simulation" built a perpendicular through the
line's OWN midpoint and asked whether it intersected the line. Always true.
Meanwhile the two real failures on CAM.112 were:
  * the line was short with open floor at its ends -- only 2 of 67 tracks
    crossed the segment while 9 changed side of the infinite line
  * the plant mask covered the line, and masked detections are DROPPED, so
    nobody could be seen crossing it
"""
from kevacv.preflight import validate_entry_line

FRAME = (1920, 1080)


def _cfg(line, polys=None):
    return {"entry_line": line, "polygons": polys or {}}


def test_line_through_a_mask_is_an_error():
    """The measured CAM.112 failure: entry line inside 'plant area mask'."""
    mask = [[995, 190], [1430, 190], [1430, 1079], [995, 1079]]
    err, _warn = validate_entry_line(
        _cfg([[1280, 742], [1396, 1068]], {"plant area mask": mask}), FRAME)
    assert any("mask" in e.lower() for e in err), err
    assert any("DROPPED" in e for e in err), err


def test_line_clear_of_the_mask_is_fine():
    mask = [[995, 190], [1200, 190], [1200, 900], [995, 900]]
    err, _warn = validate_entry_line(
        _cfg([[1600, 540], [1690, 1075]], {"plant area mask": mask}), FRAME)
    assert not any("mask" in e.lower() for e in err), err


def test_short_line_is_rejected_in_frame_fractions_not_pixels():
    """20 absolute px is 0.5% of a 4K frame -- effectively no check."""
    err, _warn = validate_entry_line(_cfg([[900, 500], [930, 530]]), FRAME)
    assert err and "too short" in err[0].lower(), err


def test_open_ends_are_warned_about():
    """A line floating mid-frame has floor to walk around at both ends."""
    _err, warn = validate_entry_line(_cfg([[700, 400], [700, 700]]), FRAME)
    assert any("around it" in w for w in warn), warn


def test_line_anchored_to_edges_is_not_warned():
    _err, warn = validate_entry_line(_cfg([[1690, 0], [1690, 1080]]), FRAME)
    assert not any("around it" in w for w in warn), warn


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)


def test_mask_over_line_is_fatal_only_while_the_mask_deletes():
    """The overlap is FATAL at MASK_REQUIRE_MOTION=0 and survivable above it.

    Without this, --strict refuses to run the very config that fixes the
    problem: turning on motion-gating does not move the polygon, so the
    geometric overlap is still there and the ERROR could never be cleared.
    """
    import kevacv.config as C
    from kevacv.preflight import validate_entry_line

    zones = {
        "frame_size": [3840, 2160],
        "entry_lines": {"entry line": [[2560, 1484], [2792, 2136]]},
        "polygons": {"plant area mask": [[1990, 380], [2860, 380],
                                         [2860, 1700], [2760, 2159],
                                         [2060, 2159], [1990, 1700]]},
    }
    old = C.MASK_REQUIRE_MOTION
    try:
        C.MASK_REQUIRE_MOTION = 0.0
        errs, warns = validate_entry_line(zones, (3840, 2160))
        assert any("mask zone" in e for e in errs), \
            "deleting mask over the line must be an ERROR"

        C.MASK_REQUIRE_MOTION = 12.0
        errs, warns = validate_entry_line(zones, (3840, 2160))
        assert not any("mask zone" in e for e in errs), \
            "motion-gated mask must not be fatal -- --strict could never pass"
        assert any("mask zone" in w for w in warns), \
            "still worth a warning: a standing guest can be suppressed"
    finally:
        C.MASK_REQUIRE_MOTION = old
