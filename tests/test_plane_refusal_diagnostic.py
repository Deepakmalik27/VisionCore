"""The plane refusal must name WHICH filter emptied the sample set.

A bare "insufficient" is why a mis-tuned aspect gate looked like a quiet clip
for the whole life of this module: the run fell back to the drifting fit and
said nothing about the median box being 1.58 h/w on an oblique camera.
"""
from kevacv.geometry_calibration import fit_robust_ground_plane


def _wide(n):
    """Boxes a real oblique camera produces: aspect ~1.5, below the 2.0 gate."""
    return [(100.0, 500.0 + i, 200.0, 650.0 + i) for i in range(n)]


def test_refusal_reports_counts_and_median():
    gp = fit_robust_ground_plane(_wide(50), (1920, 1080))
    assert not gp.ok
    why = gp.why if isinstance(getattr(gp, "why", None), str) else str(gp.__dict__)
    assert "0 of 50" in why, why          # none passed the aspect gate
    assert "1.50" in why, why             # and the median says why


def test_refusal_survives_degenerate_boxes():
    """Zero-width boxes must not divide-by-zero the diagnostic."""
    gp = fit_robust_ground_plane([(10.0, 10.0, 10.0, 10.0)] * 12, (1920, 1080))
    assert not gp.ok


if __name__ == "__main__":
    test_refusal_reports_counts_and_median()
    test_refusal_survives_degenerate_boxes()
    print("ok")
