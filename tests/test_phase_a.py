"""test_phase_a.py — Phase A Verification Test Suite.

Verifies:
 1. Preflight Validation (Fail-Loud Engine):
    - Passes on valid zones file (CAM.112_zone.json).
    - Fails loudly on corrupted or short entry line / missing polygons.
 2. Graph-Based Identity Fusion Engine:
    - Successfully associates fragmented tracklets across spatial/temporal gaps.
    - Correctly enforces physical speed limits and spatial consistency.
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))

import json
import unittest
from pathlib import Path
import numpy as np

from kevacv import (
    PreflightValidationError,
    run_preflight_checks,
    FusionWeights,
    solve_graph_fusion,
    GroundPlane
)


class TestPreflightValidation(unittest.TestCase):
    def setUp(self):
        # WAS CAM.112_zone.json, which is NOT valid and has not been since the
        # entry line was found lying inside 'plant area mask' -- the fault that
        # made the door read "IN 0 | OUT 0" while people walked through it.
        self.valid_zone_path = (_P(__file__).resolve().parent.parent
                                / "zones" / "CAM.112_zone_v5.json")
        with open(self.valid_zone_path, "r") as f:
            self.valid_cfg = json.load(f)

    def test_valid_configuration(self):
        """Valid configuration should pass preflight without errors."""
        report = run_preflight_checks(self.valid_cfg, strict=False)
        self.assertEqual(report["status"], "PASSED")
        self.assertEqual(len(report["errors"]), 0)

    def test_short_entry_line_fail_loud(self):
        """Entry line shorter than threshold should raise PreflightValidationError."""
        bad_cfg = json.loads(json.dumps(self.valid_cfg))
        # Shorten entry line to 5px length
        bad_cfg["entry_line"] = [[100, 100], [103, 104]]

        with self.assertRaises(PreflightValidationError) as ctx:
            run_preflight_checks(bad_cfg, strict=True)
        
        err_msg = str(ctx.exception)
        # the message now names WHICH line, since a venue has several
        self.assertIn("is too short", err_msg)

    def test_invalid_polygon_vertices(self):
        """Polygon with fewer than 3 vertices should fail preflight."""
        bad_cfg = json.loads(json.dumps(self.valid_cfg))
        bad_cfg["polygons"]["corrupted_zone"] = [[10, 10], [20, 20]]  # Only 2 points

        with self.assertRaises(PreflightValidationError) as ctx:
            run_preflight_checks(bad_cfg, strict=True)
        
        err_msg = str(ctx.exception)
        self.assertIn("must be a polygon with at least 3 vertices", err_msg)


class TestGraphFusion(unittest.TestCase):
    def test_fusion_association(self):
        """Test global association solver on matching vs non-matching tracklets."""
        # Create two tracklets representing the SAME person fragmented over a 5s gap
        emb_person_1 = np.array([0.9, 0.1, 0.05, 0.4], dtype=np.float32)
        hsv_person_1 = np.array([0.8, 0.2], dtype=np.float32)

        tracklet_1 = {
            "id": 101,
            "t_span": (10.0, 25.0),
            "entry_pos": (200, 300),
            "exit_pos": (500, 600),
            "reid_emb": emb_person_1,
            "hsv_hist": hsv_person_1
        }

        # Similar appearance, nearby exit/entry position 3 seconds later
        tracklet_2 = {
            "id": 102,
            "t_span": (28.0, 45.0),
            "entry_pos": (520, 610),
            "exit_pos": (900, 1000),
            "reid_emb": emb_person_1 + np.array([0.01, -0.01, 0.0, 0.01]),
            "hsv_hist": hsv_person_1
        }

        # Different person entirely
        emb_person_2 = np.array([-0.5, 0.8, 0.3, -0.1], dtype=np.float32)
        tracklet_3 = {
            "id": 103,
            "t_span": (30.0, 50.0),
            "entry_pos": (100, 100),
            "exit_pos": (200, 200),
            "reid_emb": emb_person_2,
            "hsv_hist": np.array([0.1, 0.9], dtype=np.float32)
        }

        tracklets = [tracklet_1, tracklet_2, tracklet_3]
        weights = FusionWeights(max_temporal_gap_s=60.0)

        canon_map = solve_graph_fusion(tracklets, weights=weights, cost_cutoff=0.45)

        # Tracklet 101 and 102 should be merged into the same canonical ID
        self.assertEqual(canon_map[101], canon_map[102])
        # Tracklet 103 should remain separate
        self.assertNotEqual(canon_map[101], canon_map[103])


if __name__ == "__main__":
    unittest.main()
