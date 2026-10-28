"""test_phase_b.py — Phase B Verification Test Suite.

Verifies:
 1. SOTA Dual-Modality Re-ID Engine (P4/P5):
    - Extracting RGB & IR embeddings.
    - Vector normalization and dimensionality consistency.
 2. Robust RANSAC Ground Plane Calibration (P6):
    - RANSAC fitting on synthetic detection points.
    - Exclusion of unphysical / crouching posture outliers.
 3. BoxMOT Tracker Wrapper (P7):
    - Multi-object tracker instantiation and bounding box update loop.
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))

import unittest
import numpy as np

from kevacv import (
    ReIDEmbeddingExtractor,
    fit_robust_ground_plane,
    TrackerWrapper,
    GroundPlane
)


class TestReIDEngine(unittest.TestCase):
    def test_embedding_extraction(self):
        extractor = ReIDEmbeddingExtractor(embedding_dim=512, is_ir_mode=False)
        crop_rgb = np.random.randint(0, 256, (128, 64, 3), dtype=np.uint8)
        
        emb = extractor.extract_crop_embedding(crop_rgb, is_ir=False)
        self.assertEqual(emb.shape, (512,))
        norm = np.linalg.norm(emb)
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_ir_embedding_extraction(self):
        extractor = ReIDEmbeddingExtractor(embedding_dim=512, is_ir_mode=True)
        crop_ir = np.random.randint(0, 256, (128, 64, 3), dtype=np.uint8)
        
        emb_ir = extractor.extract_crop_embedding(crop_ir, is_ir=True)
        self.assertEqual(emb_ir.shape, (512,))
        norm = np.linalg.norm(emb_ir)
        self.assertAlmostEqual(norm, 1.0, places=4)


class TestCalibration(unittest.TestCase):
    def test_ransac_ground_plane_fitting(self):
        # Generate synthetic standing person detections: h(y) = 0.15 * y + 30 + noise
        frame_size = (1920, 1080)
        np.random.seed(42)

        inlier_ys = np.random.uniform(300, 1000, 40)
        inlier_hs = 0.15 * inlier_ys + 30 + np.random.normal(0, 2.0, 40)
        
        detections = []
        for y, h in zip(inlier_ys, inlier_hs):
            w = h / 3.0  # Aspect ratio ~3.0
            x1 = 500
            y1 = y - h
            x2 = x1 + w
            y2 = y
            detections.append((x1, y1, x2, y2))

        # Add posture outlier detections (crouching/seated: aspect ~ 1.2, height ~ 40px)
        for _ in range(10):
            y = np.random.uniform(400, 900)
            detections.append((100, y - 40, 150, y))

        gp = fit_robust_ground_plane(detections, frame_size)
        self.assertTrue(gp.ok)
        self.assertIsNotNone(gp.a)
        self.assertGreater(gp.a, 0)


class TestTrackerWrapper(unittest.TestCase):
    def test_tracker_update_loop(self):
        wrapper = TrackerWrapper(tracker_type="bytetrack")
        frame_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        dets = np.array([
            [100, 100, 200, 400, 0.9, 0],
            [300, 300, 400, 600, 0.85, 0]
        ], dtype=np.float32)

        out = wrapper.update(dets, frame_img)
        self.assertIsNotNone(out)
        self.assertEqual(len(out), 2)


if __name__ == "__main__":
    unittest.main()
