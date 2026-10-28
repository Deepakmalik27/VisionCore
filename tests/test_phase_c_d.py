"""test_phase_c_d.py — Phase C & D Verification Test Suite.

Verifies:
 1. Zone Anomaly Baseline Detector (P10):
    - Baseline fitting and Z-score calculation.
    - Identification of anomalous occupancy counts.
 2. Dataset Collector & Pseudo-Label Exporter (P13):
    - Exporting YOLO format labels and images.
    - Dataset YAML configuration generation.
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))

import tempfile
import unittest
from pathlib import Path
import numpy as np

from kevacv import (
    ZoneAnomalyDetector,
    DatasetCollector
)


class TestAnomalyBaseline(unittest.TestCase):
    def test_anomaly_detection(self):
        detector = ZoneAnomalyDetector(z_threshold=2.5, min_history_samples=5)
        
        # Fit baseline with 10 historical records of ~5 people
        records = [
            {"zone_id": "reception", "hour": 19, "is_weekend": False, "count": 5.0 + np.random.normal(0, 0.5)}
            for _ in range(10)
        ]
        detector.fit_history(records)

        # Normal query (count=5) -> Not an anomaly
        res_normal = detector.score_occupancy("reception", 19, count=5.0, is_weekend=False)
        self.assertFalse(res_normal["is_anomaly"])

        # Anomalous query (count=25) -> Anomaly detected
        res_surge = detector.score_occupancy("reception", 19, count=25.0, is_weekend=False)
        self.assertTrue(res_surge["is_anomaly"])
        self.assertGreater(res_surge["z_score"], 2.5)


class TestDatasetCollector(unittest.TestCase):
    def test_pseudo_label_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DatasetCollector(output_dir=tmpdir)
            
            fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            boxes = [(100, 100, 200, 400), (500, 200, 600, 600)]

            success = collector.save_frame_pseudo_labels(fake_frame, boxes)
            self.assertTrue(success)

            yaml_path = collector.export_dataset_yaml("test_venue")
            self.assertTrue(yaml_path.exists())


if __name__ == "__main__":
    unittest.main()
