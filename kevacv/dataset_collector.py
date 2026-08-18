"""dataset_collector.py — Phase D P13 Venue-Specific Dataset Collector & Pseudo-Labeling Engine.

WHY THIS EXISTS
    Verkada's 2025 AI whitepaper demonstrated that a 400M fine-tuned venue-specific model 
    substantially outperforms a 2B general model in both accuracy and speed.

THE PRINCIPLE
    Automated Data Bank & Pseudo-Labeling:
    1. Passively collects high-confidence crops and frame samples during video analytics runs.
    2. Exports pseudo-labeled training sets in YOLO format (images + txt labels).
    3. Enables fine-tuning YOLO / DINOv2 adapters specifically for the venue's camera angles,
       lighting conditions, and occlusion profiles.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import cv2
    _HAVE_CV2 = True
except ImportError:
    _HAVE_CV2 = False


class DatasetCollector:
    """Venue-specific training sample collector and pseudo-label exporter."""

    def __init__(self, output_dir: Union[str, Path] = "./venue_dataset"):
        self.output_dir = Path(output_dir)
        self.img_dir = self.output_dir / "images"
        self.lbl_dir = self.output_dir / "labels"
        
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.lbl_dir.mkdir(parents=True, exist_ok=True)
        self.counter = 0

    def save_frame_pseudo_labels(self,
                                 frame: np.ndarray,
                                 boxes: List[Tuple[float, float, float, float]],
                                 confidences: Optional[List[float]] = None,
                                 class_id: int = 0) -> bool:
        """Save a frame image and normalized YOLO format bounding box label file.
        boxes: List of (x1, y1, x2, y2)
        """
        if not _HAVE_CV2 or frame is None or len(boxes) == 0:
            return False

        self.counter += 1
        stem = f"frame_{self.counter:06d}"
        
        img_path = self.img_dir / f"{stem}.jpg"
        lbl_path = self.lbl_dir / f"{stem}.txt"

        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return False

        # Write image
        cv2.imwrite(str(img_path), frame)

        # Write YOLO label format: class_id x_center y_center width height (normalized)
        yolo_lines = []
        for idx, (x1, y1, x2, y2) in enumerate(boxes):
            bw = float(x2 - x1)
            bh = float(y2 - y1)
            xc = float(x1 + x2) / 2.0 / w
            yc = float(y1 + y2) / 2.0 / h
            nw = bw / w
            nh = bh / h

            # Clamp normalized coordinates to [0.0, 1.0]
            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        lbl_path.write_text("\n".join(yolo_lines), encoding="utf-8")
        return True

    def export_dataset_yaml(self, venue_name: str = "delilah_la",
                            val_frac: float = 0.2) -> Path:
        """Export dataset.yaml for Ultralytics fine-tuning, with a REAL split.

        This used to emit `train: images` AND `val: images` — the same
        directory for both. Validating on the training set makes the reported
        mAP meaningless: it measures memorisation, and it goes UP as the model
        overfits, so the number that is supposed to tell you when to stop is
        the one that misleads you into continuing.

        The split is deterministic (every 1-in-N by sorted filename) rather
        than random, so re-running produces the same split and two training
        runs stay comparable.
        """
        imgs = sorted(p.name for p in self.img_dir.glob("*.jpg"))
        step = max(2, int(round(1.0 / max(val_frac, 1e-6))))
        val = [n for i, n in enumerate(imgs) if i % step == 0]
        train = [n for i, n in enumerate(imgs) if i % step != 0]
        (self.output_dir / "train.txt").write_text(
            "\n".join(f"images/{n}" for n in train), encoding="utf-8")
        (self.output_dir / "val.txt").write_text(
            "\n".join(f"images/{n}" for n in val), encoding="utf-8")

        yaml_content = f"""# Venue-Specific Dataset Config for {venue_name}
#
# READ THIS BEFORE TRAINING ON IT
#   These labels are the PIPELINE'S OWN PREDICTIONS (pseudo-labels). Training
#   on them teaches the model to reproduce what it already does — including
#   every phantom, every merged box and every miss. Used raw, this makes the
#   detector more confidently wrong, and the errors become invisible because
#   the model and the labels now agree.
#
#   They are a STARTING POINT FOR CORRECTION, exactly like tools/gt_kit.py's
#   CVAT seed: delete the phantoms, add the missed people, fix the boxes that
#   cover two bodies. Correcting is ~5-10x less work than drawing from
#   scratch, and the corrected set is what you fine-tune on.
#
#   {len(train)} train / {len(val)} val, split deterministically so two runs
#   remain comparable.
path: {self.output_dir.absolute()}
train: train.txt
val: val.txt

names:
  0: person
"""
        yaml_path = self.output_dir / "dataset.yaml"
        yaml_path.write_text(yaml_content, encoding="utf-8")
        return yaml_path
