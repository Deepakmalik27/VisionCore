"""tracker_wrapper.py — Phase B P7 BoxMOT Tracker Interface & Configuration.

WHY THIS EXISTS
    The pipeline relies on BoxMOT for Multi-Object Tracking. Upgrading BoxMOT (e.g. v19 -> v25+
    with BoostTrack++ / Soft BIoU) requires a resilient wrapper that abstracts API differences
    across BoxMOT versions and handles numpy/torch array conversions cleanly.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

_log = logging.getLogger(__name__)

try:
    import boxmot
    _HAVE_BOXMOT = True
except ImportError:
    _HAVE_BOXMOT = False


class TrackerWrapper:
    """Unified wrapper around BoxMOT trackers (BoostTrack++, BoT-SORT, StrongSORT, ByteTrack)."""

    SUPPORTED_TRACKERS = ("boosttrack", "botsort", "strongsort", "bytetrack", "ocsort")

    def __init__(self,
                 tracker_type: str = "boosttrack",
                 reid_weights: Optional[Union[str, Path]] = None,
                 device: str = "cuda:0",
                 conf: float = 0.25,
                 iou: float = 0.45,
                 track_high_thresh: float = 0.5,
                 track_low_thresh: float = 0.1,
                 new_track_thresh: float = 0.6,
                 track_buffer: int = 60):
        self.tracker_type = tracker_type.lower()
        self.reid_weights = Path(reid_weights) if reid_weights else None
        self.device = device
        self.conf = conf
        self.iou = iou
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.track_buffer = track_buffer

        self.tracker = self._init_tracker()

    def _init_tracker(self) -> Any:
        if not _HAVE_BOXMOT:
            return None

        # Determine class name dynamically from boxmot
        try:
            if self.tracker_type in ("boosttrack", "boosttrack++"):
                if hasattr(boxmot, "BoostTrack"):
                    return boxmot.BoostTrack(
                        reid_weights=self.reid_weights,
                        device=self.device,
                        conf=self.conf,
                        iou=self.iou,
                        track_buffer=self.track_buffer
                    )
            elif self.tracker_type == "botsort":
                if hasattr(boxmot, "BotSort"):
                    return boxmot.BotSort(
                        reid_weights=self.reid_weights,
                        device=self.device,
                        conf=self.conf,
                        iou=self.iou,
                        track_buffer=self.track_buffer
                    )
            elif self.tracker_type == "bytetrack":
                if hasattr(boxmot, "ByteTrack"):
                    return boxmot.ByteTrack(
                        track_thresh=self.track_high_thresh,
                        track_buffer=self.track_buffer,
                        match_thresh=self.iou
                    )
        except Exception as e:
            _log.warning("BoxMOT tracker init failed (%s: %s). "
                         "Falling back to dummy per-frame ID assignment.", type(e).__name__, e)

        return None

    def update(self, dets: np.ndarray, img: np.ndarray) -> np.ndarray:
        """Update tracker with frame detections [x1, y1, x2, y2, conf, cls].
        Returns tracked objects [x1, y1, x2, y2, track_id, conf, cls, idx].
        """
        if self.tracker is not None:
            try:
                result = self.tracker.update(dets, img)
                if result is not None:
                    return result
            except Exception as e:
                _log.warning("BoxMOT tracker.update() failed (%s: %s). "
                             "Using fallback ID assignment for this frame.", type(e).__name__, e)

        # Fallback dummy tracking format if boxmot is absent or failed
        if len(dets) == 0:
            return np.empty((0, 8), dtype=np.float32)

        out = []
        for idx, d in enumerate(dets):
            x1, y1, x2, y2 = d[:4]
            c = d[4] if len(d) > 4 else 1.0
            cls_id = d[5] if len(d) > 5 else 0.0
            out.append([x1, y1, x2, y2, idx + 1, c, cls_id, idx])
        return np.array(out, dtype=np.float32)
