"""anomaly_baseline.py — Phase C P10 Learned Anomaly Baselines (Avigilon UMD-Style Semantic Anomaly Engine).

WHY THIS EXISTS
    Static alert thresholds require manual hand-tuning per venue and fail to adapt to day-of-week
    or hour-of-day traffic patterns. 

THE PRINCIPLE
    1. Per-Zone Seasonal Baseline Model: Learns typical occupancy distribution (mean & std)
       per (zone_id, hour_of_day, is_weekend).
    2. Residual Z-Score Scoring: Scores actual occupancy count against learned baseline:
       z = (actual - mean) / std.
    3. Flags statistically significant anomalies (|z| > threshold, default 2.5) with semantic context:
       - "Host stand unstaffed during peak ramp"
       - "Queue depth surge"
       - "Unusual lounge area crowd"
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


class ZoneAnomalyDetector:
    """Learned anomaly baseline engine for per-zone occupancy time-series."""

    def __init__(self, z_threshold: float = 2.5, min_history_samples: int = 5):
        self.z_threshold = z_threshold
        self.min_history_samples = min_history_samples
        # Store history: (zone_id, hour, is_weekend) -> list of occupancy counts
        self.history: Dict[Tuple[str, int, bool], List[float]] = defaultdict(list)

    def fit_history(self, records: List[Dict[str, Any]]) -> None:
        """Fit history from records: [{zone_id, hour, is_weekend, count}, ...]"""
        for r in records:
            key = (r["zone_id"], int(r["hour"]), bool(r.get("is_weekend", False)))
            self.history[key].append(float(r["count"]))

    def score_occupancy(self,
                        zone_id: str,
                        hour: int,
                        count: float,
                        is_weekend: bool = False) -> Dict[str, Any]:
        """Score current zone occupancy count against baseline model.
        Returns dict with z_score, is_anomaly, baseline_mean, baseline_std, and explanation.
        """
        key = (zone_id, hour, is_weekend)
        samples = self.history.get(key, [])

        if len(samples) < self.min_history_samples:
            # Update history and return unflagged baseline learning state
            self.history[key].append(float(count))
            return {
                "zone_id": zone_id,
                "hour": hour,
                "count": count,
                "is_anomaly": False,
                "z_score": 0.0,
                "baseline_mean": count,
                "baseline_std": 0.0,
                "note": f"Learning baseline ({len(samples)}/{self.min_history_samples} samples)"
            }

        mean = sum(samples) / len(samples)
        variance = sum((x - mean) ** 2 for x in samples) / len(samples)
        if variance < 1e-6:
            # All historical values are near-identical — z-score is meaningless
            std = 1.0
            is_constant_baseline = True
        else:
            std = math.sqrt(variance)
            is_constant_baseline = False

        z_score = (count - mean) / std
        is_anomaly = abs(z_score) >= self.z_threshold

        # Update running history (limit max memory 500 samples per slot)
        self.history[key].append(float(count))
        if len(self.history[key]) > 500:
            self.history[key].pop(0)

        explanation = ""
        if is_anomaly:
            direction = "SURGE" if z_score > 0 else "DROP/UNSTAFFED"
            explanation = (f"Zone '{zone_id}' occupancy {count:.0f} deviates ({direction}) "
                           f"from normal baseline {mean:.1f} ± {std:.1f} (Z={z_score:+.2f})")
        elif is_constant_baseline and abs(count - mean) > 0.5:
            explanation = (f"Zone '{zone_id}' baseline has zero variance (all prior samples ~{mean:.0f}). "
                           f"Current count {count:.0f} differs but z-score uses artificial std=1.")

        return {
            "zone_id": zone_id,
            "hour": hour,
            "count": count,
            "is_anomaly": is_anomaly,
            "z_score": round(z_score, 2),
            "baseline_mean": round(mean, 1),
            "baseline_std": round(std, 1),
            "constant_baseline": is_constant_baseline,
            "explanation": explanation
        }
