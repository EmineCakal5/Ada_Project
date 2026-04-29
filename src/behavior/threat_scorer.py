# -*- coding: utf-8 -*-
"""
Threat Scorer — Kural çıktılarından feature vektörü ve birleşik tehlike skoru üretir.
Implementation Plan v2'deki feature vektör yapısını uygular.
"""

import numpy as np
import math
import logging
from typing import Dict, List, Tuple
from src.tracker.track_history import TrackInfo

logger = logging.getLogger(__name__)

# Tehdit seviyeleri
THREAT_LEVELS = {
    0: "LOW",
    1: "MEDIUM",
    2: "HIGH",
    3: "CRITICAL"
}

THREAT_THRESHOLDS = {
    "LOW":      (0.00, 0.35),
    "MEDIUM":   (0.35, 0.60),
    "HIGH":     (0.60, 0.80),
    "CRITICAL": (0.80, 1.01),
}


class ThreatScorer:
    """
    Kural çıktılarını normalize edip ağırlıklı tehlike skoru hesaplar.

    Feature vektörü (8 boyut):
        [0] zone_violation_score      — 0.0 - 1.0
        [1] dwell_time_normalized     — saniye / max_threshold
        [2] velocity_magnitude        — normalize edilmiş px/s
        [3] trajectory_variance       — normalize edilmiş
        [4] loitering_score           — 0.0 - 1.0
        [5] abandoned_object_score    — 0.0 - 1.0
        [6] time_of_day_encoded       — sin(hour * 2π/24)
        [7] object_class_risk         — sınıfa göre risk (person=0.5, vehicle=0.3 vb.)

    Kullanım:
        scorer = ThreatScorer(config)
        result = scorer.compute(track, zone_result, loitering_result, abandoned_score)
    """

    # Normalize sabitleri
    MAX_DWELL_SECONDS = 300.0     # 5 dakika
    MAX_VELOCITY_PX   = 200.0    # px/frame
    MAX_VARIANCE       = 50000.0

    # Sınıf risk skoru
    CLASS_RISK = {
        "person":     0.5,
        "car":        0.3,
        "truck":      0.4,
        "bus":        0.3,
        "motorbike":  0.35,
        "bicycle":    0.2,
        "backpack":   0.6,
        "handbag":    0.5,
        "suitcase":   0.7,
        "default":    0.3,
    }

    def __init__(self, config: dict):
        weights_cfg = config["behavior"]["threat_scorer"]["weights"]
        self.w_zone           = weights_cfg.get("zone_violation",        0.25)
        self.w_loitering      = weights_cfg.get("loitering",             0.20)
        self.w_abandoned      = weights_cfg.get("abandoned_object",      0.25)
        self.w_velocity       = weights_cfg.get("velocity_anomaly",      0.10)
        self.w_reconnaissance = weights_cfg.get("reconnaissance",        0.10)
        self.w_coordinated    = weights_cfg.get("coordinated_movement",  0.10)

        logger.info(f"ThreatScorer başlatıldı: weights={weights_cfg}")

    def compute(
        self,
        track: TrackInfo,
        zone_result: Dict,
        loitering_result: Dict,
        abandoned_score: float = 0.0,
        reconnaissance_score: float = 0.0,
        coordinated_score: float = 0.0,
    ) -> Tuple[np.ndarray, float, str]:
        """
        Tehdit skoru ve feature vektörü hesapla.

        Returns:
            (feature_vector, threat_score, threat_level)
        """
        fv = self.build_feature_vector(
            track, zone_result, loitering_result,
            abandoned_score, reconnaissance_score, coordinated_score,
        )
        score = self.weighted_score(fv, zone_result, loitering_result, abandoned_score, track)
        level = self.score_to_level(score)
        return fv, score, level

    def build_feature_vector(
        self,
        track: TrackInfo,
        zone_result: Dict,
        loitering_result: Dict,
        abandoned_score: float,
        reconnaissance_score: float = 0.0,
        coordinated_score: float = 0.0,
    ) -> np.ndarray:
        """10-boyutlu feature vektörü üret.

        [0] zone_violation_score
        [1] dwell_time_normalized
        [2] velocity_magnitude
        [3] trajectory_variance
        [4] loitering_score
        [5] abandoned_object_score
        [6] time_of_day_encoded
        [7] object_class_risk
        [8] reconnaissance_score
        [9] coordinated_movement_score
        """
        import datetime
        hour = datetime.datetime.now().hour

        fv = np.array([
            float(zone_result.get("violation_score", 0.0)),                    # [0] zone
            min(track.dwell_time / self.MAX_DWELL_SECONDS, 1.0),               # [1] dwell
            min(track.velocity / self.MAX_VELOCITY_PX, 1.0),                   # [2] velocity
            min(track.trajectory_variance / self.MAX_VARIANCE, 1.0),           # [3] variance
            float(loitering_result.get("loitering_score", 0.0)),               # [4] loitering
            float(abandoned_score),                                             # [5] abandoned
            math.sin(hour * 2 * math.pi / 24),                                # [6] time-of-day
            self.CLASS_RISK.get(track.class_name, self.CLASS_RISK["default"]), # [7] class risk
            float(reconnaissance_score),                                        # [8] reconnaissance
            float(coordinated_score),                                           # [9] coordinated
        ], dtype=np.float32)

        return fv

    def weighted_score(
        self,
        fv: np.ndarray,
        zone_result: Dict,
        loitering_result: Dict,
        abandoned_score: float,
        track: TrackInfo
    ) -> float:
        """Ağırlıklı kural tabanlı skor (MLP yokken fallback)."""
        score = (
            self.w_zone          * fv[0] +   # zone violation
            self.w_loitering     * fv[4] +   # loitering
            self.w_abandoned     * fv[5] +   # abandoned
            self.w_velocity      * fv[2] +   # velocity anomaly
            self.w_reconnaissance * fv[8] +  # reconnaissance
            self.w_coordinated   * fv[9]     # coordinated movement
        )
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def score_to_level(score: float) -> str:
        """Skor → seviye."""
        for level, (lo, hi) in THREAT_THRESHOLDS.items():
            if lo <= score < hi:
                return level
        return "LOW"

    @staticmethod
    def level_color(level: str) -> Tuple[int, int, int]:
        """Tehdit seviyesi → BGR rengi."""
        colors = {
            "LOW":      (0, 200, 0),
            "MEDIUM":   (0, 165, 255),
            "HIGH":     (0, 0, 255),
            "CRITICAL": (0, 0, 180),
        }
        return colors.get(level, (0, 200, 0))

    @staticmethod
    def level_to_index(level: str) -> int:
        mapping = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        return mapping.get(level, 0)
