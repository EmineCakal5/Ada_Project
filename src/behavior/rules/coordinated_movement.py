"""
Coordinated Movement Detection — Koordineli Hareket Tespiti.
Birden fazla nesnenin uyumlu şekilde birlikte hareket etmesini tespit eder.
"""

import numpy as np
import time
import logging
from typing import Dict, List, Set, Tuple
from itertools import combinations

logger = logging.getLogger(__name__)

PERSON_CLASS_NAMES = {"person", "pedestrian", "people"}


class CoordinatedMovementDetector:
    """
    Koordineli hareket tespiti.

    Algoritma:
    1. Aktif kişi tracklerini ikili olarak karşılaştır.
    2. Her track için son N frame'deki ortalama hız vektörünü hesapla.
    3. Cosine similarity > eşik AND kişiler arası mesafe < proximity_px → koordineli çift.
    4. Bu durum min_duration saniye sürekli devam ederse → alarm.
    """

    def __init__(self, config: dict):
        cfg = config["behavior"].get("coordinated_movement", {})
        self.velocity_sim_threshold = cfg.get("velocity_similarity_threshold", 0.85)
        self.proximity_px           = cfg.get("proximity_px", 200)
        self.min_duration_seconds   = cfg.get("min_duration_seconds", 5.0)
        self.min_velocity_px        = cfg.get("min_velocity_px", 3.0)

        # (tid_a, tid_b) → {"start": float, "warned": bool}
        self._pair_states: Dict[Tuple[int, int], dict] = {}
        logger.info(
            "CoordinatedMovementDetector: sim_thr=%.2f, proximity=%dpx, min_dur=%.1fs",
            self.velocity_sim_threshold, self.proximity_px, self.min_duration_seconds,
        )

    # ------------------------------------------------------------------
    def _velocity_vector(self, trajectory) -> np.ndarray:
        """Son 8 frame üzerinden ortalama hız vektörü."""
        pts = list(trajectory)
        if len(pts) < 4:
            return np.zeros(2, dtype=np.float32)
        recent = pts[-min(8, len(pts)):]
        vecs   = [np.array(recent[i + 1]) - np.array(recent[i])
                  for i in range(len(recent) - 1)]
        return np.mean(vecs, axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    def check(self, tracks: List) -> List[Dict]:
        """
        Tüm aktif trackler arasında koordineli hareket ara.

        Args:
            tracks: List[TrackInfo]

        Returns:
            Alert dict listesi
        """
        alerts: List[Dict] = []
        now    = time.time()

        person_tracks = [
            t for t in tracks
            if t.class_name in PERSON_CLASS_NAMES
            and t.velocity >= self.min_velocity_px
            and len(t.trajectory) >= 4
        ]

        active_pairs: Set[Tuple[int, int]] = set()

        for t_a, t_b in combinations(person_tracks, 2):
            pair_key = (min(t_a.track_id, t_b.track_id),
                        max(t_a.track_id, t_b.track_id))

            # Mesafe kontrolü
            dist = float(np.linalg.norm(
                np.array(t_a.center) - np.array(t_b.center)
            ))
            if dist > self.proximity_px:
                continue

            # Hız vektörü kosinüs benzerliği
            v_a    = self._velocity_vector(t_a.trajectory)
            v_b    = self._velocity_vector(t_b.trajectory)
            norm_a = np.linalg.norm(v_a)
            norm_b = np.linalg.norm(v_b)
            if norm_a < 0.01 or norm_b < 0.01:
                continue

            similarity = float(np.dot(v_a, v_b) / (norm_a * norm_b))
            if similarity < self.velocity_sim_threshold:
                continue

            active_pairs.add(pair_key)

            state    = self._pair_states.setdefault(pair_key, {"start": now, "warned": False})
            duration = now - state["start"]
            score    = min(duration / max(self.min_duration_seconds, 1), 1.0)

            if duration >= self.min_duration_seconds and not state["warned"]:
                state["warned"] = True
                alerts.append({
                    "type":       "coordinated_movement",
                    "track_id":   pair_key[0],
                    "track_ids":  list(pair_key),
                    "score":      round(score, 3),
                    "similarity": round(similarity, 3),
                    "distance_px": round(dist, 1),
                    "duration_s": round(duration, 1),
                    "alert_msg": (
                        f"👥 KOORDİNELİ HAREKET: #{pair_key[0]} ve #{pair_key[1]} "
                        f"birlikte hareket ediyor "
                        f"[benzerlik={similarity:.2f}, mesafe={dist:.0f}px, {duration:.0f}s]"
                    ),
                })

        # Artık aktif olmayan çiftleri temizle
        stale = [p for p in self._pair_states if p not in active_pairs]
        for p in stale:
            del self._pair_states[p]

        return alerts

    def get_score(self, track_id: int) -> float:
        """Belirli bir track'in koordineli hareket skoru (0.0-1.0)."""
        now  = time.time()
        best = 0.0
        for pair_key, state in self._pair_states.items():
            if track_id in pair_key:
                duration = now - state["start"]
                score    = min(duration / max(self.min_duration_seconds, 1), 1.0)
                best     = max(best, score)
        return best

    def update_active(self, active_ids: List[int]) -> None:
        stale = [p for p in self._pair_states
                 if p[0] not in active_ids and p[1] not in active_ids]
        for p in stale:
            del self._pair_states[p]

    def apply_camera_motion(self, H) -> None:
        # Hız vektörleri TrackInfo.trajectory üzerinden hesaplanıyor.
        pass
