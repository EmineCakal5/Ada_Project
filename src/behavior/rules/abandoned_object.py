"""
Abandoned Object Detection — Terk Edilmiş Nesne Tespiti.
Bir nesnenin (çanta, kutu vb.) sahibinden ayrılarak hareketsiz kalmasını tespit eder.
"""

import time
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Set

logger = logging.getLogger(__name__)

# COCO sınıfları: hangileri "potansiyel terk edilmiş nesne" sayılır
OBJECT_CLASS_IDS = {24, 26, 28, 63, 56, 73}  # backpack, handbag, suitcase, laptop, chair, book
OBJECT_CLASS_NAMES = {"backpack", "handbag", "suitcase", "laptop", "chair", "book",
                      "bottle", "phone", "keyboard"}
PERSON_CLASS_NAMES = {"person"}


class AbandonedObjectDetector:
    """
    Terk edilmiş nesne tespiti.

    Algoritma:
    1. Hareketsiz nesneleri takip et (küçük displacement).
    2. En yakın kişi owner_distance'tan uzaklaşırsa → "terk edilmiş aday".
    3. confirm_seconds kadar bu durum devam ederse → alarm ver.

    Kullanım:
        detector = AbandonedObjectDetector(config)
        alerts = detector.check(tracks)  # tüm aktif track'ler
    """

    def __init__(self, config: dict):
        cfg = config["behavior"]["abandoned_object"]
        self.owner_distance = cfg.get("owner_distance", 150)    # piksel
        self.confirm_seconds = cfg.get("confirm_seconds", 10)   # saniye

        # Hareketsiz nesne state'i: track_id → {center, last_move, candidate_since, warned}
        self._object_states: Dict[int, dict] = {}

        # Hareketsizlik eşiği (piksel/frame)
        self.stationary_threshold = 20

        logger.info(
            f"AbandonedObjectDetector: owner_dist={self.owner_distance}px, "
            f"confirm={self.confirm_seconds}s"
        )

    def check(self, tracks: List) -> List[Dict]:
        """
        Tüm aktif track'leri analiz et.

        Args:
            tracks: List[TrackInfo] — track_history'den gelen aktif trackler

        Returns:
            Alert listesi, her biri dict

        """
        alerts = []

        # Nesneleri ve kişileri ayır
        objects = [t for t in tracks if t.class_name in OBJECT_CLASS_NAMES]
        persons = [t for t in tracks if t.class_name in PERSON_CLASS_NAMES]

        active_object_ids = {t.track_id for t in objects}

        # Artık var olmayan nesneleri temizle
        dead = [tid for tid in self._object_states if tid not in active_object_ids]
        for tid in dead:
            del self._object_states[tid]

        for obj in objects:
            tid = obj.track_id
            center = obj.center
            velocity = obj.velocity

            if tid not in self._object_states:
                self._object_states[tid] = {
                    "center": center,
                    "last_move": time.time(),
                    "candidate_since": None,
                    "warned": False
                }
                continue

            state = self._object_states[tid]

            # Hareket kontrolü
            dist_moved = np.linalg.norm(np.array(center) - np.array(state["center"]))
            if dist_moved > self.stationary_threshold:
                # Nesne hareket etti — sıfırla
                state["center"] = center
                state["last_move"] = time.time()
                state["candidate_since"] = None
                state["warned"] = False
                continue

            # Nesne sabit — en yakın kişiyi bul
            min_dist = self._nearest_person_distance(center, persons)

            if min_dist > self.owner_distance:
                # Sahibi uzaklaştı — aday başlat
                if state["candidate_since"] is None:
                    state["candidate_since"] = time.time()

                elapsed = time.time() - state["candidate_since"]

                if elapsed >= self.confirm_seconds and not state["warned"]:
                    state["warned"] = True
                    alert = {
                        "type": "abandoned_object",
                        "track_id": tid,
                        "class_name": obj.class_name,
                        "center": center,
                        "abandoned_seconds": round(elapsed, 1),
                        "score": min(elapsed / (self.confirm_seconds * 3), 1.0),
                        "alert_msg": (
                            f"🎒 TERK EDİLMİŞ NESNE: {obj.class_name} #{tid} "
                            f"{int(elapsed)}s boyunca sahipsiz"
                        )
                    }
                    alerts.append(alert)
            else:
                # Sahibi geri döndü
                state["candidate_since"] = None
                state["warned"] = False

        return alerts

    def get_score(self, track_id: int) -> float:
        """Belirli bir nesne için terk skoru döner (0.0-1.0)."""
        state = self._object_states.get(track_id)
        if state is None or state["candidate_since"] is None:
            return 0.0
        elapsed = time.time() - state["candidate_since"]
        return min(elapsed / (self.confirm_seconds * 3), 1.0)

    @staticmethod
    def _nearest_person_distance(center: Tuple, persons: List) -> float:
        """En yakın kişiye olan mesafeyi döner."""
        if not persons:
            return float("inf")
        dists = [
            np.linalg.norm(np.array(center) - np.array(p.center))
            for p in persons
        ]
        return float(min(dists))
