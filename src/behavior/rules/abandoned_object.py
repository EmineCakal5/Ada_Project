# -*- coding: utf-8 -*-
"""
Abandoned Object Detection — Terk Edilmiş Nesne Tespiti.
Bir nesnenin (çanta, kutu vb.) sahibinden ayrılarak hareketsiz kalmasını tespit eder.
"""

import time
import numpy as np
import logging
from typing import Dict, List, Tuple

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
    2. k. en yakın kişi (nearest_person_rank) owner_distance'tan uzaksa → "terk edilmiş aday"
       — kalabalıkta 1. en yakın her zaman çanta yanında kalabilir; k>1 ile ayarlanır.
    3. confirm_seconds kadar bu durum devam ederse → alarm ver.

    Kullanım:
        detector = AbandonedObjectDetector(config)
        alerts = detector.check(tracks)  # tüm aktif track'ler
    """

    def __init__(self, config: dict):
        cfg = config["behavior"]["abandoned_object"]
        self.owner_distance = cfg.get("owner_distance", 200)    # piksel
        self.confirm_seconds = cfg.get("confirm_seconds", 10)   # saniye
        self.nearest_person_rank = max(1, int(cfg.get("nearest_person_rank", 1)))

        # Hareketsiz nesne state'i: track_id → {center, last_move, candidate_since, warned}
        self._object_states: Dict[int, dict] = {}

        # Piksel: merkez kayması bu değerin altındaysa "sabit" say (bbox/GMC jitter)
        self.stationary_threshold = float(cfg.get("stationary_threshold_px", 28))

        logger.info(
            f"AbandonedObjectDetector: owner_dist={self.owner_distance}px, "
            f"confirm={self.confirm_seconds}s, "
            f"stationary_px={self.stationary_threshold}, "
            f"person_rank_k={self.nearest_person_rank}"
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

        if logger.isEnabledFor(logging.DEBUG):
            for t in tracks:
                logger.debug(
                    "Nesne sinifi: %s, ID: %s (cls_id=%s, vel=%.3f)",
                    t.class_name,
                    t.track_id,
                    getattr(t, "class_id", -1),
                    float(t.velocity),
                )

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

            # Nesne sabit — k. en yakın kişi (kalabalık UAV görüntüsü için)
            min_dist = self._k_nearest_person_distance(
                center, persons, self.nearest_person_rank
            )

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
                            f"Abandoned object: {obj.class_name} #{tid} "
                            f"unattended for {int(elapsed)}s"
                        )
                    }
                    alerts.append(alert)
            else:
                # Sahibi geri döndü
                state["candidate_since"] = None
                state["warned"] = False

        if logger.isEnabledFor(logging.DEBUG):
            parts: List[str] = []
            for obj in objects:
                tid = obj.track_id
                st = self._object_states.get(tid)
                md = self._k_nearest_person_distance(
                    obj.center, persons, self.nearest_person_rank
                )
                if st is None:
                    parts.append(
                        f"id={tid} cls={obj.class_name} state=new "
                        f"near_person={md:.0f}px/{self.owner_distance}px"
                    )
                    continue
                cand = st.get("candidate_since")
                cand_sec = (time.time() - cand) if cand is not None else 0.0
                disp = float(
                    np.linalg.norm(np.array(obj.center) - np.array(st["center"]))
                )
                parts.append(
                    f"id={tid} cls={obj.class_name} near_person={md:.0f}px "
                    f"cand={cand_sec:.1f}s/{self.confirm_seconds}s "
                    f"disp={disp:.1f}px(stationary<={self.stationary_threshold}) "
                    f"vel={float(obj.velocity):.3f} warned={st.get('warned')}"
                )
            logger.debug(
                "abandoned_monitored: %s",
                " | ".join(parts) if parts else "(nesne yok - OBJECT_CLASS_NAMES disi veya tespit yok)",
            )

        return alerts

    def get_score(self, track_id: int) -> float:
        """Belirli bir nesne için terk skoru döner (0.0-1.0)."""
        state = self._object_states.get(track_id)
        if state is None or state["candidate_since"] is None:
            return 0.0
        elapsed = time.time() - state["candidate_since"]
        return min(elapsed / (self.confirm_seconds * 3), 1.0)

    # ------------------------------------------------------------------
    # GMC (drone/IHA) — sabit nesne referans noktalarini warp et
    # ------------------------------------------------------------------
    def apply_camera_motion(self, H) -> None:
        """
        AbandonedObjectDetector, her sabit nesnenin son gordugu merkezi
        `state["center"]` olarak tutar. Kamera kaydiginda bu merkez de
        warp edilmeli; aksi halde asla hareket etmemis bir çanta bile
        "hareket etti" olarak damgalanir, aday state sifirlanir ve alarm
        hiç çalmaz.
        """
        if H is None or not self._object_states:
            return
        Hf = np.asarray(H, dtype=np.float32)
        if Hf.shape != (2, 3):
            return
        for state in self._object_states.values():
            cx, cy = state["center"]
            nx = Hf[0, 0] * cx + Hf[0, 1] * cy + Hf[0, 2]
            ny = Hf[1, 0] * cx + Hf[1, 1] * cy + Hf[1, 2]
            state["center"] = (float(nx), float(ny))

    @staticmethod
    def _k_nearest_person_distance(
        center: Tuple, persons: List, k: int
    ) -> float:
        """Merkeze göre k. en yakın kişinin mesafesi (k=1: en yakın)."""
        if not persons:
            return float("inf")
        dists = sorted(
            float(np.linalg.norm(np.array(center) - np.array(p.center)))
            for p in persons
        )
        rank = min(max(1, k), len(dists))
        return dists[rank - 1]
