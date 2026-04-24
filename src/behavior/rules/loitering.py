# -*- coding: utf-8 -*-
"""
Loitering Detection — Anormal Bekleme Tespiti.
Bir nesnenin belirli bir süre aynı alanda kalmasını tespit eder.
"""

import time
import numpy as np
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class LoiteringDetector:
    """
    Anormal bekleme tespiti.

    Algoritma:
    1. Track başlangıç pozisyonunu kaydet.
    2. Her frame'de toplam yer değiştirmeye bak.
    3. Yer değiştirme min_displacement'tan azsa → "duruyor" sayılır.
    4. Dwell time threshold'u aşılırsa → alarm ver.

    Kullanım:
        detector = LoiteringDetector(config)
        result = detector.check(track_id, center, dwell_time, velocity)
    """

    def __init__(self, config: dict):
        cfg = config["behavior"]["loitering"]
        self.threshold_seconds = cfg.get("threshold_seconds", 60)
        self.min_displacement = cfg.get("min_displacement", 50)   # piksel

        # track_id → {"anchor": center, "start": time, "warned": bool}
        self._states: Dict[int, dict] = {}
        logger.info(
            f"LoiteringDetector: threshold={self.threshold_seconds}s, "
            f"min_displacement={self.min_displacement}px"
        )

    def check(self, track_id: int, center: Tuple[float, float],
              dwell_time: float, velocity: float,
              class_name: str = "person") -> Dict:
        """
        Tek bir track için loitering kontrolü.

        Args:
            track_id: İz kimliği
            center: Güncel merkez koordinatı (px, py)
            dwell_time: Kaç saniyedir var (TrackInfo.dwell_time)
            velocity: Piksel/saniye hızı
            class_name: Nesne sınıfı

        Returns:
            {
                "loitering": bool,
                "loitering_score": 0.0-1.0,
                "dwell_seconds": float,
                "alert_msg": str or None
            }
        """
        result = {
            "loitering": False,
            "loitering_score": 0.0,
            "dwell_seconds": dwell_time,
            "alert_msg": None
        }

        if track_id not in self._states:
            self._states[track_id] = {
                "anchor": center,
                "start": time.time(),
                "warned": False,
                "stationary_start": time.time()
            }
            return result

        state = self._states[track_id]
        anchor = state["anchor"]

        # Mevcut hareketi anchor'a göre ölç
        displacement = np.linalg.norm(
            np.array(center) - np.array(anchor)
        )

        # Çok fazla hareket ettiyse anchor'ı yenile
        if displacement > self.min_displacement * 2:
            state["anchor"] = center
            state["stationary_start"] = time.time()
            state["warned"] = False
            return result

        # Küçük hareket — durağan süreyi ölç
        stationary_duration = time.time() - state["stationary_start"]

        # Skor: 0 → threshold, skor lineer artar
        score = min(stationary_duration / self.threshold_seconds, 1.0)
        result["loitering_score"] = round(score, 3)
        result["dwell_seconds"] = round(stationary_duration, 1)

        if stationary_duration >= self.threshold_seconds:
            result["loitering"] = True
            if not state["warned"]:
                state["warned"] = True
                result["alert_msg"] = (
                    f"Loitering: {class_name} #{track_id} stationary for "
                    f"{int(stationary_duration)}s"
                )

        return result

    def remove_track(self, track_id: int):
        """Takip bittiğinde state'i temizle."""
        self._states.pop(track_id, None)

    def update_active(self, active_ids):
        """Artık aktif olmayan track'lerin state'ini temizle."""
        dead = [tid for tid in self._states if tid not in active_ids]
        for tid in dead:
            del self._states[tid]

    def get_stationary_duration(self, track_id: int) -> float:
        """Belirli bir track'in durağan süresini döner."""
        if track_id not in self._states:
            return 0.0
        return time.time() - self._states[track_id]["stationary_start"]

    # ------------------------------------------------------------------
    # GMC (drone/IHA) — anchor noktalarini kamera hareketine gore kaydir
    # ------------------------------------------------------------------
    def apply_camera_motion(self, H) -> None:
        """
        Loitering anchor'lari ekran koordinatinda saklanir; kamera
        kaydiginda bu noktalar da warp edilmeli, aksi halde aslinda
        sabit duran bir yaya "hareket ediyormus" gibi algilanir ve
        anchor surekli yenilenir (loitering alarmi hiç tetiklenmez).
        """
        if H is None or not self._states:
            return
        Hf = np.asarray(H, dtype=np.float32)
        if Hf.shape != (2, 3):
            return
        for state in self._states.values():
            ax, ay = state["anchor"]
            nx = Hf[0, 0] * ax + Hf[0, 1] * ay + Hf[0, 2]
            ny = Hf[1, 0] * ax + Hf[1, 1] * ay + Hf[1, 2]
            state["anchor"] = (float(nx), float(ny))
