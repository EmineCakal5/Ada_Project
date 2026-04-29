"""
Reconnaissance Detection — Keşif/Gözetleme Davranışı Tespiti.
Bir nesnenin belirli bir alanı sistematik olarak taramasını/gezmesini tespit eder.
"""

import numpy as np
import logging
from collections import deque
from typing import Dict, List

logger = logging.getLogger(__name__)

PERSON_CLASS_NAMES = {"person", "pedestrian", "people"}


class ReconnaissanceDetector:
    """
    Keşif davranışı tespiti.

    Algoritma:
    1. Track'in tüm yol uzunluğunu hesapla (path_length).
    2. İlk → son nokta doğrusal mesafesini hesapla (displacement).
    3. path_efficiency = displacement / path_length
       Düşük değer → ileri-geri / zigzag / sistematik tarama hareketi.
    4. Trajektori bounding-box alanını hesapla → geniş alan kapsama.
    5. Koşullar: min_time geçti + efficiency < eşik + coverage > eşik → alarm.
    """

    def __init__(self, config: dict):
        cfg = config["behavior"].get("reconnaissance", {})
        self.min_time_seconds     = cfg.get("min_time_seconds", 30)
        self.efficiency_threshold = cfg.get("efficiency_threshold", 0.35)
        self.min_coverage_px2     = cfg.get("min_coverage_px2", 8000)
        self.person_only          = cfg.get("person_only", True)

        # track_id → {"warned": bool}
        self._states: Dict[int, dict] = {}
        logger.info(
            "ReconnaissanceDetector: min_time=%ds, efficiency_thr=%.2f, "
            "min_coverage=%dpx²",
            self.min_time_seconds, self.efficiency_threshold, self.min_coverage_px2,
        )

    def check(self, track_id: int, trajectory: deque,
              dwell_time: float, class_name: str = "person") -> Dict:
        """
        Tek track için keşif davranışı kontrolü.

        Returns:
            {
                "reconnaissance": bool,
                "reconnaissance_score": 0.0-1.0,
                "path_efficiency": float,
                "coverage_area": float,
                "alert_msg": str or None
            }
        """
        result = {
            "reconnaissance": False,
            "reconnaissance_score": 0.0,
            "path_efficiency": 1.0,
            "coverage_area": 0.0,
            "alert_msg": None,
        }

        if self.person_only and class_name not in PERSON_CLASS_NAMES:
            return result

        if dwell_time < self.min_time_seconds or len(trajectory) < 15:
            return result

        pts = np.array(list(trajectory), dtype=np.float32)

        # Toplam yol uzunluğu
        path_length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        if path_length < 1.0:
            return result

        # Doğrusal yer değiştirme
        displacement = float(np.linalg.norm(pts[-1] - pts[0]))
        efficiency   = displacement / path_length
        result["path_efficiency"] = round(efficiency, 3)

        # Kapsama alanı (trajektori bounding-box)
        min_pt   = pts.min(axis=0)
        max_pt   = pts.max(axis=0)
        coverage = float((max_pt[0] - min_pt[0]) * (max_pt[1] - min_pt[1]))
        result["coverage_area"] = round(coverage, 1)

        # Keşif skoru
        if efficiency < self.efficiency_threshold:
            eff_score = (self.efficiency_threshold - efficiency) / self.efficiency_threshold
        else:
            eff_score = 0.0
        cov_score  = min(coverage / max(self.min_coverage_px2, 1), 1.0)
        recon_score = round(0.6 * eff_score + 0.4 * cov_score, 3)
        result["reconnaissance_score"] = recon_score

        if efficiency < self.efficiency_threshold and coverage > self.min_coverage_px2:
            result["reconnaissance"] = True
            state = self._states.setdefault(track_id, {"warned": False})
            if not state["warned"]:
                state["warned"] = True
                result["alert_msg"] = (
                    f"🔍 KEŞİF DAVRANIŞI: {class_name} #{track_id} "
                    f"alanı sistematik olarak tarıyor "
                    f"[verimlilik={efficiency:.2f}, kapsama={coverage:.0f}px²]"
                )

        return result

    def update_active(self, active_ids: List[int]) -> None:
        dead = [tid for tid in self._states if tid not in active_ids]
        for tid in dead:
            del self._states[tid]

    def apply_camera_motion(self, H) -> None:
        # Trajectory warp TrackInfo üzerinde yapılıyor; burada ek state yok.
        pass
