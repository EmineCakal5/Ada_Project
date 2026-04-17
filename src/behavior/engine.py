"""
Behavior Engine — Tüm davranış kurallarını orkestre eden ana motor.
Zone ihlali + Loitering + Terk edilmiş nesne → Tehdit skoru + MLP sınıflandırma.
"""

import time
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np

from src.tracker.track_history import TrackHistory, TrackInfo
from src.behavior.rules.zone_violation import ZoneViolationDetector
from src.behavior.rules.loitering import LoiteringDetector
from src.behavior.rules.abandoned_object import AbandonedObjectDetector
from src.behavior.threat_scorer import ThreatScorer
from src.behavior.threat_mlp import ThreatMLPClassifier

logger = logging.getLogger(__name__)


class Alert:
    """Tek bir güvenlik alarmı."""
    def __init__(self, alert_type: str, track_id: int, message: str,
                 threat_level: str, score: float, timestamp: float = None):
        self.alert_type  = alert_type
        self.track_id    = track_id
        self.message     = message
        self.threat_level = threat_level
        self.score       = score
        self.timestamp   = timestamp or time.time()

    def to_dict(self) -> dict:
        return {
            "type":         self.alert_type,
            "track_id":     self.track_id,
            "message":      self.message,
            "threat_level": self.threat_level,
            "score":        round(self.score, 3),
            "time":         time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        }


class BehaviorEngine:
    """
    Ana davranış analiz motoru.

    Her frame'de:
    1. Track history üzerinden aktif track'leri al.
    2. Zone ihlali kontrol et.
    3. Loitering kontrol et.
    4. Terk edilmiş nesne kontrol et.
    5. Tehdit skoru hesapla (kural tabanlı).
    6. MLP ile sınıfla (derin öğrenme katmanı).
    7. Alert listesi döndür.

    Kullanım:
        engine = BehaviorEngine(config)
        alerts = engine.process(track_history, frame_w, frame_h)
    """

    def __init__(self, config: dict, zones_path: str = None):
        self.config = config
        zones_path = zones_path or config["behavior"]["zones_file"]

        # Alt modüller
        self.zone_detector    = None   # Frame boyutu gelince başlatılır
        self.loitering        = LoiteringDetector(config)
        self.abandoned        = AbandonedObjectDetector(config)
        self.scorer           = ThreatScorer(config)
        self.mlp              = None   # Lazy load

        self._zones_path      = zones_path
        self._frame_w         = None
        self._frame_h         = None

        # Alert geçmişi (son 100)
        self.alert_history: List[Alert] = []

        # Tekrar alarm bastırma: track_id → son alarm zamanı
        self._last_alert_time: Dict[int, float] = {}
        self.alert_cooldown = 15.0  # saniye

        logger.info("BehaviorEngine başlatıldı")

    def _init_zones(self, frame_w: int, frame_h: int):
        """Zone detector'ı frame boyutuyla başlat."""
        if self.zone_detector is None or self._frame_w != frame_w:
            self.zone_detector = ZoneViolationDetector(self._zones_path, frame_w, frame_h)
            self._frame_w = frame_w
            self._frame_h = frame_h

    def _get_mlp(self):
        """MLP'yi lazy initialize et (ilk kullanımda)."""
        if self.mlp is None:
            try:
                self.mlp = ThreatMLPClassifier(self.config)
            except Exception as e:
                logger.warning(f"MLP yüklenemedi, rule-based kullanılacak: {e}")
        return self.mlp

    # ------------------------------------------------------------------
    # GMC (drone / IHA) — tum davranis alt modullerine tek noktadan yay
    # ------------------------------------------------------------------
    def apply_camera_motion(self, H) -> None:
        """
        Pipeline her frame'de EgoMotionCompensator'dan gelen 2x3 affine
        matrisi bu metoda verir. BehaviorEngine da zone poligonlarini,
        loitering anchor'larini ve abandoned-object merkezlerini bu
        matrise gore senkronize eder. Boylece kurallar kamera hareketli
        iken dahi "yeryuzune sabitlenmis" gibi davranir.
        """
        if H is None:
            return
        if self.zone_detector is not None:
            self.zone_detector.apply_camera_motion(H)
        self.loitering.apply_camera_motion(H)
        self.abandoned.apply_camera_motion(H)

    def process(self, track_history: TrackHistory,
                frame_w: int, frame_h: int) -> Tuple[List[Alert], Dict]:
        """
        Tüm aktif track'leri analiz et.

        Returns:
            (new_alerts_this_frame, per_track_info)
            per_track_info: {track_id: {zone, loitering, threat_score, ...}}
        """
        self._init_zones(frame_w, frame_h)
        mlp = self._get_mlp()

        active_tracks = track_history.get_all_active()
        new_alerts: List[Alert] = []
        per_track: Dict = {}

        # Terk edilmiş nesne kontrolü (tüm track'ler gerekli)
        abandoned_alerts = self.abandoned.check(active_tracks)
        for aa in abandoned_alerts:
            self._add_alert(new_alerts, Alert(
                alert_type="abandoned_object",
                track_id=aa["track_id"],
                message=aa["alert_msg"],
                threat_level="HIGH",
                score=aa["score"]
            ))

        # Abandoned score map
        abandoned_scores = {aa["track_id"]: aa["score"] for aa in abandoned_alerts}

        # Her track için analiz
        for track in active_tracks:
            tid = track.track_id

            # 1. Zone ihlali
            zone_result = self.zone_detector.check(tid, track.center, track.class_name)

            # 2. Loitering
            loiter_result = self.loitering.check(
                tid, track.center, track.dwell_time,
                track.velocity, track.class_name
            )

            # 3. Abandoned score for this track
            ab_score = self.abandoned.get_score(tid)

            # 4. Feature vector + rule score
            fv, rule_score, rule_level = self.scorer.compute(
                track, zone_result, loiter_result, ab_score
            )

            # 5. MLP sınıflama
            if mlp is not None:
                try:
                    mlp_level, mlp_probs = mlp.predict(fv)
                    # Hibrit: %60 MLP + %40 kural
                    mlp_score = float(np.dot(mlp_probs, [0.0, 0.33, 0.66, 1.0]))
                    final_score = 0.6 * mlp_score + 0.4 * rule_score
                    final_level = ThreatScorer_level(final_score)
                except Exception:
                    final_score = rule_score
                    final_level = rule_level
            else:
                final_score = rule_score
                final_level = rule_level

            # Track'e yaz
            track.threat_score = final_score
            track.threat_level = final_level
            track.zone_id = zone_result.get("zone_id")

            per_track[tid] = {
                "zone":           zone_result,
                "loitering":      loiter_result,
                "abandoned_score": ab_score,
                "feature_vector": fv.tolist(),
                "threat_score":   final_score,
                "threat_level":   final_level,
            }

            # Alarm kontrolleri
            if zone_result.get("violation") and zone_result.get("alert_msg"):
                self._add_alert(new_alerts, Alert(
                    alert_type="zone_violation",
                    track_id=tid,
                    message=zone_result["alert_msg"],
                    threat_level="HIGH",
                    score=zone_result.get("violation_score", 0.8)
                ))

            if loiter_result.get("loitering") and loiter_result.get("alert_msg"):
                self._add_alert(new_alerts, Alert(
                    alert_type="loitering",
                    track_id=tid,
                    message=loiter_result["alert_msg"],
                    threat_level="MEDIUM",
                    score=loiter_result.get("loitering_score", 0.5)
                ))

        # Loitering: aktif olmayan track'leri temizle
        active_ids = [t.track_id for t in active_tracks]
        self.loitering.update_active(active_ids)

        # Alert geçmişine ekle
        self.alert_history.extend(new_alerts)
        self.alert_history = self.alert_history[-100:]  # Son 100

        return new_alerts, per_track

    def _add_alert(self, alert_list: List[Alert], alert: Alert):
        """Cooldown kontrolü yaparak alert ekle."""
        tid = alert.track_id
        now = time.time()
        last = self._last_alert_time.get(tid, 0)
        if now - last >= self.alert_cooldown:
            alert_list.append(alert)
            self._last_alert_time[tid] = now

    def draw_overlays(self, frame, track_history: TrackHistory, per_track: Dict):
        """Tüm bölge ve tehdit overlay'lerini frame üzerine çiz."""
        import cv2
        from src.behavior.threat_scorer import ThreatScorer

        # Bölgeleri çiz
        if self.zone_detector:
            frame = self.zone_detector.draw_zones(frame)

        # Track overlay
        for track in track_history.get_all_active():
            tid = track.track_id
            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            info = per_track.get(tid, {})
            level = info.get("threat_level", "LOW")
            score = info.get("threat_score", 0.0)

            color = ThreatScorer.level_color(level)

            # BBox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Trajektori
            pts = list(track.trajectory)
            for i in range(1, len(pts)):
                p1 = (int(pts[i-1][0]), int(pts[i-1][1]))
                p2 = (int(pts[i][0]), int(pts[i][1]))
                cv2.line(frame, p1, p2, color, 1)

            # Label
            label = f"#{tid} {track.class_name} [{level}] {score:.2f}"
            lw, lh = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            cv2.rectangle(frame, (x1, y1 - 18), (x1 + lw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def get_recent_alerts(self, n: int = 10) -> List[Dict]:
        return [a.to_dict() for a in self.alert_history[-n:]]


def ThreatScorer_level(score: float) -> str:
    """Skor → Seviye (engine içi yardımcı)."""
    from src.behavior.threat_scorer import THREAT_THRESHOLDS
    for level, (lo, hi) in THREAT_THRESHOLDS.items():
        if lo <= score < hi:
            return level
    return "LOW"
