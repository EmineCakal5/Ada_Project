"""
Track History — Takip edilen nesnelerin trajektori ve süre bilgisi.
Her track_id için pozisyon geçmişi, dwell time ve hız hesaplar.
"""

import numpy as np
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional


@dataclass
class TrackInfo:
    """Tek bir track'in durum bilgisi."""
    track_id: int
    class_id: int
    class_name: str
    bbox: List[float]                        # Güncel [x1, y1, x2, y2]
    center: Tuple[float, float]              # Güncel merkez
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    trajectory: deque = field(default_factory=lambda: deque(maxlen=60))
    confidence: float = 1.0
    is_active: bool = True

    # Davranış analizi için
    zone_id: Optional[str] = None           # Şu an hangi bölgede
    threat_score: float = 0.0
    threat_level: str = "LOW"               # LOW / MEDIUM / HIGH / CRITICAL
    alerts: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.trajectory.append(self.center)

    @property
    def dwell_time(self) -> float:
        """Kaç saniyedir takip ediliyor."""
        return self.last_seen - self.first_seen

    @property
    def velocity(self) -> float:
        """Piksel/saniye cinsinden ortalama hız."""
        if len(self.trajectory) < 2:
            return 0.0
        pts = list(self.trajectory)
        if len(pts) < 2:
            return 0.0
        recent = pts[-min(10, len(pts)):]
        total_dist = sum(
            np.linalg.norm(np.array(recent[i+1]) - np.array(recent[i]))
            for i in range(len(recent)-1)
        )
        return total_dist / max(len(recent) - 1, 1)

    @property
    def trajectory_variance(self) -> float:
        """Hareket düzensizliği: yüksek = düzensiz hareket."""
        if len(self.trajectory) < 5:
            return 0.0
        pts = np.array(list(self.trajectory))
        return float(np.var(pts[:, 0]) + np.var(pts[:, 1]))

    @property
    def total_displacement(self) -> float:
        """İlk ve son nokta arasındaki toplam mesafe."""
        if len(self.trajectory) < 2:
            return 0.0
        pts = list(self.trajectory)
        return float(np.linalg.norm(np.array(pts[-1]) - np.array(pts[0])))

    def update(self, bbox: List[float], center: Tuple[float, float],
               confidence: float = 1.0, class_id: int = -1, class_name: str = ""):
        """Track bilgisini güncelle."""
        self.bbox = bbox
        self.center = center
        self.last_seen = time.time()
        self.trajectory.append(center)
        self.confidence = confidence
        self.is_active = True
        if class_id >= 0:
            self.class_id = class_id
        if class_name:
            self.class_name = class_name

    def add_alert(self, alert_type: str):
        """Alert ekle, tekrar ekleme."""
        if alert_type not in self.alerts:
            self.alerts.append(alert_type)

    # ------------------------------------------------------------------
    # GMC (drone/IHA) — kamera hareketi varken geçmiş noktaları warp et
    # ------------------------------------------------------------------
    def apply_camera_motion(self, H: np.ndarray) -> None:
        """
        Kamera ego-motion matrisi ile trajectory, center ve bbox'i yeni
        frame koordinat sistemine tasi. Bu sayede loitering / velocity /
        displacement gibi davranissal metrikler kamera kaymasindan
        etkilenmez (yeryuzune sabitlenmis gibi).

        Args:
            H: 2x3 float32 affine matris.
        """
        if H is None:
            return
        import cv2
        Hf = np.asarray(H, dtype=np.float32)
        if Hf.shape != (2, 3):
            return

        # ---- Trajectory geçmişi ----
        if len(self.trajectory) > 0:
            pts = np.array(self.trajectory, dtype=np.float32).reshape(-1, 1, 2)
            warped = cv2.transform(pts, Hf).reshape(-1, 2)
            # deque'un maxlen'ini koruyarak yeniden doldur
            self.trajectory.clear()
            for p in warped:
                self.trajectory.append((float(p[0]), float(p[1])))

        # ---- Merkez (son pozisyon) ----
        cx, cy = self.center
        new_cx = Hf[0, 0] * cx + Hf[0, 1] * cy + Hf[0, 2]
        new_cy = Hf[1, 0] * cx + Hf[1, 1] * cy + Hf[1, 2]
        self.center = (float(new_cx), float(new_cy))

        # ---- Bbox köşeleri ----
        x1, y1, x2, y2 = self.bbox
        corners = np.array([[x1, y1], [x2, y2]], dtype=np.float32).reshape(-1, 1, 2)
        warped = cv2.transform(corners, Hf).reshape(-1, 2)
        self.bbox = [
            float(warped[0, 0]), float(warped[0, 1]),
            float(warped[1, 0]), float(warped[1, 1]),
        ]

    def to_dict(self) -> dict:
        """JSON serializable dict döner."""
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "bbox": self.bbox,
            "center": list(self.center),
            "dwell_time": round(self.dwell_time, 2),
            "velocity": round(self.velocity, 2),
            "threat_score": round(self.threat_score, 3),
            "threat_level": self.threat_level,
            "zone_id": self.zone_id,
            "alerts": self.alerts,
        }


class TrackHistory:
    """
    Tüm aktif ve geçmiş track'leri yöneten sınıf.

    Kullanım:
        history = TrackHistory()
        history.update(tracks)  # tracker çıktısı
        track = history.get(track_id)
    """

    def __init__(self, max_lost_frames: int = 30):
        self.tracks: Dict[int, TrackInfo] = {}       # Aktif track'ler
        self.lost_tracks: Dict[int, TrackInfo] = {}  # Kaybedilen track'ler
        self.max_lost_frames: int = max_lost_frames
        self._lost_counters: Dict[int, int] = {}

    def update(self, track_id: int, bbox: List[float], center: Tuple[float, float],
               class_id: int = 0, class_name: str = "person",
               confidence: float = 1.0) -> TrackInfo:
        """Bir track'i oluştur veya güncelle."""
        if track_id in self.tracks:
            self.tracks[track_id].update(bbox, center, confidence, class_id, class_name)
        else:
            # Kayıp listesinden geri al veya yeni oluştur
            if track_id in self.lost_tracks:
                track = self.lost_tracks.pop(track_id)
                track.update(bbox, center, confidence, class_id, class_name)
                self.tracks[track_id] = track
            else:
                self.tracks[track_id] = TrackInfo(
                    track_id=track_id,
                    class_id=class_id,
                    class_name=class_name,
                    bbox=bbox,
                    center=center,
                    confidence=confidence
                )
            if track_id in self._lost_counters:
                del self._lost_counters[track_id]

        return self.tracks[track_id]

    def mark_missing(self, active_ids: List[int]):
        """
        Aktif listede olmayan track'leri kayıp olarak işaretle.
        Çok uzun süre kayıp olanları temizle.
        """
        current_ids = set(self.tracks.keys())
        missing = current_ids - set(active_ids)

        for tid in missing:
            self._lost_counters[tid] = self._lost_counters.get(tid, 0) + 1
            if self._lost_counters[tid] >= self.max_lost_frames:
                self.lost_tracks[tid] = self.tracks.pop(tid)
                del self._lost_counters[tid]

    def get(self, track_id: int) -> Optional[TrackInfo]:
        return self.tracks.get(track_id)

    def get_all_active(self) -> List[TrackInfo]:
        return list(self.tracks.values())

    def get_by_class(self, class_name: str) -> List[TrackInfo]:
        return [t for t in self.tracks.values() if t.class_name == class_name]

    def count_active(self) -> int:
        return len(self.tracks)

    def clear(self):
        self.tracks.clear()
        self.lost_tracks.clear()
        self._lost_counters.clear()

    # ------------------------------------------------------------------
    # GMC (drone/IHA) — tum aktif track geçmişlerine warp uygula
    # ------------------------------------------------------------------
    def apply_camera_motion(self, H: np.ndarray) -> None:
        """
        Kamera hareketi varken hem aktif hem kayip track'lerin trajectory
        ve anchor verilerini tek noktadan warp eder. Pipeline tracker
        update'inden ONCE cagirmalidir.
        """
        if H is None:
            return
        for t in self.tracks.values():
            t.apply_camera_motion(H)
        # Kayıp track'ler de aynı uzayda tutulmalı ki geri yakalandığında
        # trajectory'si tutarlı olsun:
        for t in self.lost_tracks.values():
            t.apply_camera_motion(H)
