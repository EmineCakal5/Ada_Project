"""
Zone Violation — Bölge ihlali tespiti.
Poligon noktası-içi testi ile nesne pozisyonunu bölgelerle karşılaştırır.
"""

import cv2
import numpy as np
import json
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class Zone:
    """Tek bir güvenlik bölgesi."""

    def __init__(self, zone_data: dict, frame_w: int = 960, frame_h: int = 540):
        self.id = zone_data["id"]
        self.name = zone_data["name"]
        self.zone_type = zone_data.get("type", "monitor")  # restricted / monitor / safe
        self.color = tuple(zone_data.get("color", [255, 0, 0]))
        self.alpha = zone_data.get("alpha", 0.3)
        self.alert_on_entry = zone_data.get("alert_on_entry", False)
        self.description = zone_data.get("description", "")

        # Koordinatları hedef çözünürlüğe ölçekle
        ref_w = zone_data.get("ref_w", 960)
        ref_h = zone_data.get("ref_h", 540)
        raw_pts = np.array(zone_data["points"], dtype=np.float32)
        scale_x = frame_w / ref_w
        scale_y = frame_h / ref_h
        scaled = (raw_pts * [scale_x, scale_y]).astype(np.float32)

        # Dinamik bolgeler icin iki kopyayi da tutuyoruz:
        #   * self.points       -> ekranda cizim ve pointPolygonTest icin
        #                          kullanilan GUNCEL (kamera hareketi ile
        #                          her frame warp edilen) poligon.
        #   * self._points_f    -> ayni verinin float hali; hassas warp zinciri
        #                          boyunca yuvarlama hatalarini onler.
        self._points_f: np.ndarray = scaled.copy()
        self.points: np.ndarray = scaled.astype(np.int32)

    # ------------------------------------------------------------------
    # Dinamik bolge (GMC) — drone / IHA kamerasi icin
    # ------------------------------------------------------------------
    def update_polygon(self, transform_matrix: np.ndarray) -> None:
        """
        Kamera hareketi 2x3 affine matrisi ile bolge noktalarini warp eder.
        Boylece bolge ekran koordinatinda kayar ama yeryuzunde (arka planda)
        sabit kalmis gibi gorunur.

        Args:
            transform_matrix: 2x3 float32 affine (EgoMotionCompensator ciktisi).
                              None veya identity verilirse no-op.
        """
        if transform_matrix is None:
            return
        H = np.asarray(transform_matrix, dtype=np.float32)
        if H.shape != (2, 3):
            return

        # cv2.transform (N, 1, 2) sekline ihtiyaç duyar
        pts_in = self._points_f.reshape(-1, 1, 2)
        pts_out = cv2.transform(pts_in, H)
        self._points_f = pts_out.reshape(-1, 2).astype(np.float32)
        self.points = self._points_f.astype(np.int32)

    def contains_point(self, point: Tuple[float, float]) -> bool:
        """Nokta bu bölge içinde mi? (cv2.pointPolygonTest)"""
        pt = (float(point[0]), float(point[1]))
        result = cv2.pointPolygonTest(self.points, pt, measureDist=False)
        return result >= 0

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Bolgeyi frame uzerine seffaf olarak ciz + dinamik (ground-lock)
        capa isaretleri (HUD hissiyati)."""
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.points], self.color)
        cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0, frame)

        # Sinir cizgisi
        border_color = tuple(max(0, c - 50) for c in self.color)
        cv2.polylines(frame, [self.points], isClosed=True,
                      color=border_color, thickness=2)

        # --- Dinamik kose capa (+) isaretleri: zone'in yeryuzune kilitli
        # oldugunu gorsel olarak belli eder. Kamera kaydikca bu '+' larin
        # sahnedeki ayni piksel referansinda kaldigi izlenir.
        anchor_color = (255, 220, 0)   # #00dcff neon mavi (BGR)
        arm = 5
        for pt in self.points:
            x, y = int(pt[0]), int(pt[1])
            cv2.line(frame, (x - arm, y), (x + arm, y), anchor_color, 1, cv2.LINE_AA)
            cv2.line(frame, (x, y - arm), (x, y + arm), anchor_color, 1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 2, anchor_color, -1, cv2.LINE_AA)

        # --- Merkez capa: kucuk hedef halkasi + isim etiketi
        centroid = self.points.mean(axis=0).astype(int)
        cx, cy = int(centroid[0]), int(centroid[1])
        cv2.circle(frame, (cx, cy), 6, anchor_color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx - 9, cy), (cx - 3, cy), anchor_color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx + 3, cy), (cx + 9, cy), anchor_color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - 9), (cx, cy - 3), anchor_color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy + 3), (cx, cy + 9), anchor_color, 1, cv2.LINE_AA)

        # Isim (uppercase, teknik)
        label = f"◈ {self.name.upper()}"
        cv2.putText(frame, label, (cx + 12, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return frame


class ZoneViolationDetector:
    """
    Bölge ihlali tespiti.
    Nesnelerin hangi bölgede olduğunu ve ihlal yapıp yapmadığını tespit eder.

    Kullanım:
        detector = ZoneViolationDetector(zones_path, frame_w, frame_h)
        alerts = detector.check(track_id, center, class_name)
    """

    ZONE_TYPES = {"restricted": 1.0, "monitor": 0.4, "safe": 0.0}

    def __init__(self, zones_path: str, frame_w: int = 960, frame_h: int = 540):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.zones: List[Zone] = []
        self._load_zones(zones_path)
        logger.info(f"ZoneViolationDetector: {len(self.zones)} bölge yüklendi")

    def _load_zones(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            ref_w = data.get("frame_width", 960)
            ref_h = data.get("frame_height", 540)
            for zd in data.get("zones", []):
                zd["ref_w"] = ref_w
                zd["ref_h"] = ref_h
                self.zones.append(Zone(zd, self.frame_w, self.frame_h))
        except Exception as e:
            logger.error(f"Zone dosyası yüklenemedi: {e}")

    def get_zone(self, center: Tuple[float, float]) -> Optional[Zone]:
        """Bir noktanın içinde olduğu ilk bölgeyi döner."""
        for zone in self.zones:
            if zone.contains_point(center):
                return zone
        return None

    def check(self, track_id: int, center: Tuple[float, float],
              class_name: str = "person") -> Dict:
        """
        Bir track'in bölge durumunu kontrol et.

        Returns:
            {
                "zone_id": "zone_1" or None,
                "zone_name": ...,
                "zone_type": ...,
                "violation": True/False,
                "violation_score": 0.0-1.0,
                "alert_msg": "..." or None
            }
        """
        zone = self.get_zone(center)

        result = {
            "zone_id": None,
            "zone_name": None,
            "zone_type": None,
            "violation": False,
            "violation_score": 0.0,
            "alert_msg": None
        }

        if zone is None:
            return result

        result["zone_id"] = zone.id
        result["zone_name"] = zone.name
        result["zone_type"] = zone.zone_type
        score = self.ZONE_TYPES.get(zone.zone_type, 0.0)
        result["violation_score"] = score

        if zone.zone_type == "restricted" and zone.alert_on_entry:
            result["violation"] = True
            result["alert_msg"] = (
                f"⚠️ BÖLGE İHLALİ: {class_name} #{track_id} "
                f"yasak bölgeye girdi [{zone.name}]"
            )

        return result

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Tüm bölgeleri frame üzerine çiz."""
        for zone in self.zones:
            frame = zone.draw(frame)
        return frame

    def get_all_zones(self) -> List[Zone]:
        return self.zones

    def apply_camera_motion(self, transform_matrix: np.ndarray) -> None:
        """
        Kamera hareket matrisi geldiginde tum zone poligonlarini dinamik
        olarak yeryuzune sabitlenmis gibi kaydir. Drone / IHA senaryosunda
        BehaviorEngine tarafindan cagrilir.
        """
        if transform_matrix is None:
            return
        for zone in self.zones:
            zone.update_polygon(transform_matrix)
