# -*- coding: utf-8 -*-
"""
ByteTrack Tracker — Hız öncelikli, sabit kamera için optimize edilmiş tracker.
ByteTrack algoritması: yüksek confidence + düşük confidence detection'ları birleştirir.
Referans: ByteTrack (Zhang et al., 2022)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging

from src.tracker.ego_motion import decompose_affine

logger = logging.getLogger(__name__)

# ─── Kalman Filtresi ──────────────────────────────────────────


class KalmanBoxTracker:
    """
    Tek bir nesneyi Kalman filtresi ile takip eder.
    State: [cx, cy, w, h, vx, vy, vw, vh]
    """
    count = 0

    def __init__(self, bbox: List[float]):
        from filterpy.kalman import KalmanFilter
        self.kf = KalmanFilter(dim_x=8, dim_z=4)

        # Transition matrix
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ], dtype=float)

        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ], dtype=float)

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = self._xyxy_to_cxcywh(bbox).reshape(4, 1)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.last_class_id = 0
        self.last_confidence = 1.0

    def update(self, bbox: List[float], class_id: int = 0, confidence: float = 1.0):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.last_class_id = class_id
        self.last_confidence = confidence
        self.kf.update(self._xyxy_to_cxcywh(bbox).reshape(4, 1))

    def predict(self) -> np.ndarray:
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self._cxcywh_to_xyxy(self.kf.x[:4].flatten())

    def get_state(self) -> np.ndarray:
        return self._cxcywh_to_xyxy(self.kf.x[:4].flatten())

    # ------------------------------------------------------------------
    # GMC (Global Motion Compensation) — drone / IHA kamerasi icin
    # ------------------------------------------------------------------
    def apply_camera_motion(self, H: np.ndarray) -> None:
        """
        Kamera (ego) hareketi tespit edildiginde Kalman state'ini yeni
        frame'e hizala. Bu, kamera kaydiginda ID Switch olmasini engeller.

        Mantik:
            * (cx, cy) merkezine affine uygulanir.
            * (w, h) uniform scale ile carpilir (kamera zoom / irtifa
              degisimi senaryosu).
            * (vx, vy) hizina rotation+scale (A kismi) uygulanir; translation
              degil, zira translation bir hiz degil, tek sefer oteleme.
            * (vw, vh) boyut hiz terimleri de ayni scale ile olceklenir.

        Args:
            H: 2x3 float32 affine matris (EgoMotionCompensator.estimate).
        """
        if H is None:
            return

        A = H[:, :2].astype(float)           # rotasyon + olcek kismi
        t = H[:, 2].astype(float)            # oteleme kismi
        scale, _ = decompose_affine(H)

        x = self.kf.x.flatten().copy()       # state (8,)

        # Merkez: affine ile warp
        cx_new = A[0, 0] * x[0] + A[0, 1] * x[1] + t[0]
        cy_new = A[1, 0] * x[0] + A[1, 1] * x[1] + t[1]

        # Boyut: uniform scale
        w_new = x[2] * scale
        h_new = x[3] * scale

        # Hiz: sadece rotasyon + scale (oteleme hizi degistirmez)
        vx_new = A[0, 0] * x[4] + A[0, 1] * x[5]
        vy_new = A[1, 0] * x[4] + A[1, 1] * x[5]
        vw_new = x[6] * scale
        vh_new = x[7] * scale

        self.kf.x = np.array(
            [cx_new, cy_new, w_new, h_new, vx_new, vy_new, vw_new, vh_new],
            dtype=float,
        ).reshape(-1, 1)

        # Kovaryans da benzer donusum gerektirir. Tam dogru matematik:
        # P' = M P M^T  (M: 8x8 blok-diagonal [A,A,A,A]).
        # Basit ve stabil yaklasim: P'yi hafifce sisir, kamera hareketine
        # bagli belirsizligi temsil etsin. Jetson'da matris carpim maliyetini
        # de dusurur.
        self.kf.P[:2, :2] *= (scale * scale)     # merkez konum belirsizligi
        self.kf.P[2:4, 2:4] *= (scale * scale)   # boyut belirsizligi
        # Ek kamera-kaynakli gurultu (ortalama 2 piksel std^2)
        self.kf.P[0, 0] += 4.0
        self.kf.P[1, 1] += 4.0

    @staticmethod
    def _xyxy_to_cxcywh(bbox):
        x1, y1, x2, y2 = bbox
        return np.array([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1])

    @staticmethod
    def _cxcywh_to_xyxy(cx):
        cx_, cy_, w, h = cx
        return np.array([cx_-w/2, cy_-h/2, cx_+w/2, cy_+h/2])


# ─── Track Nesnesi ────────────────────────────────────────────


@dataclass
class Track:
    """ByteTrack çıktı track nesnesi."""
    track_id: int
    bbox: List[float]          # [x1, y1, x2, y2]
    class_id: int
    class_name: str
    confidence: float
    center: Tuple[float, float]

    @classmethod
    def from_kalman(cls, tracker: KalmanBoxTracker, class_id: int,
                    class_name: str, names: dict) -> "Track":
        bbox = tracker.get_state().tolist()
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        return cls(
            track_id=tracker.id,
            bbox=bbox,
            class_id=class_id,
            class_name=class_name,
            confidence=tracker.last_confidence,
            center=(cx, cy)
        )


# ─── ByteTrack ────────────────────────────────────────────────


class ByteTracker:
    """
    Basitleştirilmiş ByteTrack implementasyonu.
    Yüksek confidence + düşük confidence detection'ları iki aşamada eşleştirir.

    Kullanım:
        tracker = ByteTracker(config)
        tracks = tracker.update(detections, frame)
    """

    def __init__(self, config: dict):
        cfg = config["tracker"]["bytetrack"]
        self.track_thresh = cfg.get("track_thresh", 0.5)
        self.track_buffer = cfg.get("track_buffer", 30)
        self.match_thresh = cfg.get("match_thresh", 0.8)
        self.frame_rate = cfg.get("frame_rate", 15)
        self.min_hits = int(cfg.get("min_hits", 3))

        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
        self.max_age = self.track_buffer
        KalmanBoxTracker.count = 0

        # Sınıf isimlerini YOLO modelinden alacağız
        self.class_names: dict = {}

        logger.info(
            f"ByteTracker başlatıldı: thresh={self.track_thresh}, buffer={self.track_buffer}, "
            f"min_hits={self.min_hits}"
        )

    def set_class_names(self, names: dict):
        self.class_names = names

    def update(self, detections, frame: np.ndarray = None,
               transform: Optional[np.ndarray] = None) -> List[Track]:
        """
        Detections listesini alir, Track listesi doner.

        Args:
            detections: List[Detection] — yolo_detector ciktisi
            frame: opsiyonel, gelecekteki Re-ID icin
            transform: opsiyonel 2x3 affine matris (EgoMotionCompensator ciktisi).
                       Kamera hareketli ise (drone/IHA), Kalman state'leri
                       predict() oncesi bu matrise gore warp edilir. None ise
                       sabit kamera modu (mevcut davranis) korunur.

        Returns:
            List[Track] — aktif track'ler
        """
        self.frame_count += 1

        # ─── GMC: Kalman state'lerini yeni frame'e hizala ─────
        # Onemli: predict()'ten ONCE uygulanmali ki F*x sonrasi olusan
        # beklenen pozisyon zaten warp edilmis kordinat sisteminde olsun.
        if transform is not None:
            for trk in self.trackers:
                trk.apply_camera_motion(transform)

        # Detection'lari iki gruba ayir
        high_det = [d for d in detections if d.confidence >= self.track_thresh]
        low_det = [d for d in detections if d.confidence < self.track_thresh]

        # Kalman predict
        predicted_boxes = []
        for trk in self.trackers:
            pred = trk.predict()
            predicted_boxes.append(pred)

        # 1. Aşama: Yüksek confidence ile eşleştir
        matched_h, unmatched_trks, unmatched_det_h = self._associate(
            self.trackers, high_det, self.match_thresh
        )

        # Eşleşenleri güncelle
        for trk_idx, det_idx in matched_h:
            det = high_det[det_idx]
            self.trackers[trk_idx].update(det.bbox, det.class_id, det.confidence)

        # 2. Aşama: Kalan tracker'ları düşük confidence ile dene
        remaining_trks_idx = [i for i in unmatched_trks]
        remaining_trks = [self.trackers[i] for i in remaining_trks_idx]

        matched_l, still_unmatched_trks, _ = self._associate(
            remaining_trks, low_det, 0.5
        )

        for trk_idx, det_idx in matched_l:
            real_idx = remaining_trks_idx[trk_idx]
            det = low_det[det_idx]
            self.trackers[real_idx].update(det.bbox, det.class_id, det.confidence)

        # Eşleşmeyen yüksek confidence detection'lardan yeni tracker oluştur
        for det_idx in unmatched_det_h:
            det = high_det[det_idx]
            new_trk = KalmanBoxTracker(det.bbox)
            new_trk.last_class_id = det.class_id
            new_trk.last_confidence = det.confidence
            self.trackers.append(new_trk)

        # Çok uzun süredir güncellenmeyenleri kaldır
        active_tracks = []
        for trk in self.trackers:
            if trk.time_since_update < self.max_age and trk.hits >= self.min_hits:
                bbox = trk.get_state().tolist()
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                cls_id = trk.last_class_id
                cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
                active_tracks.append(Track(
                    track_id=trk.id,
                    bbox=bbox,
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=trk.last_confidence,
                    center=(cx, cy)
                ))

        self.trackers = [t for t in self.trackers if t.time_since_update < self.max_age]

        return active_tracks

    def _associate(self, trackers, detections, threshold):
        """IoU tabanlı Hungarian eşleştirme."""
        if len(trackers) == 0 or len(detections) == 0:
            return [], list(range(len(trackers))), list(range(len(detections)))

        iou_matrix = self._compute_iou_matrix(trackers, detections)

        try:
            import lap
            _, row_ind, col_ind = lap.lapjv(-iou_matrix, extend_cost=True, cost_limit=-threshold)
            matched = [(r, c) for r, c in zip(range(len(row_ind)), row_ind) if c >= 0]
        except ImportError:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            matched = [(r, c) for r, c in zip(row_ind, col_ind) if iou_matrix[r, c] >= threshold]

        matched_trks = {m[0] for m in matched}
        matched_dets = {m[1] for m in matched}

        unmatched_trks = [i for i in range(len(trackers)) if i not in matched_trks]
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]

        return matched, unmatched_trks, unmatched_dets

    def _compute_iou_matrix(self, trackers, detections) -> np.ndarray:
        trk_boxes = [t.get_state() for t in trackers]
        det_boxes = [d.bbox for d in detections]
        iou_mat = np.zeros((len(trk_boxes), len(det_boxes)))
        for i, tb in enumerate(trk_boxes):
            for j, db in enumerate(det_boxes):
                iou_mat[i, j] = self._iou(tb, db)
        return iou_mat

    @staticmethod
    def _iou(box1, box2) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        union = a1 + a2 - inter
        return inter / (union + 1e-6)

    def reset(self):
        self.trackers.clear()
        self.frame_count = 0
        KalmanBoxTracker.count = 0
