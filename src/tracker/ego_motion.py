# -*- coding: utf-8 -*-
"""
Ego-Motion Compensator — IHA / Drone Kamerasi icin Global Motion Compensation (GMC)
===================================================================================

Ada_Project -> Drone Guvenlik Modulu

Amac:
    Kamera hareketli oldugu icin ekrandaki her piksel, nesne gercekten hareket
    etmese bile yer degistirir. Bu sinif, ardisik iki frame arasindaki kamera
    hareketini (oteleme + donme + olcek) bir 2x3 Affine matrisi olarak tahmin
    eder. Bu matris daha sonra:
        * ByteTrack Kalman state'lerini,
        * TrackHistory trajectory geçmişini,
        * Loitering / AbandonedObject anchor'larini,
        * Zone poligonlarini
    'yeryuzune sabitlenmis' gibi ters kaydirmak icin kullanilir.

Algoritma:
    1) Frame downscale edilir (performans icin, Jetson'da kritiktir).
    2) goodFeaturesToTrack ile grid-benzeri (her hucreden ayri) korner bul.
    3) calcOpticalFlowPyrLK ile bu kornerleri yeni frame'e tasi.
    4) Geri-dogru flow kontrolu (forward-backward error) ile outlier temizle.
    5) estimateAffinePartial2D + RANSAC ile 2x3 affine cikar.
    6) Downscale nedeniyle olcek duzeltmesi uygulanir (translation / s).

Notlar:
    * `estimateAffinePartial2D`: 4 DoF (tx, ty, scale, rotation). Drone'da
      shear olmadigi icin `Affine2D`'den daha stabildir ve RANSAC dostudur.
    * Ilk frame'de identity matris doner (henuz ref frame yok).
    * Yetersiz feature / eslesme -> identity + uyari logu.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# Identity affine (2x3) — dışa da kullanışlı
IDENTITY_AFFINE: np.ndarray = np.array(
    [[1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0]],
    dtype=np.float32,
)


def warp_points(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    2x3 affine ile nokta dizisini warp eder.
    `points`: shape (N, 2) veya (N, 1, 2), float kabul eder.
    Returns: ayni sekil, warp edilmis koordinatlar (float32).
    """
    if points is None or len(points) == 0:
        return points
    pts = np.asarray(points, dtype=np.float32)
    orig_shape = pts.shape
    if pts.ndim == 2:
        pts = pts.reshape(-1, 1, 2)
    out = cv2.transform(pts, H.astype(np.float32))
    return out.reshape(orig_shape)


def decompose_affine(H: np.ndarray) -> Tuple[float, float]:
    """
    2x3 affine'den (uniform_scale, rotation_rad) cikarir.
    Kalman state'teki w, h ve velocity vektörlerine uygulamak icin gerekli.
    """
    a, b = float(H[0, 0]), float(H[0, 1])
    c, d = float(H[1, 0]), float(H[1, 1])
    sx = np.sqrt(a * a + c * c)
    sy = np.sqrt(b * b + d * d)
    uniform_scale = float((sx + sy) * 0.5)
    rotation = float(np.arctan2(c, a))
    return uniform_scale, rotation


class EgoMotionCompensator:
    """
    Ardisik iki frame arasindaki kamera hareketini tahmin eder.

    Kullanim:
        ego = EgoMotionCompensator(downscale=0.5)
        H = ego.estimate(frame_bgr)     # ilk frame icin identity doner
        # H (2x3): pt_new ~= H @ [pt_old_x, pt_old_y, 1]

    Jetson / Edge optimizasyonu:
        * downscale=0.5 default (genelde 2-3x hizlandirir).
        * max_features degistirilerek dogruluk-hiz trade-off ayarlanabilir.
        * grid parametresi: feature'larin sahnede homojen dagilmasini saglar;
          tek bir bolgede yogunlasip genel hareketi yansitmamasini engeller.
    """

    def __init__(
        self,
        downscale: float = 0.5,
        max_features: int = 500,
        grid: Tuple[int, int] = (4, 4),
        quality_level: float = 0.01,
        min_distance: int = 8,
        ransac_thresh: float = 3.0,
        fb_error_thresh: float = 1.5,
    ):
        """
        Args:
            downscale: 0 < s <= 1. 0.5 -> goruntu yariya inip hizlanir.
            max_features: sahneden toplam kac kose alinacak.
            grid: (cols, rows). Sahneyi gride bolup her hucreden dengeli
                  feature cikarir; esit dagilim -> daha saglam affine.
            quality_level: goodFeaturesToTrack kalite esigi.
            min_distance: kornerler arasi min piksel mesafe.
            ransac_thresh: estimateAffinePartial2D RANSAC piksel esigi.
            fb_error_thresh: forward-backward optical flow hata toleransi.
        """
        if not (0.0 < downscale <= 1.0):
            raise ValueError("downscale 0 ile 1 arasinda olmali")

        self.downscale = float(downscale)
        self.max_features = int(max_features)
        self.grid = grid
        self.quality_level = float(quality_level)
        self.min_distance = int(min_distance)
        self.ransac_thresh = float(ransac_thresh)
        self.fb_error_thresh = float(fb_error_thresh)

        # LK parametreleri
        self._lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        self._prev_gray: Optional[np.ndarray] = None
        self._last_H: np.ndarray = IDENTITY_AFFINE.copy()
        self._frame_idx: int = 0

        logger.info(
            "EgoMotionCompensator hazir | downscale=%.2f, max_feat=%d, grid=%s",
            self.downscale, self.max_features, grid,
        )

    # ----- public API ------------------------------------------------------

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Yeni gelen frame icin kamera hareketini tahmin et.

        Returns:
            H (2x3, float32): pt_new = H @ [pt_old; 1]
            - Ilk frame'de / hesaplanamadigi durumlarda identity doner.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return IDENTITY_AFFINE.copy()

        gray_ds = self._prepare_gray(frame_bgr)

        if self._prev_gray is None:
            self._prev_gray = gray_ds
            self._frame_idx += 1
            return IDENTITY_AFFINE.copy()

        try:
            H = self._compute_affine(self._prev_gray, gray_ds)
        except cv2.error as e:  # pragma: no cover — OpenCV corner case
            logger.debug("Ego-motion cv2 hata: %s", e)
            H = IDENTITY_AFFINE.copy()

        # Downscale telafisi: translation tam boy uzayinda olmali.
        if self.downscale != 1.0:
            H = H.copy()
            H[:, 2] /= self.downscale  # t_full = t_ds / s

        self._prev_gray = gray_ds
        self._last_H = H.astype(np.float32)
        self._frame_idx += 1
        return self._last_H.copy()

    def reset(self) -> None:
        """Sahne kesildiginde / yeni video basladiginda cagir."""
        self._prev_gray = None
        self._last_H = IDENTITY_AFFINE.copy()
        self._frame_idx = 0

    @property
    def last_transform(self) -> np.ndarray:
        return self._last_H.copy()

    # ----- internals -------------------------------------------------------

    def _prepare_gray(self, frame_bgr: np.ndarray) -> np.ndarray:
        """BGR -> gri + downscale. Downscale fx/fy OpenCV'nin en hizli yolu."""
        if self.downscale != 1.0:
            frame_bgr = cv2.resize(
                frame_bgr, None,
                fx=self.downscale, fy=self.downscale,
                interpolation=cv2.INTER_AREA,
            )
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    def _grid_features(self, gray: np.ndarray) -> np.ndarray:
        """
        Sahneyi grid hucrelere bolup her hucreden esit sayida korner cikarir.
        Homojen dagilim, tek bir hareketli nesnenin estimate'i bozmasini engeller.
        """
        h, w = gray.shape[:2]
        cols, rows = self.grid
        per_cell = max(4, self.max_features // (cols * rows))

        all_pts = []
        for r in range(rows):
            for c in range(cols):
                x0 = int(c * w / cols)
                x1 = int((c + 1) * w / cols)
                y0 = int(r * h / rows)
                y1 = int((r + 1) * h / rows)
                cell = gray[y0:y1, x0:x1]
                if cell.size == 0:
                    continue
                pts = cv2.goodFeaturesToTrack(
                    cell,
                    maxCorners=per_cell,
                    qualityLevel=self.quality_level,
                    minDistance=self.min_distance,
                    blockSize=3,
                )
                if pts is None:
                    continue
                pts = pts.reshape(-1, 2)
                pts[:, 0] += x0
                pts[:, 1] += y0
                all_pts.append(pts)

        if not all_pts:
            return np.empty((0, 1, 2), dtype=np.float32)
        merged = np.concatenate(all_pts, axis=0).astype(np.float32)
        return merged.reshape(-1, 1, 2)

    def _compute_affine(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
        """Sparse LK + RANSAC affine fit, forward-backward filtreli."""
        pts0 = self._grid_features(prev_gray)
        if pts0.shape[0] < 8:
            logger.debug("Yetersiz feature (%d) - identity donuluyor", pts0.shape[0])
            return IDENTITY_AFFINE.copy()

        # Forward flow
        pts1, status1, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, pts0, None, **self._lk_params
        )
        # Backward flow — outlier temizligi icin
        pts0_bw, status2, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, prev_gray, pts1, None, **self._lk_params
        )

        if pts1 is None or pts0_bw is None:
            return IDENTITY_AFFINE.copy()

        fb_err = np.linalg.norm(pts0.reshape(-1, 2) - pts0_bw.reshape(-1, 2), axis=1)
        good = (
            (status1.ravel() == 1)
            & (status2.ravel() == 1)
            & (fb_err < self.fb_error_thresh)
        )

        src = pts0.reshape(-1, 2)[good]
        dst = pts1.reshape(-1, 2)[good]

        if len(src) < 8:
            logger.debug("FB filtresi sonrasi yetersiz nokta (%d)", len(src))
            return IDENTITY_AFFINE.copy()

        H, inliers = cv2.estimateAffinePartial2D(
            src, dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thresh,
            maxIters=500,
            confidence=0.99,
        )
        if H is None:
            return IDENTITY_AFFINE.copy()
        return H.astype(np.float32)
