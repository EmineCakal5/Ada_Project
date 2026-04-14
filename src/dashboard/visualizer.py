"""
Visualizer — OpenCV frame üzerine profesyonel görselleştirme.
FPS sayacı, track ID'leri, tehdit skoru, overlay panel vb.
"""

import cv2
import numpy as np
import time
import logging
from typing import List, Dict, Optional
from collections import deque

logger = logging.getLogger(__name__)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL  = 0.45
FONT_MEDIUM = 0.6
FONT_LARGE  = 0.8


class FPSCounter:
    """Kayan pencere ile gerçek zamanlı FPS hesaplar."""
    def __init__(self, window: int = 30):
        self._times = deque(maxlen=window)

    def tick(self):
        self._times.append(time.time())

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / max(elapsed, 1e-6)


class Visualizer:
    """
    Frame üzerine tüm bilgileri çizen sınıf.

    Kullanım:
        viz = Visualizer(config)
        frame = viz.render(frame, tracks, alerts, per_track, fps)
    """

    # Tehdit renk paleti (BGR)
    LEVEL_COLORS = {
        "LOW":      (50, 205, 50),
        "MEDIUM":   (0, 165, 255),
        "HIGH":     (0, 60, 255),
        "CRITICAL": (0, 0, 200),
    }

    LEVEL_BG = {
        "LOW":      (20, 80, 20),
        "MEDIUM":   (20, 60, 120),
        "HIGH":     (20, 20, 150),
        "CRITICAL": (10, 10, 100),
    }

    def __init__(self, config: dict):
        self.fps_counter = FPSCounter()
        self.show_fps    = config["dashboard"].get("fps_display", True)
        self.title       = config["dashboard"].get("title", "Security System")

    def render(self, frame: np.ndarray, track_history, alerts: List,
               per_track: Dict, extra_info: dict = None) -> np.ndarray:
        """
        Tam görselleştirme uygula.

        Args:
            frame: BGR frame
            track_history: TrackHistory nesnesi
            alerts: Bu frame'deki yeni alertler
            per_track: {track_id: {threat_level, threat_score, ...}}
            extra_info: {'scenario': 'Senaryo Adı', ...}
        """
        self.fps_counter.tick()
        frame = self._draw_tracks(frame, track_history, per_track)
        frame = self._draw_status_bar(frame, track_history, extra_info)
        frame = self._draw_alert_panel(frame, alerts)
        return frame

    def _draw_tracks(self, frame: np.ndarray, track_history, per_track: Dict) -> np.ndarray:
        """Her aktif track için bbox + trajektori + label çiz."""
        for track in track_history.get_all_active():
            tid   = track.track_id
            info  = per_track.get(tid, {})
            level = info.get("threat_level", "LOW")
            score = info.get("threat_score", 0.0)
            color = self.LEVEL_COLORS.get(level, (100, 100, 100))

            x1, y1, x2, y2 = [int(v) for v in track.bbox]

            # BBox — köşe çizgileri (modern look)
            self._draw_corner_box(frame, x1, y1, x2, y2, color, thickness=2, corner_len=15)

            # Trajektori
            pts = list(track.trajectory)
            for i in range(1, min(len(pts), 25)):
                alpha = i / 25
                pt1 = (int(pts[-i+1][0] if i > 1 else pts[0][0]),
                       int(pts[-i+1][1] if i > 1 else pts[0][1]))
                pt2 = (int(pts[-i][0]), int(pts[-i][1]))
                c = tuple(int(c * alpha) for c in color)
                cv2.line(frame, pt1, pt2, c, 1)

            # Tehdit skoru çubuğu (bbox altında)
            bar_w = x2 - x1
            bar_filled = int(bar_w * score)
            cv2.rectangle(frame, (x1, y2 + 2), (x2, y2 + 6), (50, 50, 50), -1)
            cv2.rectangle(frame, (x1, y2 + 2), (x1 + bar_filled, y2 + 6), color, -1)

            # Label
            label = f"#{tid} {track.class_name}"
            sub   = f"{level} {score:.2f}"
            lw = max(
                cv2.getTextSize(label, FONT, FONT_SMALL, 1)[0][0],
                cv2.getTextSize(sub,   FONT, FONT_SMALL, 1)[0][0]
            ) + 8

            # Label arka plan
            bg = self.LEVEL_BG.get(level, (30, 30, 30))
            cv2.rectangle(frame, (x1, y1 - 36), (x1 + lw, y1), bg, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 22), FONT, FONT_SMALL, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, sub,   (x1 + 4, y1 - 6),  FONT, FONT_SMALL, color, 1, cv2.LINE_AA)

            # Merkez noktası
            cx, cy = int(track.center[0]), int(track.center[1])
            cv2.circle(frame, (cx, cy), 3, color, -1)

        return frame

    def _draw_corner_box(self, frame, x1, y1, x2, y2, color, thickness=2, corner_len=10):
        """Köşe çizgili modern kutu."""
        # Sol üst
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)
        # Sağ üst
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)
        # Sol alt
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)
        # Sağ alt
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)

    def _draw_status_bar(self, frame: np.ndarray, track_history, extra_info: dict = None) -> np.ndarray:
        """Üst durum çubuğu: başlık, FPS, track sayısı."""
        h, w = frame.shape[:2]
        bar_h = 36

        # Yarı saydam arka plan
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (15, 15, 25), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Başlık
        cv2.putText(frame, "◉ " + self.title, (10, 24),
                    FONT, FONT_MEDIUM, (0, 220, 255), 1, cv2.LINE_AA)

        # FPS
        if self.show_fps:
            fps_text = f"FPS: {self.fps_counter.fps:.1f}"
            cv2.putText(frame, fps_text, (w - 120, 24),
                        FONT, FONT_MEDIUM, (100, 255, 100), 1, cv2.LINE_AA)

        # Track sayısı
        n_tracks = track_history.count_active()
        track_text = f"Tracks: {n_tracks}"
        cv2.putText(frame, track_text, (w - 240, 24),
                    FONT, FONT_MEDIUM, (200, 200, 200), 1, cv2.LINE_AA)

        # Senaryo modu
        if extra_info and extra_info.get("scenario"):
            scn = f"▶ {extra_info['scenario']}"
            cv2.putText(frame, scn, (w // 2 - 100, 24),
                        FONT, FONT_MEDIUM, (0, 255, 200), 1, cv2.LINE_AA)

        return frame

    def _draw_alert_panel(self, frame: np.ndarray, alerts: List) -> np.ndarray:
        """Sağ alt köşeye anlık alertleri çiz."""
        if not alerts:
            return frame

        h, w = frame.shape[:2]
        panel_w = 380
        row_h   = 22
        padding = 8
        panel_h = len(alerts) * row_h + padding * 2

        px1 = w - panel_w - 10
        py1 = h - panel_h - 10
        px2 = w - 10
        py2 = h - 10

        # Arka plan
        overlay = frame.copy()
        cv2.rectangle(overlay, (px1, py1), (px2, py2), (15, 15, 30), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (px1, py1), (px2, py2), (80, 80, 120), 1)

        for i, alert in enumerate(alerts[-6:]):
            msg = alert.message if hasattr(alert, "message") else str(alert)
            color = self.LEVEL_COLORS.get(
                alert.threat_level if hasattr(alert, "threat_level") else "LOW",
                (200, 200, 200)
            )
            y = py1 + padding + (i + 1) * row_h - 4
            cv2.putText(frame, msg[:52], (px1 + 6, y),
                        FONT, 0.4, color, 1, cv2.LINE_AA)

        return frame
