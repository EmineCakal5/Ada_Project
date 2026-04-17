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

# SENTINEL HUD paleti (BGR) — havacilik / uzay teknik konsol hissiyati
HUD_NEON    = (255, 220,   0)  # #00dcff neon mavi
HUD_CYBER   = (165, 255,   0)  # #00ffa5 siber yesil
HUD_AMBER   = (  0, 170, 255)  # #ffaa00 uyari ambarli
HUD_DIM     = (180, 180, 180)
HUD_LINE    = (120, 120, 140)


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
        # HUD katmani: crosshair + ust telemetri + sag alt GMC kutusu
        frame = self._draw_crosshair(frame)
        frame = self._draw_status_bar(frame, track_history, extra_info)
        frame = self._draw_gmc_hud(frame, extra_info)
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

    # ------------------------------------------------------------------
    # SENTINEL HUD — Ust telemetri seridi (uzay/havacilik hissiyati)
    # ------------------------------------------------------------------
    def _draw_status_bar(self, frame: np.ndarray, track_history,
                         extra_info: dict = None) -> np.ndarray:
        """Ust HUD serit: baslik, FPS, track sayisi, GMC gostergesi, senaryo."""
        h, w = frame.shape[:2]
        bar_h = 40

        # Yari saydam siyah + alt cizgi (neon mavi tiktrack)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (10, 12, 18), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        cv2.line(frame, (0, bar_h), (w, bar_h), HUD_NEON, 1)

        # Sol: baslik + "SENTINEL" mimikeri
        cv2.putText(frame, "[ SENTINEL ] " + self.title, (10, 26),
                    FONT, FONT_MEDIUM, HUD_NEON, 1, cv2.LINE_AA)

        # Orta: senaryo etiketi (varsa)
        if extra_info and extra_info.get("scenario"):
            scn = f"> MISSION: {extra_info['scenario']}"
            cv2.putText(frame, scn, (w // 2 - 120, 26),
                        FONT, FONT_MEDIUM, HUD_CYBER, 1, cv2.LINE_AA)

        # Sag: GMC durum rozeti + tracks + FPS
        x = w - 10
        if self.show_fps:
            fps_txt = f"FPS {self.fps_counter.fps:4.1f}"
            (tw, _), _ = cv2.getTextSize(fps_txt, FONT, FONT_MEDIUM, 1)
            x -= tw
            cv2.putText(frame, fps_txt, (x, 26),
                        FONT, FONT_MEDIUM, HUD_CYBER, 1, cv2.LINE_AA)
            x -= 18

        n_tracks = track_history.count_active()
        trk_txt = f"TRK {n_tracks:02d}"
        (tw, _), _ = cv2.getTextSize(trk_txt, FONT, FONT_MEDIUM, 1)
        x -= tw
        cv2.putText(frame, trk_txt, (x, 26),
                    FONT, FONT_MEDIUM, HUD_DIM, 1, cv2.LINE_AA)
        x -= 18

        # GMC rozeti
        gmc_active = bool(extra_info and extra_info.get("gmc_active"))
        badge = "GMC ACTIVE" if gmc_active else "GMC OFF"
        badge_color = HUD_NEON if gmc_active else HUD_LINE
        (tw, th), _ = cv2.getTextSize(badge, FONT, FONT_SMALL, 1)
        x -= tw + 12
        cv2.rectangle(frame, (x - 6, 10), (x + tw + 6, 30), (20, 24, 32), -1)
        cv2.rectangle(frame, (x - 6, 10), (x + tw + 6, 30), badge_color, 1)
        cv2.putText(frame, badge, (x, 24),
                    FONT, FONT_SMALL, badge_color, 1, cv2.LINE_AA)
        # Canli pulse noktasi
        if gmc_active:
            pulse_r = 3 if int(time.time() * 2) % 2 == 0 else 4
            cv2.circle(frame, (x - 14, 20), pulse_r, HUD_CYBER, -1)

        return frame

    # ------------------------------------------------------------------
    # HUD — Ekran merkezi nisan (crosshair)
    # ------------------------------------------------------------------
    def _draw_crosshair(self, frame: np.ndarray) -> np.ndarray:
        """Ekran merkezine ince bir + ve aci referansi cizer (HUD hissiyati)."""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        gap, length = 8, 14

        # Artı kollar
        cv2.line(frame, (cx - gap - length, cy), (cx - gap, cy), HUD_NEON, 1, cv2.LINE_AA)
        cv2.line(frame, (cx + gap, cy), (cx + gap + length, cy), HUD_NEON, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - gap - length), (cx, cy - gap), HUD_NEON, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy + gap), (cx, cy + gap + length), HUD_NEON, 1, cv2.LINE_AA)
        # Orta nokta
        cv2.circle(frame, (cx, cy), 1, HUD_NEON, -1)
        # Disaridaki koseli kose isaretleri (sinyal kutusu hissi)
        corner = 22
        for sx, sy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            x0 = cx + sx * corner
            y0 = cy + sy * corner
            cv2.line(frame, (x0, y0), (x0 - sx * 6, y0), HUD_LINE, 1, cv2.LINE_AA)
            cv2.line(frame, (x0, y0), (x0, y0 - sy * 6), HUD_LINE, 1, cv2.LINE_AA)
        return frame

    # ------------------------------------------------------------------
    # HUD — Sag alt GMC telemetri kutusu (dx, dy, rot, scale)
    # ------------------------------------------------------------------
    def _draw_gmc_hud(self, frame: np.ndarray, extra_info: dict = None) -> np.ndarray:
        """
        Sag alt kosede GMC (Global Motion Compensation) telemetri paneli.
        `extra_info` icinden beklenen anahtarlar (hepsi opsiyonel):
            - gmc_active: bool
            - ego_dx, ego_dy: float piksel (anlik)
            - ego_rot_deg: float derece
            - ego_scale: float (1.0 = sabit irtifa)
        """
        if not extra_info:
            return frame
        if not extra_info.get("gmc_active"):
            return frame

        dx = float(extra_info.get("ego_dx", 0.0))
        dy = float(extra_info.get("ego_dy", 0.0))
        rot = float(extra_info.get("ego_rot_deg", 0.0))
        scl = float(extra_info.get("ego_scale", 1.0))

        h, w = frame.shape[:2]
        pw, ph = 200, 88
        x1, y1 = 10, h - ph - 10
        x2, y2 = x1 + pw, y1 + ph

        # Panel arka plani (neon cerceve + yari saydam)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (8, 12, 20), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), HUD_NEON, 1)
        # Sol dikey vurgu seridi
        cv2.rectangle(frame, (x1, y1), (x1 + 3, y2), HUD_NEON, -1)

        # Baslik
        cv2.putText(frame, "EGO-MOTION / GMC", (x1 + 10, y1 + 16),
                    FONT, 0.42, HUD_NEON, 1, cv2.LINE_AA)
        cv2.line(frame, (x1 + 10, y1 + 22), (x2 - 10, y1 + 22), HUD_LINE, 1)

        # Veri satirlari (monospace hissi icin sabit genislikli font simulasyonu)
        row = y1 + 38
        cv2.putText(frame, f"dx  {dx:+6.2f} px", (x1 + 10, row),
                    FONT, 0.45, HUD_CYBER, 1, cv2.LINE_AA)
        cv2.putText(frame, f"dy  {dy:+6.2f} px", (x1 + 10, row + 16),
                    FONT, 0.45, HUD_CYBER, 1, cv2.LINE_AA)
        cv2.putText(frame, f"rot {rot:+5.2f} deg", (x1 + 105, row),
                    FONT, 0.45, HUD_AMBER, 1, cv2.LINE_AA)
        cv2.putText(frame, f"scl {scl:5.3f}", (x1 + 105, row + 16),
                    FONT, 0.45, HUD_AMBER, 1, cv2.LINE_AA)

        # Hareket yonu mini okcugu
        ax, ay = x2 - 22, y2 - 14
        norm = max(1.0, np.hypot(dx, dy))
        ux, uy = dx / norm, dy / norm
        tip = (int(ax + ux * 10), int(ay + uy * 10))
        cv2.arrowedLine(frame, (ax, ay), tip, HUD_NEON, 1, cv2.LINE_AA, tipLength=0.4)

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
