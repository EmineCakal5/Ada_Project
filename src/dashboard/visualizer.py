# -*- coding: utf-8 -*-
"""
Visualizer — BGR frame overlays: tracks, HUD, GMC telemetry.
English labels; PIL (`src.pil_text`) for correct Unicode rendering.
"""

import cv2
import numpy as np
import time
import logging
from typing import List, Dict, Optional, Tuple
from collections import deque

from src.pil_text import draw_text_bgr, _text_size

logger = logging.getLogger(__name__)

# Neon accent #00D4FF (BGR)
HUD_NEON = (255, 212, 0)
HUD_CYBER = (136, 255, 0)   # #00FF88
HUD_AMBER = (0, 140, 255)   # #FF8C00-ish
HUD_DIM = (180, 186, 198)
HUD_LINE = (90, 96, 110)

# Re-export for dashboard
__all__ = ["Visualizer", "FPSCounter", "draw_text_bgr"]


class FPSCounter:
    """Sliding-window FPS estimate."""

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
    Draw tracks, corner boxes, top label line, status bar, and GMC HUD.
    Threat colors: LOW green, MEDIUM gold, HIGH red, CRITICAL purple.
    """

    LEVEL_COLORS = {
        "LOW": (136, 255, 0),       # #00FF88
        "MEDIUM": (0, 215, 255),    # #FFD700
        "HIGH": (68, 68, 255),      # #FF4444
        "CRITICAL": (255, 64, 200),  # purple / magenta
    }

    def __init__(self, config: dict):
        self.fps_counter = FPSCounter()
        d = config.get("dashboard", {}) or {}
        self.show_fps = d.get("fps_display", True)
        self.title = d.get("title", "Aerial Surveillance")
        self.aerial_view = d.get("aerial_view", True)
        self.bbox_scale = float(d.get("aerial_bbox_scale", 0.7))
        self.bbox_scale = max(0.35, min(1.0, self.bbox_scale))

    def render(self, frame: np.ndarray, track_history, alerts: List,
               per_track: Dict, extra_info: dict = None) -> np.ndarray:
        self.fps_counter.tick()
        frame = self._draw_tracks(frame, track_history, per_track)
        frame = self._draw_crosshair(frame)
        frame = self._draw_status_bar(frame, track_history, extra_info)
        frame = self._draw_gmc_hud(frame, extra_info)
        frame = self._draw_alert_panel(frame, alerts)
        if self.aerial_view:
            frame = self._draw_aerial_badge(frame)
        return frame

    def _draw_tracks(self, frame: np.ndarray, track_history, per_track: Dict) -> np.ndarray:
        for track in track_history.get_all_active():
            tid = track.track_id
            info = per_track.get(tid, {})
            level = str(info.get("threat_level", "LOW")).upper()
            if level not in self.LEVEL_COLORS:
                level = "LOW"
            score = float(info.get("threat_score", 0.0) or 0.0)
            color = self.LEVEL_COLORS.get(level, (120, 120, 120))

            x1, y1, x2, y2 = [int(v) for v in track.bbox]

            cl = max(4, int(round(12 * self.bbox_scale)))
            th = 1 if self.bbox_scale >= 0.55 else 1
            self._draw_corner_box(
                frame, x1, y1, x2, y2, color, thickness=th, corner_len=cl,
            )

            pts = list(track.trajectory)
            for i in range(1, min(len(pts), 25)):
                alpha = i / 25
                pt1 = (int(pts[-i + 1][0] if i > 1 else pts[0][0]),
                       int(pts[-i + 1][1] if i > 1 else pts[0][1]))
                pt2 = (int(pts[-i][0]), int(pts[-i][1]))
                c = tuple(int(c * alpha) for c in color)
                cv2.line(frame, pt1, pt2, c, 1, cv2.LINE_AA)

            raw_cls = track.class_name or "object"
            cls_en = raw_cls.replace("_", " ").upper()
            line = f"#{tid} {cls_en} | {level} {score:.2f}"
            fz = max(9, int(round(13 * self.bbox_scale)))
            tw, th = _text_size(line, fz)
            pad_x, pad_y = 6, 4
            bg_y0 = max(0, y1 - th - 2 * pad_y)
            bg_x1 = x1 + tw + 2 * pad_x
            bg_y1 = y1
            cv2.rectangle(frame, (x1, bg_y0), (bg_x1, bg_y1), (18, 22, 32), -1)
            cv2.rectangle(frame, (x1, bg_y0), (bg_x1, bg_y1), color, 1)
            draw_text_bgr(
                frame, line, (x1 + pad_x, y1 - pad_y), fz, (245, 248, 252),
                stroke_width=1, stroke_color=(10, 12, 18),
            )

            cx, cy = int(track.center[0]), int(track.center[1])
            cv2.circle(frame, (cx, cy), 3, color, -1, cv2.LINE_AA)

        return frame

    def _draw_corner_box(self, frame, x1, y1, x2, y2, color, thickness=1, corner_len=10):
        """Minimal corner brackets."""
        tl = (x1, y1), (x1 + corner_len, y1), (x1, y1 + corner_len)
        tr = (x2, y1), (x2 - corner_len, y1), (x2, y1 + corner_len)
        bl = (x1, y2), (x1 + corner_len, y2), (x1, y2 - corner_len)
        br = (x2, y2), (x2 - corner_len, y2), (x2, y2 - corner_len)
        cv2.line(frame, tl[0], tl[1], color, thickness, cv2.LINE_AA)
        cv2.line(frame, tl[0], tl[2], color, thickness, cv2.LINE_AA)
        cv2.line(frame, tr[0], tr[1], color, thickness, cv2.LINE_AA)
        cv2.line(frame, tr[0], tr[2], color, thickness, cv2.LINE_AA)
        cv2.line(frame, bl[0], bl[1], color, thickness, cv2.LINE_AA)
        cv2.line(frame, bl[0], bl[2], color, thickness, cv2.LINE_AA)
        cv2.line(frame, br[0], br[1], color, thickness, cv2.LINE_AA)
        cv2.line(frame, br[0], br[2], color, thickness, cv2.LINE_AA)

    def _draw_status_bar(self, frame: np.ndarray, track_history,
                         extra_info: dict = None) -> np.ndarray:
        h, w = frame.shape[:2]
        bar_h = 40
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (10, 14, 26), -1)
        cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)
        cv2.line(frame, (0, bar_h), (w, bar_h), HUD_NEON, 1, cv2.LINE_AA)

        fz_med, fz_sm = 17, 12
        title_txt = "[ SENTINEL UAV ] " + str(self.title)
        draw_text_bgr(frame, title_txt, (10, 26), fz_med, HUD_NEON,
                      stroke_width=1, stroke_color=(8, 10, 18))

        if extra_info and extra_info.get("scenario"):
            scn = f"> SCENARIO: {extra_info['scenario']}"
            tw, _ = _text_size(scn, fz_med)
            draw_text_bgr(frame, scn, (w // 2 - tw // 2, 26), fz_med, HUD_CYBER,
                          stroke_width=1, stroke_color=(8, 10, 18))

        x = w - 10
        if self.show_fps:
            fps_txt = f"FPS {self.fps_counter.fps:4.1f}"
            tw, _ = _text_size(fps_txt, fz_med)
            x -= tw
            draw_text_bgr(frame, fps_txt, (x, 26), fz_med, HUD_CYBER,
                          stroke_width=1, stroke_color=(8, 10, 18))
            x -= 16

        n_tracks = track_history.count_active()
        trk_txt = f"TRACKS {n_tracks:02d}"
        tw, _ = _text_size(trk_txt, fz_med)
        x -= tw
        draw_text_bgr(frame, trk_txt, (x, 26), fz_med, HUD_DIM,
                      stroke_width=1, stroke_color=(8, 10, 18))
        x -= 16

        gmc_active = bool(extra_info and extra_info.get("gmc_active"))
        badge = "GMC ON" if gmc_active else "GMC OFF"
        badge_color = HUD_NEON if gmc_active else HUD_LINE
        tw, _ = _text_size(badge, fz_sm)
        x -= tw + 12
        cv2.rectangle(frame, (x - 6, 10), (x + tw + 6, 30), (16, 20, 32), -1)
        cv2.rectangle(frame, (x - 6, 10), (x + tw + 6, 30), badge_color, 1)
        draw_text_bgr(frame, badge, (x, 24), fz_sm, badge_color,
                      stroke_width=1, stroke_color=(8, 10, 18))
        if gmc_active:
            pulse_r = 3 if int(time.time() * 2) % 2 == 0 else 4
            cv2.circle(frame, (x - 14, 20), pulse_r, HUD_CYBER, -1, cv2.LINE_AA)

        return frame

    def _draw_crosshair(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        gap, length = 8, 14
        cv2.line(frame, (cx - gap - length, cy), (cx - gap, cy), HUD_NEON, 1, cv2.LINE_AA)
        cv2.line(frame, (cx + gap, cy), (cx + gap + length, cy), HUD_NEON, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - gap - length), (cx, cy - gap), HUD_NEON, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy + gap), (cx, cy + gap + length), HUD_NEON, 1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 1, HUD_NEON, -1, cv2.LINE_AA)
        corner = 22
        for sx, sy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            x0 = cx + sx * corner
            y0 = cy + sy * corner
            cv2.line(frame, (x0, y0), (x0 - sx * 6, y0), HUD_LINE, 1, cv2.LINE_AA)
            cv2.line(frame, (x0, y0), (x0, y0 - sy * 6), HUD_LINE, 1, cv2.LINE_AA)
        return frame

    def _draw_gmc_hud(self, frame: np.ndarray, extra_info: dict = None) -> np.ndarray:
        if not extra_info or not extra_info.get("gmc_active"):
            return frame

        dx = float(extra_info.get("ego_dx", 0.0))
        dy = float(extra_info.get("ego_dy", 0.0))
        rot = float(extra_info.get("ego_rot_deg", 0.0))
        scl = float(extra_info.get("ego_scale", 1.0))

        h, w = frame.shape[:2]
        pw, ph = 220, 108
        x1, y1 = 10, h - ph - 10
        x2, y2 = x1 + pw, y1 + ph

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (12, 16, 28), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), HUD_NEON, 1)
        cv2.rectangle(frame, (x1, y1), (x1 + 3, y2), HUD_NEON, -1)

        fz_h, fz_d = 12, 13
        draw_text_bgr(
            frame, "GMC · CAMERA DRIFT", (x1 + 10, y1 + 16), fz_h, HUD_NEON,
            stroke_width=1, stroke_color=(0, 0, 0),
        )
        cv2.line(frame, (x1 + 10, y1 + 22), (x2 - 10, y1 + 22), HUD_LINE, 1)
        dmag = float(np.hypot(dx, dy))
        draw_text_bgr(
            frame, f"DRIFT  {dmag:5.2f} px", (x1 + 10, y1 + 30), 11, HUD_NEON,
            stroke_width=1, stroke_color=(0, 0, 0),
        )

        row = y1 + 48
        draw_text_bgr(frame, f"dx  {dx:+6.2f} px", (x1 + 10, row), fz_d, HUD_CYBER,
                      stroke_width=1, stroke_color=(0, 0, 0))
        draw_text_bgr(frame, f"dy  {dy:+6.2f} px", (x1 + 10, row + 16), fz_d, HUD_CYBER,
                      stroke_width=1, stroke_color=(0, 0, 0))
        draw_text_bgr(frame, f"ROT {rot:+5.2f} deg", (x1 + 108, row), fz_d, HUD_AMBER,
                      stroke_width=1, stroke_color=(0, 0, 0))
        draw_text_bgr(frame, f"SCL {scl:5.3f}", (x1 + 108, row + 16), fz_d, HUD_AMBER,
                      stroke_width=1, stroke_color=(0, 0, 0))

        ax, ay = x2 - 22, y2 - 14
        norm = max(1.0, np.hypot(dx, dy))
        ux, uy = dx / norm, dy / norm
        tip = (int(ax + ux * 10), int(ay + uy * 10))
        cv2.arrowedLine(frame, (ax, ay), tip, HUD_NEON, 1, cv2.LINE_AA, tipLength=0.35)

        return frame

    def _draw_aerial_badge(self, frame: np.ndarray) -> np.ndarray:
        label = "AERIAL VIEW"
        fz = 14
        tw, th = _text_size(label, fz)
        pad = 8
        h, w = frame.shape[:2]
        x0, y0 = pad, h - th - 2 * pad
        cv2.rectangle(
            frame,
            (x0 - 2, y0 - 2),
            (x0 + tw + 14, y0 + th + 4),
            (10, 14, 26),
            -1,
        )
        cv2.rectangle(
            frame,
            (x0 - 2, y0 - 2),
            (x0 + tw + 14, y0 + th + 4),
            HUD_NEON,
            1,
            cv2.LINE_AA,
        )
        draw_text_bgr(
            frame, label, (x0 + 6, y0 + th),
            fz, HUD_NEON, stroke_width=1, stroke_color=(6, 8, 14),
        )
        return frame

    def _draw_alert_panel(self, frame: np.ndarray, alerts: List) -> np.ndarray:
        if not alerts:
            return frame

        h, w = frame.shape[:2]
        panel_w = 400
        row_h = 22
        padding = 8
        panel_h = min(len(alerts), 6) * row_h + padding * 2

        px1 = w - panel_w - 10
        py1 = h - panel_h - 10
        px2 = w - 10
        py2 = h - 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (px1, py1), (px2, py2), (14, 18, 30), -1)
        cv2.addWeighted(overlay, 0.76, frame, 0.24, 0, frame)
        cv2.rectangle(frame, (px1, py1), (px2, py2), HUD_LINE, 1)

        fz_al = 11
        for i, alert in enumerate(alerts[-6:]):
            msg = alert.message if hasattr(alert, "message") else str(alert)
            lvl = alert.threat_level if hasattr(alert, "threat_level") else "LOW"
            color = self.LEVEL_COLORS.get(str(lvl).upper(), (200, 200, 200))
            y = py1 + padding + (i + 1) * row_h - 4
            draw_text_bgr(frame, msg[:118], (px1 + 6, y), fz_al, color,
                          stroke_width=1, stroke_color=(8, 10, 18))

        return frame
