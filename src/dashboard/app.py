# -*- coding: utf-8 -*-
"""
Streamlit dashboard — landing page, video analysis, alert panel, replay, metrics.

Run:
    streamlit run src/dashboard/app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import yaml
import sys
import os
import time
import logging
from html import escape as html_escape
from typing import List
import plotly.graph_objects as go
import pandas as pd

DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(DASHBOARD_DIR, "..", ".."))
sys.path.insert(0, ROOT)


def is_stream_url(s: str) -> bool:
    return s.startswith(("rtsp://", "rtmp://", "http://", "https://"))


def resolve_user_video_path(user_input: str) -> str:
    s = (user_input or "").strip()
    if not s:
        return s
    if is_stream_url(s):
        return s
    s = os.path.expanduser(s)
    norm = os.path.normpath(s)
    if os.path.isabs(norm):
        return os.path.normpath(os.path.abspath(norm))
    return os.path.normpath(os.path.join(DASHBOARD_DIR, "..", "..", norm))


from src.detector.yolo_detector import YOLODetector
from src.tracker.bytetrack_tracker import ByteTracker
from src.tracker.track_history import TrackHistory
from src.tracker.ego_motion import EgoMotionCompensator, decompose_affine
from src.behavior.engine import BehaviorEngine
from src.dashboard.visualizer import Visualizer, draw_text_bgr
from src.dashboard.alert_system import AlertSystem
from src.dashboard.replay_mode import ReplayManager

# ─── Theme constants ────────────────────────────────────────────────────────

BG       = "#0A0E1A"
BG2      = "#0d1222"
ACCENT   = "#00D4FF"
CRIT     = "#FF4444"
HIGH_C   = "#FF8C00"
MED_C    = "#FFD700"
NORM_C   = "#00FF88"
BORDER   = "rgba(0, 212, 255, 0.15)"

st.set_page_config(
    page_title="Derin Öğrenme ile Güvenlik İhlali Tespiti",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    h1, h2, h3, h4, h5, .font-mono-head {{
        font-family: 'JetBrains Mono', 'Consolas', monospace !important;
    }}

    .stApp {{
        background-color: {BG};
        color: #e8eaef;
    }}

    header[data-testid="stHeader"] {{
        background: #070b14 !important;
        border-bottom: 1px solid rgba(0,212,255,0.08) !important;
        height: 2.75rem !important;
        min-height: 2.75rem !important;
    }}
    /* Sadece deploy/menü butonlarını gizle — sidebar toggle'a dokunma */
    .stDeployButton {{ display: none !important; }}
    [data-testid="stMainMenu"] {{ display: none !important; }}
    [data-testid="stStatusWidget"] {{ display: none !important; }}
    /* Collapsed sidebar aç butonu: koyu temada görünür yap */
    [data-testid="collapsedControl"] {{
        display: block !important;
        visibility: visible !important;
        z-index: 9999 !important;
    }}
    [data-testid="collapsedControl"] button {{
        background: rgba(0,212,255,0.10) !important;
        border: 1px solid rgba(0,212,255,0.40) !important;
    }}
    [data-testid="collapsedControl"] svg {{
        fill: {ACCENT} !important;
        color: {ACCENT} !important;
    }}
    .block-container {{
        padding-top: 3.25rem;
        padding-bottom: 2rem;
        max-width: 1600px;
    }}
    [data-testid="stSidebar"] {{
        background-color: #070b14 !important;
        border-right: 1px solid rgba(0, 212, 255, 0.12);
    }}
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown {{
        color: #b8c0d4 !important;
    }}

    [data-testid="stSidebar"] [data-testid="stSelectbox"] *,
    [data-testid="stSidebar"] [data-testid="stSelectbox"] *::before,
    [data-testid="stSidebar"] [data-testid="stSelectbox"] *::after {{
        cursor: default !important;
    }}

    div[data-testid="stVerticalBlock"] > div:has(> label) label {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.04em;
    }}

    .stButton > button {{
        background: linear-gradient(180deg, rgba(0,212,255,0.12) 0%, rgba(0,212,255,0.04) 100%) !important;
        border: 1px solid rgba(0, 212, 255, 0.45) !important;
        color: {ACCENT} !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase;
        border-radius: 6px !important;
        padding: 0.65rem 1rem !important;
        transition: all 0.18s ease;
    }}
    .stButton > button:hover {{
        border-color: {ACCENT} !important;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.25);
        background: rgba(0, 212, 255, 0.18) !important;
    }}

    .stImage > img {{
        padding: 6px;
        background: #070b14;
        border-radius: 10px;
        border: 1px solid rgba(0, 212, 255, 0.15);
    }}

    .metrics-footer {{
        background: linear-gradient(180deg, #0d1222 0%, #080c18 100%);
        border: 1px solid rgba(0, 212, 255, 0.12);
        padding: 1rem 0.5rem;
        display: flex;
        justify-content: space-around;
        align-items: flex-end;
        border-radius: 10px;
        margin-top: 0.75rem;
    }}
    .metric-box {{
        display: flex;
        flex-direction: column;
        text-align: center;
        min-width: 0;
    }}
    .metric-box label {{
        font-size: 0.62rem;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 600;
        color: #7d8aa3;
        text-transform: uppercase;
        letter-spacing: 0.14em;
    }}
    .metric-box .metric-val {{
        font-size: 1.65rem;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700;
        color: {ACCENT};
        line-height: 1.15;
        margin-top: 0.15rem;
    }}

    .metrics-uav-hero {{
        display: flex;
        gap: 1rem;
        margin-top: 0.75rem;
        margin-bottom: 0.35rem;
    }}
    .uav-hero-box {{
        flex: 1;
        min-width: 0;
        background: linear-gradient(180deg, #0f1528 0%, #0a0e1a 100%);
        border: 1px solid rgba(0, 212, 255, 0.28);
        border-radius: 10px;
        padding: 0.75rem 1rem 0.65rem;
        text-align: center;
    }}
    .uav-hero-box label {{
        font-size: 0.68rem;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700;
        color: #9aa8c4;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        display: block;
        margin-bottom: 0.25rem;
    }}
    .uav-hero-gmc .metric-hero-val {{
        font-size: 2.35rem;
        line-height: 1.1;
    }}
    .uav-hero-drift .metric-hero-val {{
        font-size: 2rem;
        line-height: 1.1;
    }}
    .metric-hero-val {{
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700;
    }}
    .uav-hero-drift .drift-dxdy {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #6b7a94;
        letter-spacing: 0.04em;
        margin-top: 0.2rem;
    }}

    .section-label {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.2em;
        color: #7d8aa3 !important;
        text-transform: uppercase;
        margin-bottom: 0.35rem !important;
    }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_config():
    config_path = os.path.join(ROOT, "config", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_resource
def init_components(_config):
    detector  = YOLODetector(_config)
    tracker   = ByteTracker(_config)
    if hasattr(detector, "model") and hasattr(detector.model, "names"):
        tracker.set_class_names(detector.model.names)
    history   = TrackHistory()
    engine    = BehaviorEngine(_config)
    visualizer = Visualizer(_config)
    alert_sys  = AlertSystem(_config)
    replay     = ReplayManager(_config)

    ego_cfg = (_config.get("tracker", {}) or {}).get("ego_motion", {}) or {}
    if ego_cfg.get("enabled", False):
        ego = EgoMotionCompensator(
            downscale=float(ego_cfg.get("downscale", 0.5)),
            max_features=int(ego_cfg.get("max_features", 500)),
            grid=tuple(ego_cfg.get("grid", [4, 4])),
            ransac_thresh=float(ego_cfg.get("ransac_thresh", 3.0)),
        )
    else:
        ego = None
    return detector, tracker, history, engine, visualizer, alert_sys, replay, ego


def init_state():
    defaults = {
        "page":               "Home",
        "running":            False,
        "replay_mode":        False,
        "current_scenario":   None,
        "alerts":             [],
        "frame_count":        0,
        "fps":                0.0,
        "track_count":        0,
        "threat_history":     [],
        "pipeline_frame_count": 0,
        "video_frame_pos":    0,
        "last_cap_source":    None,
        "threat_plot_ema":    0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─── Landing Page ────────────────────────────────────────────────────────────

def render_landing(replay: ReplayManager):
    scenarios = replay.list_scenarios()

    # Senaryo satırlarını oluştur
    sc_rows = ""
    for s in scenarios:
        dot   = f'<span style="color:{NORM_C};">●</span>' if s.exists else '<span style="color:#4a5568;">○</span>'
        durum = "Hazır" if s.exists else "Eksik"
        sc_rows += f"""
        <div class="sc-row">
          <div class="sc-top">
            <span class="sc-name">{html_escape(s.name)}</span>
            <span class="sc-badge">{dot} {durum}</span>
          </div>
          <div class="sc-desc">{html_escape(s.description)}</div>
        </div>"""

    page_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<style>
  *{{box-sizing:border-box;margin:0;padding:0;}}
  html,body{{
    background:{BG};
    font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;
    color:#e2e8f0;
    -webkit-font-smoothing:antialiased;
    font-size:15px;
    line-height:1.6;
  }}
  .page{{display:flex;flex-direction:column;gap:20px;padding:4px 2px 12px;}}

  /* ── HERO ── */
  .hero{{
    background:linear-gradient(135deg,#080d1a 0%,#0d1628 60%,#071524 100%);
    border:1px solid rgba(0,212,255,0.25);border-radius:16px;
    padding:2.2rem 2.5rem 2rem;position:relative;overflow:hidden;
  }}
  .hero::after{{
    content:'';position:absolute;top:-80px;right:-60px;
    width:340px;height:340px;border-radius:50%;
    background:radial-gradient(circle,rgba(0,212,255,0.08) 0%,transparent 65%);
    pointer-events:none;
  }}
  .pill{{
    display:inline-flex;align-items:center;gap:7px;
    background:rgba(0,212,255,0.07);border:1px solid rgba(0,212,255,0.28);
    border-radius:100px;padding:4px 14px;margin-bottom:1rem;
  }}
  .pill-dot{{width:6px;height:6px;border-radius:50%;background:{ACCENT};box-shadow:0 0 6px {ACCENT};}}
  .pill-txt{{font-size:0.72rem;font-weight:700;letter-spacing:0.18em;color:{ACCENT};text-transform:uppercase;}}
  .hero h1{{
    font-size:1.9rem;font-weight:700;color:#fff;
    letter-spacing:-0.01em;line-height:1.2;margin-bottom:0.65rem;
  }}
  .hero h1 em{{color:{ACCENT};font-style:normal;}}
  .hero p{{font-size:0.95rem;color:#94a3b8;line-height:1.7;max-width:640px;margin-bottom:1.5rem;}}
  .meta{{display:flex;flex-wrap:wrap;gap:8px;}}
  .meta-item{{
    background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);
    border-radius:8px;padding:8px 14px;
  }}
  .meta-label{{font-size:0.68rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:3px;}}
  .meta-val{{font-size:0.85rem;font-weight:600;color:#cbd5e1;}}

  /* ── ÖZELLİK KARTLARI ── */
  .features{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;}}
  .fc{{
    background:linear-gradient(160deg,#0f1830 0%,#0a0e1a 100%);
    border:1px solid rgba(0,212,255,0.12);border-radius:12px;
    padding:1.1rem 1rem;display:flex;flex-direction:column;gap:8px;
  }}
  .fc-badge{{
    display:inline-block;font-size:0.68rem;font-weight:700;
    letter-spacing:0.1em;padding:3px 9px;border-radius:5px;
    width:fit-content;
  }}
  .fc-title{{font-size:0.88rem;font-weight:700;color:#e2e8f0;}}
  .fc-desc{{font-size:0.78rem;color:#94a3b8;line-height:1.6;flex:1;}}

  /* ── ALT SATIRLAR ── */
  .row2{{display:grid;grid-template-columns:1fr 1fr;gap:16px;}}

  /* Mimari */
  .box{{
    background:linear-gradient(160deg,#0f1830 0%,#0a0e1a 100%);
    border:1px solid rgba(0,212,255,0.12);border-radius:14px;padding:1.4rem 1.5rem;
  }}
  .sec-hdr{{
    font-size:0.72rem;font-weight:700;letter-spacing:0.2em;
    color:{ACCENT};text-transform:uppercase;margin-bottom:1rem;
  }}
  .pipe{{display:flex;align-items:center;gap:6px;margin-bottom:12px;}}
  .node{{
    flex:1;background:#0b1120;border:1px solid rgba(0,212,255,0.18);
    border-radius:8px;padding:0.6rem 0.75rem;text-align:center;
  }}
  .node-t{{font-size:0.75rem;font-weight:700;color:{ACCENT};margin-bottom:4px;}}
  .node-s{{font-size:0.7rem;color:#64748b;line-height:1.4;}}
  .arrow{{color:rgba(0,212,255,0.45);font-size:1.1rem;flex-shrink:0;}}
  .hybrid{{
    background:rgba(0,212,255,0.04);border:1px solid rgba(0,212,255,0.14);
    border-radius:8px;padding:0.65rem 1rem;
    display:flex;align-items:center;gap:8px;flex-wrap:wrap;
  }}
  .hy-lbl{{font-size:0.7rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;}}
  .hy-v{{font-size:0.85rem;font-weight:700;color:{ACCENT};}}
  .hy-sep{{color:rgba(0,212,255,0.3);}}
  .hy-lvl{{margin-left:auto;font-size:0.7rem;color:#64748b;}}

  /* Senaryolar */
  .sc-row{{
    background:#080c18;border:1px solid rgba(0,212,255,0.1);
    border-radius:8px;padding:0.65rem 0.9rem;margin-bottom:8px;
  }}
  .sc-top{{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;}}
  .sc-name{{font-size:0.8rem;font-weight:700;color:#cbd5e1;}}
  .sc-badge{{font-size:0.72rem;color:#64748b;white-space:nowrap;}}
  .sc-desc{{font-size:0.75rem;color:#64748b;line-height:1.5;}}
  .hint{{font-size:0.75rem;color:#475569;margin-top:10px;line-height:1.55;}}
  .hint strong{{color:#94a3b8;}}

  /* Kullanım */
  .steps{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;}}
  .step{{display:flex;gap:10px;align-items:flex-start;}}
  .step-n{{
    min-width:26px;height:26px;border-radius:50%;flex-shrink:0;
    background:rgba(0,212,255,0.1);border:1px solid rgba(0,212,255,0.28);
    display:flex;align-items:center;justify-content:center;
    font-size:0.72rem;font-weight:700;color:{ACCENT};margin-top:1px;
  }}
  .step-t{{font-size:0.82rem;color:#94a3b8;line-height:1.55;}}
  .step-t strong{{color:#e2e8f0;}}
</style>
</head><body>
<div class="page">

  <!-- HERO -->
  <div class="hero">
    <div class="pill"><div class="pill-dot"></div><span class="pill-txt">Hava Gözetleme · Davranışsal Yapay Zeka</span></div>
    <h1>Derin Öğrenme ile <em>Güvenlik İhlali</em> Tespiti</h1>
    <p>Güvenlik kamerası ve İHA görüntülerinden gerçek zamanlı davranışsal anomali tespiti.
    YOLOv8 + ByteTrack üzerine inşa edilmiş, kural tabanlı ve MLP hibrit tehdit analizi motoru.</p>
    <div class="meta">
      <div class="meta-item">
        <div class="meta-label">Proje</div>
        <div class="meta-val">OSTİMTECH Ar-Ge ve İnovasyon Proje Pazarı II</div>
      </div>
      <div class="meta-item">
        <div class="meta-label">Geliştiriciler</div>
        <div class="meta-val">Azra Karakaya · Emine Çakal</div>
      </div>
      <div class="meta-item">
        <div class="meta-label">Model</div>
        <div class="meta-val">YOLOv8s · VisDrone 80 Epoch</div>
      </div>
    </div>
  </div>

  <!-- ÖZELLİKLER -->
  <div class="features">
    <div class="fc" style="border-left:3px solid {HIGH_C};">
      <div class="fc-badge" style="color:{HIGH_C};background:rgba(255,140,0,0.13);">YÜKSEK</div>
      <div class="fc-title">Bölge İhlali</div>
      <div class="fc-desc">Kişi veya araç yasak bölgeye girdiğinde anında alarm üretir. Poligon tabanlı dinamik bölge haritası.</div>
    </div>
    <div class="fc" style="border-left:3px solid {MED_C};">
      <div class="fc-badge" style="color:{MED_C};background:rgba(255,215,0,0.12);">ORTA</div>
      <div class="fc-title">Anormal Bekleme</div>
      <div class="fc-desc">Nesne 20+ saniye aynı konumda kalırsa bekleme (loitering) tespiti tetiklenir.</div>
    </div>
    <div class="fc" style="border-left:3px solid {HIGH_C};">
      <div class="fc-badge" style="color:{HIGH_C};background:rgba(255,140,0,0.13);">YÜKSEK</div>
      <div class="fc-title">Terk Edilmiş Nesne</div>
      <div class="fc-desc">Çanta veya bavul sahibinden uzaklaştıktan sonra hareketsiz kalırsa 10 saniyelik onay ile alarm verilir.</div>
    </div>
    <div class="fc" style="border-left:3px solid {HIGH_C};">
      <div class="fc-badge" style="color:{HIGH_C};background:rgba(255,140,0,0.13);">YÜKSEK</div>
      <div class="fc-title">Keşif Davranışı</div>
      <div class="fc-desc">Düşük yol verimliliğiyle (&#60; 0.35) geniş alan tarayan kişi sistematik keşif adayı olarak işaretlenir.</div>
    </div>
    <div class="fc" style="border-left:3px solid {CRIT};">
      <div class="fc-badge" style="color:{CRIT};background:rgba(255,68,68,0.13);">KRİTİK</div>
      <div class="fc-title">Koordineli Hareket</div>
      <div class="fc-desc">İki veya daha fazla kişi uyumlu hız vektörüyle (cos &#62; 0.85) 5+ saniye hareket ettiğinde tespit edilir.</div>
    </div>
  </div>

  <!-- MİMARİ + SENARYOLAR -->
  <div class="row2">
    <div class="box">
      <div class="sec-hdr">Sistem Mimarisi</div>
      <div class="pipe">
        <div class="node">
          <div class="node-t">YOLOv8s</div>
          <div class="node-s">VisDrone ince ayar<br/>9 sınıf · 640px</div>
        </div>
        <div class="arrow">→</div>
        <div class="node">
          <div class="node-t">ByteTrack</div>
          <div class="node-s">GMC kompanzasyon<br/>ego-motion</div>
        </div>
        <div class="arrow">→</div>
        <div class="node">
          <div class="node-t">Davranış Motoru</div>
          <div class="node-s">5 kural + MLP<br/>10 boyutlu vektör</div>
        </div>
      </div>
      <div class="hybrid">
        <span class="hy-lbl">Hibrit Karar</span>
        <span class="hy-v">%60 MLP</span>
        <span class="hy-sep">+</span>
        <span class="hy-v">%40 Kural</span>
        <span class="hy-lvl">→ DÜŞÜK / ORTA / YÜKSEK / KRİTİK</span>
      </div>
    </div>

    <div class="box">
      <div class="sec-hdr">Demo Senaryoları</div>
      {sc_rows}
      <div class="hint">Sol panelden <strong>Tekrar Modu</strong>'nu seçerek senaryoları başlatın.</div>
    </div>
  </div>

  <!-- KULLANIM KILAVUZU -->
  <div class="box">
    <div class="sec-hdr">Kullanım Kılavuzu</div>
    <div class="steps">
      <div class="step">
        <div class="step-n">1</div>
        <div class="step-t"><strong>Analiz</strong> sekmesine geçin ve sol panelden mod seçin.</div>
      </div>
      <div class="step">
        <div class="step-n">2</div>
        <div class="step-t"><strong>Tekrar Modu</strong>'nda senaryo seçin ya da Video Kütüphanesi'nden dosya yükleyin.</div>
      </div>
      <div class="step">
        <div class="step-n">3</div>
        <div class="step-t"><strong>Başlat</strong> tuşuna basın. Gerçek zamanlı tehdit analizi başlar.</div>
      </div>
      <div class="step">
        <div class="step-n">4</div>
        <div class="step-t"><strong>Alarm paneli</strong> ve tehdit skoru grafiğini takip edin.</div>
      </div>
    </div>
  </div>

</div>
</body></html>"""

    components.html(page_html, height=1080, scrolling=False)


# ─── Surveillance page helpers ───────────────────────────────────────────────

def render_header():
    html_doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<style>
  html,body{{margin:0;background:transparent;color:#e8eaef;
    font-family:'Inter',system-ui,sans-serif;-webkit-font-smoothing:antialiased;}}
  .bar{{
    display:flex;justify-content:space-between;align-items:center;
    padding:14px 22px;background:rgba(10,14,26,0.96);
    border:1px solid rgba(0,212,255,0.22);border-radius:10px;
    box-shadow:0 8px 32px rgba(0,0,0,0.45);
  }}
  h1{{margin:0;font-family:'JetBrains Mono','Consolas',monospace;font-size:1.45rem;
    font-weight:700;color:{ACCENT};letter-spacing:0.03em;}}
  .sub{{font-size:0.62rem;color:#7d8aa3;text-transform:uppercase;
    letter-spacing:0.22em;margin-top:6px;}}
  .right{{text-align:right;}}
  #clock{{font-family:'JetBrains Mono',monospace;font-size:0.95rem;color:#e8eaef;}}
  .live-row{{display:flex;align-items:center;justify-content:flex-end;gap:8px;margin-top:6px;}}
  .pulse{{
    width:8px;height:8px;background:{NORM_C};border-radius:50%;
    animation:pulse 1.15s ease-in-out infinite;
    box-shadow:0 0 10px rgba(0,255,136,0.55);
  }}
  @keyframes pulse{{0%,100%{{opacity:1;transform:scale(1);}}50%{{opacity:0.45;transform:scale(0.88);}}}}
  .live-txt{{
    font-family:'JetBrains Mono',monospace;font-size:0.68rem;font-weight:700;
    letter-spacing:0.18em;color:{NORM_C};
  }}
</style></head><body>
<div class="bar">
  <div>
    <h1>Davranışsal Analiz</h1>
    <div class="sub">Derin Öğrenme ile Güvenlik İhlali Tespiti · v2.0</div>
  </div>
  <div class="right">
    <div id="clock">--:--:--</div>
    <div class="live-row"><span class="pulse"></span><span class="live-txt">ANALİZ AKTİF</span></div>
  </div>
</div>
<script>
function tick(){{
  const el=document.getElementById('clock');
  const d=new Date();
  const t=d.toLocaleTimeString('en-GB',{{hour12:false}});
  const dt=d.toLocaleDateString('en-GB',{{day:'2-digit',month:'short',year:'numeric'}});
  el.textContent=t+' · '+dt;
}}
tick(); setInterval(tick,1000);
</script>
</body></html>"""
    components.html(html_doc, height=104)


def render_metrics(alert_sys: AlertSystem, fps: float, track_count: int, extra_info: dict = None):
    stats = alert_sys.get_stats()
    info  = extra_info or {}
    gmc_active = bool(info.get("gmc_active"))
    dx = float(info.get("ego_dx", 0.0))
    dy = float(info.get("ego_dy", 0.0))
    drift_mag = (dx * dx + dy * dy) ** 0.5

    gmc_label = "ON"  if gmc_active else "OFF"
    gmc_color = NORM_C if gmc_active else "#5c6578"
    gmc_dot   = "● "  if gmc_active else "○ "
    drift_color = ACCENT if drift_mag < 3.0 else HIGH_C

    st.markdown(f"""
    <div class="metrics-uav-hero">
        <div class="uav-hero-box uav-hero-gmc">
            <label>GMC Durumu</label>
            <span class="metric-hero-val" style="color:{gmc_color};">{gmc_dot}{gmc_label}</span>
        </div>
        <div class="uav-hero-box uav-hero-drift">
            <label>Kamera Kayması (px)</label>
            <span class="metric-hero-val" style="color:{drift_color};">{drift_mag:.2f}</span>
            <div class="drift-dxdy">kare Δ &nbsp; dx {dx:+.2f} · dy {dy:+.2f}</div>
        </div>
    </div>
    <div class="metrics-footer">
        <div class="metric-box">
            <label>Aktif Takip</label>
            <span class="metric-val" style="color:{ACCENT};">{track_count}</span>
        </div>
        <div class="metric-box">
            <label>Akış FPS</label>
            <span class="metric-val" style="color:#e8eaef;">{fps:.1f}</span>
        </div>
        <div class="metric-box">
            <label>Kritik</label>
            <span class="metric-val" style="color:{CRIT};">{stats['critical']}</span>
        </div>
        <div class="metric-box">
            <label>Toplam Alarm</label>
            <span class="metric-val" style="color:#e8eaef;">{stats['total']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_alerts(alert_sys: AlertSystem):
    recent = list(reversed(alert_sys.get_recent(n=80)))
    cards_html: List[str] = []

    for a in recent[:80]:
        level_str = str(a.get("threat_level", "LOW")).upper()
        card_mod = "low"
        if level_str == "CRITICAL":
            card_mod = "critical"
        elif level_str == "HIGH":
            card_mod = "high"
        elif level_str in ("MEDIUM", "MODERATE"):
            card_mod = "medium"

        time_str = html_escape(str(a.get("time_str", "--:--:--")))
        type_str = html_escape(str(a.get("alert_type", "event")).upper())
        msg      = html_escape(str(a.get("message", "Unknown event")))
        row = (
            f'<span class="lvl lvl-{card_mod}">[{html_escape(level_str)}]</span>'
            f'<span class="sep">|</span>'
            f'<span class="tme">{time_str}</span>'
            f'<span class="sep">|</span>'
            f'<span class="typ">{type_str}</span>'
            f'<span class="sep">|</span>'
            f'<span class="msg">{msg}</span>'
        )
        cards_html.append(
            f'<div class="card card-{card_mod}"><div class="card-row">{row}</div></div>'
        )

    inner = (
        '<p class="empty">Aktif güvenlik olayı yok.</p>'
        if not recent
        else "".join(cards_html)
    )

    css = f"""
    html,body{{margin:0;padding:0;background:{BG};color:#e8eaef;
      font-family:'Inter',system-ui,sans-serif;-webkit-font-smoothing:antialiased;}}
    .wrap{{padding:12px 14px 16px;box-sizing:border-box;max-height:520px;overflow-y:auto;}}
    .hdr{{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;
      padding-bottom:10px;border-bottom:1px solid rgba(0,212,255,0.15);position:sticky;top:0;background:{BG};z-index:2;}}
    .hdr h2{{font-family:'JetBrains Mono','Consolas',monospace;font-size:11px;font-weight:700;margin:0;
      text-transform:uppercase;letter-spacing:0.24em;color:{ACCENT};}}
    .live{{font-family:'JetBrains Mono',monospace;background:rgba(0,255,136,0.12);color:{NORM_C};
      font-size:9px;padding:4px 10px;border-radius:4px;font-weight:700;letter-spacing:0.15em;border:1px solid rgba(0,255,136,0.35);}}
    .card{{background:#0d1222;border-radius:8px;padding:11px 12px 11px 14px;margin-bottom:10px;
      border:1px solid rgba(0,212,255,0.08);border-left:4px solid #5c6578;
      box-shadow:0 4px 16px rgba(0,0,0,0.35);}}
    .card-critical{{border-left-color:{CRIT};background:rgba(255,68,68,0.06);}}
    .card-high{{border-left-color:{HIGH_C};}}
    .card-medium{{border-left-color:{MED_C};}}
    .card-low{{border-left-color:{NORM_C};}}
    .card-row{{font-family:'JetBrains Mono','Consolas',monospace;font-size:10px;line-height:1.55;
      display:flex;flex-wrap:wrap;align-items:baseline;gap:6px;}}
    .sep{{color:#4a5366;}}
    .lvl{{font-weight:700;letter-spacing:0.06em;}}
    .lvl-critical{{color:{CRIT};}}
    .lvl-high{{color:{HIGH_C};}}
    .lvl-medium{{color:{MED_C};}}
    .lvl-low{{color:{NORM_C};}}
    .tme{{color:#8b95a8;white-space:nowrap;}}
    .typ{{color:{ACCENT};font-weight:600;}}
    .msg{{color:#c8d0e0;flex:1;min-width:140px;}}
    .empty{{font-size:12px;color:#7d8aa3;margin:12px 0;}}
    .wrap::-webkit-scrollbar{{width:6px;}}
    .wrap::-webkit-scrollbar-thumb{{background:rgba(0,212,255,0.25);border-radius:3px;}}
    """

    html_doc = (
        '<!DOCTYPE html><html><head><meta charset="utf-8"><style>'
        + css
        + '</style></head><body><div class="wrap">'
        '<div class="hdr"><h2>Aktif Güvenlik Olayları</h2><span class="live">CANLI</span></div>'
        + inner
        + "</div></body></html>"
    )
    components.html(html_doc, height=560, scrolling=False)


def _sidebar_model_info(config):
    st.sidebar.divider()
    st.sidebar.markdown("#### Model")
    st.sidebar.markdown(f"**Dedektör:** `{config['detector']['model']}`")
    st.sidebar.markdown(f"**Takipçi:** `{config['tracker']['type']}`")
    st.sidebar.markdown(f"**Cihaz:** `{config['detector']['device']}`")
    st.sidebar.caption(
        "GMC ego-motion kompanzasyonlu İHA/UAV görüntülerine uyarlanmış ML modeli."
    )


def render_sidebar(config, replay: ReplayManager):
    st.sidebar.markdown("### Kontrol Paneli")

    mode = st.sidebar.radio(
        "Mod",
        ["Tekrar Modu", "Video Analizi", "IP Kamera"],
        key="mode_select",
    )

    st.sidebar.divider()

    if mode == "IP Kamera":
        st.sidebar.markdown("#### IP Kamera / RTSP Akışı")
        rtsp = st.sidebar.text_input(
            "Akış URL'si",
            placeholder="rtsp://kullanici:sifre@192.168.1.100:554/stream",
            key="rtsp_url",
        ).strip()
        st.sidebar.caption("RTSP, RTMP ve HTTP stream URL'leri desteklenir.")
        if rtsp and not is_stream_url(rtsp):
            st.sidebar.warning("Geçerli bir akış URL'si girin (rtsp:// ile başlamalı).")
            _sidebar_model_info(config)
            return None, None
        _sidebar_model_info(config)
        return rtsp or None, None

    if mode == "Tekrar Modu":
        st.sidebar.markdown("#### Demo Senaryo")
        scenarios      = replay.list_scenarios()
        scenario_names = [s.name for s in scenarios]
        sel      = st.sidebar.selectbox("Senaryo", scenario_names, key="scenario_sel")
        selected = next((s for s in scenarios if s.name == sel), None)

        if selected:
            st.sidebar.caption(selected.description)
            if not selected.exists:
                st.sidebar.warning("Video dosyası bulunamadı.")
                st.sidebar.markdown(ReplayManager.get_download_instructions())
                _sidebar_model_info(config)
                return None, None
            st.sidebar.success("Video hazır")

        _sidebar_model_info(config)
        source = selected.video_path if selected else None
        return source, sel if selected else None

    # ── Video Analizi modu ─────────────────────────────────────────────────
    st.sidebar.markdown("#### Video Kaynağı")
    video_dir = os.path.join(ROOT, "data", "test_videos")
    try:
        available_videos = [
            f for f in os.listdir(video_dir)
            if f.endswith((".mp4", ".avi", ".mkv", ".mov"))
        ]
    except FileNotFoundError:
        available_videos = []

    src_type = st.sidebar.radio(
        "Giriş",
        ["Video kütüphanesi", "Manuel yol"],
    )

    if src_type == "Video kütüphanesi":
        if not available_videos:
            st.sidebar.warning("`data/test_videos/` klasöründe video yok.")
            st.sidebar.caption("`.mp4` / `.avi` dosyalarını o klasöre ekleyin.")
            _sidebar_model_info(config)
            return None, None
        selected_file = st.sidebar.selectbox("Test videoları", available_videos)
        abs_pick = os.path.normpath(os.path.join(ROOT, "data", "test_videos", selected_file))
        _sidebar_model_info(config)
        return abs_pick, None

    raw = st.sidebar.text_input("Dosya yolu", "data/test_videos/test.mp4").strip()
    if not raw:
        _sidebar_model_info(config)
        return None, None
    path = resolve_user_video_path(raw)
    _sidebar_model_info(config)
    return path, None


def render_video_transport():
    c1, c2, c3, _ = st.columns([1, 1, 1, 4])
    with c1:
        start = st.button("▶  Başlat", use_container_width=True, key="transport_start")
    with c2:
        stop  = st.button("■  Durdur", use_container_width=True, key="transport_stop")
    with c3:
        clear = st.button("↺  Temizle", use_container_width=True, key="transport_clear")
    return start, stop, clear


def render_threat_chart(history: list):
    if history:
        df = pd.DataFrame(history[-120:], columns=["time", "score"])
    else:
        df = pd.DataFrame({"time": [0, 1], "score": [0.0, 0.0]})

    fig = go.Figure()
    x_vals = df["time"]
    fig.add_trace(
        go.Scatter(
            x=x_vals, y=df["score"],
            mode="lines", fill="tozeroy",
            line=dict(color=ACCENT, width=2),
            fillcolor="rgba(0, 212, 255, 0.18)",
        )
    )

    x0 = float(x_vals.iloc[0])  if len(x_vals) else 0
    x1 = float(x_vals.iloc[-1]) if len(x_vals) else 1
    fig.add_shape(type="line", x0=x0, y0=0.6, x1=x1, y1=0.6,
                  line=dict(color=CRIT, width=2, dash="dash"))
    fig.add_annotation(x=x1, y=0.6, xref="x", yref="y",
                       text="Threshold 0.6", showarrow=False,
                       xanchor="right", yshift=8,
                       font=dict(size=10, color=CRIT, family="JetBrains Mono, monospace"))

    fig.update_layout(
        title=dict(text="TEHDİT SKORU ZAMANLAMASı",
                   font=dict(family="JetBrains Mono, monospace", size=13, color=ACCENT),
                   x=0, xanchor="left"),
        height=200,
        margin=dict(l=48, r=12, t=40, b=36),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7, 11, 20, 0.85)",
        showlegend=False,
        xaxis=dict(title="Kare", gridcolor="rgba(0,212,255,0.06)",
                   tickfont=dict(color="#7d8aa3", size=10), zeroline=False),
        yaxis=dict(range=[0, 1], title="Skor", gridcolor="rgba(0,212,255,0.06)",
                   tickfont=dict(color="#7d8aa3", size=10), zeroline=False),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── Page: Home ──────────────────────────────────────────────────────────────

def page_home(replay: ReplayManager):
    render_landing(replay)

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
    if st.button("Analiz Ekranına Geç →", use_container_width=False):
        st.session_state.page = "Surveillance"
        st.rerun()


# ─── Page: Surveillance ──────────────────────────────────────────────────────

def page_surveillance(config, detector, tracker, history, engine, visualizer,
                      alert_sys, replay, ego):
    render_header()

    source, scenario_name = render_sidebar(config, replay)

    col_video, col_panel = st.columns([3, 1])

    with col_video:
        start, stop, clear = render_video_transport()
        video_placeholder = st.empty()
        st.markdown('<p class="section-label">TEHDİT SKORU ZAMANLAMASı</p>', unsafe_allow_html=True)
        chart_placeholder = st.empty()

    with col_panel:
        st.markdown('<p class="section-label">Olaylar</p>', unsafe_allow_html=True)
        alert_placeholder = st.empty()

    if start:
        st.session_state.running = True
    if stop:
        st.session_state.running = False
    if clear:
        alert_sys.clear()
        st.session_state.alerts        = []
        st.session_state.threat_history  = []
        st.session_state.threat_plot_ema = 0.0
        st.session_state.pipeline_frame_count = 0

    st.divider()
    metric_placeholder = st.empty()

    if st.session_state.running and source is not None:
        source_path = source
        _is_stream = isinstance(source_path, str) and is_stream_url(source_path)
        _is_file   = isinstance(source_path, str) and not _is_stream

        if _is_file and not os.path.isfile(source_path):
            st.error(f"Video bulunamadı: {source_path}")
            st.session_state.running = False
        else:
            src_key = str(source_path) if isinstance(source_path, str) else "webcam:0"
            if st.session_state.get("last_cap_source") != src_key:
                st.session_state.video_frame_pos  = 0
                st.session_state.last_cap_source  = src_key

            cap = cv2.VideoCapture(source_path)
            if _is_file:
                pos = int(st.session_state.get("video_frame_pos", 0) or 0)
                if pos > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

            target_w = config["dashboard"].get("frame_width",  960)
            target_h = config["dashboard"].get("frame_height", 540)
            fps_counter_times = []
            ego_dx_hist, ego_dy_hist = [], []
            if ego is not None:
                ego.reset()

            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    if _is_stream:
                        st.warning("Akış kesildi. Bağlantı kontrol edin.")
                        st.session_state.running = False
                        break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    st.session_state.video_frame_pos = 0
                    history.clear()
                    tracker.reset()
                    if ego is not None:
                        ego.reset()
                    continue

                if _is_file:
                    st.session_state.video_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                frame_count = st.session_state.pipeline_frame_count + 1
                st.session_state.pipeline_frame_count = frame_count
                frame = cv2.resize(frame, (target_w, target_h))

                H = None
                ego_dx = ego_dy = 0.0
                ego_rot_deg = 0.0
                ego_scale   = 1.0
                if ego is not None:
                    H = ego.estimate(frame)
                    ego_dx   = float(H[0, 2])
                    ego_dy   = float(H[1, 2])
                    ego_scale, rot_rad = decompose_affine(H)
                    ego_rot_deg = float(np.degrees(rot_rad))
                    history.apply_camera_motion(H)
                    engine.apply_camera_motion(H)

                detections = detector.detect(frame)
                tracks     = tracker.update(detections, frame, transform=H)

                active_ids = []
                for t in tracks:
                    history.update(t.track_id, t.bbox, t.center,
                                   t.class_id, t.class_name, t.confidence)
                    active_ids.append(t.track_id)
                history.mark_missing(active_ids)

                # Zone overlay devre dışı — test videoları kalibre edilmemiş koordinatlar içeriyor

                alerts, per_track = engine.process(history, target_w, target_h)
                if alerts:
                    alert_sys.add_all(alerts, st.session_state.pipeline_frame_count)

                fps_counter_times.append(time.time())
                fps_counter_times = fps_counter_times[-30:]
                fps = (
                    (len(fps_counter_times) - 1)
                    / max(fps_counter_times[-1] - fps_counter_times[0], 1e-6)
                    if len(fps_counter_times) > 1 else 0
                )

                ego_dx_hist.append(ego_dx)
                ego_dy_hist.append(ego_dy)
                ego_dx_hist = ego_dx_hist[-15:]
                ego_dy_hist = ego_dy_hist[-15:]
                ego_dx_avg  = float(np.mean(ego_dx_hist)) if ego_dx_hist else 0.0
                ego_dy_avg  = float(np.mean(ego_dy_hist)) if ego_dy_hist else 0.0

                extra = {
                    "scenario":   scenario_name,
                    "gmc_active": ego is not None,
                    "ego_dx":     ego_dx_avg,
                    "ego_dy":     ego_dy_avg,
                    "ego_rot_deg": ego_rot_deg,
                    "ego_scale":  ego_scale,
                }
                frame_out = visualizer.render(frame, history, alerts, per_track, extra)

                max_rule   = max((float(v.get("rule_score",   0) or 0) for v in per_track.values()), default=0.0)
                max_threat = max((float(v.get("threat_score", 0) or 0) for v in per_track.values()), default=0.0)
                max_alert  = max((float(getattr(a, "score",   0) or 0) for a in alerts),             default=0.0)
                instant    = max(max_rule, max_alert, max_threat)
                prev_plot  = float(st.session_state.get("threat_plot_ema", 0.0))
                plot_val   = instant if instant >= prev_plot else 0.88 * prev_plot + 0.12 * instant
                st.session_state.threat_plot_ema = plot_val
                st.session_state.threat_history.append(
                    (st.session_state.pipeline_frame_count, plot_val)
                )
                st.session_state.threat_history = st.session_state.threat_history[-120:]

                frame_rgb = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, use_container_width=True)

                with chart_placeholder:
                    render_threat_chart(st.session_state.threat_history)
                with alert_placeholder:
                    render_alerts(alert_sys)
                with metric_placeholder:
                    render_metrics(alert_sys, fps, history.count_active(), extra)

            cap.release()
            alert_sys.force_save()

    elif not st.session_state.running:
        placeholder_img = np.zeros((540, 960, 3), dtype=np.uint8)
        draw_text_bgr(
            placeholder_img,
            "Sol menüden video seçin ve Başlat'a tıklayın…",
            (130, 280), 20, (90, 96, 110),
            stroke_width=1, stroke_color=(10, 14, 26),
        )
        video_placeholder.image(placeholder_img, channels="BGR", use_container_width=True)

        with chart_placeholder:
            render_threat_chart(st.session_state.threat_history)
        with alert_placeholder:
            render_alerts(alert_sys)
        with metric_placeholder:
            render_metrics(alert_sys, 0.0, 0,
                           extra_info={"gmc_active": ego is not None,
                                       "ego_dx": 0.0, "ego_dy": 0.0})


# ─── Sidebar navigation ──────────────────────────────────────────────────────

def render_nav():
    st.sidebar.markdown(
        f"""<div style="padding:1rem 0.5rem 0.5rem;text-align:center;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
               font-weight:700;letter-spacing:0.22em;color:{ACCENT};
               text-transform:uppercase;margin-bottom:0.6rem;">Navigasyon</div>
        </div>""",
        unsafe_allow_html=True,
    )
    pages = {"Home": "◈  Ana Sayfa", "Surveillance": "◉  Analiz"}
    for key, label in pages.items():
        active = st.session_state.page == key
        if st.sidebar.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key
            st.rerun()
    st.sidebar.divider()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    init_state()
    config = load_config()

    _log_cfg = config.get("logging", {}) or {}
    _lvl = getattr(logging, str(_log_cfg.get("level", "INFO")).upper(), logging.INFO)
    logging.basicConfig(
        level=_lvl,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

    detector, tracker, history, engine, visualizer, alert_sys, replay, ego = init_components(config)

    render_nav()

    if st.session_state.page == "Home":
        page_home(replay)
    else:
        page_surveillance(config, detector, tracker, history, engine,
                          visualizer, alert_sys, replay, ego)


if __name__ == "__main__":
    main()
