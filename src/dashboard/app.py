# -*- coding: utf-8 -*-
"""
Streamlit dashboard — live video, alert panel, replay, metrics.

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


def resolve_user_video_path(user_input: str) -> str:
    """Map manual input to a filesystem path: absolute as-is, else relative to project root."""
    s = (user_input or "").strip()
    if not s:
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

# ─── Theme constants ───────────────────────────────────────────

BG = "#0A0E1A"
ACCENT = "#00D4FF"
CRIT = "#FF4444"
HIGH_C = "#FF8C00"
MED_C = "#FFD700"
NORM_C = "#00FF88"

st.set_page_config(
    page_title="SENTINEL UAV — Aerial Surveillance",
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

    header[data-testid="stHeader"] {{ display: none; }}
    .block-container {{
        padding-top: 1.25rem;
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

    /* Sidebar selectboxes (e.g. Test videos): native arrow, no BaseWeb/text cursor */
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
    detector = YOLODetector(_config)
    tracker = ByteTracker(_config)
    if hasattr(detector, "model") and hasattr(detector.model, "names"):
        tracker.set_class_names(detector.model.names)
    history = TrackHistory()
    engine = BehaviorEngine(_config)
    visualizer = Visualizer(_config)
    alert_sys = AlertSystem(_config)
    replay = ReplayManager(_config)

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
        "running": False,
        "replay_mode": False,
        "current_scenario": None,
        "alerts": [],
        "frame_count": 0,
        "fps": 0.0,
        "track_count": 0,
        "threat_history": [],
        "pipeline_frame_count": 0,
        "video_frame_pos": 0,
        "last_cap_source": None,
        "threat_plot_ema": 0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


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
    <h1>SENTINEL UAV COMMAND</h1>
    <div class="sub">Aerial Surveillance System · Behavioral Analysis v2.0.4</div>
  </div>
  <div class="right">
    <div id="clock">--:--:--</div>
    <div class="live-row"><span class="pulse"></span><span class="live-txt">UAV LINK LIVE</span></div>
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
    info = extra_info or {}
    gmc_active = bool(info.get("gmc_active"))
    dx = float(info.get("ego_dx", 0.0))
    dy = float(info.get("ego_dy", 0.0))
    drift_mag = (dx * dx + dy * dy) ** 0.5

    if gmc_active:
        gmc_label = "ON"
        gmc_color = NORM_C
        gmc_dot = "● "
    else:
        gmc_label = "OFF"
        gmc_color = "#5c6578"
        gmc_dot = "○ "

    drift_color = ACCENT if drift_mag < 3.0 else HIGH_C

    st.markdown(f"""
    <div class="metrics-uav-hero">
        <div class="uav-hero-box uav-hero-gmc">
            <label>GMC status</label>
            <span class="metric-hero-val" style="color:{gmc_color};">{gmc_dot}{gmc_label}</span>
        </div>
        <div class="uav-hero-box uav-hero-drift">
            <label>Camera drift (px)</label>
            <span class="metric-hero-val" style="color:{drift_color};">{drift_mag:.2f}</span>
            <div class="drift-dxdy">frame Δ &nbsp; dx {dx:+.2f} · dy {dy:+.2f}</div>
        </div>
    </div>
    <div class="metrics-footer">
        <div class="metric-box">
            <label>Active tracks</label>
            <span class="metric-val" style="color:{ACCENT};">{track_count}</span>
        </div>
        <div class="metric-box">
            <label>Stream FPS</label>
            <span class="metric-val" style="color:#e8eaef;">{fps:.1f}</span>
        </div>
        <div class="metric-box">
            <label>Critical</label>
            <span class="metric-val" style="color:{CRIT};">{stats['critical']}</span>
        </div>
        <div class="metric-box">
            <label>Total alerts</label>
            <span class="metric-val" style="color:#e8eaef;">{stats['total']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_alerts(alert_sys: AlertSystem):
    """Newest-first alert cards; scroll after ~8 cards."""
    recent = alert_sys.get_recent(n=80)
    recent = list(reversed(recent))
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
        msg = html_escape(str(a.get("message", "Unknown event")))
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
        '<p class="empty">No active security events.</p>'
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
        '<div class="hdr"><h2>Active Security Events</h2><span class="live">LIVE</span></div>'
        + inner
        + "</div></body></html>"
    )
    components.html(html_doc, height=560, scrolling=False)


def _sidebar_model_info(config):
    st.sidebar.divider()
    st.sidebar.markdown("#### Model")
    st.sidebar.markdown(f"**Detector:** `{config['detector']['model']}`")
    st.sidebar.markdown(f"**Tracker:** `{config['tracker']['type']}`")
    st.sidebar.markdown(f"**Device:** `{config['detector']['device']}`")
    st.sidebar.caption(
        "ML features may be trained on fixed CCTV-style feeds (e.g. UCF); "
        "rules + GMC are tuned for UAV / aerial motion."
    )


def render_sidebar(config, replay: ReplayManager):
    st.sidebar.markdown("### Control Panel")

    mode = st.sidebar.radio(
        "Mode",
        ["Live Video", "Replay Mode"],
        key="mode_select",
    )

    st.sidebar.divider()

    if mode == "Replay Mode":
        st.sidebar.markdown("#### Scenario")
        scenarios = replay.list_scenarios()
        scenario_names = [s.name for s in scenarios]
        sel = st.sidebar.selectbox("Scenario", scenario_names, key="scenario_sel")
        selected = next((s for s in scenarios if s.name == sel), None)

        if selected:
            st.sidebar.caption(selected.description)
            if not selected.exists:
                st.sidebar.warning("Video file not found.")
                st.sidebar.markdown(ReplayManager.get_download_instructions())
                start, stop, clear = render_sidebar_transport()
                _sidebar_model_info(config)
                return None, None, start, stop, clear
            st.sidebar.success("Video ready")

        start, stop, clear = render_sidebar_transport()
        _sidebar_model_info(config)
        source = selected.video_path if selected else None
        return source, sel if selected else None, start, stop, clear

    st.sidebar.markdown("#### Video source")
    video_dir = os.path.join(ROOT, "data", "test_videos")
    try:
        available_videos = [
            f for f in os.listdir(video_dir)
            if f.endswith((".mp4", ".avi", ".mkv", ".mov"))
        ]
    except FileNotFoundError:
        available_videos = []

    src_type = st.sidebar.radio(
        "Input",
        ["Webcam", "Video library", "Manual path"],
    )

    if src_type == "Webcam":
        start, stop, clear = render_sidebar_transport()
        _sidebar_model_info(config)
        return 0, None, start, stop, clear
    if src_type == "Video library":
        if not available_videos:
            st.sidebar.warning("No videos in `data/test_videos`.")
            start, stop, clear = render_sidebar_transport()
            _sidebar_model_info(config)
            return None, None, start, stop, clear
        selected_file = st.sidebar.selectbox("Test videos", available_videos)
        rel_pick = os.path.join("data", "test_videos", selected_file)
        abs_pick = os.path.normpath(os.path.join(ROOT, rel_pick))
        start, stop, clear = render_sidebar_transport()
        _sidebar_model_info(config)
        return abs_pick, None, start, stop, clear

    raw = st.sidebar.text_input("File path", "data/test_videos/test.mp4").strip()
    if not raw:
        start, stop, clear = render_sidebar_transport()
        _sidebar_model_info(config)
        return None, None, start, stop, clear
    path = resolve_user_video_path(raw)
    start, stop, clear = render_sidebar_transport()
    _sidebar_model_info(config)
    return path, None, start, stop, clear


def render_sidebar_transport():
    """Start / Stop / Clear — full-width buttons so labels don't wrap in the narrow sidebar."""
    st.sidebar.markdown(
        '<p class="section-label" style="margin-bottom:0.35rem;">Session control</p>',
        unsafe_allow_html=True,
    )
    start = st.sidebar.button("Start", use_container_width=True, key="transport_start")
    stop = st.sidebar.button("Stop", use_container_width=True, key="transport_stop")
    clear = st.sidebar.button("Clear", use_container_width=True, key="transport_clear")
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
            x=x_vals,
            y=df["score"],
            mode="lines",
            fill="tozeroy",
            line=dict(color=ACCENT, width=2),
            fillcolor="rgba(0, 212, 255, 0.18)",
        )
    )

    x0 = float(x_vals.iloc[0]) if len(x_vals) else 0
    x1 = float(x_vals.iloc[-1]) if len(x_vals) else 1
    fig.add_shape(
        type="line",
        x0=x0,
        y0=0.6,
        x1=x1,
        y1=0.6,
        line=dict(color=CRIT, width=2, dash="dash"),
    )
    fig.add_annotation(
        x=x1,
        y=0.6,
        xref="x",
        yref="y",
        text="Threshold 0.6",
        showarrow=False,
        xanchor="right",
        yshift=8,
        font=dict(size=10, color=CRIT, family="JetBrains Mono, monospace"),
    )

    fig.update_layout(
        title=dict(
            text="THREAT SCORE TIMELINE",
            font=dict(family="JetBrains Mono, monospace", size=13, color=ACCENT),
            x=0,
            xanchor="left",
        ),
        height=200,
        margin=dict(l=48, r=12, t=40, b=36),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7, 11, 20, 0.85)",
        showlegend=False,
        xaxis=dict(
            title="Frame",
            gridcolor="rgba(0,212,255,0.06)",
            tickfont=dict(color="#7d8aa3", size=10),
            zeroline=False,
        ),
        yaxis=dict(
            range=[0, 1],
            title="Score",
            gridcolor="rgba(0,212,255,0.06)",
            tickfont=dict(color="#7d8aa3", size=10),
            zeroline=False,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    init_state()
    config = load_config()
    _log_cfg = config.get("logging", {}) or {}
    _lvl = getattr(
        logging,
        str(_log_cfg.get("level", "INFO")).upper(),
        logging.INFO,
    )
    logging.basicConfig(
        level=_lvl,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

    detector, tracker, history, engine, visualizer, alert_sys, replay, ego = init_components(config)

    render_header()

    source, scenario_name, start, stop, clear = render_sidebar(config, replay)

    if start:
        st.session_state.running = True

    if stop:
        st.session_state.running = False

    if clear:
        alert_sys.clear()
        st.session_state.alerts = []
        st.session_state.threat_history = []
        st.session_state.threat_plot_ema = 0.0
        st.session_state.pipeline_frame_count = 0

    col_video, col_panel = st.columns([3, 1])

    with col_video:
        video_placeholder = st.empty()
        st.markdown('<p class="section-label">THREAT SCORE TIMELINE</p>', unsafe_allow_html=True)
        chart_placeholder = st.empty()

    with col_panel:
        st.markdown('<p class="section-label">Events</p>', unsafe_allow_html=True)
        alert_placeholder = st.empty()

    st.divider()
    metric_placeholder = st.empty()

    if st.session_state.running and source is not None:
        source_path = source if isinstance(source, str) else source
        if isinstance(source_path, str) and not os.path.isfile(source_path) and source_path != 0:
            st.error(f"Video not found: {source_path}")
            st.session_state.running = False
        else:
            src_key = str(source_path) if isinstance(source_path, str) else "webcam:0"
            if st.session_state.get("last_cap_source") != src_key:
                st.session_state.video_frame_pos = 0
                st.session_state.last_cap_source = src_key

            cap = cv2.VideoCapture(source_path)
            if isinstance(source_path, str) and os.path.isfile(source_path):
                pos = int(st.session_state.get("video_frame_pos", 0) or 0)
                if pos > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

            target_w = config["dashboard"].get("frame_width", 960)
            target_h = config["dashboard"].get("frame_height", 540)
            fps_counter_times = []

            ego_dx_hist, ego_dy_hist = [], []
            if ego is not None:
                ego.reset()

            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    st.session_state.video_frame_pos = 0
                    history.clear()
                    tracker.reset()
                    if ego is not None:
                        ego.reset()
                    continue

                if isinstance(source_path, str) and os.path.isfile(source_path):
                    st.session_state.video_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                frame_count = st.session_state.pipeline_frame_count + 1
                st.session_state.pipeline_frame_count = frame_count

                frame = cv2.resize(frame, (target_w, target_h))

                H = None
                ego_dx = ego_dy = 0.0
                ego_rot_deg = 0.0
                ego_scale = 1.0
                if ego is not None:
                    H = ego.estimate(frame)
                    ego_dx = float(H[0, 2])
                    ego_dy = float(H[1, 2])
                    ego_scale, rot_rad = decompose_affine(H)
                    ego_rot_deg = float(np.degrees(rot_rad))
                    history.apply_camera_motion(H)
                    engine.apply_camera_motion(H)

                detections = detector.detect(frame)
                tracks = tracker.update(detections, frame, transform=H)

                active_ids = []
                for t in tracks:
                    history.update(
                        t.track_id, t.bbox, t.center,
                        t.class_id, t.class_name, t.confidence,
                    )
                    active_ids.append(t.track_id)
                history.mark_missing(active_ids)

                if engine.zone_detector:
                    frame = engine.zone_detector.draw_zones(frame)

                alerts, per_track = engine.process(history, target_w, target_h)
                if alerts:
                    alert_sys.add_all(alerts, st.session_state.pipeline_frame_count)

                fps_counter_times.append(time.time())
                fps_counter_times = fps_counter_times[-30:]
                fps = (
                    (len(fps_counter_times) - 1)
                    / max(fps_counter_times[-1] - fps_counter_times[0], 1e-6)
                    if len(fps_counter_times) > 1
                    else 0
                )

                ego_dx_hist.append(ego_dx)
                ego_dy_hist.append(ego_dy)
                ego_dx_hist = ego_dx_hist[-15:]
                ego_dy_hist = ego_dy_hist[-15:]
                ego_dx_avg = float(np.mean(ego_dx_hist)) if ego_dx_hist else 0.0
                ego_dy_avg = float(np.mean(ego_dy_hist)) if ego_dy_hist else 0.0

                extra = {
                    "scenario": scenario_name,
                    "gmc_active": ego is not None,
                    "ego_dx": ego_dx_avg,
                    "ego_dy": ego_dy_avg,
                    "ego_rot_deg": ego_rot_deg,
                    "ego_scale": ego_scale,
                }
                frame_out = visualizer.render(frame, history, alerts, per_track, extra)

                max_rule = max(
                    (float(v.get("rule_score", 0) or 0) for v in per_track.values()),
                    default=0.0,
                )
                max_threat = max(
                    (float(v.get("threat_score", 0) or 0) for v in per_track.values()),
                    default=0.0,
                )
                max_alert = max(
                    (float(getattr(a, "score", 0) or 0) for a in alerts),
                    default=0.0,
                )
                instant = max(max_rule, max_alert, max_threat)
                prev_plot = float(st.session_state.get("threat_plot_ema", 0.0))
                if instant >= prev_plot:
                    plot_val = instant
                else:
                    plot_val = 0.88 * prev_plot + 0.12 * instant
                st.session_state.threat_plot_ema = plot_val
                st.session_state.threat_history.append(
                    (st.session_state.pipeline_frame_count, plot_val),
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
            "Press Start for aerial surveillance…",
            (200, 280),
            20,
            (90, 96, 110),
            stroke_width=1,
            stroke_color=(10, 14, 26),
        )
        video_placeholder.image(placeholder_img, channels="BGR", use_container_width=True)

        with chart_placeholder:
            render_threat_chart(st.session_state.threat_history)

        with alert_placeholder:
            render_alerts(alert_sys)

        with metric_placeholder:
            render_metrics(
                alert_sys,
                0.0,
                0,
                extra_info={"gmc_active": ego is not None, "ego_dx": 0.0, "ego_dy": 0.0},
            )


if __name__ == "__main__":
    main()
