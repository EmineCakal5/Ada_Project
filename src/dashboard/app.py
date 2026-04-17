"""
Streamlit Dashboard — Davranışsal Güvenlik Analiz Sistemi
Canlı video + Alert Paneli + Replay Mode + Metrikler

Çalıştırmak için:
    streamlit run src/dashboard/app.py
"""

import streamlit as st
import cv2
import numpy as np
import yaml
import sys
import os
import time
import json
import threading
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from PIL import Image

# Path ayarla
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.detector.yolo_detector import YOLODetector
from src.tracker.bytetrack_tracker import ByteTracker
from src.tracker.track_history import TrackHistory
from src.tracker.ego_motion import EgoMotionCompensator, decompose_affine
from src.behavior.engine import BehaviorEngine
from src.dashboard.visualizer import Visualizer
from src.dashboard.alert_system import AlertSystem
from src.dashboard.replay_mode import ReplayManager

# ─── Sayfa Yapılandırması ──────────────────────────────────────

st.set_page_config(
    page_title="Davranışsal Güvenlik Analiz Sistemi",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ──────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');

    * { font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4, h5, .font-space { font-family: 'Space Grotesk', sans-serif !important; }

    /* SENTINEL Color Palette & Base */
    .stApp {
        background-color: #111318;
        color: #e2e2e9;
    }
    
    header[data-testid="stHeader"] { display: none; }
    .block-container { padding-top: 2rem; max-width: 1400px; }
    [data-testid="stSidebar"] {
        background-color: #111318 !important;
        border-right: 1px solid rgba(59, 73, 77, 0.2);
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
        color: #bac9ce !important;
    }

    /* Header */
    .sentinel-topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.2rem 2rem;
        background: rgba(17, 19, 24, 0.8);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(59, 73, 77, 0.2);
        border-radius: 8px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .sentinel-title {
        color: #00dcff;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700;
        font-size: 1.4rem;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .sentinel-subtitle {
        color: #bac9ce;
        font-size: 0.65rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        margin-top: 4px;
    }

    /* Alerts */
    .alert-container {
        background: #1a1b21;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid rgba(59, 73, 77, 0.2);
        max-height: 600px;
        overflow-y: auto;
    }
    .alert-card {
        background: #282a2f;
        padding: 0.8rem 1rem;
        margin-bottom: 0.8rem;
        border-radius: 4px;
        position: relative;
        overflow: hidden;
        border-left: 3px solid transparent;
        transition: background 0.2s;
    }
    .alert-card:hover { background: #33353a; }
    
    .critical-bg { border-left-color: #aa0a1b !important; }
    .high-bg { border-left-color: #ffb7b1 !important; }
    .medium-bg { border-left-color: #00a2e6 !important; }
    .low-bg { border-left-color: #859397 !important; }

    .alert-header { display: flex; justify-content: space-between; margin-bottom: 0.4rem; }
    .alert-level {
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        font-family: 'Space Grotesk', sans-serif !important;
    }
    .critical-text { color: #aa0a1b; }
    .high-text { color: #ffb7b1; }
    .medium-text { color: #00a2e6; }
    .low-text { color: #859397; }

    .alert-time { font-size: 0.65rem; font-family: monospace; color: #bac9ce; }
    .alert-title { font-size: 0.85rem; font-weight: 700; color: #e2e2e9; margin: 0 0 0.2rem 0; line-height: 1.2; }
    .alert-desc { font-size: 0.75rem; color: #bac9ce; margin: 0; line-height: 1.4; }

    /* Buttons */
    .stButton > button {
        background-color: transparent !important;
        border: 1px solid rgba(59, 73, 77, 0.5) !important;
        color: #00dcff !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.8rem !important;
        letter-spacing: 1px;
        text-transform: uppercase;
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover {
        background-color: rgba(0, 220, 255, 0.1) !important;
        border-color: #00dcff !important;
        box-shadow: 0 0 10px rgba(0,220,255,0.2) !important;
    }

    /* Video & Chart Overlays */
    .stImage > img { padding: 4px; background: #1a1b21; border-radius: 8px; border: 1px solid rgba(59,73,77,0.3); }
    
    /* Metrics panel */
    .metrics-footer {
        background: #1a1b21;
        border: 1px solid rgba(59,73,77,0.2);
        padding: 1.2rem;
        display: flex;
        justify-content: space-around;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .metric-box { display: flex; flex-direction: column; text-align: center; }
    .metric-box label {
        font-size: 0.65rem;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700;
        color: #bac9ce;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .metric-box .metric-val {
        font-size: 1.8rem;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 900;
        color: #00dcff;
        line-height: 1.2;
        margin-top: 0.2rem;
    }

    @keyframes pulse-dot {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .live-dot {
        height: 6px; width: 6px; background-color: #ffb4ab; border-radius: 50%;
        display: inline-block; margin-right: 6px; animation: pulse-dot 1.5s infinite;
    }
</style>
""", unsafe_allow_html=True)


# ─── Config Yükleme ───────────────────────────────────────────

@st.cache_resource
def load_config():
    config_path = os.path.join(ROOT, "config", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_resource
def init_components(_config):
    """Pipeline bileşenlerini bir kez başlat."""
    detector  = YOLODetector(_config)
    tracker   = ByteTracker(_config)
    if hasattr(detector, 'model') and hasattr(detector.model, 'names'):
        tracker.set_class_names(detector.model.names)
    history   = TrackHistory()
    engine    = BehaviorEngine(_config)
    visualizer = Visualizer(_config)
    alert_sys  = AlertSystem(_config)
    replay    = ReplayManager(_config)

    # Ego-Motion (GMC) — drone/IHA modu icin; sabit kamera modunda None kalir
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


# ─── Session State ────────────────────────────────────────────

def init_state():
    defaults = {
        "running":         False,
        "replay_mode":     False,
        "current_scenario": None,
        "alerts":          [],
        "frame_count":     0,
        "fps":             0.0,
        "track_count":     0,
        "threat_history":  [],   # [(time, score)]
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─── Header ───────────────────────────────────────────────────

def render_header():
    st.markdown("""
    <div class="sentinel-topbar">
        <div>
            <h1 class="sentinel-title">SENTINEL COMMAND</h1>
            <div class="sentinel-subtitle">Behavioral Security Analysis v2.0.4</div>
        </div>
        <div style="display: flex; gap: 1rem; align-items: center;">
            <span class="live-dot"></span>
            <span style="font-family: 'Space Grotesk'; font-weight: 700; color: #ffb4ab; font-size: 0.75rem; letter-spacing: 1px;">SYSTEM LIVE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Metrik Kartlar ───────────────────────────────────────────

def render_metrics(alert_sys: AlertSystem, fps: float, track_count: int,
                   extra_info: dict = None):
    """Alt metrik seridi. GMC / Drone modunda kamera sapmasi ve GMC
    durumunu da canli olarak gosterir."""
    stats = alert_sys.get_stats()

    info = extra_info or {}
    gmc_active = bool(info.get("gmc_active"))
    dx = float(info.get("ego_dx", 0.0))
    dy = float(info.get("ego_dy", 0.0))
    drift_mag = (dx * dx + dy * dy) ** 0.5

    # GMC status gorsel durumu
    if gmc_active:
        gmc_label = "ACTIVE"
        gmc_color = "#00ffa5"  # siber yesil
        gmc_dot   = "● "
    else:
        gmc_label = "OFFLINE"
        gmc_color = "#859397"
        gmc_dot   = "○ "

    # Ciddi sapma rengi (> ~3 px/frame dikkatli)
    drift_color = "#00dcff" if drift_mag < 3.0 else "#ffaa00"

    st.markdown(f"""
    <div class="metrics-footer">
        <div class="metric-box">
            <label>Active Trackers</label>
            <span class="metric-val" style="color: #00dcff;">{track_count}</span>
        </div>
        <div class="metric-box">
            <label>FPS Rate</label>
            <span class="metric-val" style="color: #e2e2e9;">{fps:.1f}</span>
        </div>
        <div class="metric-box">
            <label>Camera Drift (px)</label>
            <span class="metric-val" style="color: {drift_color};">{drift_mag:.2f}</span>
            <span style="font-family:'Space Grotesk';font-size:0.6rem;color:#bac9ce;letter-spacing:0.1em;">
                dx {dx:+.2f} &nbsp; dy {dy:+.2f}
            </span>
        </div>
        <div class="metric-box">
            <label>GMC Status</label>
            <span class="metric-val" style="color: {gmc_color};font-size:1.3rem;">{gmc_dot}{gmc_label}</span>
        </div>
        <div class="metric-box">
            <label>Critical Alerts</label>
            <span class="metric-val" style="color: #aa0a1b;">{stats['critical']}</span>
        </div>
        <div class="metric-box">
            <label>Total Events</label>
            <span class="metric-val" style="color: #e2e2e9;">{stats['total']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Alert Paneli ────────────────────────────────────────────

def render_alerts(alert_sys: AlertSystem):
    recent = alert_sys.get_recent(n=15)
    
    html = '<div class="alert-container">'
    html += '''
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem; border-bottom: 1px solid rgba(59,73,77,0.2); padding-bottom: 0.5rem;">
            <h2 style="font-size: 0.85rem; font-family: 'Space Grotesk'; font-weight: 700; margin: 0; color: #e2e2e9; text-transform: uppercase; letter-spacing: 1px;">Active Security Events</h2>
            <span style="background: #93000a; color: #ffdad6; font-size: 0.6rem; padding: 2px 6px; border-radius: 4px; font-weight: bold; animation: pulse-dot 2s infinite;">LIVE</span>
        </div>
    '''

    if not recent:
        html += '<p style="color: #bac9ce; font-size: 0.8rem;">No anomalous events detected.</p></div>'
        st.markdown(html, unsafe_allow_html=True)
        return

    for a in reversed(recent):
        level_str = str(a.get("threat_level", "LOW")).upper()
        
        bg_class = "low-bg"
        text_class = "low-text"
        if level_str == "CRITICAL":
            bg_class = "critical-bg"
            text_class = "critical-text"
        elif level_str == "HIGH":
            bg_class = "high-bg"
            text_class = "high-text"
        elif level_str in ["MEDIUM", "MODERATE"]:
            bg_class = "medium-bg"
            text_class = "medium-text"
            
        time_str = a.get("time_str", "00:00:00")
        msg = a.get("message", "Unknown Event")
        title = msg.split(':', 1)[0] if ':' in msg else f"{level_str} EVENT"
        desc = msg.split(':', 1)[1].strip() if ':' in msg else msg
        
        html += f"""
        <div class="alert-card {bg_class}">
            <div class="alert-header">
                <span class="alert-level {text_class}">[{level_str}]</span>
                <span class="alert-time">{time_str}</span>
            </div>
            <p class="alert-title">{title}</p>
            <p class="alert-desc">{desc}</p>
        </div>
        """
        
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ─── Senaryo Sidebar ──────────────────────────────────────────

def render_sidebar(config, replay: ReplayManager):
    st.sidebar.markdown("### ⚙️ Kontrol Paneli")

    mode = st.sidebar.radio(
        "Mod",
        ["🎥 Canlı Video", "▶ Replay Mode"],
        key="mode_select"
    )

    st.sidebar.divider()

    if "Replay" in mode:
        st.sidebar.markdown("#### 📼 Senaryo Seçimi")
        scenarios = replay.list_scenarios()
        scenario_names = [s.name for s in scenarios]
        sel = st.sidebar.selectbox("Senaryo", scenario_names, key="scenario_sel")
        selected = next((s for s in scenarios if s.name == sel), None)

        if selected:
            st.sidebar.markdown(f"*{selected.description}*")
            if not selected.exists:
                st.sidebar.warning("⚠️ Video dosyası bulunamadı!")
                st.sidebar.markdown(ReplayManager.get_download_instructions())
                return None, None
            else:
                st.sidebar.success("✅ Video hazır")

        source = selected.video_path if selected else None
        return source, sel if selected else None
    else:
        st.sidebar.markdown("#### 📷 Video Kaynağı")
        
        video_dir = os.path.join(ROOT, "data", "test_videos")
        try:
            available_videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))]
        except FileNotFoundError:
            available_videos = []
            
        src_type = st.sidebar.radio("Kaynak", ["Webcam", "Videolarım (Klasörden Seç)", "Manuel Yol Yaz"])
        
        if src_type == "Webcam":
            return 0, None
        elif src_type == "Videolarım (Klasörden Seç)":
            if not available_videos:
                st.sidebar.warning("data/test_videos klasöründe hiç video bulunamadı.")
                return None, None
            selected_file = st.sidebar.selectbox("Test Videoları", available_videos)
            return os.path.join("data", "test_videos", selected_file), None
        else:
            path = st.sidebar.text_input("Video Yolu", "data/test_videos/test.mp4")
            return path, None

    st.sidebar.divider()
    st.sidebar.markdown("#### 🧠 Model")
    st.sidebar.markdown(f"**Detector**: {config['detector']['model']}")
    st.sidebar.markdown(f"**Tracker**: {config['tracker']['type']}")
    st.sidebar.markdown(f"**Device**: {config['detector']['device']}")


# ─── Threat Skoru Grafiği ─────────────────────────────────────

def render_threat_chart(history: list):
    if len(history) < 2:
        return
    df = pd.DataFrame(history[-60:], columns=["time", "score"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["score"],
        fill="tozeroy",
        line=dict(color="#00dcff", width=2),
        fillcolor="rgba(0,220,255,0.1)"
    ))
    fig.update_layout(
        height=150,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.2)",
        showlegend=False,
        xaxis=dict(showticklabels=False, gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(range=[0, 1], tickfont=dict(color="#94a3b8", size=10),
                   gridcolor="rgba(255,255,255,0.05)")
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── Ana Uygulama ─────────────────────────────────────────────

def main():
    init_state()
    config = load_config()
    detector, tracker, history, engine, visualizer, alert_sys, replay, ego = init_components(config)

    render_header()

    # Sidebar
    source, scenario_name = render_sidebar(config, replay)

    # Ana layout
    col_video, col_panel = st.columns([3, 1])

    with col_video:
        video_placeholder = st.empty()
        st.markdown("##### 📊 Anlık Tehdit Skoru")
        chart_placeholder = st.empty()

    with col_panel:
        st.markdown("##### 🚨 Alert Geçmişi")
        alert_placeholder = st.empty()

    # Metrik satırı
    st.divider()
    metric_placeholder = st.empty()

    # Kontrol butonları
    btn_col1, btn_col2, btn_col3, _ = st.columns([1, 1, 1, 3])
    with btn_col1:
        start = st.button("▶ Başlat", use_container_width=True)
    with btn_col2:
        stop = st.button("⏹ Durdur", use_container_width=True)
    with btn_col3:
        clear = st.button("🗑 Temizle", use_container_width=True)

    if start:
        st.session_state.running = True
        history.clear()
        tracker.reset()

    if stop:
        st.session_state.running = False

    if clear:
        alert_sys.clear()
        st.session_state.alerts = []
        st.session_state.threat_history = []

    # ─── Video İşleme Döngüsü ──────────────────────────────
    if st.session_state.running and source is not None:
        source_path = source if isinstance(source, str) else source
        if isinstance(source_path, str) and not os.path.isfile(source_path) and source_path != 0:
            st.error(f"Video bulunamadı: {source_path}")
            st.session_state.running = False
        else:
            cap = cv2.VideoCapture(source_path)
            target_w = config["dashboard"].get("frame_width", 960)
            target_h = config["dashboard"].get("frame_height", 540)
            fps_counter_times = []

            frame_count = 0
            # Ego-motion kisa vadeli ortalama: HUD/metrikler icin stabil gozukur
            ego_dx_hist, ego_dy_hist = [], []
            if ego is not None:
                ego.reset()

            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    history.clear()
                    tracker.reset()
                    if ego is not None:
                        ego.reset()
                    continue

                frame_count += 1
                t0 = time.time()

                # Boyut ayarla
                frame = cv2.resize(frame, (target_w, target_h))

                # 0. EGO-MOTION (Drone/IHA). Sabit kamera modunda no-op.
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

                # 1. Pipeline
                detections = detector.detect(frame)
                tracks = tracker.update(detections, frame, transform=H)

                active_ids = []
                for t in tracks:
                    history.update(t.track_id, t.bbox, t.center,
                                   t.class_id, t.class_name, t.confidence)
                    active_ids.append(t.track_id)
                history.mark_missing(active_ids)

                # Zone overlay (dinamik capa isaretleriyle birlikte)
                if engine.zone_detector:
                    frame = engine.zone_detector.draw_zones(frame)

                alerts, per_track = engine.process(history, target_w, target_h)
                if alerts:
                    alert_sys.add_all(alerts, frame_count)

                # FPS
                fps_counter_times.append(time.time())
                fps_counter_times = fps_counter_times[-30:]
                fps = (len(fps_counter_times)-1)/max(fps_counter_times[-1]-fps_counter_times[0], 1e-6) if len(fps_counter_times) > 1 else 0

                # Ego-motion ortalamasi (son ~15 frame) — HUD titremesin diye
                ego_dx_hist.append(ego_dx); ego_dy_hist.append(ego_dy)
                ego_dx_hist = ego_dx_hist[-15:]; ego_dy_hist = ego_dy_hist[-15:]
                ego_dx_avg = float(np.mean(ego_dx_hist)) if ego_dx_hist else 0.0
                ego_dy_avg = float(np.mean(ego_dy_hist)) if ego_dy_hist else 0.0

                extra = {
                    "scenario":    scenario_name,
                    "gmc_active":  ego is not None,
                    "ego_dx":      ego_dx_avg,
                    "ego_dy":      ego_dy_avg,
                    "ego_rot_deg": ego_rot_deg,
                    "ego_scale":   ego_scale,
                }
                frame_out = visualizer.render(frame, history, alerts, per_track, extra)

                # Threat history
                max_score = max((v.get("threat_score", 0) for v in per_track.values()), default=0)
                st.session_state.threat_history.append((frame_count, max_score))
                st.session_state.threat_history = st.session_state.threat_history[-120:]

                # Streamlit güncelle
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
        # Bekleme ekranı
        placeholder_img = np.zeros((540, 960, 3), dtype=np.uint8)
        cv2.putText(placeholder_img, "Baslat butonuna basin...", (300, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 100), 2)
        video_placeholder.image(placeholder_img, channels="BGR", use_container_width=True)

        with alert_placeholder:
            render_alerts(alert_sys)

        with metric_placeholder:
            render_metrics(alert_sys, 0.0, 0,
                           extra_info={"gmc_active": ego is not None,
                                       "ego_dx": 0.0, "ego_dy": 0.0})


if __name__ == "__main__":
    main()
