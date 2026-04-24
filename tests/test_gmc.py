# -*- coding: utf-8 -*-
"""
GMC (Global Motion Compensation) Smoke Testleri
================================================
Ada_Project / Drone - IHA modulu.

Bu testler:
  1) EgoMotionCompensator'in bilinen bir kayma (shift) icin dogru affine
     ciktigini dogrular.
  2) KalmanBoxTracker.apply_camera_motion, TrackHistory.apply_camera_motion,
     Zone.update_polygon, Loitering.apply_camera_motion ve
     AbandonedObjectDetector.apply_camera_motion zincirlerinin bozulmadan
     calistigini ve koordinatlari dogru warp ettigini dogrular.

Cikti: terminale ✓ / ✗ satirlari. Tek bir hata varsa exit code 1.

Calistirma:
    python tests/test_gmc.py
"""

from __future__ import annotations

import os
import sys
import time
import tempfile
import traceback
from pathlib import Path

import numpy as np
import cv2

# Proje kokunu path'e ekle
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.tracker.ego_motion import EgoMotionCompensator, warp_points
from src.tracker.bytetrack_tracker import KalmanBoxTracker
from src.tracker.track_history import TrackHistory


# ---------------------------------------------------------------------------
# Yardimcilar
# ---------------------------------------------------------------------------
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    RESULTS.append((name, cond, extra))
    mark = "[OK] " if cond else "[FAIL]"
    msg = f"{mark} {name}"
    if extra:
        msg += f"  ({extra})"
    print(msg)


def make_textured_frame(seed: int = 42, size=(540, 960)) -> np.ndarray:
    """
    goodFeaturesToTrack'in bol kose bulabilecegi rastgele ama tekrar
    uretilebilir bir desen. 2D perlin-benzeri karelik mozaik + gurultu.
    """
    rng = np.random.default_rng(seed)
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # 20x20 renkli kareler
    step = 20
    for y in range(0, h, step):
        for x in range(0, w, step):
            color = rng.integers(40, 230, size=3)
            img[y:y + step, x:x + step] = color
    # Hafif gurultu (LK'yi gercekci test etmek icin)
    noise = rng.integers(-8, 8, size=img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def shift_frame(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Bilinen bir piksel kaymasi uygular (kamera panoramasini simule eder)."""
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        img, M, (img.shape[1], img.shape[0]),
        borderMode=cv2.BORDER_REFLECT
    )


# ---------------------------------------------------------------------------
# TEST 1 — EgoMotionCompensator'in bilinen kayma icin dogrulugu
# ---------------------------------------------------------------------------
def test_ego_motion_known_shift() -> None:
    print("\n--- TEST 1: EgoMotion bilinen kayma tahmini ---")
    frame0 = make_textured_frame(seed=1)
    DX, DY = 12, -7
    frame1 = shift_frame(frame0, DX, DY)

    ego = EgoMotionCompensator(downscale=0.5, max_features=500)

    H_first = ego.estimate(frame0)                 # ilk frame -> identity
    check(
        "ilk frame -> identity",
        np.allclose(H_first, np.eye(2, 3), atol=1e-5),
    )

    H = ego.estimate(frame1)                       # ikinci frame -> (DX, DY)
    tx, ty = float(H[0, 2]), float(H[1, 2])
    scale_x = float(H[0, 0])
    rot = float(np.arctan2(H[1, 0], H[0, 0]))

    # 1 piksel tolerans (downscale + LK subpixel + borderReflect nedeniyle)
    check(
        "tx ~= DX (+-1 px)",
        abs(tx - DX) < 1.5,
        f"tx={tx:.2f} DX={DX}",
    )
    check(
        "ty ~= DY (+-1 px)",
        abs(ty - DY) < 1.5,
        f"ty={ty:.2f} DY={DY}",
    )
    check(
        "scale ~= 1.0",
        abs(scale_x - 1.0) < 0.02,
        f"a00={scale_x:.4f}",
    )
    check(
        "rotation ~= 0",
        abs(rot) < 0.02,
        f"rot_rad={rot:.4f}",
    )


# ---------------------------------------------------------------------------
# TEST 2 — warp_points helper
# ---------------------------------------------------------------------------
def test_warp_points() -> None:
    print("\n--- TEST 2: warp_points helper ---")
    H = np.array([[1, 0, 10], [0, 1, -5]], dtype=np.float32)
    pts = np.array([[0, 0], [100, 200]], dtype=np.float32)
    out = warp_points(pts, H)
    check(
        "warp_points shift dogru",
        np.allclose(out, [[10, -5], [110, 195]]),
        f"out={out.tolist()}",
    )


# ---------------------------------------------------------------------------
# TEST 3 — KalmanBoxTracker.apply_camera_motion
# ---------------------------------------------------------------------------
def test_kalman_warp() -> None:
    print("\n--- TEST 3: KalmanBoxTracker.apply_camera_motion ---")
    KalmanBoxTracker.count = 0
    trk = KalmanBoxTracker([100, 100, 200, 200])   # cx=150, cy=150, w=100, h=100
    H = np.array([[1, 0, 20], [0, 1, 30]], dtype=np.float32)
    trk.apply_camera_motion(H)
    state = trk.kf.x.flatten()
    cx, cy, w, h = state[:4]
    check("Kalman cx 20 px shift", abs(cx - 170) < 1e-3, f"cx={cx}")
    check("Kalman cy 30 px shift", abs(cy - 180) < 1e-3, f"cy={cy}")
    check("Kalman w preserved (scale=1)", abs(w - 100) < 1e-3, f"w={w}")
    check("Kalman h preserved (scale=1)", abs(h - 100) < 1e-3, f"h={h}")


# ---------------------------------------------------------------------------
# TEST 4 — TrackHistory trajectory warp
# ---------------------------------------------------------------------------
def test_track_history_warp() -> None:
    print("\n--- TEST 4: TrackHistory.apply_camera_motion ---")
    hist = TrackHistory(max_lost_frames=5)
    # 3 frame boyunca ayni yerde duran bir track olustur
    hist.update(1, [50, 50, 100, 100], (75.0, 75.0), class_id=0, class_name="person")
    hist.update(1, [50, 50, 100, 100], (75.0, 75.0), class_id=0, class_name="person")
    hist.update(1, [50, 50, 100, 100], (75.0, 75.0), class_id=0, class_name="person")

    H = np.array([[1, 0, 10], [0, 1, 20]], dtype=np.float32)
    hist.apply_camera_motion(H)

    t = hist.get(1)
    check("center shift (cx)", abs(t.center[0] - 85) < 1e-3, f"cx={t.center[0]}")
    check("center shift (cy)", abs(t.center[1] - 95) < 1e-3, f"cy={t.center[1]}")
    check("bbox x1 shift", abs(t.bbox[0] - 60) < 1e-3, f"x1={t.bbox[0]}")
    check("bbox y2 shift", abs(t.bbox[3] - 120) < 1e-3, f"y2={t.bbox[3]}")
    # Trajectory'deki tum noktalar warp edilmeli
    last = t.trajectory[-1]
    check("trajectory son nokta warp", abs(last[0] - 85) < 1e-3 and abs(last[1] - 95) < 1e-3,
          f"last={last}")


# ---------------------------------------------------------------------------
# TEST 5 — Zone.update_polygon + ZoneViolationDetector
# ---------------------------------------------------------------------------
def test_zone_warp(tmp_dir: Path) -> None:
    print("\n--- TEST 5: Zone.update_polygon + Detector.apply_camera_motion ---")
    import json
    zones_json = {
        "frame_width": 960,
        "frame_height": 540,
        "zones": [
            {
                "id": "z1",
                "name": "Test",
                "type": "restricted",
                "alert_on_entry": True,
                "points": [[100, 100], [200, 100], [200, 200], [100, 200]],
            }
        ],
    }
    zones_path = tmp_dir / "zones.json"
    zones_path.write_text(json.dumps(zones_json), encoding="utf-8")

    from src.behavior.rules.zone_violation import ZoneViolationDetector
    det = ZoneViolationDetector(str(zones_path), frame_w=960, frame_h=540)
    zone = det.zones[0]

    # Warp oncesi (150,150) zone icinde olmali
    check("zone baslangic iceri", zone.contains_point((150.0, 150.0)))

    # Kaymayi 51 piksel yapiyoruz ki test noktasi zone sinirina dusmesin
    # (pointPolygonTest sinir icin 0 doner ve bizim >=0 kontrolumuz iceride sayar).
    H = np.array([[1, 0, 51], [0, 1, 51]], dtype=np.float32)
    det.apply_camera_motion(H)

    # Warp sonrasi zone (151,151) -> (251,251)'e kaydi:
    check("zone warp sonrasi (150,150) DIsarida", not zone.contains_point((150.0, 150.0)))
    check("zone warp sonrasi (200,200) ICeride", zone.contains_point((200.0, 200.0)))


# ---------------------------------------------------------------------------
# TEST 6 — Loitering anchor warp
# ---------------------------------------------------------------------------
def test_loitering_anchor_warp() -> None:
    print("\n--- TEST 6: LoiteringDetector.apply_camera_motion ---")
    from src.behavior.rules.loitering import LoiteringDetector

    cfg = {"behavior": {"loitering": {"threshold_seconds": 60, "min_displacement": 50}}}
    loi = LoiteringDetector(cfg)

    # Anchor olustur
    loi.check(track_id=7, center=(100.0, 100.0), dwell_time=0.1, velocity=0.0)
    anchor_before = loi._states[7]["anchor"]
    check("anchor baslangic", anchor_before == (100.0, 100.0), f"{anchor_before}")

    H = np.array([[1, 0, 30], [0, 1, -10]], dtype=np.float32)
    loi.apply_camera_motion(H)

    anchor_after = loi._states[7]["anchor"]
    check(
        "anchor warp sonrasi",
        abs(anchor_after[0] - 130.0) < 1e-3 and abs(anchor_after[1] - 90.0) < 1e-3,
        f"{anchor_after}",
    )


# ---------------------------------------------------------------------------
# TEST 7 — AbandonedObject merkez warp
# ---------------------------------------------------------------------------
def test_abandoned_warp() -> None:
    print("\n--- TEST 7: AbandonedObjectDetector.apply_camera_motion ---")
    from src.behavior.rules.abandoned_object import AbandonedObjectDetector
    from src.tracker.track_history import TrackInfo
    from collections import deque

    cfg = {"behavior": {"abandoned_object": {"owner_distance": 150, "confirm_seconds": 10}}}
    det = AbandonedObjectDetector(cfg)

    # Sahte bir "backpack" track'i (abandoned aday olmasi icin state olustur)
    obj = TrackInfo(
        track_id=42, class_id=24, class_name="backpack",
        bbox=[300, 300, 340, 340], center=(320.0, 320.0),
    )
    det.check([obj])                                   # ilk gormek -> state olusur
    before = det._object_states[42]["center"]
    check("abandoned merkez baslangic", before == (320.0, 320.0), f"{before}")

    H = np.array([[1, 0, -40], [0, 1, 15]], dtype=np.float32)
    det.apply_camera_motion(H)

    after = det._object_states[42]["center"]
    check(
        "abandoned merkez warp sonrasi",
        abs(after[0] - 280.0) < 1e-3 and abs(after[1] - 335.0) < 1e-3,
        f"{after}",
    )


# ---------------------------------------------------------------------------
# TEST 8 — BehaviorEngine.apply_camera_motion yayimi
# ---------------------------------------------------------------------------
def test_engine_propagation(tmp_dir: Path) -> None:
    print("\n--- TEST 8: BehaviorEngine.apply_camera_motion zinciri ---")
    import json
    zones_json = {
        "frame_width": 960, "frame_height": 540,
        "zones": [{
            "id": "z1", "name": "Test", "type": "restricted",
            "alert_on_entry": True,
            "points": [[100, 100], [200, 100], [200, 200], [100, 200]],
        }],
    }
    zones_path = tmp_dir / "engine_zones.json"
    zones_path.write_text(json.dumps(zones_json), encoding="utf-8")

    from src.behavior.engine import BehaviorEngine
    cfg = {
        "behavior": {
            "zones_file": str(zones_path),
            "loitering": {"threshold_seconds": 60, "min_displacement": 50},
            "abandoned_object": {"owner_distance": 150, "confirm_seconds": 10},
            "threat_scorer": {"weights": {
                "zone_violation": 0.35, "loitering": 0.25,
                "abandoned_object": 0.30, "velocity_anomaly": 0.10,
            }},
        },
        "mlp": {"input_dim": 8, "hidden_dims": [32, 16], "output_dim": 4,
                "model_path": "models/weights/threat_mlp.pt", "threshold_train": 200},
    }
    eng = BehaviorEngine(cfg, zones_path=str(zones_path))
    eng._init_zones(960, 540)

    # Loitering + abandoned state'i uret
    eng.loitering.check(1, (100.0, 100.0), 0.1, 0.0)
    from src.tracker.track_history import TrackInfo
    obj = TrackInfo(track_id=2, class_id=24, class_name="backpack",
                    bbox=[300, 300, 340, 340], center=(320.0, 320.0))
    eng.abandoned.check([obj])

    H = np.array([[1, 0, 10], [0, 1, 20]], dtype=np.float32)
    eng.apply_camera_motion(H)

    zone_pt = eng.zone_detector.zones[0].points[0].tolist()
    check("engine -> zone noktasi warped", zone_pt == [110, 120], f"{zone_pt}")

    anch = eng.loitering._states[1]["anchor"]
    check("engine -> loitering anchor warped",
          abs(anch[0] - 110.0) < 1e-3 and abs(anch[1] - 120.0) < 1e-3,
          f"{anch}")

    cent = eng.abandoned._object_states[2]["center"]
    check("engine -> abandoned merkez warped",
          abs(cent[0] - 330.0) < 1e-3 and abs(cent[1] - 340.0) < 1e-3,
          f"{cent}")


# ---------------------------------------------------------------------------
# TEST 9 — Performans sanity check (Jetson profili): 540p @ downscale=0.5
# ---------------------------------------------------------------------------
def test_ego_motion_perf() -> None:
    print("\n--- TEST 9: Performans (540p) ---")
    frame0 = make_textured_frame(seed=3, size=(540, 960))
    ego = EgoMotionCompensator(downscale=0.5, max_features=500)
    ego.estimate(frame0)   # warm-up

    N = 30
    t0 = time.perf_counter()
    for i in range(N):
        frame_i = shift_frame(frame0, i % 5, -(i % 3))
        ego.estimate(frame_i)
    dt = time.perf_counter() - t0
    ms = dt * 1000.0 / N
    print(f"  Ortalama: {ms:.2f} ms/frame ({1000.0/ms:.1f} FPS, {N} frame)")
    # Uyarilabilir esik; basarisizlik olarak ele almiyoruz, sadece bilgi.
    check("540p @ downscale=0.5 < 50ms/frame (bilgilendirme)", ms < 50, f"{ms:.1f} ms")


# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 60)
    print("  GMC (Global Motion Compensation) Smoke Test Paketi")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        try:
            test_ego_motion_known_shift()
            test_warp_points()
            test_kalman_warp()
            test_track_history_warp()
            test_zone_warp(tmp)
            test_loitering_anchor_warp()
            test_abandoned_warp()
            test_engine_propagation(tmp)
            test_ego_motion_perf()
        except Exception:
            print("\n[FATAL] Test beklenmedik sekilde coktu:")
            traceback.print_exc()
            return 2

    print("\n" + "=" * 60)
    total = len(RESULTS)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = total - passed
    print(f"  Toplam: {total}   Gecti: {passed}   Basarisiz: {failed}")
    print("=" * 60)
    if failed:
        print("\nBaşarisiz testler:")
        for name, ok, extra in RESULTS:
            if not ok:
                print(f"  - {name}  {extra}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
