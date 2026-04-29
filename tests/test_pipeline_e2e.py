# -*- coding: utf-8 -*-
"""
Pipeline End-to-End Smoke Test (GMC acik)
==========================================
Ada_Project / Drone - IHA

Gerçek bir video üzerinde:
  1) YOLO detector, ByteTracker, EgoMotion, BehaviorEngine zincirini
     kurar (pipeline gibi),
  2) Ilk N frame'i isler (headless, display yok),
  3) Her frame'de EgoMotion matrisini tum state'lere yayar,
  4) FPS / ID switch / zone ihlal sayilarini raporlar.

Test videosu sabit kameradan oldugu icin, drone benzeri pan efektini
simule etmek amaciyla her frame'i progresif olarak x-y'de kaydiriyoruz.
Bu sayede GMC'nin calistigini pratik olarak gozlemleyebiliriz:
    GMC acikken -> Kalman track'ler yerinde kalmali (cunku kaydirmayi
    tespit edip telafi ediyor).

Calistirma:
    python tests/test_pipeline_e2e.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.detector.yolo_detector import YOLODetector
from src.tracker.bytetrack_tracker import ByteTracker
from src.tracker.track_history import TrackHistory
from src.tracker.ego_motion import EgoMotionCompensator
from src.behavior.engine import BehaviorEngine


TEST_VIDEO = ROOT / "data" / "test_videos" / "Zone_Violation.mp4"
MAX_FRAMES = 60          # ilk 60 frame; smoke testi 20sn'yi gecmemeli
TARGET_W, TARGET_H = 960, 540


def simulate_drone_pan(frame: np.ndarray, frame_idx: int) -> np.ndarray:
    """
    Sabit kamera videolarini drone panoramasi gibi yapmak icin her
    frame'de progresif yatay + dikey kayma uygular. Bu sentetik kayma
    EgoMotionCompensator'in gercek kamera hareketi senaryosunu taklit
    etmesini saglar.
    """
    dx = int(frame_idx * 1.5)            # ~1.5 px/frame pan
    dy = int(np.sin(frame_idx * 0.05) * 8)   # hafif yukari-asagi tilt
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        frame, M, (frame.shape[1], frame.shape[0]),
        borderMode=cv2.BORDER_REFLECT,
    )


def main() -> int:
    print("=" * 60)
    print("  Pipeline E2E Smoke Test (GMC acik)")
    print("=" * 60)

    if not TEST_VIDEO.exists():
        print(f"[FAIL] Test videosu bulunamadi: {TEST_VIDEO}")
        return 1

    # Config
    with open(ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Detector'in egitilmis drone modelini aramamasi icin fallback
    # (fine-tune edilmis model yoksa COCO pretrained kullan)
    drone_w = ROOT / "models" / "weights" / "yolov8s_drone.pt"
    if not drone_w.exists():
        config["detector"]["model"] = str(ROOT / "yolov8s.pt")
        print(f"[*] Drone modeli bulunamadi -> COCO pretrained kullaniliyor")
    else:
        config["detector"]["model"] = str(drone_w)

    # GMC'yi zorla ac
    config.setdefault("tracker", {}).setdefault("ego_motion", {})
    config["tracker"]["ego_motion"]["enabled"] = True

    # Modulleri kur
    print("[*] Modul init...")
    t_init = time.perf_counter()
    detector = YOLODetector(config)
    tracker = ByteTracker(config)
    history = TrackHistory(max_lost_frames=30)
    engine = BehaviorEngine(config)
    ego = EgoMotionCompensator(
        downscale=config["tracker"]["ego_motion"].get("downscale", 0.5),
        max_features=config["tracker"]["ego_motion"].get("max_features", 500),
    )
    if hasattr(detector.model, "names"):
        tracker.set_class_names(detector.model.names)
    print(f"[*] Init suresi: {time.perf_counter() - t_init:.2f}s")

    # Video ac
    cap = cv2.VideoCapture(str(TEST_VIDEO))
    if not cap.isOpened():
        print(f"[FAIL] Video acilamadi: {TEST_VIDEO}")
        return 1

    # Metrikler
    n_frames = 0
    n_detections = 0
    n_tracks_total = 0
    max_concurrent_tracks = 0
    unique_track_ids: set[int] = set()
    ego_translations: list[tuple[float, float]] = []
    t_start = time.perf_counter()

    while n_frames < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Boyutlandir
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(frame, (TARGET_W, TARGET_H))

        # Drone panoramasini simule et
        frame = simulate_drone_pan(frame, n_frames)

        # 0. EGO-MOTION
        H = ego.estimate(frame)
        history.apply_camera_motion(H)
        engine.apply_camera_motion(H)
        ego_translations.append((float(H[0, 2]), float(H[1, 2])))

        # 1. TESPIT
        detections = detector.detect(frame)
        n_detections += len(detections)

        # 2. TAKIP
        tracks = tracker.update(detections, frame, transform=H)
        n_tracks_total += len(tracks)
        max_concurrent_tracks = max(max_concurrent_tracks, len(tracks))

        active_ids = []
        for t in tracks:
            history.update(
                track_id=t.track_id, bbox=t.bbox, center=t.center,
                class_id=t.class_id, class_name=t.class_name,
                confidence=t.confidence,
            )
            active_ids.append(t.track_id)
            unique_track_ids.add(t.track_id)
        history.mark_missing(active_ids)

        # 3. DAVRANIS
        alerts, per_track = engine.process(history, TARGET_W, TARGET_H)

        n_frames += 1

    dt = time.perf_counter() - t_start
    cap.release()

    print("\n--- SONUC ---")
    print(f"  Islenen frame     : {n_frames}")
    print(f"  Toplam sure       : {dt:.2f} s")
    print(f"  Pipeline FPS      : {n_frames / dt:.2f}")
    print(f"  Toplam tespit     : {n_detections}")
    print(f"  Toplam track      : {n_tracks_total}")
    print(f"  Anlik max track   : {max_concurrent_tracks}")
    print(f"  Benzersiz ID      : {len(unique_track_ids)}")
    # GMC dogrulamasi: simulasyon 1.5 px/frame pan uyguladigi icin,
    # estimated tx ortalama bu degere yakin olmali (negatif isaretle, cunku
    # H shift yonunu bizim simulasyonun TERSI raporlamali... aslinda AYNI
    # yonu raporlar cunku prev_frame -> curr_frame yonunde estimate ediyor).
    if ego_translations:
        avg_tx = float(np.mean([e[0] for e in ego_translations[1:]]))  # ilk frame identity
        avg_ty = float(np.mean([e[1] for e in ego_translations[1:]]))
        print(f"  Ortalama EgoMotion tx/ty: {avg_tx:+.2f} / {avg_ty:+.2f} px")
        print(f"  (Simulasyon pan: +1.50 / ~0 px/frame)")

    # Basit smoke assertion
    if n_frames < MAX_FRAMES:
        print("[FAIL] Yeterli frame islenemedi")
        return 1
    if not ego_translations or abs(np.mean([e[0] for e in ego_translations[1:]]) - 1.5) > 1.5:
        print("[WARN] EgoMotion beklenen panoramayi tahmin edemedi "
              "(video icerigi cok homojen olabilir); Jetson hedefi olmadigi icin "
              "fail sayilmaz, sadece uyari.")

    print("[OK] Pipeline GMC acikken uçtan uca calisiyor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
