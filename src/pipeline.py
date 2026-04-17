"""
Pipeline — Uçtan uca video → tespit → takip → davranış → görselleştirme.
Sprint 1 sonunda çalışan demo.  Sprint 2 sonunda davranış analizi eklenir.
"""

import cv2
import yaml
import time
import logging
import argparse
import sys
import os

# Proje kökünü sys.path'e ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector.yolo_detector import YOLODetector
from src.tracker.bytetrack_tracker import ByteTracker
from src.tracker.track_history import TrackHistory
from src.tracker.ego_motion import EgoMotionCompensator
from src.behavior.engine import BehaviorEngine
from src.dashboard.visualizer import Visualizer
from src.dashboard.alert_system import AlertSystem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Pipeline:
    """
    Ana pipeline sınıfı.

    Kullanım:
        pipeline = Pipeline(config)
        pipeline.run(source)     # 0=webcam, "path/to/video.mp4"
    """

    def __init__(self, config: dict):
        self.config = config
        logger.info("Pipeline başlatılıyor...")

        # Modüller
        self.detector  = YOLODetector(config)
        self.tracker   = ByteTracker(config)
        self.history   = TrackHistory(max_lost_frames=30)
        self.engine    = BehaviorEngine(config)
        self.visualizer = Visualizer(config)
        self.alert_sys  = AlertSystem(config)

        # ─── Ego-Motion (Drone / IHA) ─────────────────────────────
        # Sabit kamera modunda enabled=false birakilabilir, ekstra yuk olmaz.
        ego_cfg = (config.get("tracker", {}) or {}).get("ego_motion", {}) or {}
        self.ego_enabled: bool = bool(ego_cfg.get("enabled", False))
        if self.ego_enabled:
            self.ego = EgoMotionCompensator(
                downscale=float(ego_cfg.get("downscale", 0.5)),
                max_features=int(ego_cfg.get("max_features", 500)),
                grid=tuple(ego_cfg.get("grid", [4, 4])),
                ransac_thresh=float(ego_cfg.get("ransac_thresh", 3.0)),
            )
            logger.info("Ego-Motion (GMC) aktif — drone/IHA modu")
        else:
            self.ego = None

        # Tracker'a sınıf isimlerini ver
        if hasattr(self.detector, 'model') and hasattr(self.detector.model, 'names'):
            self.tracker.set_class_names(self.detector.model.names)

        self.frame_no = 0
        logger.info("Pipeline hazır ✓")

    def run(self, source=None, display: bool = True,
            save_output: bool = False, scenario_name: str = None):
        """
        Video kaynağını işle.

        Args:
            source: 0 (webcam), veya video dosya yolu
            display: OpenCV penceresi göster
            save_output: Çıktı videosu kaydet
            scenario_name: Replay modunda senaryo adı
        """
        if source is None:
            source = self.config["video"].get("test_video", 0)

        logger.info(f"Video kaynağı açılıyor: {source}")
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            logger.error(f"Video açılamadı: {source}")
            return

        # Video özellikleri
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 15
        logger.info(f"Video: {orig_w}x{orig_h} @ {fps:.1f} FPS")

        # Hedef boyut
        target_w = self.config["dashboard"].get("frame_width", 960)
        target_h = self.config["dashboard"].get("frame_height", 540)

        # Video yazıcı
        writer = None
        if save_output:
            out_path = self.config["video"].get("output_dir", "data/output")
            os.makedirs(out_path, exist_ok=True)
            out_file = os.path.join(out_path, f"output_{int(time.time())}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_file, fourcc, fps, (target_w, target_h))
            logger.info(f"Çıktı videosu: {out_file}")

        if display:
            cv2.namedWindow("Security System", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Security System", target_w, target_h)

        logger.info("İşleme döngüsü başladı. 'q' ile çık.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.info("Video bitti. Yeniden başlatılıyor...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.history.clear()
                    self.tracker.reset()
                    if self.ego is not None:
                        self.ego.reset()
                    continue

                self.frame_no += 1

                # Boyut ayarla
                if orig_w != target_w or orig_h != target_h:
                    frame = cv2.resize(frame, (target_w, target_h))

                # ─── 0. EGO-MOTION (Drone / IHA) ──────
                # Sabit kamerada ego=None -> bu blok no-op. Hareketli
                # kamerada tum state'ler yeni frame'e hizalanir ve boylece
                # ID switch / yanlis loitering / yanlis zone ihlali onlenir.
                H = None
                if self.ego is not None:
                    H = self.ego.estimate(frame)
                    # Onemli: tracker.update() icinde predict() oncesi de
                    # Kalman state'leri warp edilir (transform parametresi
                    # asagida geciliyor). Burada ise track gecmisi ve
                    # davranis kurallari (zone/loitering/abandoned) senkron
                    # edilir.
                    self.history.apply_camera_motion(H)
                    self.engine.apply_camera_motion(H)

                # ─── 1. TESPIT ────────────────────────
                detections = self.detector.detect(frame)

                # ─── 2. TAKIP ─────────────────────────
                tracks = self.tracker.update(detections, frame, transform=H)

                # Track history güncelle
                active_ids = []
                for t in tracks:
                    self.history.update(
                        track_id=t.track_id,
                        bbox=t.bbox,
                        center=t.center,
                        class_id=t.class_id,
                        class_name=t.class_name,
                        confidence=t.confidence
                    )
                    active_ids.append(t.track_id)

                self.history.mark_missing(active_ids)

                # ─── 3. DAVRANIS ANALIZI ──────────────
                alerts, per_track = self.engine.process(
                    self.history, target_w, target_h
                )

                # ─── 4. ZONE OVERLAY ──────────────────
                if self.engine.zone_detector:
                    frame = self.engine.zone_detector.draw_zones(frame)

                # ─── 5. GORSELME ──────────────────────
                extra = {"scenario": scenario_name} if scenario_name else None
                frame = self.visualizer.render(frame, self.history, alerts, per_track, extra)

                # Alert kaydet
                if alerts:
                    self.alert_sys.add_all(alerts, self.frame_no)

                # ─── CIKTI ────────────────────────────
                if writer:
                    writer.write(frame)

                if display:
                    cv2.imshow("Security System", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("r"):
                        logger.info("Tracker sıfırlandı.")
                        self.history.clear()
                        self.tracker.reset()
                        if self.ego is not None:
                            self.ego.reset()

        except KeyboardInterrupt:
            logger.info("Durduruldu.")
        finally:
            cap.release()
            if writer:
                writer.release()
                logger.info("Çıktı videosu kaydedildi.")
            if display:
                cv2.destroyAllWindows()
            self.alert_sys.force_save()
            logger.info(f"Toplam işlenen frame: {self.frame_no}")
            self._print_summary()

    def _print_summary(self):
        stats = self.alert_sys.get_stats()
        logger.info(f"─── ÖZET ───────────────────────────")
        logger.info(f"  Toplam Alert  : {stats['total']}")
        logger.info(f"  Critical      : {stats['critical']}")
        logger.info(f"  High          : {stats['high']}")
        logger.info(f"  Tür bazında   : {stats['by_type']}")
        logger.info(f"───────────────────────────────────")


def main():
    parser = argparse.ArgumentParser(description="Davranışsal Güvenlik Analiz Sistemi")
    parser.add_argument("--source",   default=None,                help="Video kaynağı (0=webcam)")
    parser.add_argument("--config",   default="config/config.yaml", help="Config dosyası")
    parser.add_argument("--no-display", action="store_true",       help="Görüntü gösterme")
    parser.add_argument("--save",     action="store_true",         help="Çıktı videosu kaydet")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = Pipeline(config)

    source = args.source
    if source is not None and source.isdigit():
        source = int(source)

    pipeline.run(
        source=source,
        display=not args.no_display,
        save_output=args.save
    )


if __name__ == "__main__":
    main()
