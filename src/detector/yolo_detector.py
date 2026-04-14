"""
YOLOv8 Detector — Sabit Güvenlik Kamerası için Nesne Tespiti
COCO pretrained model ile frame → bbox listesi üretir.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import List, Optional
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Tek bir tespit sonucu."""
    bbox: List[float]          # [x1, y1, x2, y2] piksel koordinatları
    confidence: float
    class_id: int
    class_name: str
    center: tuple = field(init=False)

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

    @property
    def width(self):
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self):
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self):
        return self.width * self.height

    def to_tlwh(self):
        """[top, left, width, height] formatına dönüştür (ByteTrack uyumlu)."""
        x1, y1, x2, y2 = self.bbox
        return [x1, y1, x2 - x1, y2 - y1]

    def to_array(self):
        """[x1, y1, x2, y2, confidence] array döner."""
        return np.array([*self.bbox, self.confidence])


class YOLODetector:
    """
    YOLOv8 tabanlı nesne tespit modülü.

    Kullanım:
        detector = YOLODetector(config)
        detections = detector.detect(frame)
    """

    # COCO sınıf isimleri (ilgili olanlar)
    COCO_CLASSES = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorbike",
        5: "bus",
        16: "dog",
        24: "backpack",
        26: "handbag",
        28: "suitcase",
        39: "bottle",
        56: "chair",
        57: "couch",
        63: "laptop",
        66: "keyboard",
        67: "phone",
        73: "book",
        76: "scissors",
    }

    def __init__(self, config: dict):
        self.config = config["detector"]
        self.model_path = self.config.get("model", "yolov8s.pt")
        self.confidence = self.config.get("confidence", 0.4)
        self.iou_threshold = self.config.get("iou_threshold", 0.45)
        self.device = self.config.get("device", "cpu")
        self.classes = self.config.get("classes", None)
        self.imgsz = self.config.get("imgsz", 640)

        self._load_model()
        logger.info(f"YOLODetector başlatıldı: model={self.model_path}, device={self.device}")

    def _load_model(self):
        """YOLO modelini yükle (ilk çalıştırmada indirir)."""
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"Model yüklendi: {self.model_path}")
        except Exception as e:
            logger.error(f"Model yüklenemedi: {e}")
            raise

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Bir frame'de nesne tespiti yap.

        Args:
            frame: BGR formatında numpy array

        Returns:
            Detection listesi
        """
        if frame is None or frame.size == 0:
            return []

        try:
            results = self.model(
                frame,
                conf=self.confidence,
                iou=self.iou_threshold,
                classes=self.classes,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False
            )
        except Exception as e:
            logger.warning(f"Tespit hatası: {e}")
            return []

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                bbox = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.model.names.get(cls_id, f"class_{cls_id}")

                det = Detection(
                    bbox=bbox,
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name
                )
                detections.append(det)

        return detections

    def detect_with_visualization(self, frame: np.ndarray) -> tuple:
        """
        Tespit yap ve sonuçları frame üzerine çiz.

        Returns:
            (annotated_frame, detections)
        """
        detections = self.detect(frame)
        annotated = self._draw_detections(frame.copy(), detections)
        return annotated, detections

    def _draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Tespitleri frame üzerine çiz."""
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            color = self._get_class_color(det.class_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def _get_class_color(self, class_id: int) -> tuple:
        """Sınıf ID'ye göre renk döner."""
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255),
            (255, 255, 0), (0, 255, 255), (255, 0, 255),
            (128, 255, 0), (255, 128, 0), (0, 128, 255),
        ]
        return colors[class_id % len(colors)]

    @staticmethod
    def load_config(config_path: str) -> dict:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
