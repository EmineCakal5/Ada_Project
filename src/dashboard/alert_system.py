# -*- coding: utf-8 -*-
"""
Alert System — Alert geçmişi yönetimi, loglama ve JSON kaydetme.
"""

import json
import os
import time
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class AlertRecord:
    """Kalıcı alert kaydı."""
    alert_id:     int
    alert_type:   str
    track_id:     int
    message:      str
    threat_level: str
    score:        float
    timestamp:    float = field(default_factory=time.time)
    frame_no:     int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["time_str"] = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        return d


class AlertSystem:
    """
    Alert geçmişi yönetimi.

    Özellikler:
    - Tüm alertleri bellekte tutar (son 1000)
    - JSON dosyasına periyodik kaydeder
    - Streamlit için özet istatistikler sağlar

    Kullanım:
        system = AlertSystem(config)
        system.add(alert, frame_no)
        recent = system.get_recent(n=20)
    """

    def __init__(self, config: dict):
        save_cfg = config.get("logging", {})
        self.save_alerts  = save_cfg.get("save_alerts", True)
        self.alerts_file  = save_cfg.get("alerts_file", "data/output/alerts.json")
        self.max_alerts   = config["dashboard"].get("max_alerts", 50)

        self.records: List[AlertRecord] = []
        self._counter = 0
        self._last_save = time.time()
        self._save_interval = 30  # saniyede bir kaydet

        if self.save_alerts:
            os.makedirs(os.path.dirname(self.alerts_file), exist_ok=True)

        logger.info(f"AlertSystem başlatıldı, max_alerts={self.max_alerts}")

    def add(self, alert, frame_no: int = 0) -> AlertRecord:
        """
        Alert ekle.

        Args:
            alert: Alert nesnesi veya dict
            frame_no: Hangi frame'de oluştu

        Returns:
            AlertRecord
        """
        self._counter += 1

        if isinstance(alert, dict):
            record = AlertRecord(
                alert_id=self._counter,
                alert_type=alert.get("type", "unknown"),
                track_id=alert.get("track_id", -1),
                message=alert.get("message", ""),
                threat_level=alert.get("threat_level", "LOW"),
                score=alert.get("score", 0.0),
                frame_no=frame_no
            )
        else:
            record = AlertRecord(
                alert_id=self._counter,
                alert_type=getattr(alert, "alert_type", "unknown"),
                track_id=getattr(alert, "track_id", -1),
                message=getattr(alert, "message", ""),
                threat_level=getattr(alert, "threat_level", "LOW"),
                score=getattr(alert, "score", 0.0),
                frame_no=frame_no
            )

        self.records.append(record)
        # Son max_alerts kadar tut
        self.records = self.records[-self.max_alerts:]

        # Periyodik kaydet
        if self.save_alerts and (time.time() - self._last_save > self._save_interval):
            self._save_to_file()

        return record

    def add_all(self, alerts: List, frame_no: int = 0):
        for a in alerts:
            self.add(a, frame_no)

    def get_recent(self, n: int = 20) -> List[Dict]:
        return [r.to_dict() for r in self.records[-n:]]

    def get_by_type(self, alert_type: str) -> List[AlertRecord]:
        return [r for r in self.records if r.alert_type == alert_type]

    def get_by_level(self, level: str) -> List[AlertRecord]:
        return [r for r in self.records if r.threat_level == level]

    def get_stats(self) -> Dict:
        """Dashboard için istatistik özeti."""
        total = len(self.records)
        by_type = {}
        by_level = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}

        for r in self.records:
            by_type[r.alert_type] = by_type.get(r.alert_type, 0) + 1
            by_level[r.threat_level] = by_level.get(r.threat_level, 0) + 1

        return {
            "total":    total,
            "by_type":  by_type,
            "by_level": by_level,
            "critical": by_level.get("CRITICAL", 0),
            "high":     by_level.get("HIGH", 0),
        }

    def clear(self):
        self.records.clear()
        self._counter = 0

    def _save_to_file(self):
        try:
            data = [r.to_dict() for r in self.records]
            with open(self.alerts_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self._last_save = time.time()
        except Exception as e:
            logger.error(f"Alert kaydedilemedi: {e}")

    def force_save(self):
        """Manuel kaydet (uygulama kapanırken)."""
        if self.save_alerts:
            self._save_to_file()
