# -*- coding: utf-8 -*-
"""
Replay Mode — Önceden kaydedilmiş senaryoları dashboard'da oynatır.
Sunumda 'bir şey olmama' riskini ortadan kaldırır.

Sprint Plan v2: 3 senaryo → Bölge ihlali, Terk edilmiş nesne, Anormal bekleme
"""

import os
import json
import glob
import time
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Scenario:
    """Tek bir replay senaryosu."""
    name:        str
    video_path:  str
    description: str
    duration_s:  float = 0.0
    tags:        List[str] = field(default_factory=list)
    zone_config: Optional[str] = None   # Özel zone dosyası

    @property
    def exists(self) -> bool:
        return os.path.isfile(self.video_path)

    def to_dict(self) -> dict:
        return {
            "name":        self.name,
            "video_path":  self.video_path,
            "description": self.description,
            "duration_s":  self.duration_s,
            "tags":        self.tags,
            "exists":      self.exists,
        }


class ReplayManager:
    """
    Senaryo yönetimi ve replay kontrolü.

    Kullanım:
        rm = ReplayManager(config)
        scenarios = rm.list_scenarios()
        rm.load_scenario(scenario)
        path = rm.current_video_path
    """

    # Gömülü varsayılan senaryolar (video yoksa bilgi ver)
    DEFAULT_SCENARIOS = [
        {
            "name": "Scenario 1 — Zone violation",
            "video_path": "data/scenarios/scenario_zone_violation.mp4",
            "description": "A person enters a restricted zone; the system raises an alert.",
            "tags": ["zone_violation", "person"]
        },
        {
            "name": "Scenario 2 — Abandoned object",
            "video_path": "data/scenarios/scenario_abandoned.mp4",
            "description": "Someone leaves a bag and walks away; the system flags abandonment.",
            "tags": ["abandoned_object", "backpack"]
        },
        {
            "name": "Scenario 3 — Loitering",
            "video_path": "data/scenarios/scenario_loitering.mp4",
            "description": "A person remains stationary 60+ seconds; the system triggers loitering logic.",
            "tags": ["loitering", "person"]
        },
    ]

    def __init__(self, config: dict):
        self.scenarios_dir  = config["dashboard"].get("scenarios_dir", "data/scenarios")
        self.current        = None       # Aktif Scenario
        self.is_playing     = False
        self.start_time     = None
        self._scenarios: List[Scenario] = []

        os.makedirs(self.scenarios_dir, exist_ok=True)
        self._save_scenario_manifest()
        self._load_scenarios()

        logger.info(f"ReplayManager: {len(self._scenarios)} senaryo yüklendi")

    def _save_scenario_manifest(self):
        """Varsayılan senaryo manifest'ini oluştur."""
        manifest_path = os.path.join(self.scenarios_dir, "scenarios.json")
        if not os.path.exists(manifest_path):
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(self.DEFAULT_SCENARIOS, f, ensure_ascii=False, indent=2)
            logger.info(f"Senaryo manifest oluşturuldu: {manifest_path}")

    def _load_scenarios(self):
        """Senaryo manifest'ini yükle."""
        manifest_path = os.path.join(self.scenarios_dir, "scenarios.json")
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._scenarios = [Scenario(**s) for s in data]
        except Exception as e:
            logger.warning(f"Senaryo manifest yüklenemedi: {e}. Varsayılanlar kullanılıyor.")
            self._scenarios = [Scenario(**s) for s in self.DEFAULT_SCENARIOS]

    def list_scenarios(self) -> List[Scenario]:
        return self._scenarios

    def list_available(self) -> List[Scenario]:
        """Sadece video dosyası mevcut olanları döner."""
        return [s for s in self._scenarios if s.exists]

    def load_scenario(self, scenario: Scenario):
        """Senaryoyu aktif yap."""
        if not scenario.exists:
            logger.warning(f"Video bulunamadı: {scenario.video_path}")
            return False
        self.current    = scenario
        self.is_playing = True
        self.start_time = time.time()
        logger.info(f"Senaryo başlatıldı: {scenario.name}")
        return True

    def load_by_index(self, idx: int) -> bool:
        if 0 <= idx < len(self._scenarios):
            return self.load_scenario(self._scenarios[idx])
        return False

    def stop(self):
        self.current    = None
        self.is_playing = False
        self.start_time = None

    @property
    def current_video_path(self) -> Optional[str]:
        return self.current.video_path if self.current else None

    @property
    def current_name(self) -> Optional[str]:
        return self.current.name if self.current else None

    @property
    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    def get_status(self) -> Dict:
        return {
            "playing":      self.is_playing,
            "scenario":     self.current_name,
            "video_path":   self.current_video_path,
            "elapsed":      round(self.elapsed, 1),
            "total_count":  len(self._scenarios),
            "available":    len(self.list_available()),
        }

    @staticmethod
    def get_download_instructions() -> str:
        """How to add scenario videos."""
        return """
## Add test videos

Replay scenarios need real surveillance-style footage.

### Option A — Download with yt-dlp
```bash
pip install yt-dlp
yt-dlp "https://youtube.com/..." -o "data/scenarios/scenario_zone_violation.mp4"
```

### Option B — Stock footage
- [Pexels — surveillance](https://www.pexels.com/search/videos/security%20camera/)
- [Pixabay — CCTV](https://pixabay.com/videos/search/security%20camera/)

Place files as:
- `data/scenarios/scenario_zone_violation.mp4`
- `data/scenarios/scenario_abandoned.mp4`
- `data/scenarios/scenario_loitering.mp4`
"""
