# -*- coding: utf-8 -*-
"""
Smoke Test -- Tum modullerin import ve temel fonksiyonlarini test eder.
Calistir: python tests/smoke_test.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

traceback_mod = None

PASS = "[OK  ]"
FAIL = "[FAIL]"
results = []


def test(name, fn):
    try:
        fn()
        print(f"  {PASS} {name}")
        results.append((name, True))
    except Exception as e:
        print(f"  {FAIL} {name}: {e}")
        results.append((name, False))


print("\n=== Davranissal Guvenlik Sistemi - Smoke Test ===\n")

# -- Config --
print("[Config]")
def _cfg():
    import yaml
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert "detector" in cfg
    assert "behavior" in cfg
    assert "dashboard" in cfg

def _zones():
    import json
    with open("config/zones.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data["zones"]) > 0

test("config.yaml yuklendi", _cfg)
test("zones.json yuklendi", _zones)

# -- Tracker --
print("\n[Tracker]")
def _track_history():
    from src.tracker.track_history import TrackHistory
    h = TrackHistory()
    h.update(1, [10, 10, 50, 50], (30, 30), 0, "person", 0.9)
    t = h.get(1)
    assert t is not None
    assert t.class_name == "person"
    assert t.dwell_time >= 0

def _bytetrack_import():
    from src.tracker.bytetrack_tracker import ByteTracker
    import yaml
    with open("config/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    bt = ByteTracker(cfg)
    assert bt is not None

test("TrackHistory olusturuldu", _track_history)
test("ByteTracker import edildi", _bytetrack_import)

# -- Detector --
print("\n[Detector]")
def _detector_import():
    from src.detector.yolo_detector import Detection
    d = Detection(bbox=[0, 0, 100, 100], confidence=0.9, class_id=0, class_name="person")
    assert d.center == (50, 50)
    assert d.area == 10000

test("Detection dataclass", _detector_import)

# -- Behavior Rules --
print("\n[Davranis Kurallari]")
def _zone_violation():
    from src.behavior.rules.zone_violation import ZoneViolationDetector
    zd = ZoneViolationDetector("config/zones.json", 960, 540)
    assert len(zd.get_all_zones()) > 0
    r = zd.check(1, (100, 100), "person")
    assert "violation" in r

def _loitering():
    import yaml
    with open("config/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    from src.behavior.rules.loitering import LoiteringDetector
    ld = LoiteringDetector(cfg)
    r = ld.check(1, (100, 100), 5.0, 2.0, "person")
    assert "loitering_score" in r
    assert r["loitering"] == False

def _abandoned():
    import yaml
    with open("config/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    from src.behavior.rules.abandoned_object import AbandonedObjectDetector
    ad = AbandonedObjectDetector(cfg)
    assert ad is not None

test("ZoneViolationDetector", _zone_violation)
test("LoiteringDetector", _loitering)
test("AbandonedObjectDetector", _abandoned)

# -- Threat Scorer --
print("\n[Threat Scorer]")
def _scorer():
    import yaml
    with open("config/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    from src.behavior.threat_scorer import ThreatScorer
    from src.tracker.track_history import TrackInfo
    scorer = ThreatScorer(cfg)
    track = TrackInfo(1, 0, "person", [0, 0, 50, 50], (25, 25))
    zone_r = {"violation_score": 0.8, "violation": True}
    loop_r = {"loitering_score": 0.5}
    fv, score, level = scorer.compute(track, zone_r, loop_r, 0.3)
    assert len(fv) == 8
    assert 0.0 <= score <= 1.0
    assert level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

test("ThreatScorer.compute()", _scorer)

# -- MLP --
print("\n[MLP Siniflandirici]")
def _mlp_synthetic():
    from src.behavior.threat_mlp import generate_synthetic_data
    X, y = generate_synthetic_data(100)
    assert X.shape == (100, 8)
    assert y.shape == (100,)
    assert set(y).issubset({0, 1, 2, 3})

def _mlp_model():
    import torch
    from src.behavior.threat_mlp import ThreatMLP
    model = ThreatMLP(8, [32, 16], 4)
    x = torch.randn(1, 8)
    out = model(x)
    assert out.shape == (1, 4)

test("Sentetik veri uretimi", _mlp_synthetic)
test("ThreatMLP forward pass", _mlp_model)

# -- Alert System --
print("\n[Alert System]")
def _alerts():
    import yaml
    with open("config/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    from src.dashboard.alert_system import AlertSystem
    sys_ = AlertSystem(cfg)
    sys_.add({"type": "zone_violation", "track_id": 1,
              "message": "Test alert", "threat_level": "HIGH", "score": 0.9}, 1)
    stats = sys_.get_stats()
    assert stats["total"] == 1
    assert stats["high"] == 1

test("AlertSystem.add() & get_stats()", _alerts)

# -- Replay Mode --
print("\n[Replay Mode]")
def _replay():
    import yaml
    with open("config/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    from src.dashboard.replay_mode import ReplayManager
    rm = ReplayManager(cfg)
    scenarios = rm.list_scenarios()
    assert len(scenarios) == 3

test("ReplayManager & 3 senaryo", _replay)

# -- Ozet --
passed = sum(1 for _, ok in results if ok)
total = len(results)
print(f"\n{'='*50}")
print(f"Sonuc: {passed}/{total} test gecti")
if passed == total:
    print("Tum testler basarili! Sistem calismeye hazir.")
else:
    failed = [n for n, ok in results if not ok]
    print(f"Basarisiz: {', '.join(failed)}")
    print("   -> python -m pip install -r requirements.txt komutunu calistirin")
print(f"{'='*50}\n")
