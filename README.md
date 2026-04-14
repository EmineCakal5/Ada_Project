# Davranışsal Güvenlik Analiz Sistemi v2.0

> YOLOv8 + ByteTrack + MLP Hibrit Tehdit Sınıflandırıcı | Sabit Güvenlik Kamerası Perspektifi

---

## 🎯 Proje Özeti

Bu sistem, sabit güvenlik kameraları için gerçek zamanlı davranışsal anomali tespiti yapar:

| Özellik | Açıklama |
|---------|----------|
| **Nesne Tespiti** | YOLOv8s — COCO pretrained |
| **Nesne Takibi** | ByteTrack (birincil) + DeepSORT (fallback) |
| **Davranış Analizi** | Kural motoru + MLP hibrit |
| **Dashboard** | Streamlit — Canlı + Replay Mode |
| **Hedef FPS** | >15 FPS (CPU, YOLOv8s) |

### Tespit Edilen Davranışlar
- 🚫 **Bölge İhlali** — Yasak bölgeye giriş
- 🕐 **Anormal Bekleme** — 60s+ aynı noktada durma
- 🎒 **Terk Edilmiş Nesne** — Çanta/kutu bırakma tespiti

---

## 📁 Proje Yapısı

```
Ada_proje/
├── requirements.txt
├── config/
│   ├── config.yaml          # Ana ayarlar
│   └── zones.json           # Bölge poligonları
├── data/
│   ├── test_videos/         # Demo videoları (buraya ekle)
│   └── scenarios/           # Replay senaryoları
├── models/weights/          # threat_mlp.pt (otomatik oluşur)
├── src/
│   ├── detector/
│   │   └── yolo_detector.py
│   ├── tracker/
│   │   ├── bytetrack_tracker.py
│   │   └── track_history.py
│   ├── behavior/
│   │   ├── engine.py
│   │   ├── threat_scorer.py
│   │   ├── threat_mlp.py
│   │   └── rules/
│   │       ├── zone_violation.py
│   │       ├── loitering.py
│   │       └── abandoned_object.py
│   ├── dashboard/
│   │   ├── app.py           # Streamlit dashboard
│   │   ├── visualizer.py
│   │   ├── alert_system.py
│   │   └── replay_mode.py
│   └── pipeline.py          # CLI pipeline
└── tests/
```

---

## 🚀 Kurulum

```bash
# 1. Bağımlılıkları yükle
pip install -r requirements.txt

# 2. YOLOv8 modeli otomatik indirilir (ilk çalıştırmada)
# 3. MLP modeli sentetik veri ile otomatik eğitilir
```

---

## ▶ Çalıştırma

### Streamlit Dashboard (Önerilen)
```bash
streamlit run src/dashboard/app.py
```
Tarayıcıda `http://localhost:8501` açılır.

### CLI Pipeline
```bash
# Webcam ile
python src/pipeline.py --source 0

# Video dosyası ile
python src/pipeline.py --source data/test_videos/test.mp4

# Kayıt ile
python src/pipeline.py --source data/test_videos/test.mp4 --save
```

### Keyboard Shortcuts (CLI)
| Tuş | Aksiyon |
|-----|---------|
| `q` | Çık |
| `r` | Tracker sıfırla |

---

## 📼 Test Videosu Ekleme

Replay senaryoları için güvenlik kamerası videoları gerekli:

```bash
# YouTube'dan indirme (yt-dlp)
pip install yt-dlp
yt-dlp "VIDEO_URL" -o "data/scenarios/scenario_zone_violation.mp4"
```

**Ücretsiz kaynaklar:**
- [Pexels - Surveillance Videos](https://www.pexels.com/search/videos/security%20camera/)
- [Pixabay - CCTV](https://pixabay.com/videos/search/security%20camera/)

---

## 🧠 Mimari

```
Video Girişi (Güvenlik Kamerası / Test Videosu)
        |
        v
+---------------------+
|   YOLOv8 Detector   |  <- COCO pretrained
+---------------------+
        | [bbox, class, confidence]
        v
+---------------------+
|  ByteTrack Tracker  |  <- Hız öncelikli
+---------------------+
        | [track_id, trajectory, dwell_time]
        v
+-------------------------------+
| Behavioral Reasoning Engine   |
|  - Bölge İhlali               |
|  - Terk Edilmiş Nesne         |
|  - Anormal Bekleme            |
|  + MLP Threat Classifier      |  <- 8→32→16→4
+-------------------------------+
        | [threat_score, event_type, confidence]
        v
+---------------------+
|  Dashboard + Replay |  <- Streamlit
+---------------------+
```

---

## 📊 Performans Hedefleri

| Metrik | Hedef | Öncelik |
|--------|-------|---------|
| FPS (CPU, YOLOv8s) | > 15 FPS | KRİTİK |
| Tespit Başarısı | > 0.50 mAP@0.5 | YÜKSEK |
| ID Switch oranı | < %10 | ORTA |
| Yanlış Alarm | < %25 (MVP) | ORTA |
| Dashboard Latency | < 500ms | DÜŞÜK |

---

## 🔧 Konfigürasyon

`config/config.yaml` dosyasında ayarlar:

```yaml
detector:
  model: "yolov8s.pt"     # yolov8n (hızlı) / yolov8m (doğru)
  confidence: 0.4

behavior:
  loitering:
    threshold_seconds: 60  # Bekleme süresi eşiği
  abandoned_object:
    confirm_seconds: 10    # Terk onay süresi
```

`config/zones.json` dosyasında bölge poligonları tanımlanır.

---

## 👤 Geliştirici

**Nisan 2026** — Davranışsal Güvenlik Analiz Sistemi v2.0
Sprint tabanlı 2-3 haftalık geliştirme planı.
