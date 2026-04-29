# Derin Öğrenme ve Gerçek Zamanlı Davranışsal Analiz ile Güvenlik İhlali Tespiti

**Derin Öğrenme Tabanlı Gerçek Zamanlı Güvenlik İzleme Sistemi**

> Bu proje **OSTİMTECH Ar-Ge ve İnovasyon Proje Pazarı II** kapsamında geliştirilmiştir.

Güvenlik kameraları ve İHA görüntülerinden gerçek zamanlı davranışsal anomali tespiti yapan açık kaynaklı bir yapay zeka güvenlik sistemidir.

---

## Mimari Genel Bakış

```
Video Girişi (Güvenlik Kamerası / İHA / Test Videosu)
        │
        ▼
┌─────────────────────┐
│   YOLOv8 Detector   │  COCO pretrained · VisDrone fine-tune desteği
└─────────────────────┘
        │ [bbox · sınıf · güven skoru]
        ▼
┌─────────────────────┐
│  ByteTrack Tracker  │  Ego-motion komp. (İHA modu)
└─────────────────────┘
        │ [track_id · trajektori · dwell_time]
        ▼
┌──────────────────────────────────────────┐
│       Behavioral Reasoning Engine        │
│                                          │
│  Kural Katmanı          MLP Katmanı      │
│  ─────────────          ──────────       │
│  Bölge İhlali           10 → 32 → 16 → 4│
│  Anormal Bekleme        (PyTorch)        │
│  Terk Edilmiş Nesne                      │
│  Keşif Davranışı        Hibrit Karar:    │
│  Koordineli Hareket     60% MLP          │
│                         40% Kural        │
└──────────────────────────────────────────┘
        │ [tehdit_skoru · olay_tipi · güven]
        ▼
┌─────────────────────┐
│  Streamlit Dashboard │  Canlı + Replay Modu · JSON Alert Kaydı
└─────────────────────┘
```

---

## Tespit Edilen Davranışlar

| Senaryo | Açıklama | Alarm Seviyesi |
|---|---|---|
| **Bölge İhlali** | Kişi veya araç yasak bölgeye giriyor | HIGH |
| **Anormal Bekleme** | Nesne 60s+ aynı konumda bekliyor | MEDIUM |
| **Terk Edilmiş Nesne** | Çanta/bavul sahibinden (kişi veya araç) uzaklaştıktan sonra hareketsiz kalıyor | HIGH |
| **Keşif Davranışı** | Kişi düşük yol verimliliğiyle geniş alanı sistematik olarak tarıyor | HIGH |
| **Koordineli Hareket** | İki veya daha fazla kişi uyumlu hız vektörüyle birlikte hareket ediyor | HIGH |

---

## Tehdit Skoru

Her nesne için 10 boyutlu bir özellik vektörü hesaplanır:

| Boyut | Özellik |
|---|---|
| 0 | Bölge ihlal skoru |
| 1 | Normalleştirilmiş bekleme süresi |
| 2 | Hız büyüklüğü |
| 3 | Trajektori varyansı |
| 4 | Anormal bekleme skoru |
| 5 | Terk edilmiş nesne skoru |
| 6 | Günün saati (sin kodlama) |
| 7 | Nesne sınıfı risk katsayısı |
| 8 | Keşif davranışı skoru |
| 9 | Koordineli hareket skoru |

Bu vektör PyTorch MLP'ye girdi olarak verilir; çıktı `LOW / MEDIUM / HIGH / CRITICAL` olarak sınıflandırılır. MLP yüklenemezse sistem otomatik olarak ağırlıklı kural tabanlı skora geçer.

---

## Kurulum

```bash
pip install -r requirements.txt
```

YOLOv8 ağırlıkları (`yolov8s.pt`) ilk çalıştırmada otomatik indirilir.  
MLP modeli (`models/weights/threat_mlp.pt`) yoksa sentetik veriyle otomatik eğitilir.

---

## Çalıştırma

### Streamlit Dashboard (Önerilen)

```bash
streamlit run src/dashboard/app.py
```

Tarayıcıda `http://localhost:8501` adresine gidin.

### CLI Pipeline

```bash
# Webcam
python src/pipeline.py --source 0

# Video dosyası
python src/pipeline.py --source data/test_videos/test.mp4

# Kayıt ile
python src/pipeline.py --source data/test_videos/test.mp4 --save
```

| Tuş | Aksiyon |
|-----|---------|
| `q` | Çıkış |
| `r` | Tracker sıfırla |

---

## İHA / Drone Desteği

VisDrone2019 veri setiyle YOLOv8 fine-tuning için:

```bash
# 1. VisDrone formatını YOLO'ya dönüştür
python tools/prepare_visdrone.py --src /veri/VisDrone2019

# 2. Fine-tuning başlat (GPU gerektirir)
python tools/train_drone_yolo.py --epochs 80 --device 0
```

Eğitim sonunda `models/weights/yolov8s_drone.pt` oluşur.  
`config/config.yaml` → `detector.model` bu yola güncellenerek devreye alınır.

Ego-motion kompanzasyonu (`config.yaml` → `ego_motion.enabled: true`) aktifleştirildiğinde bölge poligonları, loitering anchor noktaları ve terk edilmiş nesne referansları kamera hareketi ile senkronize edilir; davranışsal kurallar yeryüzüne sabitlenmiş gibi çalışır.

---

## Konfigürasyon

`config/config.yaml` üzerinden tüm parametreler ayarlanabilir:

```yaml
detector:
  model: "yolov8s.pt"
  confidence: 0.4

behavior:
  loitering:
    threshold_seconds: 60
  abandoned_object:
    owner_distance: 150
    confirm_seconds: 10
  reconnaissance:
    min_time_seconds: 30
    efficiency_threshold: 0.35
  coordinated_movement:
    velocity_similarity_threshold: 0.85
    proximity_px: 200
    min_duration_seconds: 5.0
```

Yasak bölge poligonları `config/zones.json` dosyasından yüklenir.

---

## Proje Yapısı

```
Ada_Project/
├── config/
│   ├── config.yaml
│   └── zones.json
├── src/
│   ├── detector/
│   │   └── yolo_detector.py
│   ├── tracker/
│   │   ├── bytetrack_tracker.py
│   │   ├── track_history.py
│   │   └── ego_motion.py
│   ├── behavior/
│   │   ├── engine.py
│   │   ├── threat_scorer.py
│   │   ├── threat_mlp.py
│   │   └── rules/
│   │       ├── zone_violation.py
│   │       ├── loitering.py
│   │       ├── abandoned_object.py
│   │       ├── reconnaissance.py
│   │       └── coordinated_movement.py
│   ├── dashboard/
│   │   ├── app.py
│   │   ├── visualizer.py
│   │   ├── alert_system.py
│   │   └── replay_mode.py
│   └── pipeline.py
├── tools/
│   ├── prepare_visdrone.py
│   └── train_drone_yolo.py
├── tests/
├── models/weights/
└── requirements.txt
```

---

## Performans Hedefleri

| Metrik | Hedef |
|--------|-------|
| FPS (CPU, YOLOv8s) | > 15 FPS |
| Tespit Başarısı | > 0.50 mAP@0.5 |
| ID Switch Oranı | < %10 |
| Yanlış Alarm Oranı | < %25 |

---

## Geliştiriciler

| İsim | GitHub |
|------|--------|
| Azra Karakaya | [@azrakarakaya1](https://github.com/azrakarakaya1) |
| Emine Cakal | [@EmineCakal5](https://github.com/EmineCakal5) |
