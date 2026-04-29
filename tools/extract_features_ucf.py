# -*- coding: utf-8 -*-
"""
UCF-Crime Feature Extractor
===========================

UCF-Crime: data/UCF/<kategori>/ altinda ya dogrudan .mp4, ya da onceden
cikarilmis frame dosyalari (video_adi_frameindex.png, or. Robbery048_x264_0.png).
Klasorde .mp4 yoksa otomatik PNG moduna gecer; ayni video adina ait kareler
gruplanir, siralanir, her grup ayri "video" gibi islenir (track history reset).
Veriyi mevcut pipeline'dan
(YOLODetector + ByteTracker + TrackHistory + BehaviorEngine/ThreatScorer)
gecirip her aktif track icin 8 boyutlu feature vektoru uretir ve etiketli
bir dataset olusturur.

Kategori → Etiket:
    Burglary     → 2 (HIGH)
    Fighting     → 2 (HIGH)
    Robbery      → 3 (CRITICAL)
    Stealing     → 1 (MEDIUM)
    Normal*      → 0 (LOW)

Kullanim:
    # Feature cikar (varsayilan tum kategoriler):
    python tools/extract_features_ucf.py

    # Detector guven esigi sadece bu script icin (config.yaml'yi degistirmez), ornek 0.45:
    python tools/extract_features_ucf.py --conf 0.45

    # Kategori basina en fazla 20 video, video basina 200 frame:
    python tools/extract_features_ucf.py --limit 20 --max-frames 200

    # Sadece belirli kategoriler:
    python tools/extract_features_ucf.py --categories Burglary Fighting

    # MLP'yi sadece mevcut pkl uzerinden yeniden egit:
    python tools/extract_features_ucf.py --train-only

    # Hem cikar hem egit:
    python tools/extract_features_ucf.py --train

Cikti:
    data/ucf_features.pkl  — {
        "samples": [{"features": np.ndarray(8,), "label": int,
                     "video": str, "track_id": int, "frame_idx": int}, ...],
        "meta":    {kategori basina sayim, toplam sample vs.}
    }

Not (sunum / tez):
    UCF-Crime sabit güvenlik kamerası baktı; drone / İHA kuş bakışı değil.
    Özellik vektörü (YOLO+track+davranış) aynı şekilde çıkar; metin ve
    arayüzde “eğitim/özellik: CCTV tarzı; operasyon: UAV’e uyarlandı (GMC,
    coğrafi bölge, vb.)” ifadesi kullanılabilir.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import re
import sys
import time
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import yaml

# Proje kokunu sys.path'e ekle
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector.yolo_detector import YOLODetector            # noqa: E402
from src.tracker.bytetrack_tracker import ByteTracker          # noqa: E402
from src.tracker.track_history import TrackHistory             # noqa: E402
from src.behavior.engine import BehaviorEngine                 # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("extract_features_ucf")


# ─── Sabitler ──────────────────────────────────────────────────────────────

UCF_ROOT = PROJECT_ROOT / "data" / "UCF"
OUT_PATH = PROJECT_ROOT / "data" / "ucf_features.pkl"
MODEL_OUT_PATH = PROJECT_ROOT / "models" / "weights" / "threat_mlp.pt"

# Kategori klasoru → (etiket, aciklama)
UCF_LABEL_MAP: Dict[str, Tuple[int, str]] = {
    "Burglary":      (2, "HIGH"),
    "Fighting":      (2, "HIGH"),
    "Robbery":       (3, "CRITICAL"),
    "Stealing":      (1, "MEDIUM"),
    "Normal":        (0, "LOW"),     # klasor adi "Normal" olursa
    "NormalVideos":  (0, "LOW"),     # klasor adi "NormalVideos" olursa
}

# Frame adi: "<video_adi>_<frame_index>.<ext>"  (ornek: Robbery048_x264_0.png)
# video_adi = son underscore'dan onceki kisim (ornek: Robbery048_x264)
FRAME_RE = re.compile(r"^(.+)_(\d+)\.(png|jpg|jpeg|bmp|webp|tif|tiff)$", re.IGNORECASE)

TARGET_W = 960
TARGET_H = 540

# UCF video'lari 30 fps, biz her 10. frame'i kullaniyoruz → 3 fps etkin
ORIGINAL_FPS = 30.0
FRAME_STEP = 10
SIM_DT = FRAME_STEP / ORIGINAL_FPS   # saniye / adim (~0.333 s)


# ─── Config ────────────────────────────────────────────────────────────────

def load_config(path: Path = None) -> dict:
    """Pipeline config'ini yukle; UCF ayiklamasi icin ego-motion'u kapat."""
    path = path or (PROJECT_ROOT / "config" / "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # UCF sabit kamera → ego-motion gereksiz ve CPU'yu yer
    cfg.setdefault("tracker", {}).setdefault("ego_motion", {})["enabled"] = False
    return cfg


# ─── Yardimcilar ───────────────────────────────────────────────────────────

def _folder_has_mp4(folder: Path) -> bool:
    return any(f.suffix.lower() == ".mp4" for f in folder.iterdir() if f.is_file())


def group_frames_by_video(folder: Path) -> Dict[str, List[Tuple[int, Path]]]:
    """
    Kategori klasorundeki frame dosyalarini kaynak video adina gore grupla.
    Dosya adi: video_adi_frameindex.png — ayni video_adi bir grup; gruplar
    frame_index artan sirada.
    """
    buckets: Dict[str, List[Tuple[int, Path]]] = defaultdict(list)
    for p in folder.iterdir():
        if not p.is_file():
            continue
        m = FRAME_RE.match(p.name)
        if not m:
            continue
        video_name, idx_str, _ext = m.groups()
        try:
            idx = int(idx_str)
        except ValueError:
            continue
        buckets[video_name].append((idx, p))

    for v in buckets:
        buckets[v].sort(key=lambda x: x[0])
    return buckets


def load_mp4_as_frame_list(
    path: Path,
    max_output_frames: int | None = None,
) -> List[Tuple[int, np.ndarray]]:
    """
    .mp4 dosyasini ac; ORIGINAL_FPS/FRAME_STEP ile (varsayilan her 10. kare)
    ornekleyerek (idx, BGR) listesi. idx = videodaki gercek frame numarasi.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        logger.warning(f"Video acilamadi: {path}")
        return []
    out: List[Tuple[int, np.ndarray]] = []
    n = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if n % FRAME_STEP == 0:
                out.append((n, frame))
                if max_output_frames is not None and len(out) >= max_output_frames:
                    break
            n += 1
    finally:
        cap.release()
    return out


def _sync_simulated_time(history: TrackHistory, base_time: float, step: int) -> None:
    """
    TrackInfo.dwell_time wall-clock tabanli (`time.time()`) oldugu icin, frame'leri
    hizlica ardi ardina isledigimizde tum track'lerin dwell_time'i ~0 kalir.
    Bu yardimci her track icin `first_seen` / `last_seen` degerlerini, video
    icindeki ADIM numarasina gore simule edilmis bir zaman eksenine oturtur.
    Boylece feature[1] (dwell_time_normalized) gercek pipeline davranisi ile
    ayni dagilimi ogrenir.
    """
    sim_now = base_time + step * SIM_DT
    for trk in history.tracks.values():
        if not hasattr(trk, "_sim_first_seen"):
            trk._sim_first_seen = sim_now
        trk.first_seen = trk._sim_first_seen
        trk.last_seen = sim_now


def _safe_imread(path: Path) -> np.ndarray | None:
    """OpenCV imread ama bozuk dosyalarda None dondurur, exception atmaz."""
    try:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    except Exception as e:
        logger.debug(f"imread hata {path.name}: {e}")
        return None
    if img is None or img.size == 0:
        return None
    return img


# ─── Tek Video → Sample Listesi ────────────────────────────────────────────

FrameItem = Union[Path, np.ndarray]


def extract_features_for_video(
    frames: List[Tuple[int, FrameItem]],
    detector: YOLODetector,
    tracker: ByteTracker,
    history: TrackHistory,
    engine: BehaviorEngine,
    label: int,
    video_name: str,
    max_frames: int | None = None,
) -> List[dict]:
    """Bir kaynak video icin sample listesi uret. FrameItem: PNG yolu veya BGR array (.mp4)."""
    samples: List[dict] = []

    # Her video (veya frame grubu) icin tracker / history sifirla — video degisince state temiz
    tracker.reset()
    history.clear()

    # BehaviorEngine dahili state'leri (loitering anchor'lar, abandoned
    # object timer'lari, cooldown map'leri) video degisiminde sifirlansin:
    engine.loitering.update_active([])          # loitering temizle
    engine._last_alert_time.clear()             # cooldown reset
    engine.alert_history.clear()

    # abandoned object dahili state
    ab = engine.abandoned
    for attr in ("stationary_since", "owner_map", "alerted_ids", "positions"):
        if hasattr(ab, attr):
            try:
                getattr(ab, attr).clear()
            except AttributeError:
                pass

    base_time = time.time()

    iterable = frames if max_frames is None else frames[:max_frames]

    for step, (frame_idx, src) in enumerate(iterable):
        if isinstance(src, Path):
            frame = _safe_imread(src)
            if frame is None:
                logger.debug(f"  ↳ bozuk/bos frame atlandi: {src.name}")
                continue
            src_name = src.name
        else:
            frame = src
            if frame is None or frame.size == 0:
                continue
            src_name = f"frame_{frame_idx}"

        # Hedef boyuta olcekle (zone koordinatlari 960x540'a gore tanimli)
        if frame.shape[1] != TARGET_W or frame.shape[0] != TARGET_H:
            try:
                frame = cv2.resize(frame, (TARGET_W, TARGET_H))
            except Exception as e:
                logger.debug(f"  ↳ resize hata {src_name}: {e}")
                continue

        # 1) Tespit
        try:
            detections = detector.detect(frame)
        except Exception as e:
            logger.warning(f"  ↳ detect hata {src_name}: {e}")
            continue

        # 2) Takip
        try:
            tracks = tracker.update(detections, frame)
        except Exception as e:
            logger.warning(f"  ↳ tracker hata {src_name}: {e}")
            continue

        # 3) Track history guncelle
        active_ids: List[int] = []
        for t in tracks:
            history.update(
                track_id=t.track_id,
                bbox=t.bbox,
                center=t.center,
                class_id=t.class_id,
                class_name=t.class_name,
                confidence=t.confidence,
            )
            active_ids.append(t.track_id)
        history.mark_missing(active_ids)

        # dwell_time'i simule et (yoksa hep ~0 gelir)
        _sync_simulated_time(history, base_time, step)

        # 4) Davranis motoru → feature vektorleri
        try:
            _alerts, per_track = engine.process(history, TARGET_W, TARGET_H)
        except Exception as e:
            logger.warning(f"  ↳ engine.process hata {src_name}: {e}")
            continue

        # 5) Her aktif track icin sample uret
        for tid, info in per_track.items():
            fv = np.asarray(info.get("feature_vector", []), dtype=np.float32)
            if fv.shape != (8,):
                continue
            if not np.all(np.isfinite(fv)):
                continue
            samples.append({
                "features":  fv,
                "label":     int(label),
                "video":     video_name,
                "track_id":  int(tid),
                "frame_idx": int(frame_idx),
            })

    return samples


# ─── Tum Kategoriler → Dataset ─────────────────────────────────────────────

def build_dataset(
    categories: List[str] | None = None,
    videos_per_category: int | None = None,
    max_frames_per_video: int | None = None,
    out_path: Path = OUT_PATH,
    max_normal_samples: int | None = 5000,
    seed: int = 42,
    detector_confidence: float | None = None,
) -> dict:
    """Tum kategori klasorlerini tara, pipeline'dan gecir, pkl'e yaz.

    Args:
        max_normal_samples: Normal (label=0) sinifinin kacinci sample'a
            random subsample'lanacagi. UCF'de Normal anomalilerden ~10x
            daha fazla; bu sinir class imbalance'i bastirir. None = sinir yok.
    """
    cfg = load_config()
    if detector_confidence is not None:
        c = float(detector_confidence)
        if c <= 0.0 or c > 1.0:
            raise ValueError("detector_confidence / --conf 0.0-1.0 araliginda olmali (0 haric)")
        cfg.setdefault("detector", {})["confidence"] = c
        logger.info(f"UCF extract: detector confidence = {c} (config.yaml uzerine yazildi, dosya degismez)")

    if not UCF_ROOT.exists():
        raise FileNotFoundError(f"UCF klasoru bulunamadi: {UCF_ROOT}")

    # Bir kere init, tum videolar arasinda paylas (her video'da reset edilir)
    logger.info("Pipeline bilesenleri yukleniyor (YOLO + ByteTrack + Engine)...")
    detector = YOLODetector(cfg)
    tracker  = ByteTracker(cfg)
    history  = TrackHistory(max_lost_frames=30)
    engine   = BehaviorEngine(cfg)

    # Tracker'a YOLO isimlerini ver
    if hasattr(detector, "model") and hasattr(detector.model, "names"):
        tracker.set_class_names(detector.model.names)

    # Hangi kategoriler islenecek
    available = [d.name for d in UCF_ROOT.iterdir() if d.is_dir()]
    if categories:
        todo = [c for c in categories if c in available]
        missing = [c for c in categories if c not in available]
        if missing:
            logger.warning(f"Atlanan (klasor yok): {missing}")
    else:
        todo = [c for c in available if c in UCF_LABEL_MAP]

    logger.info(f"Islenecek kategoriler: {todo}")

    all_samples: List[dict] = []
    per_category_video_count: Counter = Counter()
    per_category_sample_count: Counter = Counter()
    per_category_failed_videos: Counter = Counter()

    t_start = time.time()

    for cat in todo:
        if cat not in UCF_LABEL_MAP:
            logger.warning(f"'{cat}' icin etiket tanimsiz, atlaniyor.")
            continue

        label, level = UCF_LABEL_MAP[cat]
        cat_dir = UCF_ROOT / cat
        logger.info(f"━━━ {cat}  (label={label} / {level}) ━━━")

        # .mp4 varsa dogrudan video; yoksa PNG (veya jpg) frame dosyalari — gruplu mod
        use_mp4 = _folder_has_mp4(cat_dir)
        if use_mp4:
            mp4_list = sorted(cat_dir.glob("*.mp4"), key=lambda p: p.name.lower())
            if videos_per_category is not None:
                mp4_list = mp4_list[:videos_per_category]
            logger.info(
                f"  mod=MP4  {len(mp4_list)} dosya  "
                f"(her {FRAME_STEP}. kare, ~{1.0 / SIM_DT:.1f} fps etkin)"
            )
            for v_i, mp4_path in enumerate(mp4_list, 1):
                video_key = mp4_path.stem
                frames = load_mp4_as_frame_list(mp4_path, max_output_frames=max_frames_per_video)
                n_frames = len(frames)
                t_v0 = time.time()
                try:
                    video_samples = extract_features_for_video(
                        frames=frames,
                        detector=detector,
                        tracker=tracker,
                        history=history,
                        engine=engine,
                        label=label,
                        video_name=video_key,
                        max_frames=None,
                    )
                except Exception as e:
                    logger.warning(f"  [{v_i}/{len(mp4_list)}] {mp4_path.name} BASARISIZ: {e}")
                    per_category_failed_videos[cat] += 1
                    continue
                dt = time.time() - t_v0
                logger.info(
                    f"  [{v_i:>4}/{len(mp4_list)}] {mp4_path.name}: "
                    f"{n_frames} frame → {len(video_samples)} sample  ({dt:.1f}s)"
                )
                all_samples.extend(video_samples)
                per_category_video_count[cat] += 1
                per_category_sample_count[cat] += len(video_samples)
        else:
            buckets = group_frames_by_video(cat_dir)
            if not buckets:
                logger.warning(
                    f"  mod=PNG  Klasorde .mp4 yok; "
                    f"video_adi_index.(png|jpg|...) desenine uyan frame de yok: {cat_dir}"
                )
            videos = sorted(buckets.keys())
            if videos_per_category is not None:
                videos = videos[:videos_per_category]

            n_total_frames = sum(len(buckets[v]) for v in videos) if videos else 0
            lim_note = " (kategori video limiti oncesi)" if videos_per_category is not None else ""
            logger.info(
                f"  mod=PNG  {len(videos)} video (gruplu), toplam {n_total_frames} frame{lim_note}"
            )

            for v_i, video in enumerate(videos, 1):
                frames: List[Tuple[int, FrameItem]] = buckets[video]
                n_frames = len(frames) if max_frames_per_video is None else min(
                    len(frames), max_frames_per_video
                )
                t_v0 = time.time()
                try:
                    video_samples = extract_features_for_video(
                        frames=frames,
                        detector=detector,
                        tracker=tracker,
                        history=history,
                        engine=engine,
                        label=label,
                        video_name=video,
                        max_frames=max_frames_per_video,
                    )
                except Exception as e:
                    logger.warning(f"  [{v_i}/{len(videos)}] {video} BASARISIZ: {e}")
                    per_category_failed_videos[cat] += 1
                    continue

                dt = time.time() - t_v0
                logger.info(
                    f"  [{v_i:>4}/{len(videos)}] {video}: "
                    f"{n_frames} frame → {len(video_samples)} sample  ({dt:.1f}s)"
                )
                all_samples.extend(video_samples)
                per_category_video_count[cat] += 1
                per_category_sample_count[cat] += len(video_samples)

    # ─── Class imbalance: Normal sinifi icin random subsample ──────
    # UCF-Crime'da Normal sample'lari anomalilerden cok daha yogun uretilir
    # (ornek: 21.9k Normal vs ~2.8k anomali). Cap koymazsak MLP Normal'a
    # bias yapar. Burada Normal sinifindan rastgele `max_normal_samples`
    # kadar ornek alip geri kalani atariz.
    dropped_normal = 0
    if max_normal_samples is not None and max_normal_samples > 0:
        normal_idx = [i for i, s in enumerate(all_samples) if s["label"] == 0]
        if len(normal_idx) > max_normal_samples:
            rng = np.random.default_rng(seed)
            keep = set(rng.choice(normal_idx, size=max_normal_samples, replace=False).tolist())
            before = len(all_samples)
            all_samples = [
                s for i, s in enumerate(all_samples)
                if s["label"] != 0 or i in keep
            ]
            dropped_normal = before - len(all_samples)
            logger.info(
                f"Normal cap: {len(normal_idx)} → {max_normal_samples} "
                f"(silinen {dropped_normal} sample)"
            )
            # Kategori bazli sample sayacini da guncelle (Normal klasorleri)
            for cat in list(per_category_sample_count.keys()):
                if UCF_LABEL_MAP.get(cat, (None,))[0] == 0:
                    per_category_sample_count[cat] = sum(
                        1 for s in all_samples
                        if s["label"] == 0 and s.get("video", "").startswith(cat[:6])
                    )

    # ─── Kaydet ─────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Son etiket dagilimini hesapla
    final_label_counts = Counter(s["label"] for s in all_samples)

    meta = {
        "n_samples": len(all_samples),
        "per_category_videos":  dict(per_category_video_count),
        "per_category_samples": dict(per_category_sample_count),
        "per_category_failed":  dict(per_category_failed_videos),
        "label_distribution":   dict(sorted(final_label_counts.items())),
        "normal_cap":           max_normal_samples,
        "normal_dropped":       dropped_normal,
        "label_map":            {k: v[0] for k, v in UCF_LABEL_MAP.items()},
        "created_at":           time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds":      round(time.time() - t_start, 1),
        "feature_dim":          8,
    }
    payload = {"samples": all_samples, "meta": meta}

    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ─── Ozet ────────────────────────────────────────────────────────
    logger.info("════════════ OZET ════════════")
    logger.info(f"  Toplam sample     : {len(all_samples)}")
    logger.info(f"  Sure              : {meta['elapsed_seconds']} s")
    logger.info(f"  Etiket dagilimi   : {dict(sorted(final_label_counts.items()))}")
    if dropped_normal > 0:
        logger.info(f"  Normal cap uygulandi: -{dropped_normal} sample")
    for cat in todo:
        if cat not in UCF_LABEL_MAP:
            continue
        label, level = UCF_LABEL_MAP[cat]
        logger.info(
            f"  {cat:<14} label={label} ({level:<8}) "
            f"videos={per_category_video_count.get(cat, 0):>4}  "
            f"samples={per_category_sample_count.get(cat, 0):>7}  "
            f"failed={per_category_failed_videos.get(cat, 0)}"
        )
    logger.info(f"  Kaydedildi       : {out_path}")
    logger.info("══════════════════════════════")

    return payload


# ─── MLP'yi UCF verisi ile yeniden egit ────────────────────────────────────

def train_on_real_data(
    pkl_path: Path = OUT_PATH,
    model_out: Path = MODEL_OUT_PATH,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 64,
    val_split: float = 0.2,
    seed: int = 42,
    anomaly_oversample: float = 3.0,
) -> dict:
    """
    ucf_features.pkl'i yukler ve ThreatMLP'yi gercek UCF verisiyle yeniden egitir.
    Sentetik egitime kiyasla gercek dunya dagilimini ogrenir.

    Returns:
        {"best_val_acc": float, "train_acc": float, "model_path": str,
         "per_class_f1": {...}}
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    from src.behavior.threat_mlp import ThreatMLP

    if not pkl_path.exists():
        raise FileNotFoundError(
            f"{pkl_path} bulunamadi. Once feature cikarin:\n"
            f"  python tools/extract_features_ucf.py"
        )

    logger.info(f"UCF feature dataset yukleniyor: {pkl_path}")
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)

    samples = payload["samples"]
    if not samples:
        raise RuntimeError("Pickle bos — feature yok, egitilemiyor.")

    X = np.stack([s["features"] for s in samples]).astype(np.float32)
    y = np.array([s["label"]    for s in samples], dtype=np.int64)

    # ─── Stratified split ─────────────────────────────────────────
    # Video bazli split kucuk dataset'te tum sinifin tek video'ya
    # dusmesine ve egitim setinin o sinifi hic gormemesine yol aciyor
    # (ornek: Normal tamamen val'e, anomaliler train'e dusuyor).
    # Bu yuzden sample bazli STRATIFIED split kullaniyoruz: her sinifin
    # %80'i train, %20'si val'e gider. Boylece hem train hem val her
    # sinifi temsil eder. Cross-video genelleme testi icin ileride
    # ayri bir held-out test seti yapilabilir.
    try:
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        raise ImportError(
            "scikit-learn gerekli. Kurulum: pip install scikit-learn"
        ) from e

    # Bir sinifta 1'den az sample varsa stratify patlar; emniyet kontrolu:
    uniq, counts = np.unique(y, return_counts=True)
    if (counts < 2).any():
        rare = {int(c): int(n) for c, n in zip(uniq, counts) if n < 2}
        logger.warning(
            f"Bazi sinifta stratify icin yetersiz sample var: {rare}. "
            f"Random split'e dusuluyor."
        )
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))
        split_i = int((1 - val_split) * len(X))
        X_train, X_val = X[idx[:split_i]], X[idx[split_i:]]
        y_train, y_val = y[idx[:split_i]], y[idx[split_i:]]
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=val_split,
            stratify=y,
            random_state=seed,
            shuffle=True,
        )

    logger.info(f"Train: {len(X_train)} sample / Val: {len(X_val)} sample")
    logger.info(f"Train label dagilimi: {dict(sorted(Counter(y_train.tolist()).items()))}")
    logger.info(f"Val   label dagilimi: {dict(sorted(Counter(y_val.tolist()).items()))}")

    # ─── Sinif agirliklari (dengesizlik icin) ─────────────────────
    # Yoklugu olan sinif icin weight=0 olur ve toplam ortalamaya
    # girmez. np.where + divide uyarisini onlemek icin safe hesap:
    class_counts = np.bincount(y_train, minlength=4).astype(np.float32)
    present = class_counts > 0
    class_weights = np.zeros(4, dtype=np.float32)
    class_weights[present] = 1.0 / class_counts[present]
    if class_weights.sum() > 0:
        class_weights = class_weights / class_weights.sum() * present.sum()
    logger.info(f"Class weights: {class_weights.tolist()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ThreatMLP(input_dim=8, hidden_dims=[32, 16], output_dim=4).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(device)
    )
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    Xt = torch.tensor(X_train).to(device)
    yt = torch.tensor(y_train).to(device)
    Xv = torch.tensor(X_val).to(device)
    yv = torch.tensor(y_val).to(device)

    # ─── WeightedRandomSampler: anomali sinifini oversample et ─────
    # class_weights (loss icin) dengesizligi cezalandirir ama modelin
    # gradient'i yine cogunluk sinifina dogru egilir. Sampling ile her
    # batch'te anomali ornegi sayisi fiziksel olarak artar.
    #
    # Per-sample weight:
    #   label 0 (Normal)    → 1.0
    #   label 1,2,3 (anomaly) → anomaly_oversample (varsayilan 3.0)
    #
    # Ornek: anomaly_oversample=3.0 ise her anomali sample'i Normal'in 3x
    # olasiligiyla cekilir → ~1:3 fiili oran. Replacement=True oldugu icin
    # az sample'li sinif tekrar tekrar gorunur.
    per_sample_weights = np.where(y_train == 0, 1.0, float(anomaly_oversample))
    per_sample_weights = torch.tensor(per_sample_weights, dtype=torch.double)

    # num_samples: 1 epoch'ta cekilen toplam ornek. Training setinin
    # fiili boyutunu koruyoruz ki toplam gradient adimi orantili kalsin.
    sampler = WeightedRandomSampler(
        weights=per_sample_weights,
        num_samples=len(y_train),
        replacement=True,
    )

    loader = DataLoader(
        TensorDataset(Xt, yt),
        batch_size=batch_size,
        sampler=sampler,                  # shuffle kullanilmaz (exclusive)
    )

    logger.info(
        f"WeightedRandomSampler aktif: anomaly_weight={anomaly_oversample}x, "
        f"epoch basi {len(y_train)} sample (replacement=True)"
    )

    best_val_acc = 0.0
    model_out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        scheduler.step()
        epoch_loss /= len(Xt)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_pred = model(Xv).argmax(dim=1)
                val_acc  = (val_pred == yv).float().mean().item()
                train_pred = model(Xt).argmax(dim=1)
                train_acc  = (train_pred == yt).float().mean().item()
            logger.info(
                f"Epoch {epoch:>3}/{epochs} | loss={epoch_loss:.4f} | "
                f"train_acc={train_acc:.3f} | val_acc={val_acc:.3f}"
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), model_out)
                logger.info(f"  ↳ En iyi model kaydedildi: {model_out}")

    # Sinif bazli F1
    model.eval()
    with torch.no_grad():
        val_pred = model(Xv).argmax(dim=1).cpu().numpy()
    per_class_f1 = {}
    for cls in range(4):
        tp = int(((val_pred == cls) & (y_val == cls)).sum())
        fp = int(((val_pred == cls) & (y_val != cls)).sum())
        fn = int(((val_pred != cls) & (y_val == cls)).sum())
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        per_class_f1[cls] = round(f1, 3)

    logger.info("════════════ EGITIM SONUCU ════════════")
    logger.info(f"  Best Val Acc  : {best_val_acc:.3f}")
    logger.info(f"  Per-class F1  : {per_class_f1}")
    logger.info(f"  Kaydedildi    : {model_out}")
    logger.info("══════════════════════════════════════")

    return {
        "best_val_acc": best_val_acc,
        "per_class_f1": per_class_f1,
        "model_path":   str(model_out),
        "n_train":      int(len(X_train)),
        "n_val":        int(len(X_val)),
    }


# ─── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--categories", nargs="+", default=None,
        help="Islenecek kategoriler (ornek: Burglary Fighting). Bos=hepsi.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Kategori basina en fazla kac video islensin.",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Video basina en fazla kac frame islensin.",
    )
    parser.add_argument(
        "--out", type=Path, default=OUT_PATH,
        help="Cikti pkl yolu (varsayilan: data/ucf_features.pkl).",
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Feature cikarimindan sonra MLP'yi de UCF verisiyle egit.",
    )
    parser.add_argument(
        "--train-only", action="store_true",
        help="Sadece mevcut pkl'den MLP'yi egit, feature cikarma.",
    )
    parser.add_argument(
        "--epochs", type=int, default=200,
        help="train(-only) icin epoch sayisi.",
    )
    parser.add_argument(
        "--max-normal", type=int, default=5000,
        help=("Normal (label=0) sinifi icin sample ust siniri. "
              "UCF'de Normal cok baskin; bu cap ile class imbalance "
              "bastirilir. 0 veya negatif = sinir yok. Varsayilan: 5000."),
    )
    parser.add_argument(
        "--anomaly-oversample", type=float, default=3.0,
        help=("Egitimde anomali sinif sample agirligi (WeightedRandomSampler). "
              "1.0 = kapali. Varsayilan: 3.0 → anomaliler 3x daha sik cekilir."),
    )
    parser.add_argument(
        "--conf", type=float, default=None, metavar="P",
        help=(
            "YOLO tespit guven esigi; sadece bu scriptin bellekteki config'ine uygulanir "
            "(config/config.yaml dosyasini degistirmez). Ornek: 0.45"
        ),
    )
    args = parser.parse_args()

    normal_cap = args.max_normal if args.max_normal and args.max_normal > 0 else None

    if args.train_only:
        train_on_real_data(
            pkl_path=args.out,
            epochs=args.epochs,
            anomaly_oversample=args.anomaly_oversample,
        )
        return

    build_dataset(
        categories=args.categories,
        videos_per_category=args.limit,
        max_frames_per_video=args.max_frames,
        out_path=args.out,
        max_normal_samples=normal_cap,
        detector_confidence=args.conf,
    )

    if args.train:
        train_on_real_data(
            pkl_path=args.out,
            epochs=args.epochs,
            anomaly_oversample=args.anomaly_oversample,
        )


if __name__ == "__main__":
    main()
