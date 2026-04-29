# -*- coding: utf-8 -*-
"""
YOLOv8 Drone / IHA Fine-Tuning Scripti
======================================
Ada_Project -> Drone & IHA Guvenlik Modulu

Bu script, COCO pretrained `yolov8s.pt` agirligini VisDrone2019 veri seti
uzerinde fine-tune eder. Sabit guvenlik kamerasi perspektifinden egitilmis
baz modeli, ucus dinamigine ozgu ko sullar altinda (ani manevra, irtifa
degisimi, acisal kayma) dayanikli hale getirmek icin albumentations tabanli
ozel bir veri artirma boru hatti enjekte eder.

Akis:
    1) Ultralytics'in yerlesik `Albumentations` sinifi monkey-patch ile
       drone'a ozgu donusumlerle degistirilir.
    2) `model.train(...)` ile COCO pretrained yolov8s.pt fine-tune edilir.
    3) Egitim sonunda `runs/.../weights/best.pt` -> models/weights/yolov8s_drone.pt
       yoluna kopyalanir. Mevcut pipeline bu dosyayi config.yaml uzerinden
       dogrudan yukleyebilir (bkz. README / config.yaml).

Kullanim:
    python tools/train_drone_yolo.py \
        --data   data/visdrone/visdrone.yaml \
        --epochs 80 \
        --imgsz  960 \
        --batch  16 \
        --device 0
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging — pipeline ile ayni format (src/pipeline.py ile tutarli olsun diye)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("train_drone_yolo")


# ---------------------------------------------------------------------------
# 1) DRONE'A OZGU ALBUMENTATIONS PIPELINE
# ---------------------------------------------------------------------------
# Ultralytics dahili olarak `ultralytics.data.augment.Albumentations` sinifini
# kullanir; albumentations yukluyse otomatik devreye girer. Biz bu sinifin
# __init__ metodunu override ederek drone senaryosuna ozel donusumleri enjekte
# ediyoruz. Bu yaklasim, ultralytics API'sini kirmadigi icin ileriye donuk
# surum guncellemelerinde de (v8.x) guvenlidir.
#
# Secimlerin gerekcesi:
#   * MotionBlur / GaussianBlur -> ani gimbal hareketi, rotor titresimi,
#                                   hizli yatay / dikey manevralar sirasinda
#                                   olusan hareket bulaniklii.
#   * RandomScale / Affine(scale) -> drone irtifasi degistikce nesne
#                                     olceginin dramatik biciide degismesi.
#   * Perspective                 -> kamera gimbal acisi ve roll/pitch
#                                     degisimlerinden kaynakli perspektif
#                                     sapmalari.
#   * RandomBrightnessContrast /
#     CLAHE                       -> gun icindeki isik degisimleri, havadan
#                                     cekimlerde sik rastlanan overexposure.
#   * ISONoise / ImageCompression -> ucak-yere RF aktariminda uygulanan
#                                     sikistirma ve sensor gurultusu
#                                     taklidini uretir.
# ---------------------------------------------------------------------------
def _install_drone_albumentations() -> None:
    """Ultralytics'in Albumentations sinifini drone'a ozel pipeline ile degistirir."""
    try:
        import albumentations as A
        from ultralytics.data import augment as _ul_augment
    except ImportError as e:
        logger.warning(
            "Albumentations veya ultralytics import edilemedi (%s). "
            "Drone augmentation pipeline'i devre disi.", e
        )
        return

    drone_transforms = [
        # --- Ani manevra / titresim kaynakli bulaniklik ---
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 9), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.25),

        # --- Irtifa degisimi (olcek) + acisal kayma (perspektif) ---
        A.Affine(
            scale=(0.7, 1.3),          # +-%30 irtifa kaynakli olcek simulasyonu
            translate_percent=(0.0, 0.08),
            rotate=(-10, 10),           # yaw eksenindeki kucuk sapmalar
            shear=(-5, 5),
            fit_output=False,
            p=0.5,
        ),
        A.Perspective(scale=(0.03, 0.08), keep_size=True, p=0.35),

        # --- Isik / atmosferik kosullar ---
        A.RandomBrightnessContrast(
            brightness_limit=0.25, contrast_limit=0.25, p=0.4
        ),
        A.CLAHE(clip_limit=(1.0, 3.0), tile_grid_size=(8, 8), p=0.15),
        A.HueSaturationValue(
            hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=0.2
        ),

        # --- Sensor / iletim kaynakli bozulmalar ---
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.4), p=0.15),
        A.ImageCompression(quality_lower=55, quality_upper=95, p=0.2),
    ]

    # --- Ultralytics Albumentations sinifini monkey-patch ---
    def _drone_init(self, p: float = 1.0, transforms=None, **kwargs):
        self.p = p
        self.transform = None
        # Ultralytics'in yeni surumleri bu bayragi bekliyor (spatial transform
        # varsa bbox guncellemesi yapiliyor). Perspective/Affine uzamsaldir.
        self.contains_spatial = True

        try:
            self.transform = A.Compose(
                drone_transforms,
                bbox_params=A.BboxParams(
                    format="yolo",
                    label_fields=["class_labels"],
                    min_visibility=0.1,     # kucuk nesne (VisDrone) icin toleransli
                ),
            )
            logger.info(
                "Drone albumentations pipeline yuklendi (%d transform).",
                len(drone_transforms),
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Albumentations pipeline kurulamadi: %s", exc)
            self.transform = None

    _ul_augment.Albumentations.__init__ = _drone_init


# ---------------------------------------------------------------------------
# 2) EGITIM RUTINI
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> Path:
    """Fine-tuning calistirir, en iyi .pt'yi proje agirlik klasorune kopyalar."""
    from ultralytics import YOLO  # gecikmeli import (monkey-patch sonrasi)

    logger.info("Baslangic agirligi  : %s", args.weights)
    logger.info("Dataset yaml       : %s", args.data)
    logger.info("Epoch / imgsz /batch: %d / %d / %d",
                args.epochs, args.imgsz, args.batch)
    logger.info("Device             : %s", args.device)

    model = YOLO(args.weights)

    # Ultralytics'in kendi HSV / geometri hiperparametreleri; albumentations'a
    # ek olarak calisir. VisDrone (kucuk nesne, ust bakis) icin makul degerler:
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        exist_ok=True,

        # --- Geometrik augmentasyon (albumentations'a ek olarak) ---
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,
        flipud=0.0,          # drone goruntuleri dikey cevirmek yanlistir
        fliplr=0.5,

        # --- Mozaik / mixup kucuk nesne tespitinde oldukca faydalidir ---
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,

        # --- Stabilite ---
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        patience=20,
        cos_lr=True,
        close_mosaic=10,
        workers=args.workers,
        verbose=True,
    )

    # Ultralytics, en iyi agirligi `<project>/<name>/weights/best.pt` altina yazar.
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else (
        args.project / args.name
    )
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Egitim ciktisi bulunamadi: {best_pt}")

    # Proje konvansiyonuna gore kopyala
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt, args.out)
    logger.info("En iyi agirlik kopyalandi: %s -> %s", best_pt, args.out)
    return args.out


# ---------------------------------------------------------------------------
# 3) CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]

    p = argparse.ArgumentParser(
        description="YOLOv8s -> VisDrone fine-tuning (drone perspektifi)"
    )
    p.add_argument("--data", type=Path,
                   default=root / "data" / "visdrone" / "visdrone.yaml",
                   help="prepare_visdrone.py tarafindan uretilen dataset yaml'i.")
    p.add_argument("--weights", type=Path, default=root / "yolov8s.pt",
                   help="Baslangic pretrained agirligi (varsayilan: yolov8s.pt).")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz",  type=int, default=960,
                   help="VisDrone kucuk nesneler icerdigi icin 960 onerilir.")
    p.add_argument("--batch",  type=int, default=16)
    p.add_argument("--device", type=str, default="0",
                   help="'0' / '0,1' / 'cpu'")
    p.add_argument("--workers", type=int, default=4)

    p.add_argument("--project", type=Path, default=root / "runs" / "drone",
                   help="Ultralytics runs klasoru.")
    p.add_argument("--name", type=str, default="yolov8s_visdrone")

    p.add_argument("--out", type=Path,
                   default=root / "models" / "weights" / "yolov8s_drone.pt",
                   help="Fine-tune sonrasi projeye kaydedilecek agirlik yolu.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.data.exists():
        logger.error("Dataset yaml bulunamadi: %s", args.data)
        logger.error("Once `python tools/prepare_visdrone.py --src <VisDrone-koku>` "
                     "komutunu calistirin.")
        return 1

    # Sirasi onemli: YOLO model import'undan ONCE monkey-patch uygulanmali
    # ki egitim sirasinda Albumentations sinifi bizim versiyonumuzu kullansin.
    _install_drone_albumentations()

    out_path = train(args)
    logger.info("Tamamlandi. config.yaml -> detector.model = \"%s\"",
                out_path.relative_to(Path(__file__).resolve().parents[1]).as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main())
