"""
VisDrone2019 -> YOLO Donusturucu
=================================
Ada_Project / IHA - Drone Guvenlik Modulu

Bu script, VisDrone2019-DET veri setinin "annotations/" klasorundeki
TXT etiketlerini Ultralytics YOLO formatina (normalize edilmis x_center,
y_center, width, height) cevirir ve projenin beklediği

    data/visdrone/
        images/train, images/val, images/test
        labels/train, labels/val, labels/test

klasor yapisini olusturur. Son olarak Ultralytics'in egitim sirasinda
okuyacagi `data/visdrone/visdrone.yaml` dosyasini da yazar.

VisDrone orijinal etiket formati (her satir):
    bbox_left, bbox_top, bbox_width, bbox_height, score,
    object_category, truncation, occlusion

object_category degerleri:
    0 = ignored regions  (atilir)
    1 = pedestrian
    2 = people
    3 = bicycle
    4 = car
    5 = van
    6 = truck
    7 = tricycle
    8 = awning-tricycle
    9 = bus
    10 = motor
    11 = others          (atilir)

YOLO icin 1..10 -> 0..9 araligina remap edilir (bkz. CATEGORY_MAP).

Kullanim:
    python tools/prepare_visdrone.py \
        --src  D:/datasets/VisDrone2019 \
        --dst  data/visdrone
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import yaml
from tqdm import tqdm


VISDRONE_SPLITS: Dict[str, str] = {
    "train": "VisDrone2019-DET-train",
    "val":   "VisDrone2019-DET-val",
    "test":  "VisDrone2019-DET-test-dev",
}

CATEGORY_MAP: Dict[int, int] = {
    1: 0,   # pedestrian
    2: 1,   # people
    3: 2,   # bicycle
    4: 3,   # car
    5: 4,   # van
    6: 5,   # truck
    7: 6,   # tricycle
    8: 7,   # awning-tricycle
    9: 8,   # bus
    10: 9,  # motor
}

CLASS_NAMES: List[str] = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]


def _ensure_dirs(dst_root: Path) -> None:
    """YOLO'nun beklediği images/labels klasor agacini olustur."""
    for split in VISDRONE_SPLITS.keys():
        (dst_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def _convert_line(line: str, img_w: int, img_h: int) -> str | None:
    """
    Tek bir VisDrone annotasyon satirini YOLO formatina cevirir.

    Bos satir, kategori 0 (ignored regions) veya 11 (others) durumunda
    None doner; bu ornekler egitimde tamamen atlanmalidir.
    """
    parts = [p for p in line.strip().split(",") if p != ""]
    if len(parts) < 6:
        return None

    try:
        x, y, w, h = map(int, parts[:4])
        category = int(parts[5])
    except ValueError:
        return None

    if category not in CATEGORY_MAP:
        return None
    if w <= 0 or h <= 0:
        return None

    # Koordinatlari goruntu siniri icine sikistir (VisDrone'da tasan kutular olabilir)
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))

    # YOLO: normalize merkez koordinatlari
    xc = (x + w / 2.0) / img_w
    yc = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h

    cls = CATEGORY_MAP[category]
    return f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"


def _process_split(src_split_dir: Path, dst_root: Path, split: str,
                   copy_images: bool = True) -> Tuple[int, int]:
    """
    Tek bir split (train/val/test) icin donusumu calistirir.
    Returns: (islenen_goruntu_sayisi, uretilen_etiket_sayisi)
    """
    img_dir = src_split_dir / "images"
    ann_dir = src_split_dir / "annotations"

    if not img_dir.exists() or not ann_dir.exists():
        print(f"[UYARI] '{src_split_dir}' icinde images/ veya annotations/ "
              f"bulunamadi, atlaniyor.")
        return 0, 0

    out_img_dir = dst_root / "images" / split
    out_lbl_dir = dst_root / "labels" / split

    image_paths = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if not image_paths:
        print(f"[UYARI] '{img_dir}' icinde goruntu bulunamadi.")
        return 0, 0

    n_images, n_labels = 0, 0
    for img_path in tqdm(image_paths, desc=f"[{split}] VisDrone -> YOLO"):
        ann_path = ann_dir / (img_path.stem + ".txt")
        if not ann_path.exists():
            continue

        # Goruntu boyutunu sadece basligi okuyarak degil, guvenli olsun diye
        # cv2 ile aliyoruz (bazi VisDrone goruntulerinde EXIF problemi var).
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Etiketleri donustur
        yolo_lines: List[str] = []
        with open(ann_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                conv = _convert_line(line, w, h)
                if conv is not None:
                    yolo_lines.append(conv)

        # Hic gecerli nesne yoksa bile (negative sample) goruntuyu kopyalayip
        # bos label dosyasi birakmak egitimi bozmaz; istemeyen kullanici icin
        # atlamak da bir secenek: burada sample'i atliyoruz.
        if not yolo_lines:
            continue

        out_lbl_path = out_lbl_dir / (img_path.stem + ".txt")
        out_lbl_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
        n_labels += 1

        out_img_path = out_img_dir / img_path.name
        if copy_images and not out_img_path.exists():
            shutil.copy2(img_path, out_img_path)
        n_images += 1

    return n_images, n_labels


def _write_data_yaml(dst_root: Path) -> Path:
    """
    Ultralytics'in egitim sirasinda kullanacagi dataset yaml'ini yazar.
    """
    yaml_path = dst_root / "visdrone.yaml"
    # Ultralytics `path` alanini mutlak veya gorece kabul eder; tasinabilirlik
    # icin mutlak yol tercih ediyoruz.
    data_cfg = {
        "path":  str(dst_root.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_cfg, f, sort_keys=False, allow_unicode=True)
    return yaml_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VisDrone2019 -> YOLO formati donusturucu"
    )
    p.add_argument("--src", required=True, type=Path,
                   help="VisDrone2019 koku (icinde VisDrone2019-DET-train/val/test-dev olmali).")
    p.add_argument("--dst", type=Path, default=Path("data/visdrone"),
                   help="Cikti klasoru (varsayilan: data/visdrone).")
    p.add_argument("--no-copy-images", action="store_true",
                   help="Goruntuleri kopyalama (disk tasarrufu icin symlink tercih eden kullanicilar icin).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    src_root: Path = args.src.expanduser().resolve()
    dst_root: Path = args.dst.expanduser().resolve()

    if not src_root.exists():
        print(f"[HATA] Kaynak bulunamadi: {src_root}")
        return 1

    print(f"[*] Kaynak : {src_root}")
    print(f"[*] Hedef  : {dst_root}")
    _ensure_dirs(dst_root)

    total_imgs, total_lbls = 0, 0
    for split, subdir in VISDRONE_SPLITS.items():
        src_split_dir = src_root / subdir
        n_img, n_lbl = _process_split(
            src_split_dir, dst_root, split,
            copy_images=not args.no_copy_images,
        )
        total_imgs += n_img
        total_lbls += n_lbl
        print(f"    [{split}] {n_img} goruntu, {n_lbl} etiket dosyasi")

    yaml_path = _write_data_yaml(dst_root)
    print(f"\n[OK] Donusum tamamlandi. Toplam {total_imgs} goruntu / {total_lbls} etiket.")
    print(f"[OK] Dataset YAML: {yaml_path}")
    print("     Egitim icin:  python tools/train_drone_yolo.py --data "
          f"{yaml_path.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
