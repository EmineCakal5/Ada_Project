"""
Video indirici — Pexels'ten ucretsiz guvenlik kamerasi videolari indirir.
Calistir: python tools/download_videos.py
"""

import urllib.request
import os
import sys

# Bu videolar Creative Commons / ucretsiz lisanslidir
# Pexels ve pixabay uzerindeki dogrudan mp4 linkleri
VIDEOS = [
    {
        "name": "scenario_zone_violation.mp4",
        "url": "https://videos.pexels.com/video-files/3048996/3048996-uhd_2560_1440_25fps.mp4",
        "desc": "Senaryo 1 - Bolge ihlali (yuruyucu)"
    },
    {
        "name": "scenario_loitering.mp4",
        "url": "https://videos.pexels.com/video-files/2022395/2022395-hd_1920_1080_30fps.mp4",
        "desc": "Senaryo 3 - Anormal bekleme"
    },
    {
        "name": "scenario_abandoned.mp4",
        "url": "https://videos.pexels.com/video-files/3044942/3044942-hd_1920_1080_25fps.mp4",
        "desc": "Senaryo 2 - Terk edilmis nesne"
    },
]

OUT_DIR = "data/scenarios"
os.makedirs(OUT_DIR, exist_ok=True)


def download(url, dest, desc):
    print(f"  Indiriliyor: {desc}")
    print(f"  -> {dest}")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk = 8192
            with open(dest, "wb") as f:
                while True:
                    data = response.read(chunk)
                    if not data:
                        break
                    f.write(data)
                    downloaded += len(data)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r     {pct:.1f}% ({downloaded//1024}KB / {total//1024}KB)", end="")
        print(f"\n  [OK] Indirildi: {dest}\n")
        return True
    except Exception as e:
        print(f"\n  [FAIL] Indirilemedi: {e}\n")
        # Bos placeholder olustur
        return False


def create_test_video_opencv(path, desc):
    """Gercek video bulunamazsa OpenCV ile sentetik test videosu olustur."""
    try:
        import cv2
        import numpy as np
        print(f"  Sentetik video olusturuluyor: {path}")
        w, h, fps, duration = 960, 540, 15, 30
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(path, fourcc, fps, (w, h))

        for frame_no in range(fps * duration):
            t = frame_no / fps
            frame = np.zeros((h, w, 3), dtype=np.uint8)

            # Arka plan gradyeni
            for y in range(h):
                frame[y, :] = [int(20 + y*0.05), int(20 + y*0.03), int(30 + y*0.04)]

            # Hareketli "kisi" simulasyonu
            px = int(100 + t * 25) % (w - 60)
            py = int(h // 2 + 40 * (0.5 - abs((t % 4) / 4 - 0.5)))
            cv2.rectangle(frame, (px, py - 50), (px + 30, py + 20), (180, 140, 100), -1)
            cv2.circle(frame, (px + 15, py - 65), 18, (200, 160, 120), -1)

            # Bolge gosterimi
            cv2.rectangle(frame, (300, 150), (600, 400), (0, 80, 0), 2)
            cv2.putText(frame, "IZLEME BOLGESI", (305, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

            # Bilgi
            cv2.putText(frame, desc, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, f"t={t:.1f}s", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            cv2.putText(frame, "SENTETIK VIDEO - Gercek video icin README.md oku", (20, h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

            out.write(frame)

        out.release()
        print(f"  [OK] Sentetik video olusturuldu: {path} ({fps*duration} frame)\n")
        return True
    except Exception as e:
        print(f"  [FAIL] Sentetik video olusturulamadi: {e}\n")
        return False


def main():
    print("\n=== Video Indirici / Olusturucu ===\n")
    print("Once internetten indirmeye calisacak,")
    print("basarisiz olursa sentetik test videosu olusturacak.\n")

    for v in VIDEOS:
        dest = os.path.join(OUT_DIR, v["name"])
        if os.path.exists(dest) and os.path.getsize(dest) > 10000:
            print(f"  [SKIP] Mevcut: {dest}\n")
            continue

        ok = download(v["url"], dest, v["desc"])
        if not ok or os.path.getsize(dest) < 10000:
            # Sentetik olustur
            create_test_video_opencv(dest, v["desc"])

    # Test videosu da olustur
    test_path = "data/test_videos/test.mp4"
    os.makedirs("data/test_videos", exist_ok=True)
    if not os.path.exists(test_path) or os.path.getsize(test_path) < 10000:
        create_test_video_opencv(test_path, "Test Videosu - Genel Demo")

    print("\nSonuc:")
    for v in VIDEOS:
        path = os.path.join(OUT_DIR, v["name"])
        status = "OK" if os.path.exists(path) else "EKSIK"
        size = os.path.getsize(path) // 1024 if os.path.exists(path) else 0
        print(f"  [{status}] {v['name']} ({size} KB)")

    print("\nDashboard'u calistirin:")
    print("  python -m streamlit run src/dashboard/app.py\n")


if __name__ == "__main__":
    main()
