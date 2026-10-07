#!/usr/bin/env python3
"""
Uji pipeline dengan WEBCAM SUNGGUHAN — jalankan ini di laptop Lenovo V14 G2 ITL
Anda sendiri (bukan di sandbox ini, yang tidak punya kamera).

Ini BUKAN tes otomatis (tidak ada assert) — ini alat verifikasi visual: jendela
akan terbuka menampilkan feed kamera + kotak deteksi + FPS, supaya Anda bisa
langsung lihat apakah:
  1. Kamera terbaca dengan benar (index benar, resolusi benar)
  2. Model ONNX bisa jalan di kamera real-time tanpa crash
  3. FPS yang didapat di hardware Anda sungguhan (bukan cuma sandbox)
  4. Provider ONNX Runtime yang dipakai: CUDA (MX350) atau CPU fallback

CATATAN JUJUR: karena model yang dipakai sekarang (models/yolo11n.onnx) masih
YOLO11n bawaan COCO (80 objek umum: orang, botol, buku, dst — BUKAN model obat),
jangan kaget kalau "Paracetamol" tidak terdeteksi sebagai kelas sendiri. Yang
perlu diverifikasi di tahap ini HANYA: kamera nyala, model jalan, FPS wajar,
tidak crash. Deteksi obat yang akurat baru muncul setelah Anda melatih model
sendiri (lihat scripts/train.py dan notebooks/train_on_kaggle.ipynb).

Persiapan di laptop Anda (BUKAN di sandbox ini):
    git clone / copy folder proyek ini ke laptop
    python3 -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    # opsional, untuk pakai GPU MX350:
    #   pip install onnxruntime-gpu==1.18.0   (cocok dgn CUDA 12.x, lihat dokumen desain bag. 3/9)

Cara pakai:
    python scripts/test_webcam.py                     # kamera index 0, deteksi COCO biasa
    python scripts/test_webcam.py --camera 1           # kamera index lain (mis. webcam eksternal)
    python scripts/test_webcam.py --conf 0.25          # turunkan threshold confidence
    python scripts/test_webcam.py --onnx models/medicine_yolo11n.onnx --classes data/classes.txt
                                                        # setelah Anda punya model obat sendiri

Tombol saat berjalan:
    q / ESC  -> keluar
    s        -> simpan frame saat ini ke data/test_assets/webcam_capture_NNN.png
                (berguna untuk mulai mengumpulkan foto obat asli Anda sendiri!)
"""
import argparse
import pathlib
import sys
import time

import cv2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve
from src.detector import YoloOnnxDetector
from src.pipeline import COCO_CLASSES
from src.utils import draw_detections


def load_class_names(path: str | None) -> list:
    if not path:
        return COCO_CLASSES
    lines = pathlib.Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="index kamera (default 0)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--conf", type=float, default=None, help="override conf_threshold dari config")
    ap.add_argument("--onnx", type=str, default=None, help="path model ONNX lain (mis. hasil training sendiri)")
    ap.add_argument("--classes", type=str, default=None, help="file .txt daftar nama kelas (satu per baris), untuk model custom")
    ap.add_argument("--save-dir", type=str, default="data/test_assets", help="folder simpan frame (tombol 's')")
    ap.add_argument("--config", type=str, default=None,
                     help="pakai profil config lain, mis. config/book_pipeline_config.yaml (default: "
                          "config/pipeline_config.yaml, domain obat). Script ini tidak berubah sama sekali "
                          "untuk domain baru -- cukup ganti config + --onnx + --classes.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    onnx_path = resolve(args.onnx) if args.onnx else resolve(cfg["model"]["onnx_path"])
    class_names = load_class_names(args.classes)
    conf = args.conf if args.conf is not None else cfg["model"]["conf_threshold"]

    if not onnx_path.exists():
        print(f"Error: model tidak ditemukan di {onnx_path}")
        print("Jalankan scripts/export_model.py dulu, atau tunjuk --onnx ke model Anda.")
        sys.exit(1)

    print(f"Memuat model: {onnx_path}")
    detector = YoloOnnxDetector(
        str(onnx_path),
        input_size=cfg["model"]["input_size"],
        conf_threshold=conf,
        class_names=class_names,
    )
    print(f"Provider ONNX Runtime aktif: {detector.session.get_providers()}")
    if "CUDAExecutionProvider" not in detector.session.get_providers():
        print("(CPU saja terpakai — jika Anda mengharapkan MX350/CUDA, cek instalasi onnxruntime-gpu "
              "dan driver CUDA 12.x, lihat dokumen desain bagian 3 & 9)")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"Error: tidak bisa membuka kamera index {args.camera}.")
        print("Coba index lain (--camera 1, --camera 2, ...) atau cek `ls /dev/video*` (Linux) / Device Manager (Windows).")
        sys.exit(1)

    save_dir = resolve(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_count = 0

    print("\nKamera aktif. Tekan 'q' atau ESC untuk keluar, 's' untuk simpan frame saat ini.\n")
    fps_smooth = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Gagal membaca frame dari kamera — berhenti.")
                break

            t0 = time.time()
            detections, _ = detector.infer(frame)
            dt = time.time() - t0
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            fps_smooth = inst_fps if fps_smooth == 0 else (0.9 * fps_smooth + 0.1 * inst_fps)

            vis = draw_detections(frame, detections)
            cv2.putText(vis, f"FPS: {fps_smooth:.1f}  deteksi: {len(detections)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Uji Webcam - Medicine Shelf Vision (q/ESC keluar, s simpan)", vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                out_path = save_dir / f"webcam_capture_{save_count:03d}.png"
                cv2.imwrite(str(out_path), frame)
                print(f"[simpan] {out_path}")
                save_count += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\nSelesai. Ringkasan:")
    print(f"  - Model dipakai : {onnx_path.name}")
    print(f"  - Provider      : {detector.session.get_providers()[0]}")
    print(f"  - FPS terakhir  : {fps_smooth:.1f}")
    print(f"  - Frame disimpan: {save_count}")
    if save_count > 0:
        print(f"  -> {save_count} foto tersimpan di {save_dir} — bisa jadi awal dataset foto obat asli Anda.")


if __name__ == "__main__":
    main()
