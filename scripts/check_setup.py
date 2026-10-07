#!/usr/bin/env python3
"""
Preflight check untuk Jalur A — jalankan ini di LAPTOP ANDA SENDIRI (bukan di
sandbox pengembangan ini) SEBELUM scripts/test_webcam.py, supaya masalah
instalasi/lingkungan ketahuan lewat pesan yang jelas, bukan lewat crash di
tengah pemakaian kamera.

Tidak menyentuh kamera secara visual (tidak memanggil cv2.imshow) — hanya
memeriksa environment, supaya aman dijalankan lewat SSH/terminal tanpa
display juga (hasilnya akan bilang display tidak terdeteksi, itu wajar
kalau memang begitu caranya Anda connect).

Cara pakai:
    python scripts/check_setup.py
"""
import importlib
import importlib.metadata as md
import pathlib
import shutil
import sys

PASS, WARN, FAIL = "\033[92mOK\033[0m", "\033[93mPERINGATAN\033[0m", "\033[91mGAGAL\033[0m"
ROOT = pathlib.Path(__file__).resolve().parent.parent

results = []  # (level, message)


def report(level, message):
    results.append((level, message))
    print(f"[{level}] {message}")


def check_python_version():
    v = sys.version_info
    if v >= (3, 9):
        report(PASS, f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        report(FAIL, f"Python {v.major}.{v.minor}.{v.micro} — butuh >= 3.9")


def check_package(import_name, pip_name=None, required=True):
    pip_name = pip_name or import_name
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", None) or "?"
        report(PASS, f"paket '{pip_name}' terpasang (versi {version})")
        return mod
    except ImportError:
        level = FAIL if required else WARN
        report(level, f"paket '{pip_name}' TIDAK terpasang — jalankan: pip install {pip_name}")
        return None


def check_opencv_gui(cv2_mod):
    if cv2_mod is None:
        return
    # jangan panggil cv2.imshow() di sini — di lingkungan tanpa display/GUI backend
    # ini bisa CRASH proses (SIGABRT), bukan sekadar melempar exception python yang
    # bisa ditangkap. Cukup cek dari metadata paket + build info, jauh lebih aman.
    is_headless = False
    try:
        md.version("opencv-python-headless")
        is_headless = True
    except md.PackageNotFoundError:
        pass

    build_info = cv2_mod.getBuildInformation()
    gui_backends = [b for b in ("GTK", "QT", "Cocoa", "Win32 UI", "WIN32UI") if b in build_info]

    if is_headless and not gui_backends:
        report(
            FAIL,
            "opencv-python-headless terpasang — TIDAK punya GUI backend, "
            "scripts/test_webcam.py (cv2.imshow) akan gagal. Jalankan: "
            "pip uninstall -y opencv-python-headless && pip install opencv-python",
        )
    elif gui_backends:
        report(PASS, f"OpenCV punya GUI backend: {', '.join(gui_backends)} (cv2.imshow akan berfungsi)")
    else:
        report(
            WARN,
            "Tidak bisa pastikan GUI backend OpenCV dari build info — jika "
            "scripts/test_webcam.py nanti gagal di cv2.imshow, install ulang "
            "dengan: pip install opencv-python (bukan versi -headless)",
        )


def check_tesseract():
    path = shutil.which("tesseract")
    if path:
        report(PASS, f"binary tesseract ditemukan: {path}")
    else:
        report(
            FAIL,
            "binary tesseract TIDAK ditemukan di PATH — OCR di verifier.py butuh ini. "
            "Ubuntu/Debian: sudo apt install tesseract-ocr tesseract-ocr-eng | "
            "macOS: brew install tesseract | Windows: installer dari github.com/UB-Mannheim/tesseract",
        )


def check_onnxruntime_providers(ort_mod):
    if ort_mod is None:
        return
    providers = ort_mod.get_available_providers()
    report(PASS, f"provider ONNX Runtime tersedia: {providers}")
    if "CUDAExecutionProvider" in providers:
        report(PASS, "CUDAExecutionProvider tersedia — inferensi akan pakai GPU (MX350) jika config mengizinkan")
    else:
        report(
            WARN,
            "CUDAExecutionProvider TIDAK tersedia — inferensi akan jalan di CPU saja (lebih lambat). "
            "Untuk pakai MX350: pip install onnxruntime-gpu (lepas 'onnxruntime' biasa dulu) "
            "dan pastikan driver NVIDIA + CUDA 12.6/12.8 terpasang (BUKAN CUDA 13.x — itu drop Pascal/MX350).",
        )


def check_camera_devices():
    video_devs = sorted(pathlib.Path("/dev").glob("video*")) if pathlib.Path("/dev").exists() else []
    if video_devs:
        report(PASS, f"perangkat kamera terdeteksi (Linux): {[str(p) for p in video_devs]}")
    else:
        report(
            WARN,
            "tidak ada /dev/video* terdeteksi (normal di macOS/Windows, atau jika ini "
            "dijalankan bukan di laptop dengan kamera) — scripts/test_webcam.py akan "
            "gagal membuka kamera jika memang tidak ada perangkat fisik",
        )


def check_display():
    import os

    if sys.platform.startswith("linux"):
        if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            report(PASS, "environment display terdeteksi (DISPLAY/WAYLAND_DISPLAY di-set)")
        else:
            report(
                WARN,
                "tidak ada DISPLAY/WAYLAND_DISPLAY — jika ini dijalankan lewat SSH tanpa X forwarding, "
                "cv2.imshow di test_webcam.py tidak akan bisa menampilkan jendela. "
                "Jalankan langsung di desktop laptop, atau pakai `ssh -X`.",
            )
    else:
        report(PASS, f"platform {sys.platform} — asumsi display GUI normal tersedia")


def check_model_file():
    onnx_path = ROOT / "models" / "yolo11n.onnx"
    if onnx_path.exists():
        size_mb = onnx_path.stat().st_size / 1e6
        report(PASS, f"model ditemukan: {onnx_path.relative_to(ROOT)} ({size_mb:.1f} MB)")
    else:
        report(
            FAIL,
            f"model TIDAK ditemukan di {onnx_path.relative_to(ROOT)} — jalankan: python scripts/export_model.py",
        )


def main():
    print("=== Preflight check: Medicine Shelf Vision (Jalur A) ===\n")
    check_python_version()
    cv2_mod = check_package("cv2", "opencv-python")
    np_mod = check_package("numpy")
    ort_mod = check_package("onnxruntime")
    check_package("pytesseract")
    check_package("PIL", "Pillow")
    check_package("yaml", "pyyaml")
    check_package("ultralytics", required=False)  # hanya perlu untuk export/training lokal
    print()
    check_opencv_gui(cv2_mod)
    check_tesseract()
    check_onnxruntime_providers(ort_mod)
    check_camera_devices()
    check_display()
    check_model_file()

    print("\n=== Ringkasan ===")
    n_fail = sum(1 for lvl, _ in results if lvl == FAIL)
    n_warn = sum(1 for lvl, _ in results if lvl == WARN)
    if n_fail == 0 and n_warn == 0:
        print("Semua cek lulus — lanjutkan ke: python scripts/test_webcam.py")
    elif n_fail == 0:
        print(f"{n_warn} peringatan (tidak fatal) — bisa lanjut ke scripts/test_webcam.py, "
              "tapi baca peringatan di atas dulu (biasanya soal GPU/tesseract).")
    else:
        print(f"{n_fail} masalah HARUS diperbaiki dulu sebelum scripts/test_webcam.py akan berjalan benar "
              f"({n_warn} peringatan tambahan). Lihat pesan [GAGAL] di atas untuk cara memperbaiki.")
    sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    main()
