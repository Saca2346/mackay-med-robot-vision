#!/usr/bin/env python3
"""
Correctness check for src/detector.py's hand-written ONNX decode (letterbox,
xywh->xyxy, NMS) against Ultralytics' own reference YOLO(...).predict() on the
same ONNX file and a real photo. This is the test that actually matters for
detector.py, separate from the medicine-domain smoke test in run_phase1.py:
the synthetic shelf drawings mostly yield zero detections (expected, a COCO
model doesn't know hand-drawn boxes), so THIS test is what proves the box
math and NMS are implemented correctly before you ever point it at your own
trained medicine model.

Run: python tests/test_detector_accuracy.py
"""
import pathlib
import sys

import cv2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve
from src.detector import YoloOnnxDetector
from src.pipeline import COCO_CLASSES

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"


def check(label, cond):
    print(f"[{PASS if cond else FAIL}] {label}")
    return cond


def main():
    cfg = load_config()
    onnx_path = resolve(cfg["model"]["onnx_path"])  # validasi model yang SEDANG dikonfigurasi (bukan hardcode yolo11n)
    if not onnx_path.exists():
        print(f"Model tidak ada di {onnx_path}. Jalankan scripts/export_model.py dulu.")
        sys.exit(1)
    print(f"Menguji model: {onnx_path.name}\n")

    from ultralytics.utils import ASSETS

    img_path = str(pathlib.Path(ASSETS) / "bus.jpg")
    img = cv2.imread(img_path)

    detector = YoloOnnxDetector(str(onnx_path), input_size=640, conf_threshold=0.35, class_names=COCO_CLASSES)
    ours, ms = detector.infer(img)

    from ultralytics import YOLO

    ref_model = YOLO(str(onnx_path))
    ref = ref_model.predict(img_path, conf=0.35, verbose=False)[0]

    ok = True
    ok &= check(f"jumlah deteksi sama: kita={len(ours)} vs ultralytics={len(ref.boxes)}", len(ours) == len(ref.boxes))

    for i, (mine, box) in enumerate(zip(ours, ref.boxes)):
        ref_name = ref.names[int(box.cls)]
        ref_conf = float(box.conf)
        ref_xyxy = box.xyxy[0].tolist()
        name_ok = mine.class_name == ref_name
        conf_ok = abs(mine.score - ref_conf) < 0.01
        box_ok = all(abs(a - b) <= 2.0 for a, b in zip(mine.box, ref_xyxy))  # allow 2px rounding
        ok &= check(
            f"deteksi #{i} ({ref_name}): kelas={'sama' if name_ok else 'BEDA'}, "
            f"conf {'cocok' if conf_ok else 'BEDA'} ({mine.score:.3f} vs {ref_conf:.3f}), "
            f"box {'cocok' if box_ok else 'BEDA'}",
            name_ok and conf_ok and box_ok,
        )

    # save a visual proof image
    from src.utils import draw_detections

    annotated = draw_detections(img, ours)
    out_path = resolve("data/test_assets/detector_accuracy_proof.png")
    cv2.imwrite(str(out_path), annotated)
    print(f"\nBukti visual disimpan: {out_path}")
    print(f"Waktu inferensi (CPU, sandbox ini): {ms:.1f}ms — akan jauh lebih cepat di MX350 (lihat bagian 1 dokumen desain)")

    print("\n" + ("SEMUA TES LULUS — decoder ONNX custom terbukti benar" if ok else "ADA TES YANG GAGAL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
