"""
Orchestrator wiring Camera -> Detector -> [Verifier -> DB] -> [Depth/3D] -> output,
matching design doc section 2. Each phase script below just toggles which stages run.
"""
from __future__ import annotations

import time

import cv2

from src.config import load_config, resolve
from src.database import MedicineDB
from src.depth_estimator import build_depth_backend, grasp_point_from_detection
from src.detector import YoloOnnxDetector
from src.utils import FrameSource, crop_box, draw_detections
from src.verifier import ObjectVerifier

COCO_CLASSES = [  # only used until a custom medicine-SKU model (section 4) is trained
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush",
]


class MedicineShelfPipeline:
    def __init__(self, cfg_path: str | None = None, enable_verification=False, enable_depth=False):
        self.cfg = load_config(cfg_path)
        self.enable_verification = enable_verification
        self.enable_depth = enable_depth

        model_cfg = self.cfg["model"]
        onnx_path = resolve(model_cfg["onnx_path"])
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"Model ONNX tidak ditemukan di {onnx_path}. Jalankan "
                "scripts/export_model.py dulu (lihat README)."
            )
        self.detector = YoloOnnxDetector(
            onnx_path=str(onnx_path),
            input_size=model_cfg.get("input_size", 640),
            conf_threshold=model_cfg.get("conf_threshold", 0.35),
            iou_threshold=model_cfg.get("iou_threshold", 0.45),
            providers=model_cfg.get("providers"),
            class_names=COCO_CLASSES,
        )
        print(f"[detector] ONNX Runtime execution provider aktif: {self.detector.active_provider}")

        self.source = FrameSource(self.cfg["camera"])

        self.db = None
        self.verifier = None
        if self.enable_verification:
            db_path = resolve(self.cfg["database"]["path"])
            self.db = MedicineDB(str(db_path))
            self.verifier = ObjectVerifier(self.db, self.cfg["verification"])

        self.depth_backend = None
        if self.enable_depth:
            self.depth_backend = build_depth_backend(self.cfg["depth"])

    def process_frame(self, frame):
        detections, infer_ms = self.detector.infer(frame)

        extra_labels = {}
        grasp_points_3d = []

        for i, det in enumerate(detections):
            verify_result = None
            if self.enable_verification and self.verifier is not None:
                crop = crop_box(frame, det.box)
                verify_result = self.verifier.verify(crop)
                if verify_result.sku:
                    extra_labels[i] = f"{verify_result.name} [{verify_result.confidence}]"
                else:
                    extra_labels[i] = f"obat? ocr='{verify_result.ocr_text[:16]}'"

            if self.enable_depth and self.depth_backend is not None:
                gx, gy = grasp_point_from_detection(det.box)
                if self.cfg["depth"]["source"] == "mock":
                    pt3d = self.depth_backend.estimate_from_box(det.box, frame.shape)
                else:
                    depth_frame, _ = self.depth_backend.get_depth_frame()
                    pt3d = self.depth_backend.pixel_to_3d(gx, gy, depth_frame)
                if pt3d is not None:
                    grasp_points_3d.append((i, pt3d))
                    prev = extra_labels.get(i, det.class_name)
                    extra_labels[i] = f"{prev} @ ({pt3d.x:.2f},{pt3d.y:.2f},{pt3d.z:.2f})m"

        annotated = draw_detections(frame, detections, extra_labels)
        return annotated, detections, infer_ms, grasp_points_3d

    def run(self, max_frames: int | None = None, show_window: bool = False, save_dir: str | None = None):
        frame_count = 0
        fps_log = []
        save_path = resolve(save_dir) if save_dir else None
        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)

        while True:
            frame = self.source.read()
            if frame is None:
                break
            t0 = time.perf_counter()
            annotated, detections, infer_ms, grasp_points = self.process_frame(frame)
            total_ms = (time.perf_counter() - t0) * 1000.0
            fps_log.append(1000.0 / total_ms if total_ms > 0 else 0.0)

            print(
                f"frame {frame_count:04d} | deteksi={len(detections)} | "
                f"inferensi={infer_ms:.1f}ms | total={total_ms:.1f}ms | "
                f"~{fps_log[-1]:.1f} FPS" + (f" | 3D={len(grasp_points)}" if self.enable_depth else "")
            )

            if show_window:
                cv2.imshow("Medicine Shelf Vision", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if save_path:
                cv2.imwrite(str(save_path / f"frame_{frame_count:04d}.png"), annotated)

            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break

        self.source.release()
        if show_window:
            cv2.destroyAllWindows()
        if self.db:
            self.db.close()
        if fps_log:
            avg = sum(fps_log) / len(fps_log)
            print(f"\nRata-rata: {avg:.1f} FPS di {frame_count} frame (lihat catatan hardware di README)")
        return frame_count
