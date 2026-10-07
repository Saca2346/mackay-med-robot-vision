"""Camera/frame-source abstraction + drawing helpers.

Supports three sources so the exact same pipeline code runs on the real laptop
(webcam / RealSense color stream) and in a dev sandbox with no camera attached:
  - "webcam"     : cv2.VideoCapture(device_index)
  - "video_file" : cv2.VideoCapture(video_path)
  - "synthetic"  : cycles through generated test images in synthetic_dir
"""
from __future__ import annotations

import itertools
import pathlib

import cv2
import numpy as np


class FrameSource:
    def __init__(self, cfg: dict):
        self.mode = cfg.get("source", "synthetic")
        self.width = cfg.get("width", 1280)
        self.height = cfg.get("height", 720)
        self._cap = None
        self._synthetic_iter = None

        if self.mode == "webcam":
            self._cap = cv2.VideoCapture(cfg.get("device_index", 0))
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if not self._cap.isOpened():
                raise RuntimeError(
                    "Tidak bisa membuka webcam. Set camera.source ke 'synthetic' atau "
                    "'video_file' di config/pipeline_config.yaml jika tidak ada kamera."
                )
        elif self.mode == "video_file":
            self._cap = cv2.VideoCapture(cfg["video_path"])
            if not self._cap.isOpened():
                raise RuntimeError(f"Tidak bisa membuka video: {cfg['video_path']}")
        elif self.mode == "synthetic":
            from src.config import resolve

            d = resolve(cfg.get("synthetic_dir", "data/test_assets"))
            images = sorted(pathlib.Path(d).glob("*.png"))
            if not images:
                raise RuntimeError(
                    f"Tidak ada gambar sintetis di {d}. Jalankan "
                    "scripts/generate_test_assets.py dulu."
                )
            self._synthetic_paths = images
            self._synthetic_iter = itertools.cycle(images)
        else:
            raise ValueError(f"Unknown camera source: {self.mode}")

    def read(self) -> np.ndarray | None:
        if self.mode in ("webcam", "video_file"):
            ok, frame = self._cap.read()
            if not ok:
                if self.mode == "video_file":
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = self._cap.read()
                if not ok:
                    return None
            return frame
        else:
            path = next(self._synthetic_iter)
            return cv2.imread(str(path))

    def release(self):
        if self._cap is not None:
            self._cap.release()


def draw_detections(img: np.ndarray, detections, extra_labels: dict | None = None) -> np.ndarray:
    """extra_labels: optional {index_in_detections: "extra text"} e.g. OCR/verify result."""
    out = img.copy()
    extra_labels = extra_labels or {}
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = (int(v) for v in det.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
        label = f"{det.class_name} {det.score:.2f}"
        if i in extra_labels:
            label += f" | {extra_labels[i]}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 220, 0), -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return out


def crop_box(img: np.ndarray, box: tuple, pad: int = 4) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    return img[y1:y2, x1:x2]
