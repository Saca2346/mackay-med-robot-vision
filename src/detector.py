"""
YOLO (v8/v11 nano) ONNX inference wrapper.

Matches the architecture's stage 1 ("Lightweight Vision Model") from the design doc,
section 2. Runs via onnxruntime with CUDAExecutionProvider on the real MX350 laptop,
falling back to CPUExecutionProvider automatically (e.g. in this dev sandbox).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class Detection:
    box: tuple  # (x1, y1, x2, y2) in original image pixel coords
    score: float
    class_id: int
    class_name: str


class YoloOnnxDetector:
    def __init__(
        self,
        onnx_path: str,
        input_size: int = 640,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        providers: list[str] | None = None,
        class_names: list[str] | None = None,
    ):
        providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = ort.get_available_providers()
        chosen = [p for p in providers if p in available] or ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(onnx_path, providers=chosen)
        self.active_provider = self.session.get_providers()[0]
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        # COCO-80 default; overridden once a medicine-SKU model is trained (section 4)
        self.class_names = class_names or [str(i) for i in range(1000)]

    # -- preprocessing -----------------------------------------------------
    def _letterbox(self, img: np.ndarray):
        h, w = img.shape[:2]
        r = self.input_size / max(h, w)
        nh, nw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        top = (self.input_size - nh) // 2
        left = (self.input_size - nw) // 2
        canvas[top : top + nh, left : left + nw] = resized
        return canvas, r, left, top

    def _preprocess(self, img: np.ndarray):
        canvas, r, pad_x, pad_y = self._letterbox(img)
        blob = canvas[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, normalize
        blob = blob.transpose(2, 0, 1)[None, ...]  # NCHW
        return np.ascontiguousarray(blob), r, pad_x, pad_y

    # -- postprocessing ------------------------------------------------------
    def _postprocess(self, output: np.ndarray, r, pad_x, pad_y, orig_shape):
        # Ultralytics ONNX export shape: (1, 4+num_classes, num_anchors) -> transpose
        preds = output[0]
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T  # -> (num_anchors, 4+num_classes)

        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(class_ids)), class_ids]

        keep = scores > self.conf_threshold
        boxes_xywh, scores, class_ids = boxes_xywh[keep], scores[keep], class_ids[keep]
        if len(scores) == 0:
            return []

        # xywh (center) -> xyxy, undo letterbox
        cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        x1 = (cx - w / 2 - pad_x) / r
        y1 = (cy - h / 2 - pad_y) / r
        x2 = (cx + w / 2 - pad_x) / r
        y2 = (cy + h / 2 - pad_y) / r
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        H, W = orig_shape[:2]
        boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clip(0, W - 1)
        boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clip(0, H - 1)

        idxs = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(), scores.tolist(), self.conf_threshold, self.iou_threshold
        )
        idxs = np.array(idxs).flatten() if len(idxs) else []

        detections = []
        for i in idxs:
            x1, y1, x2, y2 = boxes_xyxy[i]
            cid = int(class_ids[i])
            name = self.class_names[cid] if cid < len(self.class_names) else str(cid)
            detections.append(
                Detection(box=(float(x1), float(y1), float(x2), float(y2)), score=float(scores[i]), class_id=cid, class_name=name)
            )
        return detections

    def infer(self, img: np.ndarray) -> tuple[list[Detection], float]:
        blob, r, pad_x, pad_y = self._preprocess(img)
        t0 = time.perf_counter()
        output = self.session.run(None, {self.input_name: blob})[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        detections = self._postprocess(output, r, pad_x, pad_y, img.shape)
        return detections, elapsed_ms
