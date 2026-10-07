"""
Deteksi box + size_label memakai model YOLO hasil training sendiri.

Kenapa lewat Ultralytics (bukan decoder ONNX manual seperti src/detector.py):
model kita bertipe OBB (kotak berputar, 4 titik), dan format keluaran ONNX
OBB berbeda antar versi YOLO (YOLO11 vs YOLO26). Ultralytics membaca file
.onnx maupun .pt dan mengembalikan hasil dalam format yang sama, jadi kode ini
tidak perlu diubah saat model diganti. Inferensi tetap di CPU (onnxruntime)
sehingga jalan di laptop MX350 tanpa GPU.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from src.box_geometry import center, frac_inside, point_in_poly, poly_area, to_poly


@dataclass
class Det:
    cls: str            # "box" atau "size_label"
    conf: float         # confidence model, 0..1
    poly: np.ndarray    # (4, 2) titik sudut dalam piksel


@dataclass
class BoxGroup:
    box: Det
    labels: list[Det] = field(default_factory=list)   # size_label milik box ini (boleh >1, mis. Terumo)


class BoxDetector:
    def __init__(self, weights: str, task: str = "obb", imgsz: int = 640, device: str = "cpu",
                 conf: dict | None = None, iou: float = 0.5, class_alias: dict | None = None):
        from ultralytics import YOLO

        # imgsz boleh [tinggi, lebar] untuk model ekspor ukuran tetap (mis. OpenVINO 736x1280 untuk webcam 16:9).
        # Model .pt tidak punya ukuran tetap dan tidak bisa memakai device "intel:..." (itu khusus OpenVINO).
        if str(weights).endswith(".pt"):
            if isinstance(imgsz, (list, tuple)):
                imgsz = max(imgsz)
            if str(device).startswith("intel"):
                print(f"[detector] {device} hanya untuk model OpenVINO -> model .pt dijalankan di cpu")
                device = "cpu"
        self.model = YOLO(str(weights), task=task)
        self.imgsz = imgsz
        self.device = device
        self.conf = {"box": 0.35, "size_label": 0.25, **(conf or {})}
        self.iou = iou
        # nama kelas di model -> nama baku ("box" / "size_label"); huruf besar/kecil diabaikan
        alias = class_alias or {"box": "box", "size_label": "size_label"}
        self.alias = {str(v).lower(): k for k, v in alias.items()}
        # pemanasan: panggilan pertama selalu lambat (memuat / mengompilasi model), jangan ikut dihitung di FPS
        h, w = imgsz if isinstance(imgsz, (list, tuple)) else (imgsz, imgsz)
        self.model.predict(np.zeros((h, w, 3), dtype=np.uint8), imgsz=imgsz, device=device, verbose=False)

    def _std_name(self, name: str) -> str | None:
        return self.alias.get(str(name).lower())

    def detect(self, frame: np.ndarray) -> tuple[list[Det], float]:
        """Satu frame -> daftar deteksi + waktu inferensi (ms)."""
        t0 = time.perf_counter()
        r = self.model.predict(frame, imgsz=self.imgsz, conf=min(self.conf.values()), iou=self.iou,
                               device=self.device, verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000.0
        names = r.names
        dets: list[Det] = []
        if getattr(r, "obb", None) is not None and len(r.obb):
            polys = r.obb.xyxyxyxy.cpu().numpy()
            cls = r.obb.cls.cpu().numpy().astype(int)
            confs = r.obb.conf.cpu().numpy()
        elif getattr(r, "boxes", None) is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            polys = np.stack([xyxy[:, [0, 1]], xyxy[:, [2, 1]], xyxy[:, [2, 3]], xyxy[:, [0, 3]]], axis=1)
            cls = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
        else:
            return dets, ms
        for p, c, s in zip(polys, cls, confs):
            name = self._std_name(names.get(int(c), str(c)) if isinstance(names, dict) else names[int(c)])
            if name is None or float(s) < self.conf.get(name, 0.25):
                continue
            dets.append(Det(cls=name, conf=float(s), poly=to_poly(p)))
        return dets, ms


def group_labels(dets: list[Det], min_overlap: float = 0.5) -> tuple[list[BoxGroup], list[Det]]:
    """
    Pasangkan tiap size_label ke box induknya.
      1. Titik pusat size_label berada DI DALAM poligon box -> box itu induknya
         (kalau lebih dari satu box memuatnya, pilih box terkecil = paling spesifik).
      2. Kalau tidak ada yang memuat: box yang memuat bagian TERBESAR luas size_label, asalkan >= min_overlap.
      3. Selain itu -> size_label yatim (orphan), tidak dipakai.
    Size_label di luar box TIDAK dipasangkan ke box terdekat: di rak itu hampir selalu label kotak SEBELAH yang
    kotaknya sendiri tidak terdeteksi (terukur di rekaman w300), dan teksnya akan terbaca sebagai kotak yang salah.
    """
    boxes = [d for d in dets if d.cls == "box"]
    groups = [BoxGroup(box=b) for b in boxes]
    orphans: list[Det] = []
    for lab in (d for d in dets if d.cls == "size_label"):
        c = center(lab.poly)
        inside = [g for g in groups if point_in_poly(c, g.box.poly)]
        if inside:
            min(inside, key=lambda g: poly_area(g.box.poly)).labels.append(lab)
            continue
        best, best_f = None, 0.0
        for g in groups:
            f = frac_inside(lab.poly, g.box.poly)
            if f > best_f:
                best, best_f = g, f
        (best.labels if best is not None and best_f >= min_overlap else orphans).append(lab)
    return groups, orphans
