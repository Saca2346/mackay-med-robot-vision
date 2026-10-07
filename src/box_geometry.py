"""
Geometri poligon 4-titik untuk pipeline box + size_label.

Satu kotak = 4 titik sudut (x, y) dalam piksel. Kotak lurus (axis-aligned) juga
disimpan sebagai 4 titik, jadi semua modul memakai satu format yang sama:
  - detector    : hasil YOLO (OBB atau biasa) -> poligon 4 titik
  - verifier    : size_label milik box mana (titik pusat di dalam poligon)
  - evaluasi    : IoU poligon untuk TP / FP / FN
  - cek label   : cari label yang saling tumpang tindih (label "longgar")
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def to_poly(pts) -> np.ndarray:
    """Apa pun bentuknya (list 8 angka, (4,2), tuple) -> array float32 (4, 2)."""
    return np.asarray(pts, dtype=np.float32).reshape(4, 2)


def poly_area(poly) -> float:
    return float(abs(cv2.contourArea(np.asarray(poly, dtype=np.float32).reshape(-1, 2))))


def poly_iou(a, b) -> float:
    """IoU dua poligon cembung (convex): luas irisan / luas gabungan, 0..1."""
    a = cv2.convexHull(to_poly(a)).reshape(-1, 2)
    b = cv2.convexHull(to_poly(b)).reshape(-1, 2)
    inter, _ = cv2.intersectConvexConvex(a, b)
    if inter <= 0:
        return 0.0
    union = poly_area(a) + poly_area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def frac_inside(a, b) -> float:
    """Bagian luas poligon a yang berada di dalam poligon b, 0..1."""
    pa = cv2.convexHull(to_poly(a)).reshape(-1, 2)
    pb = cv2.convexHull(to_poly(b)).reshape(-1, 2)
    area = poly_area(pa)
    if area <= 0:
        return 0.0
    inter, _ = cv2.intersectConvexConvex(pa, pb)
    return float(max(inter, 0.0) / area)


def inside_fraction(poly, frame_size) -> float:
    """Bagian luas poligon yang ada di dalam frame (w, h), 0..1 (0,6 = 60 % kotak terlihat)."""
    w, h = frame_size
    p = cv2.convexHull(to_poly(poly)).reshape(-1, 2)
    area = poly_area(p)
    if area <= 0:
        return 0.0
    inter, _ = cv2.intersectConvexConvex(p, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    return float(max(inter, 0.0) / area)


def poly_iom(a, b) -> float:
    """Luas irisan / luas poligon yang LEBIH KECIL, 0..1. Tinggi kalau satu poligon hampir seluruhnya di dalam
    yang lain (mis. punggung kotak yang sama terdeteksi terpotong), tetap rendah untuk kotak bersebelahan."""
    a = cv2.convexHull(to_poly(a)).reshape(-1, 2)
    b = cv2.convexHull(to_poly(b)).reshape(-1, 2)
    inter, _ = cv2.intersectConvexConvex(a, b)
    small = min(poly_area(a), poly_area(b))
    return float(inter / small) if inter > 0 and small > 0 else 0.0


def center(poly) -> tuple[float, float]:
    p = to_poly(poly)
    return float(p[:, 0].mean()), float(p[:, 1].mean())


def point_in_poly(pt, poly) -> bool:
    hull = cv2.convexHull(to_poly(poly))
    return cv2.pointPolygonTest(hull, (float(pt[0]), float(pt[1])), False) >= 0


def aabb(poly) -> tuple[float, float, float, float]:
    """Kotak lurus terkecil yang memuat poligon: (x1, y1, x2, y2)."""
    p = to_poly(poly)
    return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())


def aabb_poly(poly) -> np.ndarray:
    x1, y1, x2, y2 = aabb(poly)
    return to_poly([x1, y1, x2, y1, x2, y2, x1, y2])


def rotated_rect_poly(cx: float, cy: float, length: float, width: float, theta_deg: float) -> np.ndarray:
    """Persegi panjang panjang x lebar, diputar theta derajat di sekitar (cx, cy)."""
    t = math.radians(theta_deg)
    ux, uy = math.cos(t), math.sin(t)          # arah sisi panjang
    vx, vy = -math.sin(t), math.cos(t)         # arah sisi pendek
    hl, hw = length / 2.0, width / 2.0
    pts = [
        (cx - ux * hl - vx * hw, cy - uy * hl - vy * hw),
        (cx + ux * hl - vx * hw, cy + uy * hl - vy * hw),
        (cx + ux * hl + vx * hw, cy + uy * hl + vy * hw),
        (cx - ux * hl + vx * hw, cy - uy * hl + vy * hw),
    ]
    return to_poly(pts)


def axis_deviation_deg(poly) -> float:
    """Seberapa miring kotak dari posisi lurus: 0 = lurus, 45 = paling miring."""
    (_, _), (w, h), ang = cv2.minAreaRect(to_poly(poly))
    a = abs(ang) % 90.0
    return float(min(a, 90.0 - a))


def order_points(pts) -> np.ndarray:
    """Urutkan 4 titik jadi: kiri-atas, kanan-atas, kanan-bawah, kiri-bawah."""
    p = to_poly(pts)
    s = p.sum(axis=1)
    d = np.diff(p, axis=1).ravel()
    return np.array([p[np.argmin(s)], p[np.argmin(d)], p[np.argmax(s)], p[np.argmax(d)]], dtype=np.float32)


def warp_upright(img: np.ndarray, poly, min_side: int = 48, pad: float = 0.04) -> np.ndarray:
    """
    Potong area poligon (boleh miring) dan luruskan kemiringannya. Arah hasil
    mengikuti arah foto (tidak dipaksa mendatar), karena arah teks berbeda per
    merek: angka Terumo tegak, teks iVascular / Accuforce berjalan vertikal.
    Rotasi 90 derajat untuk OCR dicoba di box_ocr.SizeTextReader.read().
    pad = tambahan tepi (fraksi) supaya huruf di pinggir tidak terpotong.
    """
    (cx, cy), (w, h), ang = cv2.minAreaRect(to_poly(poly))
    w, h = w * (1 + 2 * pad), h * (1 + 2 * pad)
    box = order_points(cv2.boxPoints(((cx, cy), (w, h), ang)))
    tl, tr, br, bl = box
    out_w = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    out_h = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    if out_w < 2 or out_h < 2:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(box, dst)
    crop = cv2.warpPerspective(img, m, (out_w, out_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    if min(crop.shape[:2]) < min_side:
        s = min_side / min(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    return crop


def parse_label_line(line: str, img_w: int, img_h: int):
    """
    Satu baris label YOLO -> (class_id, poligon piksel) atau None.
      OBB  : "cls x1 y1 x2 y2 x3 y3 x4 y4"  (9 angka, dinormalisasi 0..1)
      biasa: "cls cx cy w h"                 (5 angka, dinormalisasi 0..1)
    """
    parts = line.split()
    if len(parts) == 9:
        cls = int(float(parts[0]))
        xy = np.array([float(v) for v in parts[1:]], dtype=np.float32).reshape(4, 2)
        xy[:, 0] *= img_w
        xy[:, 1] *= img_h
        return cls, xy
    if len(parts) == 5:
        cls = int(float(parts[0]))
        cx, cy, w, h = (float(v) for v in parts[1:])
        cx, cy, w, h = cx * img_w, cy * img_h, w * img_w, h * img_h
        return cls, to_poly([cx - w / 2, cy - h / 2, cx + w / 2, cy - h / 2, cx + w / 2, cy + h / 2, cx - w / 2, cy + h / 2])
    return None


def read_label_file(path, img_w: int, img_h: int) -> list[tuple[int, np.ndarray]]:
    out = []
    try:
        text = open(path, "r", encoding="utf-8").read()
    except FileNotFoundError:
        return out
    for line in text.splitlines():
        parsed = parse_label_line(line.strip(), img_w, img_h) if line.strip() else None
        if parsed is not None:
            out.append(parsed)
    return out


def draw_poly(img: np.ndarray, poly, color, thickness: int = 2) -> None:
    cv2.polylines(img, [to_poly(poly).astype(np.int32)], True, color, thickness, cv2.LINE_AA)


def put_label(img: np.ndarray, text: str, org, color, scale: float = 0.5) -> None:
    """Teks dengan latar berwarna supaya terbaca di atas foto."""
    x, y = int(org[0]), int(org[1])
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    y = max(th + 4, y)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 2), color, -1)
    lum = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
    fg = (0, 0, 0) if lum > 140 else (255, 255, 255)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, 1, cv2.LINE_AA)
