#!/usr/bin/env python3
"""
Konversi dataset ekspor Roboflow supaya siap dipakai untuk training deteksi
(YOLO11n) dengan scripts/train.py / notebooks/train_on_kaggle.ipynb.

KENAPA SCRIPT INI PERLU ADA (temuan nyata dari dataset "Pharmacy" Roboflow
yang Anda unggah): meskipun diekspor dalam format "YOLOv8", sebagian besar
baris labelnya BUKAN bounding box biasa (class x_center y_center width height
= 5 angka), melainkan POLIGON (class lalu titik x,y berulang = 9, 11, 13, ...
angka tergantung jumlah titik) -- artinya project Roboflow itu dianotasi
sebagai instance segmentation/poligon, bukan kotak lurus. Format label deteksi
YOLO (txt: class cx cy w h) SAMA PERSIS di YOLOv5/v8/v9/v11 -- jadi ekspor
"YOLOv8" dari Roboflow tetap 100% kompatibel untuk training YOLO11n, ASALKAN
labelnya memang bounding box. Kalau berupa poligon seperti kasus ini, harus
dikonversi dulu ke kotak (bounding rectangle dari titik-titik poligon) --
itulah yang dilakukan script ini.

Yang dilakukan:
- Menyalin images/ apa adanya.
- Untuk tiap baris label:
    * 5 token (class cx cy w h)      -> disalin apa adanya (sudah bbox)
    * >5 token, genap (class x1 y1 x2 y2 ...) -> dihitung bounding rectangle
      (min/max x,y dari semua titik) lalu ditulis ulang sebagai cx cy w h
- Menulis ulang data.yaml di folder output, path relatif disesuaikan.
- Melaporkan statistik: berapa baris bbox asli vs berapa dikonversi dari poligon.

Usage:
    python scripts/convert_roboflow_dataset.py --input path/ke/hasil_unzip_roboflow --output data/pharmacy_converted
"""
import argparse
import pathlib
import shutil

import yaml

SPLITS = ("train", "valid", "test")


def convert_label_line(line: str):
    """Return (cx, cy, w, h) tuple for one YOLO-format label line's coords,
    converting a polygon to its bounding rectangle if needed. Returns None
    for a malformed/empty line."""
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls = parts[0]
    coords = [float(v) for v in parts[1:]]

    if len(coords) == 4:
        cx, cy, w, h = coords
        return cls, cx, cy, w, h

    if len(coords) % 2 != 0:
        return None  # malformed: odd number of polygon coordinate values

    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min
    return cls, cx, cy, w, h


def convert_split(src_root: pathlib.Path, dst_root: pathlib.Path, split: str):
    src_images = src_root / split / "images"
    src_labels = src_root / split / "labels"
    if not src_images.exists():
        return 0, 0, 0  # split not present (e.g. no test/ folder)

    dst_images = dst_root / split / "images"
    dst_labels = dst_root / split / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    n_bbox, n_polygon, n_bad = 0, 0, 0
    for img_path in src_images.iterdir():
        if not img_path.is_file():
            continue
        shutil.copy2(img_path, dst_images / img_path.name)

        label_path = src_labels / (img_path.stem + ".txt")
        out_lines = []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                n_coords = len(line.split()) - 1
                result = convert_label_line(line)
                if result is None:
                    n_bad += 1
                    continue
                cls, cx, cy, w, h = result
                if n_coords == 4:
                    n_bbox += 1
                else:
                    n_polygon += 1
                out_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        (dst_labels / (img_path.stem + ".txt")).write_text("\n".join(out_lines) + ("\n" if out_lines else ""))

    return n_bbox, n_polygon, n_bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="folder hasil unzip ekspor Roboflow (berisi data.yaml, train/, valid/, ...)")
    ap.add_argument("--output", required=True, help="folder tujuan dataset yang sudah dikonversi ke bbox murni")
    args = ap.parse_args()

    src_root = pathlib.Path(args.input).resolve()
    dst_root = pathlib.Path(args.output).resolve()
    data_yaml_path = src_root / "data.yaml"
    if not data_yaml_path.exists():
        print(f"Error: {data_yaml_path} tidak ditemukan -- pastikan --input menunjuk ke folder hasil unzip Roboflow.")
        raise SystemExit(1)

    src_cfg = yaml.safe_load(data_yaml_path.read_text())
    dst_root.mkdir(parents=True, exist_ok=True)

    print(f"Sumber : {src_root}")
    print(f"Tujuan : {dst_root}")
    print(f"Kelas  : {src_cfg.get('nc')} -> {src_cfg.get('names')}\n")

    totals = {"bbox": 0, "polygon": 0, "bad": 0}
    present_splits = []
    for split in SPLITS:
        n_bbox, n_polygon, n_bad = convert_split(src_root, dst_root, split)
        if n_bbox + n_polygon + n_bad == 0 and not (src_root / split / "images").exists():
            continue
        present_splits.append(split)
        totals["bbox"] += n_bbox
        totals["polygon"] += n_polygon
        totals["bad"] += n_bad
        print(f"[{split}] bbox asli: {n_bbox}, dikonversi dari poligon: {n_polygon}, baris rusak dilewati: {n_bad}")

    dst_cfg = {
        "train": "../train/images" if "train" in present_splits else None,
        "val": "../valid/images" if "valid" in present_splits else None,
        "test": "../test/images" if "test" in present_splits else None,
        "nc": src_cfg.get("nc"),
        "names": src_cfg.get("names"),
    }
    dst_cfg = {k: v for k, v in dst_cfg.items() if v is not None}
    (dst_root / "data.yaml").write_text(yaml.dump(dst_cfg, sort_keys=False))

    print(f"\nTotal: {totals['bbox']} bbox asli, {totals['polygon']} dikonversi dari poligon, "
          f"{totals['bad']} baris rusak dilewati.")
    print(f"Selesai. data.yaml baru: {dst_root / 'data.yaml'}")
    print(f"Training: python scripts/train.py --data {dst_root / 'data.yaml'} --epochs 120")


if __name__ == "__main__":
    main()
