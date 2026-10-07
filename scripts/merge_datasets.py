#!/usr/bin/env python3
"""
Menggabungkan beberapa dataset Roboflow (masing-masing sudah bbox format, hasil
scripts/convert_roboflow_dataset.py kalau sumbernya poligon) jadi SATU dataset
training dengan SATU daftar kelas gabungan.

KENAPA INI PERLU: tiap dataset Roboflow yang diekspor terpisah punya index kelas
sendiri-sendiri (mis. kelas 0 di dataset A bisa "panadol", tapi kelas 0 di dataset B
bisa "alphintern") -- menggabungkan folder begitu saja TANPA menulis ulang index label
akan bikin model belajar label yang SALAH. Script ini membangun satu daftar kelas
gabungan (union nama kelas dari semua dataset input, urutan dijaga stabil) lalu
menulis ulang setiap file label supaya menunjuk ke index yang benar di daftar gabungan.

PERINGATAN NAMA MIRIP (tidak digabung otomatis -- butuh keputusan manusia):
kalau dua dataset punya nama kelas yang MIRIP tapi tidak identik (mis. "panadol" di
satu dataset vs "panadol_extra"/"Panadol_coldflu" di dataset lain), script ini
memperlakukan keduanya sebagai kelas BERBEDA (union berbasis string persis) dan
mencetak peringatan -- karena tidak ada cara aman menebak apakah itu produk yang
sama atau varian yang benar-benar berbeda tanpa konfirmasi manusia.

Usage:
    python scripts/merge_datasets.py \
        --input data/pharmacy_v7_converted --input data/roboflow_raw/pharmacy_shelf_detect_v1 \
        --input data/roboflow_raw/pharmacy_robot_v2_converted \
        --output data/pharmacy_merged
"""
import argparse
import difflib
import pathlib
import shutil

import yaml

SPLITS = ("train", "valid", "test")


def load_names(dataset_dir: pathlib.Path) -> list[str]:
    cfg = yaml.safe_load((dataset_dir / "data.yaml").read_text())
    return list(cfg["names"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True, help="folder dataset (sudah bbox), bisa diulang")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    inputs = [pathlib.Path(p) for p in args.input]
    out_root = pathlib.Path(args.output)

    per_dataset_names = {d: load_names(d) for d in inputs}

    # Union nama kelas, urutan stabil: kelas dari dataset pertama dulu, lalu kelas
    # BARU dari dataset berikutnya (belum pernah muncul) ditambah di akhir.
    merged_names: list[str] = []
    origin_dataset: dict[str, pathlib.Path] = {}
    for d in inputs:
        for name in per_dataset_names[d]:
            if name not in merged_names:
                merged_names.append(name)
                origin_dataset[name] = d
    name_to_idx = {name: i for i, name in enumerate(merged_names)}

    # Peringatan nama mirip tapi tidak identik, HANYA ANTAR dataset sumber yang
    # berbeda (nama-nama dalam satu dataset yang sama sudah sengaja dibedakan oleh
    # yang melabeli aslinya -- bukan konflik penggabungan, jadi tidak perlu diperingatkan).
    print("=== Peringatan kemiripan nama kelas antar dataset berbeda (cek manual, TIDAK digabung otomatis) ===")
    any_warning = False
    for i, a in enumerate(merged_names):
        for b in merged_names[i + 1 :]:
            if origin_dataset[a] == origin_dataset[b]:
                continue  # dari dataset yang sama -- sudah sengaja dibedakan, bukan konflik
            ratio = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
            if ratio > 0.55 and a.lower() != b.lower():
                any_warning = True
                print(
                    f"  '{a}' ({origin_dataset[a].name}) vs '{b}' ({origin_dataset[b].name}) "
                    f"(kemiripan string: {ratio:.2f}) -- produk sama atau beda? cek manual."
                )
    if not any_warning:
        print("  (tidak ada)")
    print()

    if out_root.exists():
        shutil.rmtree(out_root)

    per_dataset_stats = {}

    for d in inputs:
        local_names = per_dataset_names[d]
        local_to_global = {i: name_to_idx[name] for i, name in enumerate(local_names)}
        n_objects = 0
        n_images = 0
        for split in SPLITS:
            src_img, src_lbl = d / split / "images", d / split / "labels"
            if not src_img.exists():
                continue
            dst_img, dst_lbl = out_root / split / "images", out_root / split / "labels"
            dst_img.mkdir(parents=True, exist_ok=True)
            dst_lbl.mkdir(parents=True, exist_ok=True)
            for img_path in src_img.iterdir():
                # prefix nama file dengan nama dataset supaya tidak tabrakan antar sumber
                dst_name = f"{d.name}__{img_path.name}"
                shutil.copy(img_path, dst_img / dst_name)
                n_images += 1
            for lbl_path in src_lbl.glob("*.txt"):
                out_lines = []
                for line in lbl_path.read_text().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    local_cls = int(parts[0])
                    global_cls = local_to_global[local_cls]
                    out_lines.append(" ".join([str(global_cls)] + parts[1:]))
                    n_objects += 1
                dst_lbl_name = f"{d.name}__{lbl_path.name}"
                (dst_lbl / dst_lbl_name).write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
        per_dataset_stats[d.name] = (n_images, n_objects)

    data_yaml = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(merged_names),
        "names": merged_names,
    }
    (out_root / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    print("=== Ringkasan penggabungan ===")
    for d in inputs:
        n_images, n_objects = per_dataset_stats[d.name]
        print(f"  {d.name}: {n_images} gambar, {n_objects} objek, {len(per_dataset_names[d])} kelas lokal")
    print(f"\nTotal kelas gabungan: {len(merged_names)}")
    print(f"Daftar kelas: {merged_names}")
    print(f"\nSelesai. Dataset gabungan di: {out_root}")
    print(f"Training: python scripts/train.py --data {out_root}/data.yaml --epochs 120")


if __name__ == "__main__":
    main()
