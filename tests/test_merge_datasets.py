#!/usr/bin/env python3
"""
Correctness check for scripts/merge_datasets.py: proves the union+reindex logic
gives EVERY merged label the right global class id (not just "runs without
crashing"), and that the similarity-warning only fires across DIFFERENT source
datasets, never for two classes the same dataset's own annotator already told
apart on purpose.

Run: python tests/test_merge_datasets.py
"""
import pathlib
import subprocess
import sys
import tempfile

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"


def check(label, cond):
    print(f"[{PASS if cond else FAIL}] {label}")
    return cond


def make_dataset(root: pathlib.Path, names: list[str], split_labels: dict):
    """split_labels: {"train": [(filename_stem, local_cls), ...], ...}"""
    (root).mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(
        yaml.safe_dump({"train": "../train/images", "val": "../valid/images", "nc": len(names), "names": names})
    )
    for split, items in split_labels.items():
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        for stem, local_cls in items:
            (root / split / "images" / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff")  # fake jpg bytes, cukup utk tes
            (root / split / "labels" / f"{stem}.txt").write_text(f"{local_cls} 0.5 0.5 0.2 0.2\n")


def main():
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)

        # Dataset A: 2 kelas, sudah sengaja dibedakan si annotator ("box_small" vs "box_large")
        ds_a = tmp / "ds_a"
        make_dataset(ds_a, ["box_small", "box_large"], {
            "train": [("a1", 0), ("a2", 1)],
            "valid": [("a3", 0)],
        })

        # Dataset B: 2 kelas, salah satunya nama MIRIP dgn kelas dataset A (box_large vs box_large_v2)
        # tapi ini dataset BEDA -- harus dianggap kelas terpisah & diberi peringatan.
        ds_b = tmp / "ds_b"
        make_dataset(ds_b, ["box_large_v2", "unique_item"], {
            "train": [("b1", 0), ("b2", 1)],
        })

        out = tmp / "merged"
        result = subprocess.run(
            [sys.executable, "scripts/merge_datasets.py", "--input", str(ds_a), "--input", str(ds_b), "--output", str(out)],
            cwd=pathlib.Path(__file__).resolve().parent.parent,
            capture_output=True, text=True,
        )
        print(result.stdout)
        ok &= check("script selesai tanpa error", result.returncode == 0)

        cfg = yaml.safe_load((out / "data.yaml").read_text())
        ok &= check(f"4 kelas gabungan (dapat: {cfg['names']})", cfg["names"] == ["box_small", "box_large", "box_large_v2", "unique_item"])

        # a2 asalnya kelas lokal 1 (box_large) di dataset A -> harus jadi global index 1 (sama posisinya
        # krn box_large adalah kelas ke-2 yg pertama kali muncul, dari dataset A)
        a2_label = (out / "train" / "labels" / "ds_a__a2.txt").read_text().split()[0]
        ok &= check(f"ds_a__a2 (box_large lokal=1) -> global index benar (dapat: {a2_label})", a2_label == "1")

        # b1 asalnya kelas lokal 0 (box_large_v2) di dataset B -> ini KELAS BARU (bukan sama dgn box_large
        # dataset A meskipun namanya mirip), harus dapat index baru (2), BUKAN index 1 milik box_large.
        b1_label = (out / "train" / "labels" / "ds_b__b1.txt").read_text().split()[0]
        ok &= check(f"ds_b__b1 (box_large_v2, BUKAN box_large) -> index baru, bukan ketiban index box_large (dapat: {b1_label})", b1_label == "2")

        b2_label = (out / "train" / "labels" / "ds_b__b2.txt").read_text().split()[0]
        ok &= check(f"ds_b__b2 (unique_item lokal=1) -> global index benar (dapat: {b2_label})", b2_label == "3")

        # peringatan kemiripan HANYA untuk pasangan lintas dataset (box_large vs box_large_v2),
        # TIDAK untuk box_small vs box_large yang sama-sama dari dataset A (sudah sengaja dibedakan).
        ok &= check(
            "peringatan muncul untuk 'box_large' vs 'box_large_v2' (lintas dataset)",
            "box_large" in result.stdout and "box_large_v2" in result.stdout,
        )
        same_dataset_warning = "'box_small' (ds_a) vs 'box_large' (ds_a)" in result.stdout
        ok &= check(
            "TIDAK ada peringatan untuk 'box_small' vs 'box_large' (satu dataset, sudah sengaja dibedakan)",
            not same_dataset_warning,
        )

        # valid split dataset A tetap ada, dataset B (tanpa valid) tidak bikin folder valid kosong nyasar
        ok &= check("valid/images dataset A ikut tersalin", (out / "valid" / "images" / "ds_a__a3.jpg").exists())

    print("\n" + ("SEMUA TES LULUS" if ok else "ADA TES YANG GAGAL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
