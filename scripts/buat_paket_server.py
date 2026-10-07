#!/usr/bin/env python3
"""
Paket untuk server (Remmina): kode + tes + data yang diperlukan demo dan uji kamera, dengan daftar file TETAP
dan sha256 per file, supaya tidak ada file yang tertinggal saat dipindahkan.

  python scripts/buat_paket_server.py                    # -> ~/Downloads/box_pipeline_demo_<tanggal>.zip
  python scripts/buat_paket_server.py --keluar paket.zip

Di server, setelah unzip:  sha256sum -c paket_sha256.txt   lalu   python -m pytest tests -q
Katalog GTIN (data/gtin_catalog.csv) dan model (runs/*.pt) TIDAK ikut: katalog dikirim terpisah, model sudah di server.
Paket menolak dibuat kalau ada `from src.X import` yang modulnya tidak ikut terbawa.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import pathlib
import re
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent

PATTERNS = [
    "src/__init__.py", "src/config.py", "src/box_*.py",
    "tests/test_box_*.py", "tests/test_enroll.py", "tests/test_webcam_apply.py",
    "scripts/demo_request.py", "scripts/uji_checklist.py", "scripts/test_webcam_box.py", "scripts/stream_webcam.py",
    "scripts/rak_inventaris.py", "scripts/stress_plan.py", "scripts/eval_verify.py", "scripts/eval_box_manual.py",
    "scripts/enroll_gtin.py", "scripts/check_box_labels.py", "scripts/export_box_onnx.py",
    "scripts/buat_paket_server.py",
    "config/box_pipeline_config.yaml", "config/box_pipeline_config_server.yaml",
    "data/box_classes.txt", "data/verify_eval_gt.csv", "data/gtin_evidence/*.jpg",
    "data/rak/inventaris.csv", "data/rak/posisi.csv", "data/rak/punggung/*.jpg",
    "requirements_box.txt", "docs/BOX_PIPELINE.md", "docs/CHECKLIST_DEMO.md", "UJI_ROBOT_LAPTOP.bat",
]
SUMS = "paket_sha256.txt"


def collect() -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for pat in PATTERNS:
        hits = sorted(ROOT.glob(pat))
        if not hits:
            sys.exit(f"File tidak ditemukan untuk pola {pat!r}")
        files += [h for h in hits if h.is_file() and h not in files]
    return files


def missing_imports(files: list[pathlib.Path]) -> list[str]:
    have = {f.relative_to(ROOT).as_posix() for f in files}
    out = []
    for f in files:
        if f.suffix != ".py":
            continue
        for mod in re.findall(r"^\s*(?:from|import)\s+src\.(\w+)", f.read_text(encoding="utf-8"), re.M):
            if f"src/{mod}.py" not in have:
                out.append(f"{f.relative_to(ROOT).as_posix()} -> src/{mod}.py")
    return sorted(set(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keluar", default=str(pathlib.Path.home() / "Downloads"
                                            / f"box_pipeline_demo_{dt.date.today():%Y%m%d}.zip"))
    args = ap.parse_args()

    files = collect()
    miss = missing_imports(files)
    if miss:
        sys.exit("Modul yang diimpor tidak ikut paket:\n  " + "\n  ".join(miss))
    sums = []
    out = pathlib.Path(args.keluar)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            rel = f.relative_to(ROOT).as_posix()
            data = f.read_bytes()
            z.writestr(rel, data)
            sums.append(f"{hashlib.sha256(data).hexdigest()}  {rel}")
        z.writestr(SUMS, "\n".join(sums) + "\n")
    print(f"{out}  ({len(files)} file + {SUMS}, {out.stat().st_size / 1e6:.1f} MB)")
    print(f"sha256 zip: {hashlib.sha256(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
