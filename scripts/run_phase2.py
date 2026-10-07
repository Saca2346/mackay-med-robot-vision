#!/usr/bin/env python3
"""Fase 2 (roadmap section 10): + OCR + Database Lokal.

Usage:
    python scripts/run_phase2.py [--frames N]
Prasyarat: jalankan scripts/init_db.py sekali untuk membuat & mengisi database contoh.
"""
import argparse
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.pipeline import MedicineShelfPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--save-dir", type=str, default="data/test_assets/phase2_out")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    pipeline = MedicineShelfPipeline(enable_verification=True, enable_depth=False)
    pipeline.run(max_frames=args.frames, show_window=args.show, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
