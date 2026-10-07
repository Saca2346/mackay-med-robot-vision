#!/usr/bin/env python3
"""Fase 1 (roadmap section 10): Camera -> Detection -> Display.

Usage:
    python scripts/run_phase1.py [--frames N] [--save-dir data/test_assets/phase1_out]
"""
import argparse
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.pipeline import MedicineShelfPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=10, help="jumlah frame yang diproses (synthetic/video mode)")
    ap.add_argument("--save-dir", type=str, default="data/test_assets/phase1_out")
    ap.add_argument("--show", action="store_true", help="tampilkan jendela live (butuh display, non-headless)")
    args = ap.parse_args()

    pipeline = MedicineShelfPipeline(enable_verification=False, enable_depth=False)
    pipeline.run(max_frames=args.frames, show_window=args.show, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
