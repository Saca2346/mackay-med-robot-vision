#!/usr/bin/env python3
"""Fase 3 (roadmap section 10): + Depth + Posisi 3D.

Di laptop dengan RealSense D435i/D415 terpasang, set depth.source: realsense di
config/pipeline_config.yaml. Tanpa hardware (mode dev/sandbox ini), depth.source: mock
tetap memvalidasi jalur geometri dan menghasilkan koordinat 3D yang masuk akal.

Usage:
    python scripts/run_phase3.py [--frames N]
"""
import argparse
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.pipeline import MedicineShelfPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--save-dir", type=str, default="data/test_assets/phase3_out")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    pipeline = MedicineShelfPipeline(enable_verification=True, enable_depth=True)
    pipeline.run(max_frames=args.frames, show_window=args.show, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
