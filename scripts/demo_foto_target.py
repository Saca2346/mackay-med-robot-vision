"""
Run the real box pipeline on a PHOTO with a requested product (the mentor's task photos), all output in --out.

  python scripts/demo_foto_target.py PHOTO.jpg --request "Chroma, 2.75 , 33" --out results/task_X/task2 [--show]

What it does
  * runs scripts/test_webcam_box.py unchanged on the photo, so the picture has the same HUD, colours and decisions as
    the live system (green = PICK, 2 independent sources; blue = candidate, 1 source).
  * writes EVERYTHING (result picture, readings, logs) under --out. The real evidence logs are never touched.
  * extra options go straight to test_webcam_box.py, e.g. --weights models/X_openvino_model --imgsz 544,960
    --view-width 900.

Since 15-10 the two brands of the task photos are real catalogue products (src/box_verifier.py PRODUCTS, the configs'
`catalog:` lists) with REF rules from their own sources (src/box_ref.py): Biosensors CHROMA and B. Braun SeQuent Please
NEO. Until then this script added them inside its own process only.
"""
from __future__ import annotations

import argparse
import pathlib
import runpy
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photo")
    ap.add_argument("--request", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="config/box_pipeline_config_realsense.yaml")
    ap.add_argument("--view", choices=("target", "all"), default="target",
                    help="target (default): only the target box is drawn; all: every box with its state")
    ap.add_argument("--show", action="store_true",
                    help="open the result window (v = all boxes / target only, q = close); default: only save it")
    args, extra = ap.parse_known_args()             # extra options go to test_webcam_box.py (e.g. --view-width 900)

    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    photo = pathlib.Path(args.photo).resolve()
    stem = photo.stem

    cfg = yaml.safe_load(open(ROOT / args.config, encoding="utf-8"))
    cfg["output"]["screens_dir"] = str(out / "screens")
    cfg["output"]["log_csv"] = str(out / "selection_log.csv")
    cfg["output"]["verify_log_csv"] = str(out / "verify_log.csv")
    cfg_path = out / "_config_demo.yaml"
    yaml.safe_dump(cfg, open(cfg_path, "w", encoding="utf-8"), sort_keys=False)

    sys.argv = ["scripts/test_webcam_box.py", "--config", str(cfg_path), "--source", str(photo),
                "--request", args.request, "--view", args.view,
                "--debug-readings", str(out / f"bacaan_{stem}.csv")] + ([] if args.show else ["--no-window"]) + extra
    runpy.run_path(str(ROOT / "scripts" / "test_webcam_box.py"), run_name="__main__")


if __name__ == "__main__":
    main()
