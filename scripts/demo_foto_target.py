"""
DEMO ONLY: run the real box pipeline on a PHOTO with a requested product, for brands that are not in the catalog.

  python scripts/demo_foto_target.py FOTO.jpg --request "chroma 4.0 x 19" --out results/demo_20261012/foto_baru/target

What it does
  * adds two brands to the product list INSIDE THIS PROCESS ONLY (nothing in src/ or config/ is changed):
        biosensors: chroma          b braun: sequent please neo
  * adds ONE REF rule for Chroma in this process only: BCR-<diameter x10><length>, valid for diameter 4.0 only, because
    that is the only pattern visible in the photo (stacked boxes BCR-4014 / 4019 / 4024 / 4029). Other diameters are
    NOT read as REF (the rule returns nothing for them), so they can only reach "candidate", never "PICK".
  * runs scripts/test_webcam_box.py unchanged, so the picture has the same HUD, colours and decisions as the live system
    (green = PICK needs 2 independent sources, blue = candidate with 1 source, yellow = unsure, grey = not the target).
  * writes EVERYTHING (result picture, readings, logs) under --out. The real evidence logs are never touched.

Limits: not a production rule set. A product only becomes real after it is added to the catalog and its REF rule is
validated on labelled photos (see src/box_ref.py).
"""
from __future__ import annotations

import argparse
import pathlib
import runpy
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def patch_products() -> None:
    import src.box_ref as ref
    import src.box_verifier as bv

    for p in ("chroma", "sequent please neo"):
        if p not in bv.PRODUCTS:
            bv.PRODUCTS.append(p)
    bv.BRANDS.setdefault("biosensors", {"chroma"})
    bv.BRANDS.setdefault("b braun", {"sequent please neo"})
    # the word "please" is a filler word in the request parser, so the request is typed "sequent neo"
    bv.ALIASES.setdefault("sequent neo", "sequent please neo")
    bv.ALIASES.setdefault("sequent", "sequent please neo")

    def chroma_diameter(dd: str):
        return 4.0 if dd == "40" else None          # only the diameter that was seen in the photo

    ref.RULES.append(("chroma", rf"BCR[-_.]?({ref._D}{{2}})({ref._D}{{2}})", chroma_diameter))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photo")
    ap.add_argument("--request", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="results/bukti_laporan_20261008/config_realsense.yaml")
    args = ap.parse_args()

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

    patch_products()
    sys.argv = ["scripts/test_webcam_box.py", "--config", str(cfg_path), "--source", str(photo),
                "--request", args.request, "--no-window", "--debug-readings", str(out / f"bacaan_{stem}.csv")]
    runpy.run_path(str(ROOT / "scripts" / "test_webcam_box.py"), run_name="__main__")


if __name__ == "__main__":
    main()
