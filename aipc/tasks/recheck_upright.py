"""The display turn (0 / 180 degrees) of the brand and detail crops of finished runs, decided again on the laptop with
pick_parts.upright (8 Nov 2026, night): the runs of job 102 (fqzf*) chose it by the closeness of the whole string to the
text read, which turned some upright digit blocks upside down; pick_parts.upright now decides it by the tokens read.
Display only: what was cut, read and decided is unchanged; only the turn of brand.png and detail_<k>.png is decided
again (from the saved image, so either turn can come out), detail.png is stacked again and crops.json records it
("display_turn"). Same OCR models as the AI PC workers (PP-OCRv6 small).

  venv/Scripts/python.exe aipc/tasks/recheck_upright.py OUT_TAG_DIR ...
"""
import glob
import json
import os
import pathlib
import sys

import cv2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import pick_parts as pp  # noqa: E402
from src.box_ocr import SizeTextReader  # noqa: E402


def main():
    reader = SizeTextReader(threads=2)
    for d in sys.argv[1:]:
        n = turned = 0
        for cj in sorted(glob.glob(os.path.join(d, "crops", "*", "crops.json"))):
            folder = os.path.dirname(cj)
            info = json.load(open(cj, encoding="utf-8"))
            flips = []
            for key in ("product", "brand"):                     # 10 Nov: product and brand are two crops
                if info["files"].get(key):
                    img, t = pp.upright(cv2.imread(os.path.join(folder, key + ".png")), reader,
                                        info.get(key + "_text") or "")
                    cv2.imwrite(os.path.join(folder, key + ".png"), img)
                    flips.append(t)
            blocks = []
            texts = (info.get("detail_text") or "").split(" | ")
            for k, f in enumerate(sorted(glob.glob(os.path.join(folder, "detail_*.png")))):
                img, t = pp.upright(cv2.imread(f), reader, texts[k] if k < len(texts) else "")
                cv2.imwrite(f, img)
                blocks.append(img)
                flips.append(t)
            if blocks:
                cv2.imwrite(os.path.join(folder, "detail.png"), pp.stack(blocks))
            info["display_turn"] = "decided again on the laptop (recheck_upright.py: tokens read, on / off, classifier)"
            info["display_turned_again"] = flips
            json.dump(info, open(cj, "w", encoding="utf-8"), indent=1)
            n += 1
            turned += sum(flips)
        print(d, n, "crop folders,", turned, "crops turned", flush=True)


if __name__ == "__main__":
    main()
