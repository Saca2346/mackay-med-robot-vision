"""PICK crop geometry: corners must remain invertible at every box rotation."""
from types import SimpleNamespace

import cv2
import numpy as np

from src.box_crops import pick_crops, rectify, to_frame


def test_diamond_label_rectifies_without_duplicate_corners():
    # Sum/difference corner sorting ties at 45 degrees and can select one vertex twice.
    p = np.float32([[80, 40], [120, 80], [80, 120], [40, 80]])
    frame = np.zeros((160, 160, 3), np.uint8)
    cv2.fillConvexPoly(frame, p.astype(np.int32), (40, 80, 160))
    crop, transform = rectify(frame, p)
    h, w = crop.shape[:2]
    back = to_frame([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], transform)
    assert np.allclose(crop[h // 2, w // 2], [40, 80, 160])
    assert all(np.min(np.linalg.norm(back - vertex, axis=1)) < 1e-4 for vertex in p)


def test_decoded_barcode_on_diamond_box_has_finite_camera_coordinates():
    p = np.float32([[80, 40], [120, 80], [80, 120], [40, 80]])
    frame = np.full((160, 160, 3), 160, np.uint8)
    group = SimpleNamespace(box=SimpleNamespace(poly=p), labels=[])

    def decode(img, hard):
        h, w = img.shape[:2]
        return [SimpleNamespace(poly=np.float32([[w * .25, h * .25], [w * .75, h * .25],
                                                 [w * .75, h * .75], [w * .25, h * .75]]),
                                raw="decoded GS1", gtin="")]

    crops = pick_crops(frame, group, decode)
    assert crops["barcode_decoded"] and crops["barcode"].size
    assert np.isfinite(crops["barcode_px"]).all()
    assert np.allclose(np.mean(crops["barcode_px"], axis=0), [80, 80], atol=1.0)
