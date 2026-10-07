"""Contoh kasus mentor (A, B, C) dihitung dengan fungsi yang sama seperti scripts/eval_box_manual.py.
pytest tests/test_box_eval.py -q"""
import importlib.util
import pathlib

import pytest

from src.box_geometry import rotated_rect_poly

_spec = importlib.util.spec_from_file_location(
    "eval_box_manual", pathlib.Path(__file__).resolve().parent.parent / "scripts" / "eval_box_manual.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

GT = [rotated_rect_poly(100 + 150 * i, 200, 300, 60, 90) for i in range(4)]    # 4 kotak di meja


def test_case_a_all_found():
    preds = [(0.9, g) for g in GT]
    tp, fp, fn = ev.match_image(preds, GT, 0.5)
    assert (len(tp), len(fp), len(fn)) == (4, 0, 0)
    assert ev.pr(4, 0, 0)[:2] == (1.0, 1.0)


def test_case_b_one_missed():
    preds = [(0.9, g) for g in GT[:3]]
    tp, fp, fn = ev.match_image(preds, GT, 0.5)
    assert (len(tp), len(fp), len(fn)) == (3, 0, 1)
    p, r, _ = ev.pr(3, 0, 1)
    assert (p, r) == (1.0, 0.75)


def test_case_c_sticker_detected_as_box():
    sticker = rotated_rect_poly(900, 600, 80, 40, 0)
    preds = [(0.9, g) for g in GT[:3]] + [(0.6, sticker)]
    tp, fp, fn = ev.match_image(preds, GT, 0.5)
    assert (len(tp), len(fp), len(fn)) == (3, 1, 1)
    p, r, f1 = ev.pr(3, 1, 1)
    assert (p, r) == (0.75, 0.75) and f1 == pytest.approx(0.75)


def test_duplicate_box_counts_as_false_positive():
    preds = [(0.9, GT[0]), (0.8, GT[0])]          # dua prediksi untuk kotak yang sama
    tp, fp, fn = ev.match_image(preds, GT[:1], 0.5)
    assert (len(tp), len(fp), len(fn)) == (1, 1, 0)


def test_loose_box_below_iou_threshold_is_fp_and_fn():
    true_box = rotated_rect_poly(500, 500, 600, 40, 10)
    from src.box_geometry import aabb_poly
    tp, fp, fn = ev.match_image([(0.9, aabb_poly(true_box))], [true_box], 0.5)   # IoU 0.28
    assert (len(tp), len(fp), len(fn)) == (0, 1, 1)
