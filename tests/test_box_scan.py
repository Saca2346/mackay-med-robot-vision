"""pytest tests/test_box_scan.py -q  (pindai detail: DataMatrix / QR / barcode 1D -> GTIN, LOT, kedaluwarsa, seri)"""
import datetime as dt

import cv2
import numpy as np
import pytest

zxingcpp = pytest.importorskip("zxingcpp")

from src.box_catalog import Catalog  # noqa: E402
from src.box_scan import card_lines, read_all_codes, scan_box  # noqa: E402
from src.box_verifier import CANDIDATE, parse_request  # noqa: E402
from src.config import load_config, resolve  # noqa: E402


class NoOcr:
    """Pembaca OCR palsu (tes hanya menguji kode)."""

    def read_once(self, img):
        class R:
            tokens = []
        return R()


def code_image(text, fmt, scale=6):
    img = np.array(zxingcpp.create_barcode(text, fmt).to_image(scale=scale))
    img = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_datamatrix_without_brackets_like_boston_scientific():
    img = code_image("0108714729831587172701211000614H25", zxingcpp.BarcodeFormat.DataMatrix)
    codes = read_all_codes(img)
    g = next(c["gs1"] for c in codes if c["gs1"] is not None)
    assert (g.gtin, g.expiry, g.lot) == ("08714729831587", dt.date(2027, 1, 21), "00614H25")


def test_qr_digital_link_and_plain_url():
    img = np.hstack([code_image("https://id.gs1.org/01/08714729896845/10/39500350?17=290324",
                                zxingcpp.BarcodeFormat.QRCode),
                     code_image("https://www.bostonscientific.com/eifu", zxingcpp.BarcodeFormat.QRCode)])
    codes = read_all_codes(img)
    gs1 = [c["gs1"] for c in codes if c["gs1"] is not None]
    assert len(codes) == 2 and len(gs1) == 1 and gs1[0].lot == "39500350"
    assert any("bostonscientific.com" in c["text"] for c in codes)      # QR biasa tetap ditampilkan


def test_scan_card_uses_catalog_and_never_hides_draft_status():
    cfg = load_config(resolve("config/box_pipeline_config.yaml"))
    cat = Catalog(resolve("data/gtin_catalog.csv"))
    img = code_image("0108714729831587172701211000614H25", zxingcpp.BarcodeFormat.DataMatrix)
    card = scan_box(img, NoOcr(), cfg, parse_request("boston scientific agent 2.75 x 30"), cat,
                    today=dt.date(2026, 9, 28))
    k = card["kode"][0]
    assert k["gtin"] == "08714729831587" and not k["sudah_kedaluwarsa"]
    if not cat.get("08714729831587").verified:
        assert "DRAF" in k["katalog"] and card["keputusan"] != "match"   # draf tidak pernah dipakai untuk MATCH
    assert any("GTIN 08714729831587" in line for line in card_lines(card))


def test_expired_code_is_flagged():
    cfg = load_config(resolve("config/box_pipeline_config.yaml"))
    img = code_image("0108714729831587172101211000614H25", zxingcpp.BarcodeFormat.DataMatrix)   # exp 2021
    card = scan_box(img, NoOcr(), cfg, parse_request("agent 2.75 x 30"), Catalog(resolve("data/gtin_catalog.csv")),
                    today=dt.date(2026, 9, 28))
    assert card["kode"][0]["sudah_kedaluwarsa"] and card["keputusan"] == "expired"
    assert card["keputusan"] != CANDIDATE
