"""pytest tests/test_enroll.py -q  (katalog GTIN: ubah merek / REF saat review, ganti nama pemeriksa)"""
import builtins
import importlib.util
import shutil

import pytest

from src.box_catalog import Catalog


def load_enroll():
    spec = importlib.util.spec_from_file_location("enroll_gtin", "scripts/enroll_gtin.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.open_image = lambda path: None                        # jangan buka penampil gambar saat tes
    return mod


def test_review_edit_changes_brand_and_ref_and_records_verifier(tmp_path, monkeypatch):
    enroll = load_enroll()
    path = tmp_path / "cat.csv"
    shutil.copy("data/gtin_catalog.csv", path)
    cat = Catalog(path)
    for e in cat.entries.values():                            # hanya 1 entri yang ditinjau
        e.verified = e.gtin != "08714729847182"
    answers = iter(["e", "nc emerge", "5.5", "12", "Boston Scientific", "MR"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers))
    enroll.review(cat, tmp_path, "Penguji")
    e = Catalog(path).get("08714729847182")
    assert (e.product, e.diameter_mm, e.length_mm, e.brand, e.ref) == ("nc emerge", 5.5, 12.0, "Boston Scientific", "MR")
    assert e.verified and e.verified_by == "Penguji"


def test_dash_clears_ref_and_rename_verifier(tmp_path, monkeypatch):
    enroll = load_enroll()
    path = tmp_path / "cat.csv"
    shutil.copy("data/gtin_catalog.csv", path)
    cat = Catalog(path)
    for e in cat.entries.values():
        e.verified = e.gtin != "04987350720061"
    answers = iter(["e", "", "", "", "", "-"])                # Enter = tetap, "-" = kosongkan REF
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers))
    enroll.review(cat, tmp_path, "nama anda")
    e = Catalog(path).get("04987350720061")
    assert e.product == "accuforce" and e.diameter_mm == 2.5 and e.ref == ""
    cat = Catalog(path)
    n = enroll.rename_verifier(cat, "nama anda", "Budi")
    assert n >= 1 and all(x.verified_by != "nama anda" for x in cat.entries.values())


def test_placeholder_names_from_instructions_are_refused():
    enroll = load_enroll()
    for s in ("nama anda", "NamaAsliAnda", "Nama Asli Anda", "<nama Anda>", "NAMA_ASLI", "", "  ", None):
        assert enroll.is_placeholder(s), s
    for s in ("Budi", "Sari Wulandari", "Ananda", "Dr. Chen"):
        assert not enroll.is_placeholder(s), s


def test_checker_name_is_asked_when_not_given(monkeypatch):
    enroll = load_enroll()
    monkeypatch.setattr(builtins, "input", lambda prompt="": "  Sari Wulandari ")
    assert enroll.ask_name(None, "nama pemeriksa") == "Sari Wulandari"
    assert enroll.ask_name("Budi", "nama pemeriksa") == "Budi"             # argumen menang, tidak ditanya
    monkeypatch.setattr(builtins, "input", lambda prompt="": "<nama Anda>")
    with pytest.raises(SystemExit):
        enroll.ask_name(None, "nama pemeriksa")
