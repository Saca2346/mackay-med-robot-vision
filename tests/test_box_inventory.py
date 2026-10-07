"""pytest tests/test_box_inventory.py -q  (inventaris rak: identitas dari beberapa bacaan, daftar isi, FEFO, ciri)"""
import datetime as dt

import numpy as np

from src.box_barcode import parse_gs1
from src.box_catalog import Catalog, CatalogEntry
from src.box_inventory import (STATUS_DAFTAR, STATUS_DIAMBIL, STATUS_DRAF, STATUS_OK, Inventory, codes_for,
                               identify, import_list, place_scanned, sig_str, signature, similarity)
from src.box_ocr import OcrToken
from src.box_ref import parse_refs
from src.box_verifier import Evidence, Request, parse_identity

TODAY = dt.date(2026, 9, 30)
NAMES = ["angiolite", "accuforce", "permanent sled bag"]


def text(*toks):
    return parse_identity([OcrToken(t, 0.99) for t in toks], NAMES)


def catalog():
    c = Catalog()
    c.upsert(CatalogEntry(gtin="04987350720245", product="accuforce", diameter_mm=2.75, length_mm=20, verified=True))
    c.upsert(CatalogEntry(gtin="04987350720061", product="accuforce", diameter_mm=2.5, length_mm=12, verified=True))
    return c


def test_text_and_ref_from_separate_reads_identify_the_box():
    evs = [Evidence(text=text("Angiolite", "2.5", "29", "mm")), Evidence(refs=parse_refs("SCCDSR14150250029"))]
    key, src, _ = identify(evs, catalog(), today=TODAY)
    assert key == ("angiolite", 2.5, 29.0) and "teks" in src and "REF" in src


def test_single_read_is_never_enough():
    evs = [Evidence(text=text("Angiolite", "2.5", "29", "mm"), refs=parse_refs("SCCDSR14150250029"))]
    assert identify(evs, catalog(), today=TODAY)[0] is None


def test_any_contradicting_read_blocks_identity():
    evs = [Evidence(text=text("Angiolite", "2.5", "29", "mm")), Evidence(refs=parse_refs("SCCDSR14150250029")),
           Evidence(refs=parse_refs("SCCDSR14150400019"))]                  # sekali terbaca REF 4 x 19
    key, _, note = identify(evs, catalog(), today=TODAY)
    assert key is None and "bertentangan" in note


def test_barcode_plus_one_more_read_identifies_and_gives_lot_expiry():
    code = parse_gs1("(01)04987350720245(17)280930(10)251023", "Code 128")
    evs = [Evidence(codes=[code]), Evidence(text=text("Accuforce", "2.75", "20", "mm"))]
    key, src, _ = identify(evs, catalog(), today=TODAY)
    assert key == ("accuforce", 2.75, 20.0) and "barcode" in src
    assert codes_for(evs, key, catalog()) == ("04987350720245", "251023", "2028-09-30")


def test_partial_text_pieces_give_only_a_proposal():
    evs = [Evidence(text=text("Angiolite", "2.5")), Evidence(text=text("Angiolite", "29", "mm"))]
    key, src, _ = identify(evs, catalog(), today=TODAY)
    assert key == ("angiolite", 2.5, 29.0) and src.startswith("USULAN")      # 1 bukti: perawat wajib cek teliti


def test_pieces_with_ref_become_certain_and_wrong_pieces_drop_out():
    evs = [Evidence(text=text("Angiolite", "2.5")), Evidence(text=text("Angiolite", "29", "mm")),
           Evidence(refs=parse_refs("SCCDSR14150250029"))]
    key, src, _ = identify(evs, catalog(), today=TODAY)
    assert key == ("angiolite", 2.5, 29.0) and not src.startswith("USULAN")
    evs.append(Evidence(text=text("Angiolite", "4", "mm")))                    # potongan angka lain: bertentangan
    assert identify(evs, catalog(), today=TODAY)[0] is None


def test_single_piece_read_is_never_a_proposal():
    evs = [Evidence(text=text("Angiolite", "2.5", "29", "mm"))]
    assert identify(evs, catalog(), today=TODAY)[0] is None


def test_inventory_roundtrip_candidates_fefo_and_list_matching(tmp_path):
    inv = Inventory(tmp_path / "inv.csv")
    assert import_list(inv, [{"produk": "Angiolite", "diameter": "2,5", "panjang": "29", "jumlah": "2"},
                             {"produk": "accuforce", "diameter": "2.75", "panjang": "20", "jumlah": "1"}]) == 3
    sig = sig_str(np.ones(90, dtype=np.float32) / 30)
    a = place_scanned(inv, ("angiolite", 2.5, 29.0), "teks + REF (3 bacaan)", "", "2511631", "2028-07-31", 1, "", sig)
    b = place_scanned(inv, ("angiolite", 2.5, 29.0), "teks + REF (2 bacaan)", "", "2511700", "2027-01-31", 2, "", sig)
    c = place_scanned(inv, ("angiolite", 4.0, 19.0), "teks + REF (2 bacaan)", "", "", "", 3, "", sig)
    assert a.status == b.status == c.status == STATUS_DRAF
    assert "tidak ada di daftar" in c.note                              # kotak yang tidak ada di daftar isi
    assert [x.status for x in inv.boxes.values()].count(STATUS_DAFTAR) == 1   # accuforce belum ditemukan
    req = Request("angiolite", 2.5, 29)
    assert inv.candidates(req, TODAY) == []                             # draf belum diperiksa perawat: tidak dipakai
    for x in (a, b):
        x.status = STATUS_OK
    inv.save()
    inv2 = Inventory(tmp_path / "inv.csv")
    got = inv2.candidates(req, TODAY)
    assert [x.box_id for x in got] == [b.box_id, a.box_id]               # FEFO: 2027-01 sebelum 2028-07
    assert got[0].signature() is not None and got[0].diameter == 2.5
    inv2.boxes[b.box_id].status = STATUS_DIAMBIL
    assert [x.box_id for x in inv2.candidates(req, TODAY)] == [a.box_id]
    inv2.boxes[a.box_id].expiry = "2026-01-01"                          # kedaluwarsa: tidak pernah dipilih
    assert inv2.candidates(req, TODAY) == []


def test_signature_matches_same_spine_even_reversed_and_not_other_colors():
    rng = np.random.default_rng(0)
    pink = np.zeros((40, 300, 3), np.uint8)
    pink[:, :60] = (180, 40, 230)                                     # ujung pink (BGR)
    pink[:, 60:200] = (200, 200, 60)                                  # panel toska
    pink[:, 200:] = (240, 240, 240)
    noisy = np.clip(pink.astype(int) + rng.integers(-12, 12, pink.shape), 0, 255).astype(np.uint8)
    navy = np.zeros_like(pink)
    navy[:, :] = (90, 30, 10)
    navy[:, 100:180] = (60, 160, 40)
    s = signature(pink)
    assert similarity(s, signature(noisy)) > 0.9
    assert similarity(s, signature(pink[:, ::-1])) > 0.9                 # kotak diletakkan terbalik
    assert similarity(s, signature(navy)) < 0.6


def test_nurse_review_confirms_edits_and_deletes(tmp_path, monkeypatch):
    import argparse
    import builtins
    import importlib.util
    spec = importlib.util.spec_from_file_location("rak_inventaris", "scripts/rak_inventaris.py")
    rak = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rak)
    rak.open_image = lambda p: None
    inv = Inventory(tmp_path / "inv.csv")
    a = place_scanned(inv, ("angiolite", 2.5, 29.0), "teks + REF (2 bacaan)", "", "", "", 1, "", "")
    b = place_scanned(inv, ("angiolite", 4.0, 19.0), "teks + REF (2 bacaan)", "", "", "", 2, "", "")
    c = place_scanned(inv, ("accuforce", 2.5, 12.0), "teks + REF (2 bacaan)", "", "", "", 3, "", "")
    inv.save()
    answers = iter(["Sari Wulandari",
                    "y",                                                       # a benar
                    "e", "angiolite", "4", "19", "", "2512587", "2028-09-21",   # b: isi LOT dan kedaluwarsa
                    "n"])                                                      # c: tercatat dua kali -> hapus
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers))
    rak.cek(argparse.Namespace(inventaris=str(tmp_path / "inv.csv"), by=None, semua=False, tanpa_gambar=True,
                               config="config/box_pipeline_config.yaml"))
    out = Inventory(tmp_path / "inv.csv")
    assert out.boxes[a.box_id].status == STATUS_OK and out.boxes[a.box_id].checked_by == "Sari Wulandari"
    assert out.boxes[b.box_id].lot == "2512587" and out.boxes[b.box_id].expiry == "2028-09-21"
    assert c.box_id not in out.boxes


def test_sizeless_product_name_is_a_key_and_needs_second_source():
    names = ["permanent sled bag", "agent"]
    ev = Evidence(text=parse_identity([OcrToken("Permanent Sled Bag", 0.99)], names))
    key, src, _ = identify([ev, ev], catalog(), today=TODAY)
    assert key == ("permanent sled bag", None, None) and src.startswith("USULAN")


def test_distinctive_first_word_reads_multiword_product():
    names = ["conqueror nc pro", "nc emerge", "angiolite"]
    i = parse_identity([OcrToken("CONQUEROF", 0.95), OcrToken("4.00", 0.95), OcrToken("mm", 0.99)], names)
    assert i.product == "conqueror nc pro" and i.product_ok
    j = parse_identity([OcrToken("NC", 0.99), OcrToken("Pro", 0.99)], names)          # kata umum: tidak cukup
    assert j.product != "conqueror nc pro" or not j.product_ok


def test_edit_rejects_brand_as_product_rep_as_gtin_and_product_as_lot(monkeypatch):
    import builtins
    import importlib.util
    spec = importlib.util.spec_from_file_location("rak_inventaris", "scripts/rak_inventaris.py")
    rak = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rak)
    from src.box_inventory import InvBox
    b = InvBox(box_id="K08")
    answers = iter(["boston scientific", "nc emerge",          # merek ditolak -> pilih produk
                    "5,5", "12",
                    "DE-RS3024ASM", "08714729847182",           # kode REF bukan GTIN
                    "NC Emerge Monorail", "38259111",           # nama produk bukan LOT
                    "03-01-2028", "2028-01-03"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers))
    rak.edit_box(b)
    assert (b.product, b.diameter, b.length, b.brand) == ("nc emerge", 5.5, 12.0, "boston scientific")
    assert (b.gtin, b.lot, b.expiry) == ("08714729847182", "38259111", "2028-01-03")
    b2 = InvBox(box_id="K01")
    answers = iter(["sled bag", "-", "-", "2029-04-26"])            # produk tanpa ukuran: diameter/panjang dilewati
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers))
    rak.edit_box(b2)
    assert (b2.product, b2.diameter, b2.length, b2.expiry) == ("permanent sled bag", None, None, "2029-04-26")


def test_box_returned_to_shelf_restores_status(tmp_path):
    import argparse
    import importlib.util
    spec = importlib.util.spec_from_file_location("rak_inventaris", "scripts/rak_inventaris.py")
    rak = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rak)
    inv = Inventory(tmp_path / "inv.csv")
    a = place_scanned(inv, ("angiolite", 2.5, 29.0), "teks + REF (2 bacaan)", "", "", "", 1, "", "")
    b = place_scanned(inv, ("angiolite", 4.0, 19.0), "teks + REF (2 bacaan)", "", "", "", 2, "", "")
    a.status, a.checked_by, a.taken_at = STATUS_DIAMBIL, "Sari Wulandari", "2026-09-30T07:04:00"
    b.status, b.taken_at = STATUS_DIAMBIL, "2026-09-30T07:05:00"               # belum pernah diperiksa perawat
    inv.save()
    rak.kembali(argparse.Namespace(inventaris=str(tmp_path / "inv.csv"), kotak=f"{a.box_id.lower()},{b.box_id}",
                                   config="config/box_pipeline_config.yaml"))
    out = Inventory(tmp_path / "inv.csv")
    assert out.boxes[a.box_id].status == STATUS_OK and not out.boxes[a.box_id].taken_at
    assert out.boxes[b.box_id].status not in (STATUS_OK, STATUS_DIAMBIL)       # tetap harus diperiksa perawat
    assert "dikembalikan" in out.boxes[a.box_id].note


def test_running_program_does_not_overwrite_kembali_from_another_window(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("twb", "scripts/test_webcam_box.py")
    twb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(twb)
    inv = Inventory(tmp_path / "inv.csv")
    a = place_scanned(inv, ("angiolite", 2.5, 29.0), "teks + REF (2 bacaan)", "", "", "", 1, "", "")
    b = place_scanned(inv, ("angiolite", 4.0, 19.0), "teks + REF (2 bacaan)", "", "", "", 2, "", "")
    a.status = b.status = STATUS_OK
    inv.save()
    running = Inventory(tmp_path / "inv.csv")                          # dipegang program uji
    assert twb._mark_taken(running, Request("angiolite", 2.5, 29.0), None) == a.box_id
    other = Inventory(tmp_path / "inv.csv")                            # jendela lain: kotak A dikembalikan
    other.boxes[a.box_id].status = STATUS_OK
    other.save()
    assert twb._mark_taken(running, Request("angiolite", 4.0, 19.0), None) == b.box_id
    out = Inventory(tmp_path / "inv.csv")
    assert out.boxes[a.box_id].status == STATUS_OK and out.boxes[b.box_id].status == STATUS_DIAMBIL
