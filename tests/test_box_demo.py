"""pytest tests/test_box_demo.py -q  (demo permintaan teks: posisi rak/level, status, pesan TM)"""
import datetime as dt
import json

from src.box_demo import (EXPIRED, HABIS, INVALID, NO_POS, SIAP, Position, append_log, load_positions, lookup,
                          to_text, to_tm_message)
from src.box_inventory import STATUS_DIAMBIL, STATUS_OK, InvBox, Inventory

TODAY = dt.date(2026, 10, 7)


def inv(tmp_path, *boxes):
    i = Inventory(tmp_path / "inventaris.csv")
    i.boxes = {b.box_id: b for b in boxes}
    return i


def test_positions_skip_rows_without_rack_or_level(tmp_path):
    p = tmp_path / "posisi.csv"
    p.write_text("box_id,rak,level,kolom,sumber\nK01,R1,2,5,tim\nK02,,1,,\nK03,R2,,,\n", encoding="utf-8")
    pos = load_positions(p)
    assert list(pos) == ["K01"]
    assert str(pos["K01"]) == "rak R1, level 2, kolom 5"
    assert load_positions(tmp_path / "tidak_ada.csv") == {}


def test_unparseable_request_is_invalid_and_never_guessed(tmp_path):
    res = lookup("angiolite 2.5", inv(tmp_path), {}, TODAY)           # panjang tidak ada
    assert res.status == INVALID and res.box is None
    assert "belum lengkap" in res.reason
    msg = to_tm_message(res)
    assert msg["status"] == INVALID and msg["rack"] == "" and msg["needs_doctor_confirmation"] is True


def test_no_stock_is_habis(tmp_path):
    res = lookup("angiolite 2.5 x 29", inv(tmp_path), {}, TODAY)
    assert res.status == HABIS and res.reason == "tidak ada di inventaris"


def test_taken_box_is_habis_with_reason(tmp_path):
    b = InvBox("K10", "angiolite", 4.0, 19.0, "ivascular", status=STATUS_DIAMBIL)
    res = lookup("ivascular angiolite 4x19", inv(tmp_path, b), {}, TODAY)
    assert res.status == HABIS and "K10" in res.reason


def test_only_expired_stock_is_reported_not_picked(tmp_path):
    b = InvBox("K08", "nc emerge", 5.5, 12.0, "boston scientific", expiry="2026-01-03", status=STATUS_OK)
    res = lookup("nc emerge 5.5 x 12", inv(tmp_path, b), {}, TODAY)
    assert res.status == EXPIRED and res.box is None and "2026-01-03" in res.reason


def test_wrong_size_is_not_a_hit(tmp_path):
    b = InvBox("K15", "angiolite", 2.5, 29.0, "ivascular", status=STATUS_OK)
    assert lookup("angiolite 2.75 x 29", inv(tmp_path, b), {}, TODAY).status == HABIS


def test_ready_box_has_position_and_tm_fields(tmp_path):
    b = InvBox("K15", "angiolite", 2.5, 29.0, "ivascular", status=STATUS_OK)
    res = lookup("ivascular angiolite 2.5 x 29", inv(tmp_path, b), {"K15": Position("R1", "2", "15", "tim")}, TODAY)
    assert res.status == SIAP and res.box.box_id == "K15" and res.reason == ""
    msg = to_tm_message(res)
    assert (msg["box_id"], msg["rack"], msg["level"], msg["column"]) == ("K15", "R1", "2", "15")
    assert (msg["product"], msg["diameter_mm"], msg["length_mm"]) == ("angiolite", 2.5, 29.0)


def test_fefo_first_box_wins_even_if_only_a_later_box_has_a_position(tmp_path):
    early = InvBox("K03", "permanent sled bag", expiry="2029-03-10", status=STATUS_OK)
    late = InvBox("K01", "permanent sled bag", expiry="2029-04-26", status=STATUS_OK)
    res = lookup("sled bag", inv(tmp_path, early, late), {"K01": Position("R1", "1")}, TODAY)
    assert res.status == NO_POS and res.box.box_id == "K03" and res.position is None
    assert res.stock == ["K03", "K01"]


def test_fefo_order_picks_earliest_expiry(tmp_path):
    boxes = [InvBox(k, "permanent sled bag", expiry=e, status=STATUS_OK)
             for k, e in (("K01", "2029-04-26"), ("K03", "2029-03-10"), ("K17", "2029-03-12"))]
    pos = {b.box_id: Position("R1", "1", b.box_id[1:], "contoh") for b in boxes}
    res = lookup("permanent sled bag", inv(tmp_path, *boxes), pos, TODAY)
    assert res.status == SIAP and res.box.box_id == "K03" and "CONTOH" in res.reason


def test_log_is_one_json_line_per_request(tmp_path):
    i = inv(tmp_path)
    log = tmp_path / "log" / "demo.jsonl"
    for t in ("angiolite 2.5 x 29", "bukan produk"):
        append_log(log, lookup(t, i, {}, TODAY))
    lines = log.read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["status"] for x in lines] == [HABIS, INVALID]


def test_text_shows_what_was_understood(tmp_path):
    out = to_text(lookup("terumo accuforce 2,75 x 20", inv(tmp_path), {}, TODAY))
    assert "accuforce (terumo), diameter 2.75 mm, panjang 20 mm" in out
