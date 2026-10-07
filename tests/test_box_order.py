"""pytest tests/test_box_order.py -q  (pesanan beberapa barang, jumlah, ganti / tambah / lewati, bacaan dipakai ulang)"""
import pytest

from src.box_evidence import redecide
from src.box_order import OrderQueue, order_command, parse_order
from src.box_ref import parse_refs
from src.box_verifier import IGNORED, MATCH, Evidence, Identity, Request, TrackState


def keys(items):
    return [(i.req.product, i.req.diameter, i.req.length, i.qty) for i in items]


def test_single_request_formats_still_work():
    assert keys(parse_order("angiolite,2.5,29")) == [("angiolite", 2.5, 29.0, 1)]
    assert keys(parse_order("pilih terumo accuforce diameter 2.75 panjang 20")) == [("accuforce", 2.75, 20.0, 1)]
    assert keys(parse_order("permanent sled bag")) == [("permanent sled bag", None, None, 1)]


def test_quantity_and_several_items_in_one_line():
    assert keys(parse_order("2 kotak angiolite 2.5 29")) == [("angiolite", 2.5, 29.0, 2)]
    assert keys(parse_order("angiolite 2.5 x 29; accuforce 2.75 x 20 jumlah 2")) == \
        [("angiolite", 2.5, 29.0, 1), ("accuforce", 2.75, 20.0, 2)]
    assert keys(parse_order("angiolite 2.5 29 dan essential pro 3 40 + xperience pro 2.5 15")) == \
        [("angiolite", 2.5, 29.0, 1), ("essential pro", 3.0, 40.0, 1), ("xperience pro", 2.5, 15.0, 1)]
    assert keys(parse_order("angiolite 2.5 x 29 mm")) == [("angiolite", 2.5, 29.0, 1)]   # ukuran bukan jumlah


def test_one_unclear_item_rejects_the_whole_line():
    with pytest.raises(ValueError):
        parse_order("angiolite 2.5 29; produkxyz 3 40")                 # tidak ada barang yang diam-diam hilang
    with pytest.raises(ValueError):
        parse_order("30 kotak angiolite 2.5 29")                         # jumlah tidak masuk akal
    with pytest.raises(ValueError):
        parse_order("  ")


def test_order_commands_replace_add_skip():
    assert order_command("accuforce 2.75 20")[0] == "replace"
    kind, items = order_command("+ essential pro 3 40")
    assert kind == "add" and keys(items) == [("essential pro", 3.0, 40.0, 1)]
    assert order_command("tambah 2 kotak angiolite 4 19")[0] == "add"
    assert order_command("lewati") == ("skip", [])


def test_order_queue_runs_items_in_order_and_keeps_results():
    q = OrderQueue(parse_order("angiolite 2.5 29; accuforce 2.75 20 jumlah 2"))
    assert q.start_next().req.product == "angiolite"
    assert "berikutnya: accuforce" in q.describe()
    q.finish(1, "SELESAI")
    q.add(parse_order("essential pro 3 40"))
    assert q.start_next().qty == 2
    q.finish(0, "TIDAK ADA")
    q.replace(parse_order("xperience pro 2.5 15"))                      # permintaan berubah tiba-tiba
    assert q.start_next().req.product == "xperience pro" and not q.pending
    assert [r[3] for r in q.results] == ["SELESAI", "TIDAK ADA"]


def _read_angiolite_2529():
    text = Identity(product="angiolite", product_ok=True, diameter=2.5, diameter_ok=True, length=29.0,
                    length_ok=True, text="angiolite 2.5 29")
    return Evidence(text=text, refs=parse_refs("SCCDSR14150250029"))


def test_readings_are_reused_for_a_new_request():
    """Rak sudah dibaca untuk permintaan accuforce; permintaan berganti ke angiolite 2.5x29: kotak itu langsung
    MATCH dari bacaan yang sudah ada, tanpa membaca rak dari nol."""
    cfg = {"verify": {}}
    st = TrackState()
    old = Request("accuforce", 2.75, 20.0)
    for _ in range(2):
        ev = _read_angiolite_2529()
        d, r, k, sup = redecide(ev, old, cfg)
        st.add(d, k, sup, ev)
    assert st.stable(2) == IGNORED
    assert st.redecide(lambda ev: redecide(ev, Request("angiolite", 2.5, 29.0), cfg))
    assert st.stable(2) == MATCH
    assert st.redecide(lambda ev: redecide(ev, Request("angiolite", 4.0, 19.0), cfg))
    assert st.stable(2) == IGNORED                                      # saudaranya tetap bukan target


def test_readings_without_stored_evidence_are_dropped_on_request_change():
    st = TrackState()
    st.history.extend([(MATCH, ("k",)), (MATCH, ("k",))])               # bacaan lama tanpa Evidence tersimpan
    assert not st.redecide(lambda ev: (MATCH, "", ("k",), (frozenset(), False)))
    assert not st.history                                               # dibaca ulang, tidak ditebak
