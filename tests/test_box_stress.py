"""pytest tests/test_box_stress.py -q  (stress test: rencana, batas waktu per barang, catatan hasil)"""
import csv
import importlib.util
import random

from src.box_inventory import InvBox
from src.box_stress import StressRun, load_plan

PLAN = """# Bagian 1: kotak bergiliran
angiolite 2.5 29   # K15
accuforce 2.75 20  # K16

# Bagian 2: formasi 1
! formasi 1: ubah 1 kotak -> K15 balik 180
angiolite 2.5 29   # K15 (balik)
"""


def run_with(tmp_path, per_item=90.0):
    p = tmp_path / "plan.txt"
    p.write_text(PLAN, encoding="utf-8")
    s = StressRun(load_plan(p), per_item, tmp_path / "hasil.csv")
    s.advance()
    return s


def test_plan_has_sections_notes_and_pauses(tmp_path):
    s = run_with(tmp_path)
    kinds = [(st.kind, st.section) for st in s.steps]
    assert kinds == [("req", "Bagian 1: kotak bergiliran"), ("req", "Bagian 1: kotak bergiliran"),
                     ("pause", "Bagian 2: formasi 1"), ("req", "Bagian 2: formasi 1")]
    assert s.steps[0].text == "angiolite 2.5 29" and s.steps[0].note == "K15" and s.n_req == 3


def test_ambil_counts_only_after_it_holds_and_batal_resets(tmp_path):
    s = run_with(tmp_path)
    stats = {"reads": 0, "ocr_ms": 0.0}
    s.start_item(100.0, stats)
    s.on_events([(110.0, "PICK", 7, "teks + REF", None)])
    assert s.check(111.0, "FOUND", 7) is None                 # belum bertahan 2 s
    s.on_events([(111.5, "PICK CANCELLED", 7, "2 bacaan bertentangan", None)])
    assert s.check(113.0, "SEARCHING", None) is None                # AMBIL yang batal tidak dihitung
    s.on_events([(120.0, "PICK", 9, "teks + REF", None)])
    assert s.check(122.5, "FOUND", 9) == "PICK"
    stats.update(reads=10, ocr_ms=4000.0)
    row = s.finish_item(122.5, "PICK", stats, 20, "angiolite 2.5 29 SCCDSR14150250029")
    assert row["detik"] == "20.0" and row["kotak"] == 9 and row["batal"] == 1 and row["ms_per_bacaan"] == "400"


def test_timeout_and_tidak_ada_move_on(tmp_path):
    s = run_with(tmp_path, per_item=90.0)
    s.start_item(0.0, {})
    assert s.check(89.0, "SEARCHING", None) is None
    assert s.check(90.0, "SEARCHING", None) == "TIMEOUT"
    s.start_item(100.0, {})
    assert s.check(101.0, "NOT FOUND", None) is None
    assert s.check(104.5, "NOT FOUND", None) == "NOT FOUND"


def test_results_are_written_and_summarised(tmp_path):
    s = run_with(tmp_path)
    s.start_item(0.0, {})
    s.on_events([(12.0, "PICK", 3, "teks + REF", None)])
    s.finish_item(14.5, "PICK", {}, 20)
    s.advance()
    s.start_item(20.0, {})
    s.finish_item(110.0, "TIMEOUT", {}, 20)
    rows = list(csv.DictReader(open(tmp_path / "hasil.csv", encoding="utf-8")))
    assert [r["hasil"] for r in rows] == ["PICK", "TIMEOUT"] and rows[1]["detik"] == "90.0"
    # the on-screen counter and summary must count the SAME result names (08-10: counted "AMBIL" -> always 0)
    assert "found 1/2" in s.status(120.0) and "PICK 1/2" in s.summary()[-2] and "timeout 1" in s.summary()[-2]
    text = "\n".join(s.summary())
    assert "PICK 1/2" in text and "accuforce 2.75 20 (TIMEOUT)" in text
    assert s.advance().kind == "pause" and s.waiting                # jeda ganti formasi menunggu "lanjut"
    assert s.advance().kind == "req" and not s.waiting
    assert s.advance() is None


def test_stress_plan_covers_every_box_then_formations_then_random():
    spec = importlib.util.spec_from_file_location("stress_plan", "scripts/stress_plan.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    boxes = [InvBox(f"K{i:02d}", "angiolite", 2.5, 29) for i in range(1, 21)]
    boxes[0] = InvBox("K01", "permanent sled bag")
    lines = mod.build_plan(boxes, formasi=5, per_formasi=5, acak=10, rng=random.Random(1))
    reqs = [ln for ln in lines if ln and not ln.startswith(("#", "!"))]
    assert len(reqs) == 20 + 5 * 5 + 10
    assert sum(ln.startswith("!") for ln in lines) == 6               # 5 formasi + 1 sebelum permintaan acak
    assert reqs[0].startswith("permanent sled bag   # K01")           # tanpa ukuran: nama saja


def test_until_taken_item_ends_only_when_the_box_is_taken(tmp_path):
    # list mode (live test 09-10): PICK alone is not the end; the item ends at TAKEN, then the next request starts
    s = run_with(tmp_path)
    s.until_taken = True
    s.start_item(0.0, {})
    s.on_events([(10.0, "PICK", 3, "text + REF", None)])
    assert s.check(15.0, "FOUND", 3) is None                     # picked, not taken yet: keep the item
    s.on_events([(25.0, "TAKEN", 3, "empty slot verified", None)])
    assert s.check(25.1, "DONE", None) == "TAKEN"
    row = s.finish_item(25.1, "TAKEN", {}, 20)
    assert row["kotak"] == 3 and row["detik"] == "10.0" and "found 1/1" in s.status(26.0)
