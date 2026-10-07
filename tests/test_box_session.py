"""pytest tests/test_box_session.py -q  (alur penyeleksi: rak kosong, hitung, cari, TIDAK ADA, AMBIL, TERAMBIL)"""
import numpy as np

from src.box_evidence import Evidence
from src.box_ref import parse_refs
from src.box_session import (DITEMUKAN, MENCARI, PERLU_KONFIRMASI, RAK_KOSONG, SELESAI, TIDAK_ADA,
                             SelectionSession)
from src.box_verifier import (CANDIDATE, CONFIRM, EMPTY, IGNORED, MATCH, READING, Identity, Request,
                              SimpleTracker, decide_multi)

FS = (1920, 1080)
REQ = Request("angiolite", 2.5, 29.0)
KEY = ("angiolite", 2.5, 29.0, ())


def spine(x, y=300, w=40, h=400):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


SHELF = [spine(100 + 60 * i) for i in range(20)]             # 20 kotak berjejer di rak


class Clock:
    def __init__(self):
        self.t = 0.0


def run(sess, tr, polys, clock, seconds, still=True, frame=None, fps=10):
    ids = []
    for _ in range(int(seconds * fps)):
        clock.t += 1.0 / fps
        ids = tr.update(polys, None, FS)
        f = frame() if callable(frame) else frame
        sess.update(clock.t, ids, polys, tr, FS, None, still, f)
    return ids


def decide(tr, ids, rule):
    """Beri tiap kotak 2 bacaan yang sama (seperti dua kali OCR yang sepakat)."""
    for i, tid in enumerate(ids):
        d = rule(i)
        tr.state(tid).history.extend([(d, KEY if d == MATCH else None)] * 2)


def started(n=20, **kw):
    sess, tr, clock = SelectionSession(REQ, need=2, **kw), SimpleTracker(), Clock()
    ids = run(sess, tr, SHELF[:n], clock, 2.0)
    return sess, tr, clock, ids


# ------------------------------------------------------------------ 1 rak kosong / hitung
def test_empty_shelf_stops_search():
    sess, tr, clock = SelectionSession(REQ), SimpleTracker(), Clock()
    run(sess, tr, [], clock, 3.0)
    assert sess.phase == RAK_KOSONG and not sess.ocr_allowed                 # OCR tidak dijalankan
    assert any(e[1] == RAK_KOSONG for e in sess.events)
    run(sess, tr, SHELF, clock, 2.0)                                           # kamera diarahkan ke rak berisi
    assert sess.phase == MENCARI and sess.shelf_n == 20


def test_moving_camera_does_not_count():
    sess, tr, clock = SelectionSession(REQ), SimpleTracker(), Clock()
    run(sess, tr, SHELF, clock, 3.0, still=False)
    assert sess.phase != MENCARI and sess.shelf_n == 0


def test_count_then_find_the_one_requested_box():
    sess, tr, clock, ids = started()
    assert sess.phase == MENCARI and sess.shelf_n == 20
    target = ids[7]
    decide(tr, ids, lambda i: MATCH if i == 7 else IGNORED)                  # 19 lainnya pasti bukan target
    run(sess, tr, SHELF, clock, 0.2)
    assert sess.phase == DITEMUKAN and sess.pick == target
    assert "AMBIL" in sess.message and "19 kotak lain tidak diambil (19 pasti bukan target)" in sess.detail


# ------------------------------------------------------------------ 2 TIDAK ADA hanya kalau pasti
def test_all_boxes_not_target_gives_tidak_ada():
    sess, tr, clock, ids = started()
    decide(tr, ids, lambda i: IGNORED)
    run(sess, tr, SHELF, clock, 0.2)
    assert sess.phase == TIDAK_ADA and sess.color == "red"
    assert any(e[1] == TIDAK_ADA for e in sess.events)


def test_one_unsure_box_blocks_tidak_ada():
    for unsure in (CONFIRM, CANDIDATE):
        sess, tr, clock, ids = started()
        decide(tr, ids, lambda i: unsure if i == 3 else IGNORED)
        run(sess, tr, SHELF, clock, 0.5)
        assert sess.phase == PERLU_KONFIRMASI, unsure                         # tidak menebak "tidak ada"
        assert f"#{ids[3]}" in sess.message


def test_unread_box_blocks_tidak_ada():
    sess, tr, clock, ids = started()
    decide(tr, ids[:19], lambda i: IGNORED)                                   # kotak ke-20 belum dibaca
    run(sess, tr, SHELF, clock, 0.5)
    assert sess.phase == MENCARI


def test_boxes_never_checked_block_tidak_ada():
    sess, tr, clock, ids = started()
    for tid in ids[12:]:                                                      # 8 kotak hilang dari pelacakan
        tr.tracks.pop(tid)
    decide(tr, ids[:12], lambda i: IGNORED)
    run(sess, tr, SHELF[:12], clock, 0.5)
    assert sess.phase == PERLU_KONFIRMASI and "12 dari 20" in sess.message


def test_boxes_out_of_view_never_count_for_tidak_ada():
    sess, tr, clock, ids = started(n=14)
    decide(tr, ids, lambda i: IGNORED)
    for tid in ids[7:]:                                       # 7 kotak "keluar layar" (posisi perkiraan bergeser)
        tr.tracks[tid]["poly"] = np.float32(spine(2500 + 60 * tid))
        tr.tracks[tid]["was_out"] = True
    run(sess, tr, SHELF[:7], clock, 0.5)                      # hanya 7 kotak terlihat
    assert sess.phase == PERLU_KONFIRMASI and "7 dari 14" in sess.message


def test_tidak_ada_waits_for_rereading_of_doubtful_boxes():
    sess, tr, clock, ids = started()
    decide(tr, ids, lambda i: IGNORED)
    for tid in ids[:3]:
        tr.state(tid).mark_stale()                            # nomor 3 kotak sempat diragukan
    run(sess, tr, SHELF, clock, 0.5)
    assert sess.phase == MENCARI
    for tid in ids[:3]:
        tr.state(tid).history.extend([(IGNORED, None), (IGNORED, None)])   # dibaca ulang 2 kali
        tr.state(tid).stale = False
    run(sess, tr, SHELF, clock, 0.2)
    assert sess.phase == TIDAK_ADA


FULL = (frozenset({"tp", "td", "tl", "ref"}), False)        # bacaan lengkap: teks 2.5 x 29 + REF
NAME_ONLY = (frozenset({"tp"}), False)                      # "angiolite" saja (kotak saudara juga begitu)
REF_ONLY = (frozenset({"ref"}), False)
SIBLING = (frozenset({"tp"}), True)                         # angiolite 4 x 19: ukuran membantah


def events(sess, what):
    return [e for e in sess.events if e[1] == what]


def test_doubtful_box_number_holds_pick_and_rechecks_instead_of_cancelling():
    sess, tr, clock, ids = started()
    decide(tr, ids, lambda i: MATCH if i == 4 else IGNORED)
    run(sess, tr, SHELF, clock, 0.2)
    assert sess.phase == DITEMUKAN
    st = tr.state(ids[4])
    st.mark_stale()                                          # tangan meraih kotak: nomornya diragukan
    run(sess, tr, SHELF, clock, 1.0)
    assert sess.phase == DITEMUKAN and sess.pick == ids[4] and not sess.solid     # dikunci, JANGAN diambil
    assert "JANGAN diambil" in sess.message and events(sess, "CEK ULANG") and not events(sess, "BATAL AMBIL")
    st.stale = False
    st.add(CONFIRM, EMPTY, (frozenset(), False))             # tangan masih menutupi: kosong, tetap ditahan
    st.add(CANDIDATE, KEY, NAME_ONLY)                        # "angiolite" saja: belum cukup (saudaranya juga)
    run(sess, tr, SHELF, clock, 0.5)
    assert sess.phase == DITEMUKAN and not sess.solid
    st.add(CANDIDATE, KEY, REF_ONLY)                         # REF 2.5 x 29 terbaca: kotak yang sama
    run(sess, tr, SHELF, clock, 0.2)
    assert sess.phase == DITEMUKAN and sess.solid and events(sess, "AMBIL DIPASTIKAN")


def test_recheck_of_doubtful_pick_rejects_sibling_box():
    sess, tr, clock, ids = _pick_ready()
    st = tr.state(ids[4])
    st.mark_stale()
    st.stale = False
    st.add(IGNORED, KEY, SIBLING)                            # ternyata angiolite 4 x 19
    run(sess, tr, SHELF, clock, 0.3)
    assert sess.phase == DITEMUKAN and not sess.solid        # satu bantahan: tetap ditahan, dibaca lagi
    st.add(CANDIDATE, KEY, REF_ONLY)                         # satu bukti sesudah bantahan belum cukup (perlu skor 2)
    run(sess, tr, SHELF, clock, 0.3)
    assert not sess.solid
    st.add(IGNORED, KEY, SIBLING)
    run(sess, tr, SHELF, clock, 0.2)
    assert sess.phase != DITEMUKAN and events(sess, "BATAL AMBIL") and not events(sess, "AMBIL DIPASTIKAN")


def test_recheck_times_out_only_while_box_is_in_its_slot():
    sess, tr, clock, ids = _pick_ready()
    tr.state(ids[4]).mark_stale()
    run(sess, tr, SHELF, clock, 9.0)
    assert sess.phase == DITEMUKAN and not sess.solid
    run(sess, tr, SHELF, clock, 1.5)                         # > relock_s tanpa bacaan baru yang lengkap
    assert sess.phase != DITEMUKAN
    assert any("tidak terpastikan" in e[3] for e in events(sess, "BATAL AMBIL"))


def test_box_grabbed_right_after_doubt_is_taken_not_cancelled():
    # uji 30-09: nomor kotak AMBIL diragukan tepat saat tangan meraihnya -> dulu BATAL AMBIL, TERAMBIL tak tercatat
    sess, tr, clock, ids = _pick_ready()
    tr.state(ids[4]).mark_stale()
    run(sess, tr, SHELF, clock, 0.3)
    for k in range(1, 9):                                    # kotak ditarik ke atas
        polys = [spine(100 + 60 * 4, y=300 - 50 * k) if i == 4 else p for i, p in enumerate(SHELF)]
        run(sess, tr, polys, clock, 0.1)
        assert sess.phase == DITEMUKAN
    run(sess, tr, [p for i, p in enumerate(SHELF) if i != 4], clock, 2.5)
    assert sess.phase == SELESAI and events(sess, "TERAMBIL") and not events(sess, "BATAL AMBIL")


def test_changed_look_of_pick_in_slot_is_rechecked():
    sess, tr, clock, ids = started()
    decide(tr, ids, lambda i: MATCH if i == 4 else IGNORED)
    run(sess, tr, SHELF, clock, 0.5, frame=_scene("box"))
    assert sess.phase == DITEMUKAN and sess.solid
    run(sess, tr, SHELF, clock, 0.3, frame=_scene("other"))  # nomor sama, isi slot tampak lain: belum pasti
    assert sess.solid and not events(sess, "CEK ULANG")      # (bisa jadi potongan slot sesaat meleset)
    run(sess, tr, SHELF, clock, 0.5, frame=_scene("other"))  # berturut-turut -> dipastikan ulang
    assert sess.phase == DITEMUKAN and not sess.solid and events(sess, "CEK ULANG")
    tr.state(ids[4]).add(MATCH, KEY, FULL)                   # bacaan BARU: tetap kotak yang diminta
    run(sess, tr, SHELF, clock, 0.5, frame=_scene("other"))
    assert sess.phase == DITEMUKAN and sess.solid


def test_slot_stays_on_shelf_while_box_is_pulled_out():
    sess, tr, clock, ids = started()
    decide(tr, ids, lambda i: MATCH if i == 4 else IGNORED)
    run(sess, tr, SHELF, clock, 0.2)
    slot0 = np.float32(SHELF[4])
    for k in range(1, 9):                                     # kotak ditarik ke atas 50 px per frame
        polys = [spine(100 + 60 * 4, y=300 - 50 * k) if i == 4 else p for i, p in enumerate(SHELF)]
        run(sess, tr, polys, clock, 0.1)
        assert sess.phase == DITEMUKAN
    from src.box_geometry import poly_iou
    assert poly_iou(sess.slot, slot0) > 0.9                   # slot tetap di rak, tidak ikut kotak
    rest = [p for i, p in enumerate(SHELF) if i != 4]
    run(sess, tr, rest, clock, 2.5)
    assert sess.phase == SELESAI


def test_slot_follows_camera_motion():
    sess, tr, clock, ids = started()
    decide(tr, ids, lambda i: MATCH if i == 4 else IGNORED)
    run(sess, tr, SHELF, clock, 0.2)
    step = np.float32([[1, 0, 10], [0, 1, 0]])                # kamera geser 10 px per frame, terukur
    for k in range(1, 4):
        clock.t += 0.1
        moved = [np.float32(p) + (10 * k, 0) for p in SHELF]
        ids2 = tr.update(moved, step, FS)
        sess.update(clock.t, ids2, moved, tr, FS, step, True, None)
    from src.box_geometry import poly_iou
    assert sess.phase == DITEMUKAN and poly_iou(sess.slot, np.float32(SHELF[4]) + (30, 0)) > 0.9


def test_neighbour_leaning_into_gap_still_counts_as_taken():
    sess, tr, clock, ids = started()
    decide(tr, ids, lambda i: MATCH if i == 4 else IGNORED)
    run(sess, tr, SHELF, clock, 0.2)
    lean = [spine(100 + 60 * 3 + 14) if i == 3 else p for i, p in enumerate(SHELF) if i != 4]   # sebelah condong 30 %
    run(sess, tr, lean, clock, 2.5)
    assert sess.phase == SELESAI


def test_expected_count_shortfall_blocks_tidak_ada():
    sess, tr, clock, ids = started(n=18, expected=20)                         # detektor hanya melihat 18 dari 20
    decide(tr, ids, lambda i: IGNORED)
    run(sess, tr, SHELF[:18], clock, 0.5)
    assert sess.phase == PERLU_KONFIRMASI and "18 dari 20" in sess.message


# ------------------------------------------------------------------ 3 AMBIL -> TERAMBIL -> permintaan lagi
def test_taken_box_verified_then_same_request_again():
    sess, tr, clock, ids = started()
    a, b = ids[4], ids[11]                                                    # dua kotak identik yang diminta
    decide(tr, ids, lambda i: MATCH if i in (4, 11) else IGNORED)
    run(sess, tr, SHELF, clock, 0.2)
    assert sess.phase == DITEMUKAN and sess.pick == a
    rest = [p for i, p in enumerate(SHELF) if i != 4]                         # kotak a diambil
    run(sess, tr, rest, clock, 1.0)
    assert sess.phase == DITEMUKAN                                            # belum 2 detik: belum dianggap terambil
    run(sess, tr, rest, clock, 1.5)
    assert sess.phase == SELESAI and [t.track for t in sess.taken] == [a]
    assert a not in tr.tracks                                                 # nomor a dibuang
    sess.request_again(now=clock.t)                                           # permintaan yang sama lagi
    run(sess, tr, rest, clock, 0.2)
    assert sess.phase == DITEMUKAN and sess.pick == b                         # kotak lain, bukan slot kosong a
    rest2 = [p for i, p in enumerate(SHELF) if i not in (4, 11)]
    run(sess, tr, rest2, clock, 2.5)
    assert sess.phase == SELESAI and [t.track for t in sess.taken] == [a, b]
    sess.request_again(now=clock.t)
    run(sess, tr, rest2, clock, 0.5)
    assert sess.phase == TIDAK_ADA and "lagi" in sess.message                 # 18 sisanya bukan target


def test_request_again_waits_until_current_box_is_taken():
    sess, tr, clock, ids = started()
    decide(tr, ids, lambda i: MATCH if i in (4, 11) else IGNORED)
    run(sess, tr, SHELF, clock, 0.2)
    sess.request_again(now=clock.t)
    run(sess, tr, SHELF, clock, 3.0)
    assert sess.phase == DITEMUKAN and sess.pick == ids[4]                    # tetap kotak pertama sampai terambil


def test_not_taken_while_camera_moves():
    sess, tr, clock, ids = started()
    decide(tr, ids, lambda i: MATCH if i == 4 else IGNORED)
    run(sess, tr, SHELF, clock, 0.2)
    rest = [p for i, p in enumerate(SHELF) if i != 4]
    run(sess, tr, rest, clock, 5.0, still=False)
    assert sess.phase == DITEMUKAN and not sess.taken


def _scene(slot_pattern):
    """Frame sintetis: latar bertekstur tetap, kotak #5 bergaris di posisinya (atau isi lain kalau diambil)."""
    rng = np.random.default_rng(3)
    base = (rng.random((FS[1], FS[0], 3)) * 255).astype(np.uint8)
    x, y, w, h = 100 + 60 * 4, 300, 40, 400

    def make():
        f = base.copy()
        if slot_pattern == "box":
            f[y:y + h, x:x + w] = 0
            f[y:y + h:20, x:x + w] = 255                                      # garis-garis = kemasan
        elif slot_pattern == "hand":
            f[y:y + h, x:x + w] = (np.random.default_rng().random((h, w, 3)) * 255).astype(np.uint8)
        elif slot_pattern == "other":
            f[y:y + h, x:x + w] = 0
            f[y:y + h, x:x + w:8] = 255                                       # kemasan lain (garis tegak)
        else:
            f[y:y + h, x:x + w] = 90                                          # dinding rak polos
        return f
    return make


def test_detector_miss_is_not_taken_while_box_still_visible():
    sess, tr, clock, ids = started()
    decide(tr, ids, lambda i: MATCH if i == 4 else IGNORED)
    run(sess, tr, SHELF, clock, 0.5, frame=_scene("box"))                     # gambar kotak AMBIL disimpan
    rest = [p for i, p in enumerate(SHELF) if i != 4]
    run(sess, tr, rest, clock, 4.0, frame=_scene("box"))                      # detektor gagal, kotak masih ada
    assert sess.phase == DITEMUKAN and "masih terlihat" in sess.detail
    run(sess, tr, rest, clock, 2.5, frame=_scene("empty"))                    # kotak benar-benar diambil
    assert sess.phase == SELESAI


def test_hand_over_slot_is_not_taken():
    sess, tr, clock, ids = started()
    decide(tr, ids, lambda i: MATCH if i == 4 else IGNORED)
    run(sess, tr, SHELF, clock, 0.5, frame=_scene("box"))
    rest = [p for i, p in enumerate(SHELF) if i != 4]
    run(sess, tr, rest, clock, 4.0, frame=_scene("hand"))                     # isi slot bergerak terus
    assert sess.phase == DITEMUKAN and not sess.taken


def test_verification_change_cancels_pick():
    sess, tr, clock, ids = started()
    decide(tr, ids, lambda i: MATCH if i == 4 else IGNORED)
    run(sess, tr, SHELF, clock, 0.2)
    tr.state(ids[4]).history.append((CONFIRM, None))                         # bacaan baru bertentangan
    run(sess, tr, SHELF, clock, 0.2)
    assert sess.phase != DITEMUKAN and sess.pick is None
    assert any(e[1] == "BATAL AMBIL" for e in sess.events)


# ------------------------------------------------------------------ merek & nama sama, ukuran beda
def test_same_brand_same_name_other_sizes_never_match():
    sizes = [(2.0, 15), (2.25, 29), (2.5, 24), (2.5, 29), (2.75, 29), (3.0, 29), (2.5, 34), (2.5, 19), (3.5, 29),
             (2.25, 24), (2.0, 29), (2.5, 14), (3.0, 24), (4.0, 29), (2.5, 39), (2.75, 24), (3.25, 29), (2.5, 9),
             (3.0, 19), (2.25, 34)]
    decisions = []
    for d, length in sizes:
        text = Identity(product="angiolite", product_ok=True, diameter=d, diameter_ok=True, length=float(length),
                        length_ok=True, text=f"angiolite {d} x {length}")
        ref = parse_refs(f"SCCDSR14150{int(round(d * 100)):03d}{length:03d}")
        assert len(ref) == 1 and ref[0].diameter == d
        decisions.append(decide_multi(Evidence(text=text, refs=ref), REQ)[0])
    assert decisions.count(MATCH) == 1 and decisions[sizes.index((2.5, 29))] == MATCH
    assert all(x == IGNORED for i, x in enumerate(decisions) if sizes[i] != (2.5, 29))


def test_one_wrong_size_source_is_never_a_match():
    right = Identity(product="angiolite", product_ok=True, diameter=2.5, diameter_ok=True, length=29.0,
                     length_ok=True)
    wrong_ref = parse_refs("SCCDSR14150250024")                              # REF: 2.5 x 24
    d, reason, _ = decide_multi(Evidence(text=right, refs=wrong_ref), REQ)
    assert d == CONFIRM and "bertentangan" in reason                          # teks cocok tapi REF tidak -> ragu
    size_only = Identity(product="angiolite", product_ok=True, diameter=2.5, diameter_ok=True)
    assert decide_multi(Evidence(text=size_only), REQ)[0] == CANDIDATE        # panjang belum terbaca: bukan MATCH


# ------------------------------------------------------------------ bacaan kosong, kotak ditarik, penghuni slot
def test_empty_readings_do_not_break_agreement_but_repeated_ones_do():
    from src.box_verifier import EMPTY, TrackState
    st = TrackState()
    st.history.extend([(MATCH, KEY), (MATCH, KEY), (CONFIRM, EMPTY)])    # tangan menutupi sekali
    assert st.stable(2) == MATCH
    st.history.extend([(CONFIRM, EMPTY), (CONFIRM, EMPTY)])              # 3 kali berturut-turut tidak terbaca
    assert st.stable(2) == CONFIRM
    st2 = TrackState()
    st2.history.extend([(MATCH, KEY), (CANDIDATE, KEY)])                 # bacaan berisi yang lebih lemah memutus
    assert st2.stable(2) == CANDIDATE
    st3 = TrackState()
    st3.history.extend([(CONFIRM, EMPTY), (CONFIRM, EMPTY)])
    assert st3.stable(2) == CONFIRM                                      # tidak pernah terbaca: minta konfirmasi


def test_late_reading_from_before_doubt_is_discarded():
    import importlib.util
    spec = importlib.util.spec_from_file_location("twb", "scripts/test_webcam_box.py")
    twb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(twb)
    from src.box_verifier import TrackState
    st = TrackState()
    st.history.extend([(MATCH, KEY), (MATCH, KEY)])
    old_epoch = st.epoch
    st.pending = True
    st.mark_stale()                                                      # nomor diragukan saat OCR masih berjalan
    ev = Evidence()
    twb.apply_reading(st, ev, MATCH, "teks + REF", KEY, 0.0, 1, epoch=old_epoch)
    assert not st.history and st.stale and st.stable(2) == READING       # hasil crop lama dibuang


def _pick_ready():
    sess, tr, clock, ids = started()
    decide(tr, ids, lambda i: MATCH if i == 4 else IGNORED)
    run(sess, tr, SHELF, clock, 0.2)
    assert sess.phase == DITEMUKAN
    return sess, tr, clock, ids


def test_box_pulled_with_changing_number_is_taken_not_cancelled():
    sess, tr, clock, ids = _pick_ready()
    pulled = [spine(100 + 60 * 4, y=300 - 50) if i == 4 else p for i, p in enumerate(SHELF)]
    run(sess, tr, pulled, clock, 0.1)                                    # mulai ditarik (nomor sama)
    assert "sedang diambil" in sess.message
    tr.tracks.pop(ids[4])                                                # nomornya berganti saat bergerak
    for k in range(2, 5):
        pulled = [spine(100 + 60 * 4, y=300 - 50 * k) if i == 4 else p for i, p in enumerate(SHELF)]
        run(sess, tr, pulled, clock, 0.1)
        assert sess.phase == DITEMUKAN and not any(e[1] == "BATAL AMBIL" for e in sess.events)
    rest = [p for i, p in enumerate(SHELF) if i != 4]
    run(sess, tr, rest, clock, 2.5)
    assert sess.phase == SELESAI


def test_still_box_with_other_number_in_slot_is_rechecked():
    for reads, solid in (([(MATCH, KEY, FULL)], True), ([(IGNORED, None, SIBLING)] * 2, False)):
        sess, tr, clock, ids = _pick_ready()
        tr.tracks.pop(ids[4])                                            # nomor kotak AMBIL hilang, kotaknya diam
        run(sess, tr, SHELF, clock, 2.5)
        new = sess.pick
        assert sess.phase == DITEMUKAN and new not in (None, ids[4]) and not sess.solid   # dipastikan ulang dulu
        for d, k, s in reads:
            tr.state(new).add(d, k, s)
        run(sess, tr, SHELF, clock, 0.3)
        assert (sess.phase == DITEMUKAN and sess.solid) if solid else \
            (sess.phase != DITEMUKAN and events(sess, "BATAL AMBIL"))


def test_pick_held_on_one_weaker_reading_and_solid_again():
    sess, tr, clock, ids = _pick_ready()
    st = tr.state(ids[4])
    st.history.append((CANDIDATE, KEY))                                  # label sebagian tertutup tangan
    run(sess, tr, SHELF, clock, 0.2)
    assert sess.phase == DITEMUKAN and not sess.solid and "JANGAN diambil" in sess.message
    st.history.append((MATCH, KEY))
    run(sess, tr, SHELF, clock, 0.2)
    assert sess.phase == DITEMUKAN and sess.solid


def test_pick_cancelled_after_two_weaker_readings():
    sess, tr, clock, ids = _pick_ready()
    tr.state(ids[4]).history.extend([(CANDIDATE, KEY), (CANDIDATE, KEY)])
    run(sess, tr, SHELF, clock, 0.2)
    assert sess.phase != DITEMUKAN


def test_pick_survives_two_empty_readings_but_not_three():
    from src.box_verifier import EMPTY
    sess, tr, clock, ids = _pick_ready()
    st = tr.state(ids[4])
    st.history.extend([(CONFIRM, EMPTY), (CONFIRM, EMPTY)])
    run(sess, tr, SHELF, clock, 0.2)
    assert sess.phase == DITEMUKAN and not sess.solid                   # tetap dikunci, tapi jangan diambil dulu
    st.history.append((MATCH, KEY))
    run(sess, tr, SHELF, clock, 0.2)
    assert sess.solid
    st.history.extend([(CONFIRM, EMPTY)] * 3)
    run(sess, tr, SHELF, clock, 0.2)
    assert sess.phase != DITEMUKAN


# ------------------------------------------------------------------ hanya kotak DI RAK yang boleh AMBIL
def test_box_held_in_front_of_shelf_is_not_picked():
    sess, tr, clock, ids = started()
    held = spine(100 + 60 * 4 + 20, y=320)                             # menutupi separuh kotak ke-5
    polys = SHELF + [held]
    ids = run(sess, tr, polys, clock, 1.0)
    decide(tr, ids, lambda i: MATCH if i == 20 else IGNORED)
    run(sess, tr, polys, clock, 1.0)
    assert sess.phase != DITEMUKAN and "tidak di rak" in sess.message


def test_box_moving_in_hand_is_not_picked_until_it_rests_on_shelf():
    sess, tr, clock, ids = started(n=19)
    free = [np.float32(spine(100 + 60 * 19, y=300))]
    for k in range(10):                                                 # dipegang: goyang 15 px tiap frame
        clock.t += 0.1
        dx = 15 if k % 2 else -15
        polys = SHELF[:19] + [np.float32(spine(100 + 60 * 19 + dx, y=300 + dx))]
        ids = tr.update(polys, None, FS)
        if k == 0:
            decide(tr, ids, lambda i: MATCH if i == 19 else IGNORED)
        sess.update(clock.t, ids, polys, tr, FS, None, True, None)
    assert sess.phase != DITEMUKAN
    ids = run(sess, tr, SHELF[:19] + free, clock, 1.0)                 # ditaruh di rak dan diam
    st = tr.state(ids[19])
    if len(st.history) < 2:                                             # nomor sempat diragukan saat bergerak:
        st.history.extend([(MATCH, KEY), (MATCH, KEY)])                 # dibaca ulang setelah diam
    run(sess, tr, SHELF[:19] + free, clock, 0.5)
    assert sess.phase == DITEMUKAN


def test_neighbour_filling_the_gap_after_pick_is_taken_not_cancelled():
    # uji 30-09: Angiolite 2.5x29 diambil, Xperience Pro di sebelahnya miring mengisi celahnya
    sess, tr, clock, ids = _pick_ready()
    for k in range(1, 9):                                    # kotak AMBIL ditarik ke atas lalu hilang
        polys = [spine(100 + 60 * 4, y=300 - 50 * k) if i == 4 else p for i, p in enumerate(SHELF)]
        run(sess, tr, polys, clock, 0.1)
    for k in range(1, 7):                                    # kotak sebelah (#5, bukan target) bergeser ke celah
        polys = [p for i, p in enumerate(SHELF) if i not in (4, 5)] + [spine(100 + 60 * 5 - 10 * k)]
        run(sess, tr, polys, clock, 0.1)
    run(sess, tr, polys, clock, 1.5)
    assert sess.phase == SELESAI and [t.track for t in sess.taken] == [ids[4]]
    assert events(sess, "TERAMBIL") and not events(sess, "BATAL AMBIL")
    assert ids[5] in tr.tracks                               # kotak sebelah tetap ada (nomornya tidak dibuang)
    assert sess.watch is None and not sess.warning           # tidak ada peringatan "slot terisi lagi"


def test_box_held_in_hand_does_not_raise_shelf_count():
    sess, tr, clock, ids = started()
    assert sess.shelf_n == 20
    for k in range(30):                                      # kotak dipegang di samping rak, naik-turun 40 px
        clock.t += 0.1
        dy = 20 if k % 2 else -20
        polys = SHELF + [np.float32(spine(1500, y=150 + dy))]
        sess.update(clock.t, tr.update(polys, None, FS), polys, tr, FS, None, True, None)
    assert sess.shelf_n == 20
    run(sess, tr, SHELF + [spine(100 + 60 * 20)], clock, 2.0)  # kotak ke-21 diam di rak: jumlah naik
    assert sess.shelf_n == 21


def test_recheck_cannot_hang_when_box_flickers_in_and_out():
    # uji 30-09 14:04: slot berganti-ganti isi, batas waktu cek ulang di-reset terus -> macet sampai uji selesai
    sess, tr, clock, ids = _pick_ready()
    tr.state(ids[4]).mark_stale()
    rest = [p for i, p in enumerate(SHELF) if i != 4]
    for k in range(260):                                     # 26 s: terdeteksi / hilang bergantian, tak pernah terbaca
        run(sess, tr, SHELF if k % 2 else rest, clock, 0.1)
        if sess.phase != DITEMUKAN:
            break
    assert sess.phase != DITEMUKAN and clock.t < 2.5 + 21
    assert any("tidak terpastikan ulang dalam 20" in e[3] for e in events(sess, "BATAL AMBIL"))


def test_slot_number_is_taken_over_only_once():
    sess, tr, clock, ids = _pick_ready()
    tr.tracks.pop(ids[4])
    run(sess, tr, SHELF, clock, 0.5)
    first = sess.pick
    assert first not in (None, ids[4])
    tr.tracks.pop(first)                                     # nomornya berganti lagi
    run(sess, tr, SHELF, clock, 2.5)
    assert len([e for e in events(sess, "CEK ULANG") if "kini bernomor" in e[3]]) == 1
    assert sess.phase != DITEMUKAN and events(sess, "BATAL AMBIL")


def _moved_away():
    """Kotak AMBIL keluar dari slotnya dan diletakkan di ujung rak (nomor baru, belum terbaca)."""
    sess, tr, clock, ids = _pick_ready()
    moved = [p for i, p in enumerate(SHELF) if i != 4] + [spine(100 + 60 * 20)]
    new_ids = run(sess, tr, moved, clock, 3.0)
    return sess, tr, clock, ids, moved, new_ids[-1]


def test_box_moved_to_another_slot_is_not_counted_as_taken():
    # replay uji 30-09 14:09: kotak dipindah, bukan diambil -> dulu TERAMBIL dan pencarian berhenti
    sess, tr, clock, ids, moved, nid = _moved_away()
    assert sess.phase == DITEMUKAN and not events(sess, "TERAMBIL") and "dipindah" in sess.message
    tr.state(nid).history.extend([(MATCH, KEY), (MATCH, KEY)])  # kotak di tempat baru terbaca: kotak yang diminta
    run(sess, tr, moved, clock, 0.5)
    assert any("dipindah" in e[3] for e in events(sess, "BATAL AMBIL")) and not events(sess, "TERAMBIL")
    assert sess.phase == DITEMUKAN and sess.pick == nid        # dicari ulang: AMBIL di tempat barunya


def test_new_box_elsewhere_that_is_not_the_target_confirms_taken():
    sess, tr, clock, ids, moved, nid = _moved_away()
    tr.state(nid).history.extend([(IGNORED, None), (IGNORED, None)])
    run(sess, tr, moved, clock, 0.5)
    assert sess.phase == SELESAI and [t.track for t in sess.taken] == [ids[4]]


def test_unreadable_new_box_elsewhere_cancels_instead_of_guessing_taken():
    sess, tr, clock, ids, moved, nid = _moved_away()
    run(sess, tr, moved, clock, 15.0)
    assert not events(sess, "TERAMBIL") and any("dipindah?" in e[3] for e in events(sess, "BATAL AMBIL"))


def test_renumbered_or_edge_boxes_do_not_block_taken():
    # uji_01 30-09 malam: kotak benar-benar diambil, tapi benda yang nomornya terus berganti dan tidak pernah terbaca
    # (kotak kertas di samping rak, kotak di tepi layar) dianggap "kotak baru" -> TERAMBIL tertahan lalu BATAL
    sess, tr, clock, ids = _pick_ready()
    tr.tracks.pop(ids[9])                                    # kotak lama di posisinya dapat nomor baru, belum terbaca
    edge = spine(1900)                                       # benda baru terpotong di tepi kanan layar
    rest = [p for i, p in enumerate(SHELF) if i != 4] + [edge]
    run(sess, tr, rest, clock, 3.0)
    assert sess.phase == SELESAI and events(sess, "TERAMBIL") and not events(sess, "BATAL AMBIL")


def test_split_pieces_of_known_boxes_do_not_block_taken():
    # replay uji_01: lengan membelah kotak; potongan atas (nomor baru, belum terbaca) pusatnya jauh dari pusat kotak
    # aslinya -> dulu dianggap "kotak baru di rak" dan TERAMBIL tertahan
    sess, tr, clock, ids = _pick_ready()
    piece = spine(100 + 60 * 9, y=300, h=150)                # potongan atas kotak ke-10, di dalam rentangnya
    rest = [p for i, p in enumerate(SHELF) if i not in (4, 9)] + [piece, spine(100 + 60 * 9, y=520, h=180)]
    run(sess, tr, rest, clock, 3.0)
    assert sess.phase == SELESAI and events(sess, "TERAMBIL") and not events(sess, "BATAL AMBIL")


def test_taken_confirmation_is_not_cut_by_recheck_limit():
    # replay uji_01: cek ulang mulai (tangan meraih), kotak diambil beberapa detik kemudian, lalu slot dipastikan
    # kosong; batas cek ulang 20 s tidak boleh memotong tahap itu
    sess, tr, clock, ids = _pick_ready()
    tr.state(ids[4]).mark_stale()
    run(sess, tr, SHELF, clock, 8.0)                         # tangan di depan kotak, belum terbaca ulang
    moved = [p for i, p in enumerate(SHELF) if i != 4] + [spine(100 + 60 * 20)]   # kotak baru muncul di ujung rak
    run(sess, tr, moved, clock, 14.0)                        # 22 s sejak cek ulang: slot kosong, kotak baru belum terbaca
    assert sess.phase == DITEMUKAN and not events(sess, "BATAL AMBIL")
