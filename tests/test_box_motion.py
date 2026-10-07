"""pytest tests/test_box_motion.py -q  (gerakan kamera: tracker tetap memberi ID yang sama, estimasi geser benar)"""
import cv2
import numpy as np

from src.box_motion import MotionEstimator, consistent
from src.box_verifier import IGNORED, MATCH, READING, SimpleTracker


def spine(x, y=100, w=20, h=300):
    """Punggung kotak tegak sempit (seperti di rak): 4 titik."""
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def test_weak_match_of_box_number_needs_new_reading():
    t = SimpleTracker()
    a, = t.update([spine(100)])
    st = t.state(a)
    st.history.extend([(MATCH, "k"), (MATCH, "k")])
    assert t.update([spine(108)]) == [a]                              # IoU 0,43: nomor sama, tapi diragukan
    assert st.stale and st.stable(2) == READING and not st.history  # bacaan lama dibuang (bisa milik kotak lain)
    st.history.append((MATCH, "k"))                                   # bacaan baru (crop sesudah diragukan)
    st.stale = False
    assert st.stable(2) == READING                                    # perlu 2 bacaan baru yang sepakat
    st.history.append((MATCH, "k"))
    assert st.stable(2) == MATCH


def test_confident_tracking_keeps_match_while_camera_moves():
    t = SimpleTracker()
    a, = t.update([spine(100)])
    st = t.state(a)
    st.history.extend([(MATCH, "k"), (MATCH, "k")])
    x = 100
    for k in range(40):                                               # kamera geser terus, gerak terukur tepat
        dx = 7 if k % 3 else -5
        x += dx
        assert t.update([spine(x)], np.float32([[1, 0, dx], [0, 1, 0]])) == [a]
    assert not st.stale and st.stable(2) == MATCH


def test_slip_towards_neighbour_makes_both_boxes_reread():
    t = SimpleTracker()
    a, b = t.update([spine(100), spine(122)])
    for tid in (a, b):
        t.state(tid).history.extend([(IGNORED, None), (IGNORED, None)])
    # perkiraan gerak meleset 8 px: tiap nomor hanya separuh menempel ke kotaknya (IoU 0,43)
    ids = t.update([spine(100), spine(122)], np.float32([[1, 0, 8], [0, 1, 0]]))
    assert ids == [a, b]
    assert t.state(a).stable(2) == READING and t.state(b).stable(2) == READING   # bacaan lama tidak dipakai


def test_tracker_keeps_ids_when_camera_pans():
    boxes = [spine(100), spine(160), spine(220)]
    moved = [spine(100 + 45), spine(160 + 45), spine(220 + 45)]      # kamera geser 45 px: IoU tanpa kompensasi 0
    shift = np.array([[1, 0, 45], [0, 1, 0]], np.float32)

    t = SimpleTracker()
    first = t.update(boxes)
    t.state(first[1]).history.append(("MATCH", "k"))
    assert t.update(moved, shift) == first                            # ID sama, bacaan kotak #2 tidak hilang
    assert len(t.state(first[1]).history) == 1

    t2 = SimpleTracker()
    first2 = t2.update(boxes)
    assert set(t2.update(moved)).isdisjoint(first2)                   # tanpa kompensasi: semua dianggap kotak baru


def test_missed_track_follows_camera():
    t = SimpleTracker()
    a, b = t.update([spine(100), spine(300)])
    step = np.array([[1, 0, 30], [0, 1, 0]], np.float32)
    assert t.update([spine(130)], step) == [a]                        # kotak b sempat tidak terdeteksi ...
    assert t.update([spine(160), spine(360)], step) == [a, b]         # ... posisinya tetap ikut digeser


def test_truncated_detection_keeps_id_but_never_jumps_to_neighbour():
    t = SimpleTracker()
    a, b = t.update([spine(100), spine(122)])                        # dua punggung berdempetan
    assert t.update([spine(100, y=100, h=80), spine(122)]) == [a, b]  # a terlihat sebagian (IoU 0.27) -> tetap a
    t2 = SimpleTracker()
    a2, b2 = t2.update([spine(100), spine(122)])
    merged = [[100, 100], [142, 100], [142, 400], [100, 400]]          # satu deteksi menutupi dua kotak
    assert t2.update([merged])[0] not in (a2, b2)                     # ambigu -> kotak baru, tidak mewarisi bacaan


def test_duplicate_tracks_do_not_cascade_into_new_ids():
    t = SimpleTracker()
    a, = t.update([spine(100)])
    t.tracks[99] = {"poly": spine(101), "missed": 3, "state": t.state(a).__class__()}   # track lama di tempat sama
    seen = set()
    for _ in range(5):
        seen.update(t.update([spine(100)]))
    assert seen == {a}                                                 # tetap satu ID, tidak membuat ID baru terus


def test_motion_estimator_measures_shift():
    rng = np.random.default_rng(1)
    base = cv2.GaussianBlur((rng.random((1080, 1920)) * 255).astype(np.uint8), (0, 0), 2)
    base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    dx, dy = 24, -12
    moved = cv2.warpAffine(base, np.float32([[1, 0, dx], [0, 1, dy]]), (1920, 1080), borderMode=cv2.BORDER_REFLECT)
    me = MotionEstimator()
    assert me.update(base) == (None, 0.0)                             # frame pertama: belum ada pembanding
    M, shift = me.update(moved)
    assert M is not None
    assert abs(M[0, 2] - dx) < 2 and abs(M[1, 2] - dy) < 2            # translasi dalam piksel frame asli
    assert abs(shift - np.hypot(dx, dy)) < 3
    M2, still = me.update(moved)
    assert still < 1                                                   # kamera diam


def test_motion_estimator_flat_image_is_unknown():
    me = MotionEstimator()
    flat = np.full((720, 1280, 3), 128, np.uint8)
    me.update(flat)
    M, shift = me.update(flat)
    assert M is None and shift == float("inf")                        # tidak ada tekstur: anggap tidak diam


SHELF = [spine(100 + 30 * i) for i in range(12)]                      # 12 punggung berjejer, jarak 30 px


def test_wrong_motion_while_boxes_stay_put_is_rejected():
    # kamera diam, tangan di depan kamera membuat optical flow "geser 15 px" (setengah jarak antar kotak)
    wrong = np.float32([[1, 0, 15], [0, 1, 0]])
    assert not consistent(SHELF, SHELF, wrong)
    moved = [np.float32(p) + (15, 0) for p in SHELF]
    assert consistent(SHELF, moved, wrong)                                     # kamera benar-benar bergeser
    assert consistent(SHELF, SHELF, np.float32([[1, 0, 3], [0, 1, 0]]))       # geser kecil: tidak diputuskan
    assert not consistent(SHELF, SHELF[:5], wrong)                             # separuh rak tertutup tangan
    assert consistent(SHELF[:3], SHELF, wrong)                                 # terlalu sedikit kotak: tidak dinilai


def test_rejected_motion_keeps_reference_frame():
    rng = np.random.default_rng(2)
    base = cv2.cvtColor(cv2.GaussianBlur((rng.random((720, 1280)) * 255).astype(np.uint8), (0, 0), 2),
                        cv2.COLOR_GRAY2BGR)
    moved = cv2.warpAffine(base, np.float32([[1, 0, 20], [0, 1, 0]]), (1280, 720), borderMode=cv2.BORDER_REFLECT)
    me = MotionEstimator()
    me.update(base)
    me.update(moved)
    me.reject()                                                        # gerakan ke `moved` dianggap tidak terukur
    assert me.fails == 1
    M, _ = me.update(moved)                                            # diukur lagi dari frame acuan lama
    assert M is not None and abs(M[0, 2] - 20) < 2


def test_tracker_reports_why_a_box_number_is_doubted():
    t = SimpleTracker()
    a, = t.update([spine(100)])
    t.state(a).history.append((MATCH, "k"))
    t.update([spine(108)])
    assert t.doubted and t.doubted[0][0] == a and "IoU" in t.doubted[0][1]
    t.update([spine(108)])
    assert t.doubted == []


def rotated(cx, cy, deg, w=45, h=640):
    """Punggung kotak w x h berpusat (cx, cy), miring `deg` derajat."""
    return cv2.boxPoints(((cx, cy), (w, h), deg)).astype(np.float32)


def test_angle_jitter_of_still_box_keeps_number_and_readings():
    # uji tripod 30-09: kotak miring, sudut deteksinya berubah -> IoU 0,3-0,47 walau kamera geser < 1,5 px
    from src.box_geometry import poly_iou
    t = SimpleTracker()
    for _ in range(6):                                                     # kamera tripod: tenang
        a, b = t.update([rotated(500, 400, 0), rotated(560, 400, 0)], np.float32([[1, 0, 0.5], [0, 1, 0]]))
    for tid in (a, b):
        t.state(tid).history.extend([(MATCH, "k"), (MATCH, "k")])
    tilted = [rotated(500, 400, 7), rotated(560, 400, -6)]
    assert poly_iou(tilted[0], rotated(500, 400, 0)) < 0.5                  # IoU saja: diragukan
    assert t.update(tilted, np.float32([[1, 0, 0.5], [0, 1, 0]])) == [a, b]
    assert not t.state(a).stale and not t.state(b).stale and t.doubted == []
    assert t.update([rotated(500, 400, 12), rotated(560, 400, 0)]) == [a, b]  # sudut berubah besar: IoU < 0,3


def test_same_column_rule_not_used_when_camera_moves_a_lot_or_ambiguous():
    t = SimpleTracker()
    for _ in range(6):
        a, = t.update([rotated(500, 400, 0)])
    t.state(a).history.append((MATCH, "k"))
    # kamera bergeser 40 px (terukur), sudut deteksi berubah: aturan IoU biasa -> diragukan
    t.update([rotated(540, 400, 7)], np.float32([[1, 0, 40], [0, 1, 0]]))
    assert t.state(a).stale
    # baru saja bergerak: kamera belum tenang CALM_FRAMES frame -> aturan sekolom belum dipakai
    t.update([rotated(540, 400, 0)])
    t.state(a).stale = False
    t.state(a).history.append((MATCH, "k"))
    t.update([rotated(540, 400, 7)])
    assert t.state(a).stale
    # perkiraan posisi meleset bersama (gerak kamera salah ukur): pasangan pasti ikut bergeser -> tidak dipakai
    t3 = SimpleTracker()
    row = [rotated(300 + 60 * k, 400, 0) for k in range(6)]
    for _ in range(6):
        ids3 = t3.update(row)
    for tid in ids3:
        t3.state(tid).history.append((MATCH, "k"))
    shifted = [rotated(300 + 60 * k + 13, 400, 0) for k in range(5)] + [rotated(300 + 60 * 5 + 13, 400, 7)]
    t3.update(shifted)
    assert t3.state(ids3[5]).stale
    t2 = SimpleTracker()
    a2, = t2.update([rotated(500, 400, 0)])
    t2.state(a2).history.append((MATCH, "k"))
    # dua deteksi berdempetan di tempat kotak itu (ganda / terbelah): ambigu -> tidak dipakai
    ids = t2.update([rotated(496, 400, 5, w=30), rotated(522, 400, 5, w=30)])
    assert a2 not in ids or t2.state(a2).stale


def test_arm_covering_top_of_boxes_keeps_numbers_and_readings():
    # uji 30-09 14:09: lengan menutupi bagian atas 5-11 kotak -> deteksi x0,4 lalu x2,0 panjangnya, bacaan dibuang
    t = SimpleTracker()
    row = [rotated(300 + 60 * k, 400, 0) for k in range(6)]
    for _ in range(6):
        ids = t.update(row)
    for tid in ids:
        t.state(tid).history.append((MATCH, "k"))
    cut = [cv2.boxPoints(((300 + 60 * k, 400 + 192), (45, 256), 0)).astype(np.float32) for k in range(6)]
    assert t.update(cut) == ids and t.doubted == []          # hanya 40 % bagian bawah yang terlihat
    assert t.update(row) == ids and t.doubted == []          # terlihat utuh lagi
    assert all(t.state(tid).history and not t.state(tid).stale for tid in ids)


def test_arm_splitting_boxes_in_two_keeps_numbers_and_readings():
    # uji_01 30-09 malam: lengan melintang di depan rak membelah deteksi tiap kotak jadi potongan atas dan bawah
    t = SimpleTracker()
    row = [rotated(300 + 60 * k, 400, 0) for k in range(6)]            # y 80..720
    for _ in range(6):
        ids = t.update(row)
    for tid in ids:
        t.state(tid).history.append((MATCH, "k"))
    top = [cv2.boxPoints(((300 + 60 * k, 205), (45, 250), 0)).astype(np.float32) for k in range(6)]
    bottom = [cv2.boxPoints(((300 + 60 * k, 595), (45, 250), 0)).astype(np.float32) for k in range(6)]
    split = t.update(top + bottom)
    assert all(tid in split for tid in ids) and t.doubted == []          # nomor lama ikut salah satu potongan
    assert t.update(row) == ids and t.doubted == []                       # menyatu lagi: nomor lama, bukan baru
    assert all(t.state(tid).history and not t.state(tid).stale for tid in ids)
    # dua kotak BERSEBELAHAN dalam satu deteksi tetap kotak baru (tidak mewarisi bacaan)
    t2 = SimpleTracker()
    a2, b2 = t2.update([spine(100), spine(122)])
    merged = [[100, 100], [142, 100], [142, 400], [100, 400]]
    assert t2.update([merged])[0] not in (a2, b2)
