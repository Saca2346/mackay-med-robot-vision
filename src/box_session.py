"""
Alur penyeleksi SATU permintaan di rak (Tahap 2), di atas tracker + keputusan per kotak (src/box_verifier.py).

  1 CEK RAK     kamera diarahkan ke SELURUH rak dan ditahan diam; OCR belum jalan.
                0 kotak terus-menerus `empty_s` detik        -> RAK KOSONG (merah), pencarian tidak dilanjutkan
                kotak terlihat stabil `count_s` detik        -> jumlah kotak di rak = median hitungan -> MENCARI
  2 MENCARI     setiap kotak dibaca (teks ukuran + kode REF + barcode) dan diputuskan per kotak:
                ada kotak MATCH stabil yang terlihat         -> DITEMUKAN: hijau tebal "AMBIL", dikunci ke kotak itu
                SEMUA kotak yang TERLIHAT SEKARANG pasti bukan target (abu-abu) dan jumlahnya >= jumlah kotak di
                rak                                          -> TIDAK ADA (merah)
                (kotak di luar layar tidak ikut dihitung: posisinya hanya perkiraan dan bisa bergeser)
                semua sudah dibaca, tapi ada yang ragu (kuning) / baru satu bukti (biru) / belum semua kotak
                terlihat                                     -> PERLU KONFIRMASI: TIDAK menyimpulkan "tidak ada"
  3 DITEMUKAN   kotak yang dikunci diawasi. Kotak itu hilang dari tempatnya dan slotnya tetap kosong `gone_s`
                detik -> TERAMBIL: slot kosong terverifikasi, nomor kotak itu dibuang (tidak bisa menempel ke
                kotak lain). "Kosong" = slot terlihat utuh, kamera diam, kotak lain tetap terdeteksi, TIDAK ada
                deteksi di slot, DAN gambar slot berbeda dari gambar kotak itu (detektor yang sesaat gagal melihat
                kotak tidak dianggap terambil) DAN gambar slot diam (tangan yang lewat / menutupi tidak dihitung).
                masih ada permintaan yang sama (qty / request_again) -> MENCARI lagi TANPA kotak itu
                tidak ada lagi                               -> SELESAI
                slot yang baru kosong terisi lagi dalam `recheck_s` detik -> peringatan (kotak belum terambil?)
                nomor kotak diragukan (tangan meraih / menutupi), nomor lain diam di slotnya, atau tampilannya
                berubah -> CEK ULANG: tetap dikunci tapi JANGAN diambil; bacaan baru yang menunjuk tepat kotak yang
                diminta -> AMBIL DIPASTIKAN; bantahan / tidak terpastikan `relock_s` detik (selama kotak di slot)
                -> BATAL AMBIL. Kotak yang ditarik keluar tetap diawasi sampai TERAMBIL.
Kotak dengan produk/merek sama tapi ukuran beda tidak pernah MATCH: decide_multi mewajibkan produk DAN diameter
DAN panjang sama persis dari >= 2 bukti independen, dan satu bukti yang bertentangan = bukan target / ragu.
"""
from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from src.box_geometry import frac_inside, inside_fraction, poly_iou, to_poly
from src.box_verifier import (CANDIDATE, COMBINE_WINDOW, CONFIRM, EMPTY, EXPIRED, IGNORED, MATCH, READING,
                              _shift, combined_score, pick_fefo)

CEK_RAK, RAK_KOSONG, MENCARI = "CEK RAK", "RAK KOSONG", "MENCARI"
DITEMUKAN, PERLU_KONFIRMASI, TIDAK_ADA, SELESAI = "DITEMUKAN", "PERLU KONFIRMASI", "TIDAK ADA", "SELESAI"
PHASE_COLOR = {CEK_RAK: "white", RAK_KOSONG: "red", MENCARI: "white", DITEMUKAN: "green",
               PERLU_KONFIRMASI: "yellow", TIDAK_ADA: "red", SELESAI: "green"}
DONE = (IGNORED, EXPIRED)             # pasti tidak diambil
SAME_BOX_NCC = 0.5                    # gambar slot masih semirip ini dengan kotak AMBIL -> kotak masih di sana
STILL_SLOT_NCC = 0.85                 # dua gambar slot berturut-turut semirip ini = isi slot diam
PATCH = (32, 96)                      # (lebar, tinggi) gambar slot yang diluruskan untuk dibandingkan
RECENT_OUT = 25                       # frame: kotak yang baru keluar layar masih bisa ditunjuk arahnya
STILL_MIN = 6                         # frame berturut-turut diam terhadap rak sebelum boleh jadi AMBIL
MAX_OVERLAP = 0.25                    # kotak yang menutupi kotak lain > 25 % = di DEPAN rak (dipegang), bukan di rak
RELOCK_S = 10.0                       # kunci yang dipastikan ulang: batal kalau tidak terpastikan selama kotak di slot
RECHECK_MAX_S = 20.0                  # ... dan paling lama sekian detik total sejak mulai dipastikan ulang
HOLD_S = 15.0                         # kotak baru / berpindah di rak belum terbaca selama ini -> terambil tidak pasti
LOOK_OFF_FRAMES = 5                   # frame diam berturut-turut dengan tampilan slot berbeda -> kotak dipastikan ulang


@dataclass
class Taken:
    track: int
    time: float
    slot: np.ndarray
    gtin: str = ""
    lot: str = ""
    expiry: object = None


def warp(poly, motion):
    p = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if motion is None:
        return p
    A = np.asarray(motion, dtype=np.float32)
    return p @ A[:, :2].T + A[:, 2]


def slot_patch(frame, poly):
    """Gambar slot diluruskan ke ukuran tetap, abu-abu, dinormalkan (rata-rata 0, simpangan 1). None kalau datar."""
    w, h = PATCH
    src = to_poly(poly).astype(np.float32)
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    g = cv2.warpPerspective(frame, cv2.getPerspectiveTransform(src, dst), (w, h))
    g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sd = float(g.std())
    return None if sd < 2.0 else (g - g.mean()) / sd


def cover(slot, poly) -> float:
    """Bagian luas SLOT yang tertutup poligon lain, 0..1 (kotak sebelah yang condong sedikit ke celah < 0,6)."""
    a = cv2.convexHull(to_poly(slot)).reshape(-1, 2)
    b = cv2.convexHull(to_poly(poly)).reshape(-1, 2)
    area = float(abs(cv2.contourArea(a)))
    inter, _ = cv2.intersectConvexConvex(a, b)
    return float(max(inter, 0.0) / area) if area > 0 else 0.0


def shelf_shift(tracker, ids, skip=None):
    """Koreksi posisi dari kotak-kotak lain yang terlihat (median selisih deteksi - perkiraan), jadi slot ikut
    rak dan TIDAK ikut kotak yang sedang ditarik keluar."""
    res = [r for t, r in getattr(tracker, "residual", {}).items() if t in ids and t != skip]
    return np.median(np.asarray(res), axis=0) if len(res) >= 3 else np.zeros(2, np.float32)


def moving_on_shelf(tracker, tid, poly, shelf) -> bool:
    """Kotak ini bergerak TERHADAP RAK (ditarik / dipegang), bukan karena kamera? Nomor baru (belum ada selisih
    posisi) dianggap bergerak sampai terbukti diam."""
    r = getattr(tracker, "residual", {}).get(tid)
    if r is None:
        return True
    q = to_poly(poly)
    sides = [float(np.linalg.norm(q[(k + 1) % 4] - q[k])) for k in range(4)]
    return float(np.hypot(*(np.asarray(r) - shelf))) > max(0.08 * max(sides), 0.3 * min(sides))


def on_shelf(tracker, ids, polys, tid) -> tuple[bool, str]:
    """Kotak ini berdiri DI RAK (boleh jadi AMBIL)? Kotak yang dipegang tangan bergerak terhadap rak dan / atau
    menutupi kotak lain di belakangnya (terukur di video uji: kotak yang sudah diambil lalu dipegang di depan rak
    sempat dipilih lagi)."""
    t = tracker.tracks.get(tid, {})
    if t.get("still", 0) < STILL_MIN:
        return False, "bergerak / dipegang"
    p = polys[ids.index(tid)]
    for j, q in enumerate(polys):
        if ids[j] != tid and (frac_inside(q, p) > MAX_OVERLAP or frac_inside(p, q) > MAX_OVERLAP):
            return False, "menutupi kotak lain (di depan rak?)"
    return True, ""


def ncc(a, b) -> float:
    return float((a * b).mean()) if a is not None and b is not None else 0.0


def _direction(poly, frame_size) -> str:
    w, h = frame_size
    cx, cy = to_poly(poly).mean(axis=0)
    dx = "kiri" if cx < 0 else "kanan" if cx > w else ""
    dy = "atas" if cy < 0 else "bawah" if cy > h else ""
    return " ".join(d for d in (dx, dy) if d)


def pick_status(st) -> tuple[str, str]:
    """Kotak AMBIL yang sudah dikunci: "solid" (bacaan berisi terakhir MATCH -> boleh diambil), "tahan" (satu bacaan
    berisi terakhir lebih lemah tapi TIDAK bertentangan, mis. label tertutup tangan -> tunggu, jangan diambil dulu),
    atau "batal". Bertentangan (bukan target / kedaluwarsa / bukti bertentangan), nomor kotak diragukan, 2 bacaan
    berisi terakhir tidak ada yang MATCH, atau 3x berturut-turut tidak terbaca -> batal."""
    if st is None:
        return "batal", "nomor kotak hilang"
    if st.stale:
        return "batal", "nomor kotak diragukan - dibaca ulang"
    if len(st.support) == len(st.history) and st.history:
        return _pick_status_evidence(st)
    h = list(st.history)
    info = [(d, k) for d, k in h if k != EMPTY]
    trailing = 0
    for _, k in reversed(h):
        if k != EMPTY:
            break
        trailing += 1
    for d, k in info[-2:]:
        if d in (IGNORED, EXPIRED) or (d == CONFIRM and k != EMPTY):
            return "batal", f"verifikasi berubah: {d}"
    if trailing >= 3:
        return "batal", "tidak terbaca berulang kali"
    combined = st.combined_match(2)                 # bukti gabungan beberapa bacaan (TrackState.combined_match)
    if not (combined or any(d == MATCH for d, _ in info[-2:])):
        return "batal", "bacaan tidak lagi cocok"
    if trailing:
        return "tahan", "bacaan terakhir kosong - label tertutup / slot kosong?"
    solid = info and (info[-1][0] == MATCH or (combined and info[-1][0] == CANDIDATE))
    return ("solid", "") if solid else ("tahan", "bacaan terakhir kurang lengkap")


def _pick_status_evidence(st) -> tuple[str, str]:
    """Kunci AMBIL yang STABIL (bukti per sumber tiap bacaan tersedia, lihat TrackState.support). Kotak yang sudah
    terverifikasi dengan >= 2 bukti tetap dikunci selama tidak ada yang membantah: bacaan yang hanya sebagian
    ("angiolite" saja, REF saja) atau kosong BUKAN bantahan dan tidak lagi membatalkan kunci (dulu: hijau-kuning
    berkedip). Satu bacaan bertentangan (mis. "2.5 29" terbaca "125129") -> tahan (kuning, JANGAN diambil) dan dibaca
    ulang; kembali solid kalau sesudahnya >= 2 bacaan cocok dengan skor gabungan >= 2. Dua bacaan bertentangan di
    jendela bacaan terakhir, kedaluwarsa, nomor diragukan, atau 3x tidak terbaca -> batal."""
    rows = [(d, k, s) for (d, k), s in zip(st.history, st.support) if k != EMPTY]
    trailing = 0
    for _, k in reversed(st.history):
        if k != EMPTY:
            break
        trailing += 1
    if any(d == EXPIRED for d, _, _ in rows):
        return "batal", "kedaluwarsa"
    if trailing >= 3:
        return "batal", "tidak terbaca berulang kali"
    last = rows[-COMBINE_WINDOW:]
    bad = [i for i, (d, _, (_, c)) in enumerate(last) if c or d in (IGNORED, EXPIRED)]
    if len(bad) >= 2:
        return "batal", f"{len(bad)} bacaan bertentangan"
    if len(bad) == 1:
        agree = [a for d, _, (a, c) in last[bad[0] + 1:] if a and not c]
        if len(agree) >= 2 and combined_score(frozenset().union(*agree))[0] >= 2:
            return "solid", ""                            # bantahan tunggal kalah oleh bacaan sesudahnya
        return "tahan", "satu bacaan bertentangan - dibaca ulang"
    if trailing >= 2:
        return "tahan", "bacaan terakhir kosong - label tertutup / slot kosong?"
    if not any(a for _, _, (a, _) in last) and not any(d == MATCH for d, _, _ in last):
        return "tahan", "label belum terbaca lagi"
    return "solid", ""


def relock_status(rows) -> tuple[str, str]:
    """Kunci AMBIL yang sedang DIPASTIKAN ULANG: nomor kotak sempat diragukan (tangan meraih / menutupi kotak,
    kotak tersenggol), slotnya kini ditempati nomor lain, atau tampilan kotak di slot berubah. Hanya bacaan BARU
    yang dipakai: `rows` = [(keputusan, (atom, bantahan) atau None)], bacaan kosong sudah dibuang.
    Kotak di slot itu sudah terverifikasi >= 2 bukti; yang diragukan hanya "masih kotak yang sama?". Maka cukup
    bacaan baru yang menunjuk TEPAT produk + ukuran yang diminta (teks lengkap, REF, barcode atau GTIN; potongan
    beberapa bacaan boleh digabung) tanpa bantahan -> solid. Kotak saudara (mis. angiolite 4 x 19 di sebelahnya)
    selalu membantah lewat ukurannya, dan bacaan "angiolite" saja tidak pernah cukup. Satu bantahan -> bacaan
    sesudahnya harus berskor >= 2; dua bantahan / kedaluwarsa -> batal; selain itu -> cek (tunggu, JANGAN diambil)."""
    if any(d == EXPIRED for d, _ in rows):
        return "batal", "kedaluwarsa"
    bad = [i for i, (d, s) in enumerate(rows) if d == IGNORED or (s[1] if s is not None else d == CONFIRM)]
    if len(bad) >= 2:
        return "batal", f"{len(bad)} bacaan ulang bertentangan"
    after = rows[bad[0] + 1:] if bad else rows
    if any(d == MATCH for d, _ in after):
        return "solid", ""
    parts = [s[0] for _, s in after if s is not None and s[0]]
    if parts:
        score, gtins = combined_score(frozenset().union(*parts))
        if score >= (2 if bad else 1) and len(gtins) <= 1:
            return "solid", ""
    return "cek", "satu bacaan ulang bertentangan - dibaca lagi" if bad else "menunggu bacaan ulang yang lengkap"


class SelectionSession:
    def __init__(self, req, need: int = 2, qty: int = 1, cfg: dict | None = None, expected: int = 0):
        c = cfg or {}
        self.req, self.need, self.remaining = req, need, qty
        self.count_s = float(c.get("count_s", 1.5))
        self.empty_s = float(c.get("empty_s", 2.0))
        self.gone_s = float(c.get("gone_s", 2.0))
        self.recheck_s = float(c.get("recheck_s", 5.0))
        self.relock_s = float(c.get("relock_s", RELOCK_S))
        self.recheck_max_s = float(c.get("recheck_max_s", RECHECK_MAX_S))
        self.hold_s = float(c.get("hold_s", HOLD_S))
        self.expected = int(expected or c.get("expected_count", 0) or 0)
        self.phase = CEK_RAK
        self.samples: deque = deque()          # (waktu, jumlah kotak terlihat) pada frame diam
        self.zero_since = None
        self._ready = 0
        self.shelf_n = 0                       # jumlah kotak di rak sebelum ada yang diambil
        self.pick = None                       # nomor kotak AMBIL (dikunci)
        self.slot = None                       # poligon tempat kotak AMBIL, ikut digeser gerakan kamera
        self.box_patch = None                  # gambar kotak AMBIL saat terakhir terlihat (kamera diam)
        self.prev_patch = None                 # gambar slot pada frame sebelumnya (isi slot diam?)
        self.gone_since = self.occupied_since = None
        self.pulling = False                   # kotak AMBIL sudah terlihat ditarik dari slotnya
        self.solid = False                     # kotak AMBIL terverifikasi penuh saat ini (boleh diambil robot)
        self.pick_epoch = 0                    # epoch TrackState kotak AMBIL saat dikunci (berubah = nomor diragukan)
        self.recheck_since = None              # kunci sedang dipastikan ulang sejak (hanya waktu kotak di slotnya)
        self.recheck_t0 = None                 # ... mulai kapan, TIDAK pernah digeser (batas total recheck_max_s)
        self.hold_since = None                 # kotak tak di slot, menunggu kotak baru di rak terbaca (bukan dipindah?)
        self.recheck_from = None               # None: semua bacaan kotak itu baru; angka: hanya bacaan ke->= angka ini
        self.look_off = 0                      # frame diam berturut-turut dengan tampilan slot berbeda dari kotaknya
        self.adopted = None                    # (nomor, TrackState, ada gerakan?) kotak AMBIL asli saat slotnya diambil
                                               # alih nomor lain
        self.neigh: dict = {}                  # pusat kotak-kotak lain saat AMBIL dikunci (ikut gerak kamera)
        self._shelf = np.zeros(2, np.float32)
        self.taken: list[Taken] = []
        self.watch = None                      # (slot, waktu diambil, nomor, gambar kotak) slot yang masih diawasi
        self._warned = None
        self.events: list[tuple] = []          # (waktu, peristiwa, nomor, alasan, TrackState|None)
        self.message, self.detail, self.warning = "CEK RAK", "", ""
        self.stats: dict = {}

    # ------------------------------------------------------------------ luar
    @property
    def ocr_allowed(self) -> bool:
        return self.phase not in (CEK_RAK, RAK_KOSONG)

    @property
    def color(self) -> str:
        return PHASE_COLOR[self.phase]

    def request_again(self, n: int = 1, now: float = 0.0) -> None:
        """Permintaan yang sama sekali lagi. Kotak yang sedang AMBIL harus terverifikasi terambil dulu."""
        self.remaining += n
        self._event(now, "PERMINTAAN LAGI", None, f"sisa {self.remaining}")
        if self.phase in (SELESAI, TIDAK_ADA, PERLU_KONFIRMASI):
            self.phase = MENCARI

    def lost(self, now: float) -> None:
        """Posisi kamera hilang terlalu lama (semua nomor kotak dilupakan): slot AMBIL tidak bisa diikuti lagi."""
        if self.phase == DITEMUKAN:
            self._event(now, "BATAL AMBIL", self.pick, "posisi kamera hilang - diperiksa ulang")
            self._unlock()
        self.watch = None

    def drain_events(self) -> list[tuple]:
        ev, self.events = self.events, []
        return ev

    # ------------------------------------------------------------------ per frame
    def update(self, now: float, ids: list[int], polys: list, tracker, frame_size, motion=None,
               still: bool = True, frame=None) -> None:
        """ids/polys = kotak yang terdeteksi di frame ini (sudah lewat tracker.update dengan `motion` yang sama)."""
        self.warning = ""
        fix = self._shelf = shelf_shift(tracker, ids, skip=self.pick)
        if self.slot is not None:
            self.slot = warp(self.slot, motion) + fix
        if self.watch is not None:
            self.watch = (warp(self.watch[0], motion) + fix, *self.watch[1:])
        if self.neigh:
            self.neigh = {t: warp(q, motion) + fix for t, q in self.neigh.items()}
        self._count(now, len(ids), still,
                    sum(1 for t, p in zip(ids, polys) if not moving_on_shelf(tracker, t, p, fix)))
        if self.phase in (CEK_RAK, RAK_KOSONG):
            self._check_shelf(now)
        if self.phase == DITEMUKAN:
            self._follow_pick(now, ids, polys, tracker, frame_size, still, frame)
        if self.phase in (MENCARI, PERLU_KONFIRMASI, TIDAK_ADA):
            self._search(now, ids, polys, tracker, frame_size, still, frame)
        self._check_watch(now, polys, frame_size, frame)
        if self.phase == SELESAI:
            self.message = f"SELESAI: {len(self.taken)} kotak {self.req} sudah diambil, slot kosong terverifikasi"
            self.detail = "tekan r kalau ada permintaan yang sama lagi"

    # ------------------------------------------------------------------ 1 cek rak + hitung
    def _count(self, now, n, still, n_rest=None):
        """n = kotak terdeteksi; n_rest = di antaranya yang diam terhadap rak (untuk menaikkan jumlah kotak)."""
        if n > 0:
            self.zero_since = None
        elif still and self.zero_since is None:
            self.zero_since = now
        if still:
            self.samples.append((now, n, n if n_rest is None else n_rest))
        while self.samples and self.samples[0][0] < now - max(self.count_s, self.empty_s) - 1.0:
            self.samples.popleft()
        win = [(t, k, r) for t, k, r in self.samples if t >= now - self.count_s]
        ready = len(win) >= 5 and now - win[0][0] >= 0.8 * self.count_s and min(k for _, k, _ in win) > 0
        self._ready = int(statistics.median(k for _, k, _ in win)) if ready else 0
        self.stats["visible"] = n
        if self._ready and self.phase not in (CEK_RAK, RAK_KOSONG):
            # rak terlihat lebih lengkap dari saat dihitung -> jumlah kotak di rak dinaikkan (lebih ketat). Hanya kotak
            # yang DIAM di rak: kotak yang sedang dipegang / ditarik tidak menambah jumlah (uji 30-09: 20 kotak di
            # rak tercatat 22 -> akhirnya "baru 19 dari 22 terlihat")
            rest = int(statistics.median(r for _, _, r in win))
            self.shelf_n = max(self.shelf_n, rest + len(self.taken))

    def _check_shelf(self, now):
        if self._ready:
            self.shelf_n = max(self.shelf_n, self._ready)
            self._event(now, "JUMLAH KOTAK", None, f"{self.shelf_n} kotak di rak")
            self.phase = MENCARI
            return
        if self.zero_since is not None and now - self.zero_since >= self.empty_s:
            if self.phase != RAK_KOSONG:
                self._event(now, RAK_KOSONG, None, "tidak ada kotak di rak")
            self.phase = RAK_KOSONG
            self.message = "RAK KOSONG: tidak ada kotak - pencarian tidak dilanjutkan"
            self.detail = "arahkan kamera ke rak berisi kotak dan tahan diam"
            return
        self.message = f"CEK RAK: menghitung kotak ({self.stats.get('visible', 0)} terlihat)"
        self.detail = "arahkan kamera ke SELURUH rak dan tahan diam 2 detik"

    # ------------------------------------------------------------------ 2 mencari
    def _search(self, now, ids, polys, tracker, frame_size, still, frame):
        if self.remaining <= 0:
            self.phase = SELESAI
            return
        tracks = tracker.tracks
        states = {tid: t["state"] for tid, t in tracks.items()}
        need = self.need
        match_all = [tid for tid in ids if tid in states and states[tid].stable(need) == MATCH]
        shelf_ok = {tid: on_shelf(tracker, ids, polys, tid) for tid in match_all}
        match_vis = {tid: states[tid] for tid in match_all if shelf_ok[tid][0]}
        if match_all and not match_vis:
            tid = match_all[0]
            self.phase = MENCARI
            self.message = f"MENCARI {self.req}: kotak cocok #{tid} tidak di rak ({shelf_ok[tid][1]})"
            self.detail = "hanya kotak yang diam di rak yang dipilih - taruh kotak di rak / tahan kamera diam"
            return
        if match_vis:
            self.pick = pick_fefo(match_vis, need)
            self.slot = to_poly(polys[ids.index(self.pick)]).copy()
            self.box_patch = self.prev_patch = None
            self.gone_since = self.occupied_since = None
            self.pulling, self.solid = False, True
            st = states[self.pick]
            self.pick_epoch, self.recheck_since, self.recheck_from, self.look_off = st.epoch, None, None, 0
            self.recheck_t0 = self.hold_since = None
            self.adopted = None
            self.neigh = {t: to_poly(p).copy() for t, p in zip(ids, polys) if t != self.pick}
            self._event(now, "AMBIL", self.pick, st.last_reason, st)
            self.phase = DITEMUKAN
            self._follow_pick(now, ids, polys, tracker, frame_size, still, frame)
            return

        # Keputusan hanya dari kotak yang TERLIHAT sekarang. Posisi kotak di luar layar hanya perkiraan (bisa bergeser
        # saat sudut kamera berubah banyak), jadi tidak pernah dipakai untuk menyimpulkan "tidak ada".
        kinds = {k: [] for k in ("done", "reading", "confirm", "candidate")}
        for tid in ids:
            st = states[tid]
            s = st.stable(need)
            kinds["done" if s in DONE else "reading" if s == READING or st.pending else
                  "candidate" if s == CANDIDATE else "confirm"].append(tid)
        # kotak cocok yang BARU SAJA keluar layar (posisinya masih bisa dipercaya): petunjuk arah saja
        match_out = [tid for tid, t in tracks.items() if tid not in ids and t.get("was_out")
                     and t.get("out", 0) <= RECENT_OUT and states[tid].history
                     and states[tid].history[-1][0] == MATCH]
        n_taken = len(self.taken)
        need_n = max(self.shelf_n - n_taken, 0)
        shortfall = bool(self.expected) and self.shelf_n < self.expected
        if self.expected:
            need_n = max(need_n, self.expected - n_taken)
        n_done, n_vis = len(kinds["done"]), len(ids)
        n_exp = sum(1 for t in kinds["done"] if states[t].stable(need) == EXPIRED)
        self.stats.update(visible=n_vis, done=n_done, need=need_n, reading=len(kinds["reading"]),
                          confirm=len(kinds["confirm"]), candidate=len(kinds["candidate"]))

        def tags(tids, limit=5):
            out = []
            for t in tids[:limit]:
                d = _direction(tracks[t]["poly"], frame_size) if frame_size else ""
                out.append(f"#{t}" + (f" ({d})" if d else ""))
            return " ".join(out) + (" ..." if len(tids) > limit else "")

        req = str(self.req)
        if n_vis and n_done == n_vis and n_vis >= need_n and not shortfall:
            if self.phase != TIDAK_ADA:
                why = f"{n_done} kotak terlihat, semua bukan target" + (f" ({n_exp} kedaluwarsa)" if n_exp else "")
                if n_taken:
                    why += f"; {n_taken} sudah diambil"
                self._event(now, TIDAK_ADA, None, why)
            self.phase = TIDAK_ADA
            self.message = (f"TIDAK ADA: {req} tidak ada " + ("lagi " if n_taken else "")
                            + f"di rak ({n_done} kotak diperiksa, semua bukan target)")
            self.detail = (f"{n_exp} kotak kedaluwarsa (ungu) tidak diambil" if n_exp else
                           "kalau ada kotak baru di rak, arahkan kamera ke sana")
        elif match_out:
            self.phase = MENCARI
            self.message = f"MENCARI {req}: kotak cocok {tags(match_out)} baru saja keluar layar"
            self.detail = "arahkan kamera ke kotak itu dan tahan diam (dibaca ulang sebelum AMBIL)"
        elif n_vis and not kinds["reading"]:
            self.phase = PERLU_KONFIRMASI
            parts = []
            if kinds["candidate"]:
                parts.append(f"{tags(kinds['candidate'])} baru 1 bukti cocok")
            if kinds["confirm"]:
                parts.append(f"{tags(kinds['confirm'])} belum pasti")
            if shortfall:
                parts.append(f"terdeteksi {self.shelf_n} dari {self.expected} kotak")
            elif n_vis < need_n:
                parts.append(f"baru {n_vis} dari {need_n} kotak terlihat")
            self.message = f"PERLU KONFIRMASI {req}: " + ("; ".join(parts) or "belum pasti")
            if kinds["candidate"] or kinds["confirm"]:
                self.detail = "dekatkan kamera ke kotak itu / tunjukkan barcode - sistem tidak menebak 'tidak ada'"
            else:
                self.detail = "mundurkan kamera sampai SELURUH rak terlihat, tahan diam"
        else:
            self.phase = MENCARI
            self.message = (f"MENCARI {req}: {n_done} bukan target, {len(kinds['reading'])} sedang dibaca"
                            + (f", {len(kinds['confirm'])} ragu" if kinds["confirm"] else "")
                            + f" | {max(self.shelf_n - n_taken, 0)} kotak di rak")
            self.detail = ("arahkan kamera ke rak" if not n_vis else "arahkan kotak ke tengah layar dan tahan diam")

    # ------------------------------------------------------------------ 3 kotak AMBIL
    def _follow_pick(self, now, ids, polys, tracker, frame_size, still, frame):
        tid = self.pick
        st = tracker.tracks[tid]["state"] if tid in tracker.tracks else None
        req = str(self.req)
        if st is not None and st.epoch != self.pick_epoch:
            # nomor kotak diragukan (tangan meraih / menutupi kotak, kotak tersenggol): bacaan lamanya sudah dibuang.
            # Kunci TIDAK langsung dibatalkan (dulu: BATAL AMBIL tepat saat perawat mengambil kotak itu, lalu
            # TERAMBIL tidak pernah tercatat); kotak di slot ini dipastikan ulang dulu, JANGAN diambil robot.
            self.pick_epoch = st.epoch
            self._recheck(now, tid, st, "nomor kotak diragukan - dipastikan ulang")
        if self.recheck_t0 is not None and now - self.recheck_t0 >= self.recheck_max_s \
                and self.gone_since is None and self.hold_since is None:
            # (slot sudah terlihat kosong dan sedang dipastikan terambil: batasnya gone_s / hold_s sendiri; replay
            # uji_01 - kotak diambil 6 s setelah cek ulang mulai, batas 20 s memotong tahap itu -> BATAL, bukan TERAMBIL)
            # batas total, termasuk saat kotak tidak di slotnya (uji 30-09 14:04: dua deteksi bergantian di slot
            # kosong, nomor diambil alih bolak-balik dan batas waktu di-reset terus -> macet 60 s sampai uji selesai)
            self._event(now, "BATAL AMBIL", tid, f"tidak terpastikan ulang dalam {self.recheck_max_s:g} s", st)
            self._unlock()
            return
        at = poly_iou(polys[ids.index(tid)], self.slot) if tid in ids else 0.0
        pulled = at >= 0.2 and moving_on_shelf(tracker, tid, polys[ids.index(tid)], self._shelf)
        if at >= 0.5 and not pulled:                     # kotak AMBIL masih diam di slotnya
            if self.adopted is None:
                self.hold_since = None
            if self.recheck_since is not None:
                status, why = relock_status(self._fresh_rows(st))
                waited = now - self.recheck_since
                if status == "solid":
                    self.adopted = None
                    self._event(now, "AMBIL DIPASTIKAN", tid,
                                "dipastikan ulang: " + (st.combined_sources() or st.last_reason), st)
                    self.recheck_since = self.recheck_from = self.recheck_t0 = self.hold_since = None
                    self.box_patch = None                # gambar kotak diambil lagi dari kotak yang baru dipastikan
                elif status == "cek":
                    if waited >= self.relock_s:
                        status, why = "batal", f"tidak terpastikan ulang dalam {self.relock_s:g} s ({why})"
                    else:
                        why = f"{why}, {waited:.0f}/{self.relock_s:g} s"
            else:
                status, why = pick_status(st)
            if status == "batal" and self.adopted is not None and self.adopted[2] and self.adopted[0] not in ids \
                    and sum(1 for d, _ in self._fresh_rows(st) if d == IGNORED) >= 2:
                # uji 30-09: kotak AMBIL diambil, kotak sebelahnya (Xperience Pro) miring mengisi celahnya dan
                # "dipastikan ulang" -> 2x terbaca bukan target. Kotak yang lain PASTI ada di slot itu dan kotak AMBIL
                # tidak terlihat di mana pun -> sudah terambil (dulu: BATAL AMBIL, TERAMBIL tidak tercatat).
                old, old_st, _ = self.adopted
                if self._settle_taken(now, ids, polys, tracker, old, old_st, {tid}, frame_size, watch=False,
                                      reason=f"kotak tidak terlihat lagi; slotnya kini diisi kotak lain #{tid} "
                                             "(bukan target)"):
                    return
                status, why = "cek", "memastikan kotak tidak dipindah ke tempat lain di rak"
            if status == "batal":
                self._event(now, "BATAL AMBIL", tid, why, st)
                self._unlock()
                return
            differs = False
            if status == "solid" and still and frame is not None and self.box_patch is not None:
                # tampilan kotak di slot berubah tanpa nomornya diragukan (ditukar? tertutup tangan?). Hanya saat
                # kamera diam dan BERTURUT-TURUT: potongan slot yang sempit tidak pernah pas persis antar frame saat
                # kamera bergerak (replay kamera dipegang: tiap 1-2 detik "berbeda" = hijau-kuning berkedip lagi)
                differs = ncc(slot_patch(frame, self.slot), self.box_patch) < SAME_BOX_NCC
                self.look_off = self.look_off + 1 if differs else 0
                if self.look_off >= LOOK_OFF_FRAMES:
                    status, why = "cek", "kotak di slot tampak berbeda dari saat dipilih"
                    self._recheck(now, tid, st, why, from_reads=st.reads)
                    self.look_off = 0
            self.solid = status == "solid"
            if at >= 0.8:                                # hanya getaran kecil: rapikan posisi slot
                self.slot = to_poly(polys[ids.index(tid)]).copy()
            if frame is not None and self.solid and not differs and (still or self.box_patch is None):
                # hanya dari kotak yang sedang terverifikasi penuh (bukan saat tangan menutupi / sedang dicek ulang):
                # dipakai untuk "kotak masih di slot?" dan "slot kosong?" saat memastikan terambil
                self.box_patch = slot_patch(frame, self.slot)
            self.prev_patch = None
            self.gone_since = self.occupied_since = None
            others = [t for t in ids if t != tid]
            sure = sum(1 for t in others if tracker.tracks[t]["state"].stable(self.need) in DONE)
            if self.solid:
                self.message = f"DITEMUKAN: AMBIL #{tid} {req}" + (f" ({st.last_reason})" if st.last_reason else "")
            else:
                self.message = f"DITEMUKAN: #{tid} {req} - memeriksa ulang, JANGAN diambil dulu ({why})"
            self.detail = ((f"{len(others)} kotak lain tidak diambil ({sure} pasti bukan target)" if others else "")
                           + (f" | sisa permintaan {self.remaining}" if self.remaining > 1 else ""))
            return
        if self.recheck_since is not None:
            self.recheck_since = now                     # batas waktu pastikan-ulang hanya berjalan saat kotak di slot
        if at >= 0.2:                                    # kotak AMBIL sedang ditarik keluar dari slotnya
            self.pulling = True
            self.gone_since = self.occupied_since = None
            self.message = f"DITEMUKAN: #{tid} sedang diambil"
            self.detail = "setelah kotak keluar, tahan kamera diam ke arah slotnya"
            return
        # kotak AMBIL tidak lagi di slotnya: sudah diambil, tertutup tangan, detektor sesaat gagal, nomor berganti?
        visible = frame_size is not None and inside_fraction(self.slot, frame_size) >= 0.9
        others = [p for i, p in enumerate(polys) if ids[i] != tid]
        inside = [(ids[i], p) for i, p in enumerate(polys) if ids[i] != tid and cover(self.slot, p) >= 0.6]
        if inside and all(moving_on_shelf(tracker, t, p, self._shelf) for t, p in inside) and self.pulling:
            self.gone_since = self.occupied_since = None       # kotak yang sedang ditarik (nomornya berganti)
            self.message = f"DITEMUKAN: #{tid} sedang diambil"
            self.detail = "setelah kotak keluar, tahan kamera diam ke arah slotnya"
            return
        if self.adopted is None and len(inside) == 1 and tid not in ids \
                and not moving_on_shelf(tracker, *inside[0], self._shelf):
            # satu kotak DIAM di slot dengan nomor lain, nomor lama tidak terlihat di mana pun: biasanya kotak yang
            # sama (nomornya hilang saat tertutup tangan). Kunci pindah ke nomor itu dan dipastikan ulang dari
            # bacaannya sendiri (kotak lain di slot itu -> bantahan -> batal). SEKALI per kunci: slot yang nomornya
            # terus berganti (uji 30-09 14:04, #262 <-> #356 puluhan kali) jatuh ke "ada kotak lain" / batas 20 s.
            new = inside[0][0]
            nst = tracker.tracks[new]["state"]
            self._event(now, "CEK ULANG", new, f"kotak di slot #{tid} kini bernomor #{new} - dipastikan ulang", nst)
            # kotak AMBIL asli, dan apakah ada GERAKAN: kotak AMBIL sempat ditarik, atau kotak sebelah yang sudah
            # dikenal bergeser >= 0,5 x lebar ke slot ini (tanpa gerakan, bacaan yang membantah = BATAL biasa)
            q = to_poly(self.slot)
            w = min(float(np.linalg.norm(q[(k + 1) % 4] - q[k])) for k in range(4))
            q0 = self.neigh.get(new)
            moved_in = q0 is not None and \
                float(np.linalg.norm(to_poly(inside[0][1]).mean(axis=0) - q0.mean(axis=0))) >= 0.5 * w
            self.adopted = (tid, st, self.pulling or moved_in)
            self.pick, self.pick_epoch = new, nst.epoch
            self.recheck_since, self.recheck_from = now, None
            self.recheck_t0 = self.recheck_t0 if self.recheck_t0 is not None else now
            self.gone_since = self.occupied_since = None
            self.solid = False
            self.message = f"DITEMUKAN: #{new} {req} - memeriksa ulang, JANGAN diambil dulu (nomor kotak berganti)"
            return
        if inside:
            self.gone_since = None
            self.occupied_since = self.occupied_since or now
            self.message = f"DITEMUKAN: #{tid} - ada kotak lain di slotnya, memeriksa"
            if now - self.occupied_since >= self.gone_s:
                self._event(now, "BATAL AMBIL", tid, "ada kotak diam di slot dengan nomor lain - diperiksa ulang", st)
                self._unlock()
            return
        self.occupied_since = None
        cur = slot_patch(frame, self.slot) if frame is not None and visible else None
        same_box = frame is not None and ncc(cur, self.box_patch) >= SAME_BOX_NCC
        slot_still = frame is None or (cur is None and self.prev_patch is None) or             ncc(cur, self.prev_patch) >= STILL_SLOT_NCC
        self.prev_patch = cur
        if visible and still and others and not same_box and slot_still:
            self.gone_since = self.gone_since or now
            left = self.gone_s - (now - self.gone_since)
            if left <= 0:
                self._settle_taken(now, ids, polys, tracker, tid, st, {tid}, frame_size)
                return
            self.message = f"DITEMUKAN: #{tid} tidak ada di slotnya - memastikan terambil ({left:.1f} s)"
            self.detail = "tahan kamera diam, slot harus tetap kosong"
            return
        self.gone_since = None
        self.message = f"DITEMUKAN: AMBIL #{tid} {req}"
        if not visible:
            self.detail = "slot kotak itu di luar layar - arahkan kamera ke sana dan tahan diam"
        elif same_box:
            self.detail = f"kotak #{tid} masih terlihat di slotnya (belum terambil)"
        elif not slot_still:
            self.detail = "slot masih bergerak / tertutup - tunggu sampai slot terlihat diam"
        else:
            self.detail = "tahan kamera diam untuk memastikan terambil"

    def _moved_here(self, ids, polys, tracker, skip, frame_size=None) -> list:
        """Kotak DIAM di rak yang muncul di tempat yang KOSONG saat AMBIL dikunci (>= 0,5 x lebar dari setiap kotak
        yang terlihat waktu itu), utuh di layar, dan belum pasti bukan target -> [(nomor, keadaan)]. Bisa jadi kotak
        AMBIL itu sendiri yang dipindah, bukan diambil (replay uji 30-09 14:09).
        Dinilai dari POSISI, bukan nomor: uji_01 (30-09 malam) kotak benar-benar diambil, tapi kotak kertas di luar
        rak dan kotak di tepi layar yang nomornya terus berganti dan tidak pernah terbaca dianggap "kotak baru" ->
        TERAMBIL tertahan 15 s lalu BATAL."""
        known = list(self.neigh.values())
        out = []
        for t, p in zip(ids, polys):
            if t in skip or t not in tracker.tracks:
                continue
            if frame_size is not None and inside_fraction(p, frame_size) < 0.9:
                continue                                  # terpotong di tepi layar: bukan kotak yang diletakkan di rak
            s = tracker.tracks[t]["state"].stable(self.need)
            if s in DONE or moving_on_shelf(tracker, t, p, self._shelf):
                continue
            # di kolom dan rentang kotak yang sudah ada saat dikunci: kotak itu sendiri (nomornya saja berganti) atau
            # potongannya saat terbelah lengan (pusat potongan jauh dari pusat kotak, tapi tetap di dalam kotaknya)
            if any(lat <= 0.5 and along <= 0.5 for lat, along in (_shift(k, p) for k in known)):
                continue
            out.append((t, s))
        return out

    def _settle_taken(self, now, ids, polys, tracker, tid, st, skip, frame_size=None, **kw) -> bool:
        """Kotak AMBIL tidak lagi di slotnya. TERAMBIL hanya kalau tidak ada kotak baru / berpindah di rak yang mungkin
        kotak itu sendiri; kotak seperti itu terbaca target -> dipindah (BATAL, dicari ulang di tempat barunya);
        belum terbaca `hold_s` detik -> BATAL juga (terambil atau dipindah tidak bisa dipastikan). -> True = selesai."""
        moved = self._moved_here(ids, polys, tracker, skip, frame_size)
        if any(s == MATCH for _, s in moved):
            self._event(now, "BATAL AMBIL", tid, "kotak dipindah ke tempat lain di rak - dicari ulang", st)
            self._unlock()
            return True
        if moved:
            self.hold_since = self.hold_since if self.hold_since is not None else now
            if now - self.hold_since >= self.hold_s:
                self._event(now, "BATAL AMBIL", tid, "terambil atau dipindah? kotak baru di rak belum terbaca "
                                                     "- dicari ulang", st)
                self._unlock()
                return True
            self.message = (f"DITEMUKAN: #{tid} tidak di slotnya - memastikan tidak dipindah ke tempat lain "
                            f"({', '.join('#' + str(t) for t, _ in moved[:4])} dibaca)")
            self.detail = "jangan ambil kotak lain dulu; kotak yang baru diletakkan di rak sedang dibaca"
            return False
        self._taken(now, tid, st, tracker, **kw)
        return True

    def _taken(self, now, tid, st, tracker, reason="slot kosong terverifikasi", watch=True):
        self.taken.append(Taken(tid, now, self.slot.copy(), st.gtin if st else "", st.lot if st else "",
                                st.expiry if st else None))
        tracker.tracks.pop(tid, None)          # nomor ini tidak boleh menempel ke kotak lain
        self.remaining -= 1
        # slot yang sudah diisi kotak sebelah (watch=False) tidak diawasi "terisi lagi"
        self.watch = (self.slot.copy(), now, tid, self.box_patch) if watch else None
        self._event(now, "TERAMBIL", tid, reason, st)
        self.adopted, self.neigh = None, {}
        self.pick, self.slot, self.box_patch, self.prev_patch, self.gone_since = None, None, None, None, None
        self.recheck_since = self.recheck_from = self.recheck_t0 = self.hold_since = None
        self.phase = SELESAI if self.remaining <= 0 else MENCARI

    def _unlock(self):
        self.solid = False
        self.pick, self.slot, self.box_patch, self.prev_patch = None, None, None, None
        self.gone_since = self.occupied_since = None
        self.recheck_since = self.recheck_from = self.recheck_t0 = self.hold_since = None
        self.adopted = None
        self.neigh = {}
        self.phase = MENCARI

    def _recheck(self, now, tid, st, why, from_reads=None):
        """Mulai (atau lanjutkan) memastikan ulang kotak AMBIL: robot tidak boleh mengambil sampai bacaan baru
        menunjuk tepat kotak yang diminta (relock_status)."""
        if self.recheck_since is None:
            self.recheck_since = now
            self._event(now, "CEK ULANG", tid, why, st)
        if self.recheck_t0 is None:
            self.recheck_t0 = now
        self.recheck_from = from_reads
        self.solid = False

    def _fresh_rows(self, st):
        """Bacaan berisi yang BARU sejak kunci mulai dipastikan ulang: [(keputusan, (atom, bantahan) atau None)]."""
        h = list(st.history)
        sup = list(st.support) if len(st.support) == len(h) else [None] * len(h)
        rows = list(zip(h, sup))
        if self.recheck_from is not None:
            n = st.reads - self.recheck_from
            rows = rows[-n:] if n > 0 else []
        return [(d, s) for (d, k), s in rows if k != EMPTY]

    def _check_watch(self, now, polys, frame_size, frame):
        if self.watch is None:
            return
        slot, t0, tid, patch = self.watch
        if now - t0 > self.recheck_s:
            self.watch = None
            return
        refilled = any(cover(slot, p) >= 0.6 for p in polys)
        if not refilled and frame is not None and frame_size is not None and inside_fraction(slot, frame_size) >= 0.9:
            refilled = ncc(slot_patch(frame, slot), patch) >= SAME_BOX_NCC
        if refilled:
            self.warning = f"PERINGATAN: slot #{tid} yang sudah kosong terisi lagi - pastikan kotak benar-benar diambil"
            if self._warned != tid:
                self._event(now, "PERINGATAN", tid, "slot terisi lagi setelah terambil")
                self._warned = tid

    # ------------------------------------------------------------------ bantu
    def _event(self, now, what, tid, reason, st=None):
        self.events.append((now, what, tid, reason, st))
