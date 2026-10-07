"""
Inventaris rak: SETIAP kotak fisik di rak tercatat dengan identitasnya (produk, ukuran, GTIN, LOT, kedaluwarsa),
gambar punggungnya, dan siapa yang memeriksanya. Dipakai dalam tiga langkah (scripts/rak_inventaris.py):

  1. pindai  (sebelum hari operasi, tanpa tekanan waktu) - kamera menyisir rak; tiap kotak dibaca berkali-kali dan
             identitasnya baru DITETAPKAN kalau >= 2 bukti independen sepakat (aturan yang sama dengan verifikasi
             permintaan: barcode = 2, teks ukuran = 1, REF = 1, teks GTIN = 1) tanpa satu pun bacaan bertentangan.
             Hasilnya DRAF; kotak yang tidak bisa ditetapkan ditandai "perlu dicek".
  2. cek     perawat memeriksa tiap kotak (gambar + data) dan mengonfirmasi / mengoreksi. Hanya entri yang
             diperiksa orang yang dipakai saat operasi. Kalau daftar isi rak sudah ada (impor), hasil pindai
             dicocokkan ke daftar itu: kotak yang kurang / lebih langsung terlihat.
  3. operasi (scripts/test_webcam_box.py --inventaris) - permintaan "angiolite 2.5 x 29": kotak yang dicari sudah
             diketahui dari inventaris (FEFO: kedaluwarsa paling dulu), dicari di rak lewat CIRI TAMPILAN punggungnya
             (warna per bagian), bukan lewat posisi lama, jadi kotak yang dipindah tetap ketemu. Kotak itu tetap
             diverifikasi ulang (barcode / teks + REF) sebelum AMBIL; ciri tampilan hanya menentukan urutan baca.
"""
from __future__ import annotations

import csv
import datetime as dt
import pathlib
from dataclasses import asdict, dataclass, fields

import cv2
import numpy as np

from src.box_verifier import NO_SIZE, Evidence, Request, brand_of, combined_score, support

# ------------------------------------------------------------------ data
STATUS_DRAF = "draf"                  # dari pindai, belum diperiksa orang
STATUS_CEK = "perlu_dicek"            # pindai tidak bisa menetapkan identitas (bukti kurang / bertentangan)
STATUS_OK = "terverifikasi"           # diperiksa perawat: DIPAKAI saat operasi
STATUS_DAFTAR = "belum_dipetakan"     # dari daftar isi rak (impor), belum ditemukan di rak
STATUS_DIAMBIL = "diambil"            # sudah diambil robot (TERAMBIL) - tidak dicari lagi
ACTIVE = (STATUS_OK,)


@dataclass
class InvBox:
    box_id: str                                   # K01, K02, ... (nomor tetap, tidak berubah walau kotak dipindah)
    product: str = ""
    diameter: float | None = None
    length: float | None = None
    brand: str = ""
    gtin: str = ""
    lot: str = ""
    expiry: str = ""                              # YYYY-MM-DD
    status: str = STATUS_DRAF
    sources: str = ""                             # bukti saat pindai, mis. "barcode + teks + REF (5 bacaan)"
    order: int = 0                                # urutan kiri -> kanan saat dipindai (petunjuk, bukan kunci)
    thumb: str = ""                               # gambar punggung (relatif ke folder inventaris)
    sig: str = ""                                 # ciri tampilan (angka dipisah spasi)
    checked_by: str = ""
    checked_at: str = ""
    taken_at: str = ""
    note: str = ""

    def key(self):
        return self.product, self.diameter, self.length

    def label(self) -> str:
        size = "" if self.diameter is None else f" {self.diameter:g} x {self.length:g} mm"
        return f"{self.product}{size}" if self.product else "(belum diketahui)"

    def expiry_date(self) -> dt.date | None:
        try:
            return dt.date.fromisoformat(self.expiry) if self.expiry else None
        except ValueError:
            return None

    def signature(self) -> np.ndarray | None:
        return np.array([float(v) for v in self.sig.split()], dtype=np.float32) if self.sig else None


class Inventory:
    """data/rak/inventaris.csv + data/rak/punggung/<box_id>.jpg."""

    def __init__(self, path: str | pathlib.Path):
        self.path = pathlib.Path(path)
        self.dir = self.path.parent
        self.boxes: dict[str, InvBox] = {}
        if self.path.exists():
            with open(self.path, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    b = InvBox(**{k.name: r.get(k.name, "") for k in fields(InvBox)})
                    b.diameter = float(b.diameter) if b.diameter not in ("", None) else None
                    b.length = float(b.length) if b.length not in ("", None) else None
                    b.order = int(b.order or 0)
                    self.boxes[b.box_id] = b

    def reload(self) -> None:
        """Baca ulang dari file. Program uji memegang inventaris selama berjalan; tanpa ini `rak_inventaris kembali`
        yang dijalankan di jendela lain tertimpa oleh salinan lama saat TERAMBIL berikutnya disimpan."""
        self.boxes = Inventory(self.path).boxes

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[k.name for k in fields(InvBox)])
            w.writeheader()
            for b in sorted(self.boxes.values(), key=lambda b: b.box_id):
                row = asdict(b)
                row["diameter"] = "" if b.diameter is None else f"{b.diameter:g}"
                row["length"] = "" if b.length is None else f"{b.length:g}"
                w.writerow(row)
        tmp.replace(self.path)

    def new_id(self) -> str:
        n = max([int(k[1:]) for k in self.boxes if k[1:].isdigit()] + [0]) + 1
        return f"K{n:02d}"

    def thumb_path(self, b: InvBox) -> pathlib.Path:
        return self.dir / b.thumb

    def candidates(self, req: Request, today: dt.date | None = None) -> list[InvBox]:
        """Kotak TERVERIFIKASI yang cocok dengan permintaan, urut FEFO (kedaluwarsa paling dulu; tanpa tanggal
        paling akhir). Kotak kedaluwarsa tidak ikut."""
        today = today or dt.date.today()
        out = [b for b in self.boxes.values() if b.status in ACTIVE and b.product == req.product
               and (req.diameter is None or (b.diameter is not None and abs(b.diameter - req.diameter) < 1e-6
                                             and b.length is not None and abs(b.length - req.length) < 1e-6))
               and not (b.expiry_date() and b.expiry_date() < today)]
        return sorted(out, key=lambda b: (b.expiry_date() is None, b.expiry_date() or dt.date.max, b.box_id))

    def counts(self) -> dict:
        c: dict = {}
        for b in self.boxes.values():
            c[b.status] = c.get(b.status, 0) + 1
        return c


# ------------------------------------------------------------------ identitas dari beberapa bacaan (tanpa permintaan)
def candidate_keys(ev: Evidence, catalog=None) -> set:
    """Identitas (produk, diameter, panjang) yang DIUSULKAN oleh satu bacaan: teks ukuran lengkap, REF, atau GTIN
    di katalog terverifikasi."""
    keys = set()
    t = ev.text
    if t is not None and t.product_ok:
        if t.product in NO_SIZE:
            keys.add((t.product, None, None))            # produk tanpa ukuran (Sled Bag): nama saja = teks lengkap
        elif t.diameter_ok and t.length_ok:
            keys.add((t.product, t.diameter, t.length))
    for r in ev.refs:
        keys.add((r.product, r.diameter, r.length))
    for c in ev.codes:
        e = catalog.get(c.gtin) if catalog is not None else None
        if e is not None and e.verified:
            keys.add((e.product, e.diameter_mm, e.length_mm))
    return keys


def pieced_keys(evs: list) -> set:
    """Identitas yang dirangkai dari POTONGAN teks beberapa bacaan (produk dari satu bacaan, diameter / panjang dari
    bacaan lain). Kombinasi yang salah gugur sendiri di identify(): bacaan dengan angka lain = bertentangan."""
    prods, dias, lens = set(), set(), set()
    for e in evs:
        t = e.text
        if t is None:
            continue
        if t.product_ok and t.product in NO_SIZE:
            continue                                     # sudah jadi identitas utuh di candidate_keys
        if t.product_ok and t.product:
            prods.add(t.product)
        if t.diameter_ok and t.diameter is not None:
            dias.add(t.diameter)
        if t.length_ok and t.length is not None:
            lens.add(t.length)
    return {(p, d, ln) for p in prods for d in dias for ln in lens} if len(prods) == 1 else set()


def identify(evs: list, catalog=None, need: int = 2, today: dt.date | None = None) -> tuple:
    """Bacaan-bacaan SATU kotak (nomor tidak diragukan) -> (identitas | None, sumber, catatan).
    PASTI kalau: >= `need` bacaan menyumbang bukti yang cocok, skor gabungan >= 2 (barcode 2 / teks ukuran 1 /
    REF 1 / teks GTIN 1), satu GTIN saja, TIDAK ADA bacaan yang bertentangan, dan tidak ada identitas lain yang juga
    memenuhi syarat. Kalau tidak ada yang pasti: USULAN (sumber diawali "USULAN") bila tepat satu identitas punya
    teks ukuran lengkap dari >= `need` bacaan tanpa pertentangan - untuk dicek teliti oleh perawat, tidak pernah
    dipakai sebelum dicek."""
    evs = [e for e in evs if not e.too_far]
    keys = set().union(*[candidate_keys(e, catalog) for e in evs]) if evs else set()
    keys |= pieced_keys(evs)
    if not keys:
        return None, "", "belum ada bacaan yang lengkap" if evs else "belum terbaca"
    ok, weak, notes = [], [], []
    for key in sorted(keys, key=str):
        req = Request(*key)
        sups = [support(e, req, catalog, today=today) for e in evs]
        if any(c for _, c in sups):
            notes.append(f"{_fmt(key)}: ada bacaan yang bertentangan")
            continue
        contrib = [a for a, _ in sups if a]
        if len(contrib) < need:
            notes.append(f"{_fmt(key)}: baru {len(contrib)} bacaan")
            continue
        atoms = frozenset().union(*contrib)
        score, gtins = combined_score(atoms)
        if len(gtins) > 1:
            notes.append(f"{_fmt(key)}: dua GTIN")
            continue
        if score >= 2:
            ok.append((key, _sources(atoms), len(contrib)))
        elif score == 1:
            weak.append((key, _sources(atoms), len(contrib)))
            notes.append(f"{_fmt(key)}: baru 1 bukti")
    if len(ok) == 1:
        key, src, n = ok[0]
        return key, f"{src} ({n} bacaan)", ""
    if len(ok) > 1:
        return None, "", "dua identitas berbeda: " + ", ".join(_fmt(k) for k, _, _ in ok)
    if len(weak) == 1:
        key, src, n = weak[0]
        return key, f"USULAN 1 bukti: {src} ({n} bacaan) - cek teliti", ""
    return None, "", "; ".join(notes)


def codes_for(evs: list, key, catalog=None) -> tuple[str, str, str]:
    """GTIN / LOT / kedaluwarsa dari barcode yang cocok dengan identitas `key` -> (gtin, lot, 'YYYY-MM-DD')."""
    for e in reversed(evs):
        for c in e.codes:
            ent = catalog.get(c.gtin) if catalog is not None else None
            if ent is not None and (ent.product, ent.diameter_mm, ent.length_mm) == tuple(key):
                return c.gtin, c.lot or "", c.expiry.isoformat() if c.expiry else ""
    return "", "", ""


def _fmt(key) -> str:
    p, d, l = key
    return p if d is None else f"{p} {d:g}x{l:g}"


def _sources(atoms) -> str:
    names = [n for n, ok in (("barcode", any(isinstance(a, tuple) and a[0] == "bc" for a in atoms)),
                             ("teks GTIN", any(isinstance(a, tuple) and a[0] == "gt" for a in atoms)),
                             ("teks", {"tp", "td", "tl"} <= atoms), ("REF", "ref" in atoms)) if ok]
    return " + ".join(names)


# ------------------------------------------------------------------ ciri tampilan punggung
SIG_PARTS = 6          # punggung dibagi 6 bagian memanjang; tiap bagian: histogram warna
_H_BINS, _S_BINS, _V_BINS = 12, 3, 3


def spine_crop(frame: np.ndarray, poly) -> np.ndarray:
    """Punggung kotak dari frame, diluruskan dan dibuat MENDATAR (sisi panjang = lebar gambar)."""
    from src.box_geometry import warp_upright
    img = warp_upright(frame, poly, min_side=16, pad=0.0)
    if img.shape[0] > img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def signature(crop: np.ndarray) -> np.ndarray:
    """Ciri tampilan punggung: histogram warna (HSV) di tiap bagian memanjang, digabung dan dinormalkan.
    Tahan terhadap posisi / ukuran di gambar; urutan bagian mengikuti arah punggung (dibandingkan dua arah)."""
    img = cv2.resize(crop, (SIG_PARTS * 16, 24), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    parts = []
    for k in range(SIG_PARTS):
        seg = hsv[:, k * 16:(k + 1) * 16].reshape(-1, 3).astype(np.float32)
        h, s, v = seg[:, 0], seg[:, 1], seg[:, 2]
        colored = s > 60
        hh = np.histogram(h[colored], bins=_H_BINS, range=(0, 180))[0].astype(np.float32)
        ss = np.histogram(s, bins=_S_BINS, range=(0, 256))[0].astype(np.float32)
        vv = np.histogram(v, bins=_V_BINS, range=(0, 256))[0].astype(np.float32)
        part = np.concatenate([hh * 1.5, ss * 0.5, vv * 0.5])
        parts.append(part / max(float(np.linalg.norm(part)), 1e-6))
    return np.concatenate(parts)


def similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Kemiripan dua ciri (0..1), dibandingkan dua arah (kotak bisa diletakkan terbalik)."""
    if a is None or b is None or a.shape != b.shape:
        return 0.0
    n = len(a) // SIG_PARTS
    rev = np.concatenate([b[k * n:(k + 1) * n] for k in reversed(range(SIG_PARTS))])
    return max(float(np.dot(a, b)), float(np.dot(a, rev))) / SIG_PARTS


def sig_str(sig: np.ndarray) -> str:
    return " ".join(f"{v:.4f}" for v in sig)


# ------------------------------------------------------------------ daftar isi rak yang sudah ada
def import_list(inv: Inventory, rows: list[dict]) -> int:
    """Daftar isi rak (produk, diameter, panjang, jumlah [, gtin, lot, kedaluwarsa]) -> entri 'belum_dipetakan'.
    Saat pindai, kotak yang identitasnya ditetapkan dicocokkan ke entri ini."""
    n = 0
    for r in rows:
        prod = (r.get("produk") or r.get("product") or "").strip().lower()
        if not prod:
            continue
        d = r.get("diameter", "").strip()
        ln = (r.get("panjang") or r.get("length") or "").strip()
        qty = int((r.get("jumlah") or r.get("qty") or "1").strip() or 1)
        for _ in range(qty):
            b = InvBox(box_id=inv.new_id(), product=prod, diameter=float(d.replace(",", ".")) if d else None,
                       length=float(ln.replace(",", ".")) if ln else None, brand=brand_of(prod),
                       gtin=(r.get("gtin") or "").strip(), lot=(r.get("lot") or "").strip(),
                       expiry=(r.get("kedaluwarsa") or r.get("expiry") or "").strip(), status=STATUS_DAFTAR,
                       note="dari daftar isi rak")
            inv.boxes[b.box_id] = b
            n += 1
    return n


def place_scanned(inv: Inventory, key, sources: str, gtin: str, lot: str, expiry: str, order: int,
                  thumb: str, sig: str) -> InvBox:
    """Kotak hasil pindai yang identitasnya ditetapkan -> entri inventaris. Kalau ada entri daftar
    'belum_dipetakan' dengan identitas sama (dan LOT sama kalau keduanya punya LOT), entri itu yang dipakai:
    sekarang sudah dipetakan. Kalau tidak, entri draf baru (kotak yang tidak ada di daftar)."""
    p, d, l = key
    match = [b for b in inv.boxes.values() if b.status == STATUS_DAFTAR and b.key() == (p, d, l)
             and not (b.lot and lot and b.lot != lot)]
    b = min(match, key=lambda b: (not (b.lot and b.lot == lot), b.box_id)) if match else None
    if b is None:
        b = InvBox(box_id=inv.new_id(), product=p, diameter=d, length=l, brand=brand_of(p),
                   note="tidak ada di daftar isi rak" if any(x.note == "dari daftar isi rak"
                                                               for x in inv.boxes.values()) else "")
        inv.boxes[b.box_id] = b
    b.status = STATUS_DRAF
    b.sources, b.order, b.thumb, b.sig = sources, order, thumb, sig
    b.gtin, b.lot, b.expiry = gtin or b.gtin, lot or b.lot, expiry or b.expiry
    return b
