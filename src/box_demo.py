"""
Demo permintaan teks (meeting 05-10): permintaan dokter (brand + produk + spec) -> barang di inventaris -> posisi
rak / level -> pesan untuk software TM. Tanpa kamera.

Hasil demo TIDAK PERNAH "match": teks hanya menunjukkan ke mana robot pergi. Kotak tetap diverifikasi kamera
(test_webcam_box.py --inventaris ... --request ...) dan konfirmasi akhir tetap di dokter.

Status:
  SIAP      : ada stok terverifikasi + posisinya diketahui -> robot boleh menuju posisi, lalu verifikasi kamera
  NO_POS    : ada stok, tapi posisinya tidak ada di data/rak/posisi.csv -> perlu konfirmasi orang
  EXPIRED   : stoknya ada, tapi semuanya sudah kedaluwarsa -> tidak pernah diambil
  HABIS     : tidak ada stok (atau semua sudah diambil)
  INVALID   : permintaan tidak bisa dipahami / tidak lengkap -> ulangi permintaan
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import pathlib
from dataclasses import dataclass, field

from src.box_inventory import ACTIVE, STATUS_DIAMBIL, InvBox, Inventory
from src.box_verifier import Request, brand_of, describe_request, parse_request

SIAP, NO_POS, EXPIRED, HABIS, INVALID = "siap_verifikasi", "posisi_tidak_diketahui", "kedaluwarsa", "habis", \
    "permintaan_tidak_valid"
TEKS_STATUS = {SIAP: "SIAP - robot menuju posisi, lalu verifikasi kamera + konfirmasi dokter",
               NO_POS: "PERLU KONFIRMASI - stok ada, posisi tidak ada di tabel posisi",
               EXPIRED: "TIDAK DIAMBIL - stok yang ada sudah kedaluwarsa",
               HABIS: "TIDAK ADA - tidak ada stok",
               INVALID: "ULANGI - permintaan tidak dipahami"}


# ------------------------------------------------------------------ tabel posisi
@dataclass
class Position:
    rak: str
    level: str
    kolom: str = ""
    sumber: str = ""                  # "contoh" = belum dari tim, jangan dipakai di rumah sakit

    def __str__(self) -> str:
        return f"rak {self.rak}, level {self.level}" + (f", kolom {self.kolom}" if self.kolom else "")


def load_positions(path: str | pathlib.Path) -> dict[str, Position]:
    """data/rak/posisi.csv: box_id,rak,level,kolom,sumber -> {box_id: Position}. Baris tanpa rak/level dilewati."""
    path = pathlib.Path(path)
    out: dict[str, Position] = {}
    if not path.exists():
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            bid, rak, level = (r.get(k, "").strip() for k in ("box_id", "rak", "level"))
            if bid and rak and level:
                out[bid] = Position(rak, level, r.get("kolom", "").strip(), r.get("sumber", "").strip())
    return out


# ------------------------------------------------------------------ satu permintaan
@dataclass
class LookupResult:
    text: str                                      # permintaan apa adanya
    status: str
    request: Request | None = None
    box: InvBox | None = None
    position: Position | None = None
    reason: str = ""
    stock: list[str] = field(default_factory=list)  # semua box_id yang cocok (urut FEFO)
    time: str = ""


def pick_box(cands: list[InvBox], positions: dict[str, Position]) -> tuple[InvBox | None, str]:
    """Kotak yang dikirim ke robot dari `cands` (sudah urut FEFO, tidak kosong, tidak ada yang kedaluwarsa)
    -> (kotak, SIAP | NO_POS).
    FEFO ketat: selalu kotak pertama. Kalau posisinya tidak diketahui -> NO_POS (orang memutuskan); TIDAK pindah ke
    kotak berikutnya yang posisinya ada, supaya kotak yang lebih dulu kedaluwarsa tidak terlewat diam-diam."""
    first = cands[0]
    return first, (SIAP if first.box_id in positions else NO_POS)


def lookup(text: str, inv: Inventory, positions: dict[str, Position], today: dt.date | None = None) -> LookupResult:
    today = today or dt.date.today()
    res = LookupResult(text=text, status=INVALID, time=dt.datetime.now().isoformat(timespec="seconds"))
    try:
        res.request = parse_request(text)
    except ValueError as e:
        res.reason = str(e)
        return res
    req = res.request
    cands = inv.candidates(req, today)
    res.stock = [b.box_id for b in cands]
    if not cands:
        same = [b for b in inv.boxes.values() if b.status in ACTIVE + (STATUS_DIAMBIL,)
                and Request(b.product, b.diameter, b.length) == req]
        expired = [b for b in same if b.status in ACTIVE and b.expiry_date() and b.expiry_date() < today]
        taken = [b for b in same if b.status == STATUS_DIAMBIL]
        if expired:
            res.status = EXPIRED
            res.reason = "kedaluwarsa: " + ", ".join(f"{b.box_id} ({b.expiry})" for b in expired)
        else:
            res.status = HABIS
            res.reason = ("sudah diambil: " + ", ".join(b.box_id for b in taken)) if taken else "tidak ada di inventaris"
        return res
    res.box, res.status = pick_box(cands, positions)
    if res.box is not None:
        res.position = positions.get(res.box.box_id)
    if res.status == NO_POS:
        res.reason = "posisi tidak ada di tabel posisi: " + ", ".join(res.stock)
    elif res.position is not None and res.position.sumber == "contoh":
        res.reason = "posisi dari tabel CONTOH (belum dari tim)"
    return res


# ------------------------------------------------------------------ keluaran
def to_tm_message(res: LookupResult) -> dict:
    """Pesan untuk software TM / modul berikutnya (AMR, lengan). Format sementara sampai tim menyepakati kontraknya:
    satu objek JSON datar, kunci bahasa Inggris, nilai kosong = "" (bukan null) supaya mudah dibaca di sisi TM."""
    req, box, pos = res.request, res.box, res.position
    return {
        "time": res.time,
        "request_text": res.text,
        "status": res.status,
        "brand": brand_of(req.product) if req else "",
        "product": req.product if req else "",
        "diameter_mm": "" if req is None or req.diameter is None else req.diameter,
        "length_mm": "" if req is None or req.length is None else req.length,
        "box_id": box.box_id if box else "",
        "expiry": box.expiry if box else "",
        "rack": pos.rak if pos else "",
        "level": pos.level if pos else "",
        "column": pos.kolom if pos else "",
        "stock_count": len(res.stock),
        "needs_doctor_confirmation": True,      # selalu: konfirmasi akhir ada di dokter (meeting 05-10)
        "reason": res.reason,
    }


def to_text(res: LookupResult) -> str:
    """Teks pendek untuk layar dokter / log: apa yang didengar dan diputuskan sistem."""
    lines = [f"Permintaan : {res.text}"]
    if res.request is not None:
        lines.append(f"Dipahami   : {describe_request(res.request)}")
    lines.append(f"Status     : {TEKS_STATUS[res.status]}")
    if res.box is not None:
        exp = f", kedaluwarsa {res.box.expiry}" if res.box.expiry else ""
        lines.append(f"Kotak      : {res.box.box_id} ({res.box.label()}{exp})")
    if res.position is not None:
        lines.append(f"Posisi     : {res.position}")
    if res.stock:
        lines.append(f"Stok       : {len(res.stock)} kotak ({', '.join(res.stock)})")
    if res.reason:
        lines.append(f"Catatan    : {res.reason}")
    return "\n".join(lines)


def append_log(path: str | pathlib.Path, res: LookupResult) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(to_tm_message(res), ensure_ascii=False) + "\n")
