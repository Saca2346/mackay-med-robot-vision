"""
Stress test penyeleksi: daftar permintaan dijalankan bergiliran, tiap barang diberi batas waktu (default 90 s).
Barang yang belum AMBIL dalam batas waktu dilewati dan dicatat; hasil tiap barang ditulis ke CSV untuk
dibandingkan antar uji (laptop vs server, sebelum vs sesudah perbaikan).

File rencana (dibuat scripts/stress_plan.py, boleh diedit tangan), satu baris per langkah:
  # BAGIAN 1: 20 kotak bergiliran        -> judul bagian (dicatat di kolom "bagian")
  angiolite 2.5 29   # K15                -> satu permintaan (teks sesudah # hanya catatan)
  ! ubah 5 kotak: K03 balik, ...          -> JEDA: kamera tetap jalan, ketik "lanjut" di terminal untuk meneruskan
Barang tidak diambil (cukup AMBIL), jadi inventaris tidak berubah.
"""
from __future__ import annotations

import csv
import pathlib
import statistics
from dataclasses import dataclass, field

CONFIRM_S = 2.0          # AMBIL harus bertahan sekian detik (tidak langsung BATAL) supaya dihitung berhasil
TIDAK_ADA_HOLD_S = 3.0   # TIDAK ADA bertahan sekian detik -> barang berikutnya


@dataclass
class Step:
    kind: str            # "req" | "pause"
    text: str
    section: str = ""
    note: str = ""


def load_plan(path) -> list[Step]:
    steps, section = [], ""
    for raw in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            section = line.lstrip("# ").strip()
            continue
        if line.startswith("!"):
            steps.append(Step("pause", line[1:].strip(), section))
            continue
        text, _, note = line.partition("#")
        steps.append(Step("req", text.strip(), section, note.strip()))
    if not any(s.kind == "req" for s in steps):
        raise ValueError(f"rencana tanpa permintaan: {path}")
    return steps


FIELDS = ["no", "bagian", "permintaan", "catatan", "hasil", "detik", "kotak", "bukti", "teks_terbaca",
          "cek_ulang", "batal", "jumlah_kotak", "bacaan", "ms_per_bacaan"]


@dataclass
class StressRun:
    steps: list
    per_item_s: float = 90.0
    log_path: pathlib.Path | None = None
    idx: int = -1
    waiting: bool = False
    rows: list = field(default_factory=list)
    _t0: float = 0.0
    _found: tuple | None = None          # (waktu AMBIL, nomor kotak, alasan)
    _cek: int = 0
    _batal: int = 0
    _tidak_ada: float | None = None
    _base: tuple = (0, 0.0)              # (bacaan, ms OCR) saat barang dimulai

    @property
    def n_req(self) -> int:
        return sum(1 for s in self.steps if s.kind == "req")

    @property
    def current(self) -> Step | None:
        return self.steps[self.idx] if 0 <= self.idx < len(self.steps) else None

    def first_request(self) -> str:
        return next(s.text for s in self.steps if s.kind == "req")

    def advance(self) -> Step | None:
        """Langkah berikutnya (permintaan atau jeda); None = rencana selesai."""
        self.idx += 1
        step = self.current
        self.waiting = step is not None and step.kind == "pause"
        return step

    def start_item(self, now: float, stats: dict) -> None:
        self._t0, self._found, self._cek, self._batal, self._tidak_ada = now, None, 0, 0, None
        self._base = (stats.get("reads", 0), stats.get("ocr_ms", 0.0))

    def on_events(self, events) -> None:
        for t, what, tid, reason, _ in events:
            if what == "AMBIL" and self._found is None:
                self._found = (t, tid, reason)
            elif what == "CEK ULANG":
                self._cek += 1
            elif what == "BATAL AMBIL":
                self._batal += 1
                if self._found is not None and self._found[1] == tid:
                    self._found = None               # AMBIL yang batal tidak dihitung; tunggu AMBIL berikutnya

    def check(self, now: float, phase: str, pick) -> str | None:
        """Barang sekarang selesai? -> hasil ("AMBIL", "TIDAK ADA", "WAKTU HABIS") atau None (lanjut mencari)."""
        if self.waiting or self.current is None:
            return None
        if self._found is not None and now - self._found[0] >= CONFIRM_S and pick == self._found[1]:
            return "AMBIL"
        self._tidak_ada = (self._tidak_ada or now) if phase == "TIDAK ADA" else None
        if self._tidak_ada is not None and now - self._tidak_ada >= TIDAK_ADA_HOLD_S:
            return "TIDAK ADA"
        if now - self._t0 >= self.per_item_s:
            return "WAKTU HABIS"
        return None

    def elapsed(self, now: float) -> float:
        return now - self._t0

    def finish_item(self, now: float, result: str, stats: dict, count, teks: str = "") -> dict:
        step = self.current
        reads = stats.get("reads", 0) - self._base[0]
        ms = stats.get("ocr_ms", 0.0) - self._base[1]
        found = self._found if result == "AMBIL" else None
        row = {"no": sum(1 for s in self.steps[:self.idx + 1] if s.kind == "req"), "bagian": step.section,
               "permintaan": step.text, "catatan": step.note, "hasil": result,
               "detik": f"{(found[0] - self._t0) if found else now - self._t0:.1f}",
               "kotak": "" if not found else found[1], "bukti": "" if not found else found[2],
               "teks_terbaca": (teks or "")[:120], "cek_ulang": self._cek, "batal": self._batal,
               "jumlah_kotak": "" if count is None else count, "bacaan": reads,
               "ms_per_bacaan": f"{ms / reads:.0f}" if reads else ""}
        self.rows.append(row)
        if self.log_path is not None:
            new = not self.log_path.exists()
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if new:
                    w.writeheader()
                w.writerow(row)
        return row

    def status(self, now: float) -> str:
        step = self.current
        if step is None:
            return "STRESS selesai"
        if self.waiting:
            return f"STRESS JEDA: {step.text} | ketik lanjut di terminal"
        done = len(self.rows)
        return (f"STRESS {done + 1}/{self.n_req} {step.section[:30]} | {self.elapsed(now):.0f}/{self.per_item_s:.0f} s"
                f" | berhasil {sum(r['hasil'] == 'AMBIL' for r in self.rows)}/{done}")

    def summary(self) -> list[str]:
        if not self.rows:
            return ["stress test: belum ada barang yang selesai"]
        out = []
        sections = []
        for r in self.rows:
            if r["bagian"] not in sections:
                sections.append(r["bagian"])
        for sec in sections + ["SEMUA"]:
            rows = self.rows if sec == "SEMUA" else [r for r in self.rows if r["bagian"] == sec]
            ok = [float(r["detik"]) for r in rows if r["hasil"] == "AMBIL"]
            med = f", median {statistics.median(ok):.1f} s, maks {max(ok):.1f} s" if ok else ""
            out.append(f"{sec or '(tanpa bagian)'}: AMBIL {len(ok)}/{len(rows)}{med}"
                       f", waktu habis {sum(r['hasil'] == 'WAKTU HABIS' for r in rows)}"
                       f", TIDAK ADA {sum(r['hasil'] == 'TIDAK ADA' for r in rows)}"
                       f", BATAL {sum(int(r['batal']) for r in rows)}")
        fails = [r for r in self.rows if r["hasil"] != "AMBIL"]
        if fails:
            out.append("belum berhasil: " + "; ".join(f"{r['permintaan']} ({r['hasil']})" for r in fails))
        return out
