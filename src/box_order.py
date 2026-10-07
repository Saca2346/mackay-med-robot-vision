"""
Pesanan: satu atau beberapa barang, masing-masing dengan jumlahnya, dikerjakan berurutan.

Satu baris yang diketik di terminal (atau --request):
  angiolite 2.5 29                                  satu kotak
  2 kotak angiolite 2.5 29                          dua kotak yang sama ("jumlah 2" / "qty 2" juga bisa)
  angiolite 2.5 29; accuforce 2.75 20 jumlah 2      dua barang berbeda (pemisah: ";"  " + "  "dan"  "serta"  "lalu")
  + essential pro 3 40                              DITAMBAHKAN ke antrean (awalan "+" atau "tambah")
  lewati                                            barang sekarang dilewati, lanjut ke barang berikutnya
Baris biasa MENGGANTI pesanan yang sedang berjalan (permintaan yang tiba-tiba berubah). Dulu dua baris yang diketik
cepat berturut-turut membuat baris pertama hilang tanpa pemberitahuan, dan jumlah hanya bisa diatur lewat --qty.
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass

from src.box_verifier import Request, parse_request

MAX_QTY = 20
_SPLIT = re.compile(r"\s*(?:;|\n|\s\+\s|\bdan\b|\bserta\b|\blalu\b)\s*", re.I)
_QTY = (re.compile(r"\b(?:jumlah|qty|sebanyak)\s*[:=]?\s*(\d+)\b", re.I),
        re.compile(r"\b(\d+)\s*(?:kotak|buah|pcs|biji)\b", re.I))
_SKIP = {"lewati", "skip", "berikutnya", "next"}


@dataclass
class Item:
    req: Request
    qty: int = 1

    def __str__(self) -> str:
        return f"{self.req}" + (f" x{self.qty}" if self.qty > 1 else "")


def parse_item(text: str, catalog=None) -> Item:
    qty = 1
    for rx in _QTY:
        m = rx.search(text)
        if m:
            qty, text = int(m.group(1)), text[:m.start()] + " " + text[m.end():]
            break
    if not 1 <= qty <= MAX_QTY:
        raise ValueError(f"jumlah {qty} tidak masuk akal (1-{MAX_QTY})")
    return Item(parse_request(text.strip(" ,."), catalog), qty)


def parse_order(text: str, catalog=None) -> list[Item]:
    """Satu baris pesanan -> daftar barang. Satu barang tidak jelas -> ValueError untuk seluruh baris (tidak ada
    barang yang diam-diam terlewat)."""
    parts = [p for p in _SPLIT.split(text or "") if p and p.strip(" ,.")]
    if not parts:
        raise ValueError("permintaan kosong")
    return [parse_item(p, catalog) for p in parts]


def order_command(text: str, catalog=None) -> tuple[str, list[Item]]:
    """-> ("replace" | "add" | "skip", barang)."""
    t = (text or "").strip()
    if t.lower() in _SKIP:
        return "skip", []
    m = re.match(r"^(?:\+|tambahkan\b|tambah\b)\s*", t, re.I)
    if m:
        return "add", parse_order(t[m.end():], catalog)
    return "replace", parse_order(t, catalog)


class OrderQueue:
    """Barang yang masih menunggu setelah barang yang sedang dicari, plus hasil tiap barang yang sudah selesai."""

    def __init__(self, items=()):
        self.pending: deque[Item] = deque(items)
        self.current: Item | None = None
        self.results: list[tuple[str, int, int, str]] = []    # (barang, diminta, terambil, akhir)

    def start_next(self) -> Item | None:
        self.current = self.pending.popleft() if self.pending else None
        return self.current

    def finish(self, taken: int, outcome: str) -> None:
        if self.current is not None:
            self.results.append((str(self.current.req), self.current.qty, taken, outcome))

    def replace(self, items) -> None:
        self.pending = deque(items)

    def add(self, items) -> None:
        self.pending.extend(items)

    def describe(self) -> str:
        now = f"sekarang: {self.current}" if self.current is not None else "tidak ada yang dicari"
        return now + (" | berikutnya: " + ", ".join(str(i) for i in self.pending) if self.pending else "")
