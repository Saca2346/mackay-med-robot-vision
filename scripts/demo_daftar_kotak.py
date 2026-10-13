"""
Numbered list of the boxes on the shelf, for the demo menu (DEMO.bat): no need to remember or open another file.

  python scripts/demo_daftar_kotak.py --inventaris results/demo_20261012/inventaris_demo.csv          # print the table
  python scripts/demo_daftar_kotak.py --inventaris ... --resolve 12     # prints the request text of number 12
  python scripts/demo_daftar_kotak.py --inventaris ... --resolve "angiolite 4 x 19"   # not a number: printed unchanged

Identical boxes (same brand, product, size) share one number, because they are one request. The numbers do not change
while boxes are taken, so number 5 is always the same request. Taken boxes are marked with * (menu R resets them).
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.box_inventory import STATUS_DIAMBIL  # noqa: E402

# What happened at shelf distance in the camera tests of 10-13 Oct (results/bukti_laporan_20261008, results/demo_20261012).
# Only boxes with a measured PICK are listed; every other product stayed at candidate / unsure (second source unreadable).
MEASURED_PICK = {"K06": "10 Oct", "K10": "10 Oct", "K15": "13 Oct"}


def groups(path: pathlib.Path) -> list[dict]:
    """One entry per box (numbers 1..N in box-ID order). Identical boxes (same brand, product, size) also get the
    printed expiry date in their text, so a number asks for THAT box; plus one entry per group of identical boxes
    for "any of them" (FEFO), shown as number 0 when there is only one such group."""
    rows = sorted(csv.DictReader(open(path, encoding="utf-8")), key=lambda r: int(r["box_id"][1:]))
    key = lambda r: (r["brand"], r["product"], r["diameter"], r["length"])
    twins: dict[tuple, list] = {}
    for r in rows:
        twins.setdefault(key(r), []).append(r)
    lst = []
    for n, r in enumerate(rows, 1):
        size = f" {r['diameter']} x {r['length']}" if r["diameter"] and r["length"] else ""
        many = len(twins[key(r)]) > 1
        lst.append({"no": str(n), "text": f"{r['brand']} {r['product']}{size}" + (f" {r['expiry']}" if many and r["expiry"] else ""),
                    "boxes": [{"id": r["box_id"], "expiry": r["expiry"], "taken": r["status"] == STATUS_DIAMBIL}],
                    "twin": many, "generic": False})
    gen = [g for g in twins.values() if len(g) > 1]
    for i, g in enumerate(gen):
        r = g[0]
        size = f" {r['diameter']} x {r['length']}" if r["diameter"] and r["length"] else ""
        lst.append({"no": "0" if len(gen) == 1 else str(len(rows) + 1 + i), "text": f"{r['brand']} {r['product']}{size}",
                    "boxes": [{"id": b["box_id"], "expiry": b["expiry"], "taken": b["status"] == STATUS_DIAMBIL}
                              for b in sorted(g, key=lambda b: b["expiry"] or "9999")],
                    "twin": True, "generic": True})
    return lst


def show(lst: list[dict]) -> None:
    print()
    print(" BOXES ON THE SHELF - type the NUMBER at the Request prompt (or the text in the second column)")
    print(" " + "-" * 100)
    print(f" {'No':>2}  {'Type this':<48} {'Box':<20} Result at shelf distance")
    print(" " + "-" * 100)
    for g in sorted(lst, key=lambda g: int(g["no"])):
        ids = " ".join(("*" if b["taken"] else "") + b["id"] for b in g["boxes"])
        b0 = g["boxes"][0]
        if not g["twin"] and b0["expiry"]:
            ids += f" exp {b0['expiry']}"
        hit = [MEASURED_PICK[b["id"]] for b in g["boxes"] if b["id"] in MEASURED_PICK]
        if g["generic"]:
            res = f"any of the {len(g['boxes'])}: FEFO, earliest date first"
        elif g["twin"]:
            res = "PICK only after THIS date is read"
        else:
            res = f"PICK reached ({hit[0]})" if hit else "candidate only"
        if g["generic"]:
            ids = f"{len(g['boxes'])} boxes"
        print(f" {g['no']:>2}  {g['text']:<48} {ids:<20} {res}")
    print(" " + "-" * 100)
    print(" candidate only = the 2nd source (REF / barcode) is not read at shelf distance (tests 10-13 Oct).")
    if any(g["twin"] and not g["generic"] for g in lst):
        print(" Identical boxes (Sled Bag): a dated number asks for THAT box. The printed date is not readable at shelf")
        print(" distance, so the camera must be brought close to the date before it can say PICK (otherwise candidate).")
    print("\n Scenes without a camera search (answer is NOT FOUND / REPEAT):")
    print("   terumo accuforce 2.75 x 15      -> NOT FOUND (not in stock)")
    print("   xperience pro 3.0 x 18          -> NOT FOUND (that size is not in stock)")
    print("   essential pro 3                 -> REPEAT (diameter AND length are needed)")
    if any(b["taken"] for g in lst for b in g["boxes"]):
        print("\n * = taken in this demo. Menu R resets the list.")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventaris", default="data/rak/inventaris.csv")
    ap.add_argument("--resolve", default=None, help="number -> request text; anything else is printed unchanged")
    ap.add_argument("--rencana-out", default=None, help="write a stress-test plan: one request per box, box-ID order")
    args = ap.parse_args()
    lst = groups(pathlib.Path(args.inventaris))
    if args.rencana_out:
        boxes = sorted((g for g in lst if not g["generic"]), key=lambda g: int(g["no"]))
        lines = [f"# Stress test: every box in turn ({len(boxes)} requests, box-ID order)"]
        lines += [f"{g['text']}   # {g['boxes'][0]['id']}"
                  + (" - identical boxes: the printed date must be read, bring the camera close" if g["twin"] else "")
                  for g in boxes]
        pathlib.Path(args.rencana_out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"plan with {len(boxes)} requests -> {args.rencana_out}")
        return
    if args.resolve is not None:
        s = args.resolve.strip()
        if s.isdigit():
            hit = [g for g in lst if g["no"] == str(int(s))]
            if not hit:
                sys.exit(f"No number {s}: choose a number from the list or type the text")
            print(hit[0]["text"])
        else:
            print(s)
        return
    show(lst)


if __name__ == "__main__":
    main()
