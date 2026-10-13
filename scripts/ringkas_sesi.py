"""
Session report: one readable page from everything a recorded test session left in its folder.

  python scripts/ringkas_sesi.py results/sesi_20261014_101500

Writes LAPORAN_SESI.md (English) in that folder: steps done, calibration verdict, stress-test table (result per box,
time, evidence), random requests, and the list of files (screen recording, annotated videos, logs).
Only reports what the files say; nothing is re-measured here.
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys


def table(rows: list[list], head: list[str]) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    out += ["| " + " | ".join(str(c).replace("|", "/") for c in r) + " |" for r in rows]
    return out


def main() -> None:
    sd = pathlib.Path(sys.argv[1])
    md = [f"# Test session {sd.name}", "", "Every number below comes from the files in this folder.", ""]

    steps = sd / "langkah.txt"
    if steps.exists():
        md += ["## Steps (as logged by DEMO.bat)", "", "```", steps.read_text(encoding="utf-8").strip(), "```", ""]

    cal = sorted(sd.glob("*_kalibrasi.json"))
    if cal:
        c = json.loads(cal[0].read_text(encoding="utf-8"))
        md += ["## Camera calibration", "", f"Verdict: **{c.get('verdict', c.get('hasil', '?'))}**", ""]
        checks = c.get("checks") or c.get("cek") or []
        if checks and isinstance(checks, list) and isinstance(checks[0], dict):
            md += table([[k.get("check", ""), k.get("value", k.get("nilai", "")), k.get("status", ""),
                          k.get("advice", k.get("saran", ""))] for k in checks], ["Check", "Value", "Status", "Advice"])
            md.append("")

    stress = sd / "05_stress.csv"
    if stress.exists():
        rows = list(csv.DictReader(open(stress, encoding="utf-8")))
        ok = [r for r in rows if r.get("hasil") in ("PICK", "TAKEN")]
        md += ["## Stress test (every box in turn)", "",
               f"{len(ok)} of {len(rows)} requests reached PICK/TAKEN. Results: "
               + ", ".join(f"{k} {sum(1 for r in rows if r.get('hasil') == k)}"
                           for k in ("TAKEN", "PICK", "NOT FOUND", "TIMEOUT") if any(r.get("hasil") == k for r in rows)), ""]
        md += table([[r.get("no"), r.get("permintaan"), r.get("catatan"), r.get("hasil"), r.get("detik"), r.get("kotak"),
                      r.get("bukti"), r.get("cek_ulang"), r.get("batal")] for r in rows],
                    ["No", "Request", "Box", "Result", "Seconds", "Track", "Evidence", "Rechecks", "Cancelled"])
        md.append("")

    demo = sd / "demo_log.jsonl"
    if demo.exists():
        lines = [json.loads(x) for x in demo.read_text(encoding="utf-8").splitlines() if x.strip()]
        md += ["## Random requests (typed or picked by number)", ""]
        md += table([[x["time"][11:19], x["request_text"], x["status"], x.get("box_id", ""), x.get("expiry", ""),
                      x.get("reason", "")[:70]] for x in lines], ["Time", "Request", "Status", "Box", "Expiry", "Note"])
        md.append("")

    md += ["## Files", ""]
    for f in sorted(sd.rglob("*")):
        if f.is_file() and f.name != "LAPORAN_SESI.md":
            md.append(f"- `{f.relative_to(sd)}` ({f.stat().st_size / 1e6:.1f} MB)")
    (sd / "LAPORAN_SESI.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"report -> {sd / 'LAPORAN_SESI.md'}")


if __name__ == "__main__":
    main()
