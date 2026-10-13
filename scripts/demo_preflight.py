"""
Pre-flight check before the demo (about 5 seconds, does not stream the camera).

  python scripts/demo_preflight.py

Checks: model file, config, inventory (20 verified boxes, none marked taken), the demo position table, free disk
space, free RAM, and whether a RealSense is connected and on USB 3. Prints OK / WARN / FAIL per line; exit code 1 on FAIL.
"""
from __future__ import annotations

import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

results: list[tuple[str, str, str]] = []


def add(status: str, what: str, detail: str = "") -> None:
    results.append((status, what, detail))
    print(f"{status:<5} {what:<34} {detail}")


def main() -> int:
    from src.box_inventory import STATUS_DIAMBIL, STATUS_OK, Inventory
    from src.config import load_config, resolve

    try:
        cfg = load_config(resolve("config/box_pipeline_config_realsense.yaml"))
        w = resolve(cfg["model"]["weights"])
        add("OK" if w.exists() else "FAIL", "model weights", f"{w.name} {cfg['model']['imgsz']}"
            + ("" if w.exists() else "  <- missing: models/box_v4s_960_openvino_model"))
        add("OK" if cfg["camera"].get("backend") == "realsense" else "WARN", "camera backend",
            f"{cfg['camera'].get('backend')} {cfg['camera'].get('width')}x{cfg['camera'].get('height')}")
    except Exception as e:  # noqa: BLE001 - report, keep checking
        add("FAIL", "config", f"{type(e).__name__}: {e}")

    inv_path = ROOT / "data" / "rak" / "inventaris.csv"
    if inv_path.exists():
        inv = Inventory(inv_path)
        ok = sum(1 for b in inv.boxes.values() if b.status == STATUS_OK)
        taken = sum(1 for b in inv.boxes.values() if b.status == STATUS_DIAMBIL)
        add("OK" if ok == 20 and not taken else "WARN", "inventory", f"{ok} verified, {taken} taken"
            + ("" if not taken else "  <- run: scripts\\rak_inventaris.py kembali Kxx"))
    else:
        add("FAIL", "inventory", "data/rak/inventaris.csv missing")
    pos = ROOT / "data" / "rak" / "posisi.csv"
    add("OK" if pos.exists() else "WARN", "position table", "present (EXAMPLE until the team sends the real one)"
        if pos.exists() else "missing: positions will read 'unknown'")

    free = shutil.disk_usage(ROOT).free / 1e9
    add("OK" if free >= 5 else "WARN", "free disk", f"{free:.0f} GB (a 2-minute take is ~0.3-0.8 GB)")
    try:
        import psutil
        ram = psutil.virtual_memory().available / 1e9
        add("OK" if ram >= 2.5 else "WARN", "free RAM", f"{ram:.1f} GB" + ("" if ram >= 2.5 else "  <- close VS Code / browser"))
    except ImportError:
        add("INFO", "free RAM", "psutil not installed: check Task Manager (close VS Code and the browser)")

    try:
        import pyrealsense2 as rs
        devs = list(rs.context().devices)
        if not devs:
            add("FAIL", "RealSense", "no device: plug the D455 into a USB 3 port")
        for d in devs:
            usb = d.get_info(rs.camera_info.usb_type_descriptor) if d.supports(rs.camera_info.usb_type_descriptor) else "?"
            add("OK" if usb.startswith("3") else "FAIL", "RealSense " + d.get_info(rs.camera_info.name), f"USB {usb}"
                + ("" if usb.startswith("3") else "  <- USB 2 gives no colour frames: use a USB 3 port / cable"))
    except Exception as e:  # noqa: BLE001
        add("FAIL", "RealSense", f"{type(e).__name__}: {e}")

    bad = [r for r in results if r[0] == "FAIL"]
    warn = [r for r in results if r[0] == "WARN"]
    print(f"\n{'READY' if not bad else 'NOT READY'}: {len(bad)} FAIL, {len(warn)} WARN")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
