"""
Pre-flight check before the demo (briefly opens the configured webcam, no preview or recording).

  python scripts/demo_preflight.py
  python scripts/demo_preflight.py --config results/sesi_X/config_sesi.yaml --inventaris results/sesi_X/inventaris_sesi.csv

Checks: model file, config, inventory (20 verified boxes, none marked taken), the demo position table, free disk
space, free RAM, and the configured camera. USB webcams need a readable colour frame; the optional RealSense profile
checks its USB connection. Prints OK / WARN / FAIL per line; exit code 1 on FAIL.
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

results: list[tuple[str, str, str]] = []


def add(status: str, what: str, detail: str = "") -> None:
    results.append((status, what, detail))
    print(f"{status:<5} {what:<34} {detail}")


def check_camera(camera: dict) -> None:
    backend = str(camera.get("backend", "msmf")).lower()
    if backend in ("msmf", "dshow", "any"):
        from src.box_camera import open_camera
        cam = None
        try:
            cam, index, size = open_camera(camera)
            if cam is None or size is None:
                add("FAIL", "USB webcam", "no readable frame: connect the external webcam, close camera apps, "
                    "and check camera.index / backend in the webcam config")
                return
            width, height = size
            wanted = (int(camera.get("width", 1920)), int(camera.get("height", 1080)))
            low = width < wanted[0] or height < wanted[1]
            add("WARN" if low else "OK", "USB webcam", f"camera #{index}, {width}x{height} via {backend}"
                + ("; below requested resolution: check that this is the external webcam" if low else ""))
            add("INFO", "grasp output", "pixel coordinates only; a normal webcam supplies no metric depth")
        except Exception as exc:  # noqa: BLE001 - report camera failure in the preflight summary
            add("FAIL", "USB webcam", f"{type(exc).__name__}: {exc}")
        finally:
            if cam is not None:
                cam.release()
        return
    if backend != "realsense":
        add("FAIL", "camera backend", f"unsupported: {backend}; choose msmf, dshow, any or realsense")
        return
    try:
        import pyrealsense2 as rs
        devs = list(rs.context().devices)
        if not devs:
            add("FAIL", "RealSense", "no device: plug the D455 into a USB 3 port")
        for d in devs:
            usb = d.get_info(rs.camera_info.usb_type_descriptor) if d.supports(rs.camera_info.usb_type_descriptor) else "?"
            add("OK" if usb.startswith("3") else "FAIL", "RealSense " + d.get_info(rs.camera_info.name), f"USB {usb}"
                + ("" if usb.startswith("3") else "  <- use a USB 3 port / cable"))
    except Exception as exc:  # noqa: BLE001
        add("FAIL", "RealSense", f"{type(exc).__name__}: {exc}")


def main(argv=None) -> int:
    from src.box_inventory import STATUS_DIAMBIL, STATUS_OK, Inventory
    from src.config import load_config, resolve

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/box_pipeline_config_webcam.yaml")
    parser.add_argument("--inventaris", default="data/rak/inventaris.csv")
    args = parser.parse_args(argv)
    results.clear()
    camera = None

    try:
        config_path = resolve(args.config)
        cfg = load_config(config_path)
        add("OK", "config", str(config_path))
        w = resolve(cfg["model"]["weights"])
        add("OK" if w.exists() else "FAIL", "model weights", f"{w.name} {cfg['model']['imgsz']}"
            + ("" if w.exists() else f"  <- missing: {w}"))
        camera = cfg["camera"]
        add("INFO", "camera backend",
            f"{camera.get('backend')} {camera.get('width')}x{camera.get('height')}")
    except Exception as e:  # noqa: BLE001 - report, keep checking
        add("FAIL", "config", f"{type(e).__name__}: {e}")

    inv_path = resolve(args.inventaris)
    if inv_path.exists():
        inv = Inventory(inv_path)
        ok = sum(1 for b in inv.boxes.values() if b.status == STATUS_OK)
        taken = sum(1 for b in inv.boxes.values() if b.status == STATUS_DIAMBIL)
        add("OK" if ok == 20 and not taken else "WARN", "inventory", f"{ok} verified, {taken} taken; {inv_path}"
            + ("" if not taken else "  <- reset the demo copy with menu R after returning the boxes"))
    else:
        add("FAIL", "inventory", f"{inv_path} missing")
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

    if camera is not None:
        check_camera(camera)

    bad = [r for r in results if r[0] == "FAIL"]
    warn = [r for r in results if r[0] == "WARN"]
    print(f"\n{'READY' if not bad else 'NOT READY'}: {len(bad)} FAIL, {len(warn)} WARN")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
