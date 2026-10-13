"""
Screen recorder for a test session: the whole desktop (cmd window + camera window) from start to end.

  python scripts/rekam_layar.py --out results/sesi_X/00_layar_penuh.mp4 --stop-file results/sesi_X/STOP_REC

Runs until the stop file appears (DEMO.bat creates it at the end of the session), then stops ffmpeg cleanly with "q"
so the video is finalised. Uses Intel Quick Sync (h264_qsv: about 18 % of one CPU core, measured 14 Oct) and falls
back to x264 ultrafast if Quick Sync is not available. 8 frames per second, 1280 px wide: enough to read the terminal.
ffmpeg comes from the imageio-ffmpeg package in the venv (pip install imageio-ffmpeg).
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        sys.exit("ffmpeg missing: venv\\Scripts\\python.exe -m pip install imageio-ffmpeg")


def start(exe: str, out: pathlib.Path, log, encoder: list[str]) -> subprocess.Popen:
    cmd = [exe, "-hide_banner", "-loglevel", "warning", "-f", "gdigrab", "-framerate", "8", "-i", "desktop",
           "-vf", "scale=1280:-2,format=nv12", "-c:v", *encoder, "-movflags", "+frag_keyframe+empty_moov", "-y", str(out)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=log, stderr=log)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stop-file", required=True)
    args = ap.parse_args()
    out, stop = pathlib.Path(args.out), pathlib.Path(args.stop_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    stop.unlink(missing_ok=True)
    exe = ffmpeg_exe()
    log = open(out.with_suffix(".ffmpeg.log"), "w", encoding="utf-8")
    p = start(exe, out, log, ["h264_qsv", "-global_quality", "28"])
    time.sleep(2.0)
    if p.poll() is not None:                       # Quick Sync not available: CPU encoder instead
        print("Quick Sync not available, recording with x264 (more CPU)")
        p = start(exe, out, log, ["libx264", "-preset", "ultrafast", "-crf", "30"])
    print(f"RECORDING the screen -> {out}  (this window can stay minimised; it stops by itself at the end)")
    t0 = time.time()
    while not stop.exists() and p.poll() is None:
        time.sleep(0.5)
    if p.poll() is None:
        try:
            p.stdin.write(b"q")
            p.stdin.flush()
            p.wait(timeout=20)
        except Exception:  # noqa: BLE001 - last resort, the fragmented MP4 stays playable
            p.kill()
    stop.unlink(missing_ok=True)
    print(f"recording stopped after {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
