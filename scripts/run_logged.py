"""Run a command with inherited console input and a flushed stdout/stderr transcript."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def run_logged(command, logfile):
    logfile = Path(logfile)
    logfile.parent.mkdir(parents=True, exist_ok=True)
    command = list(command)
    if Path(command[0]).is_file():
        command[0] = str(Path(command[0]).resolve())
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    with logfile.open("a", encoding="utf-8") as log:
        def emit(text):
            log.write(text)
            log.flush()
            sys.stdout.write(text)
            sys.stdout.flush()
        emit(f"[session] UTC {datetime.now(timezone.utc).isoformat()}\n")
        emit(f"[session] cwd: {Path.cwd()}\n[session] command: {json.dumps(command)}\n")
        emit(f"[session] CMD transcript: {logfile.resolve()}\n")
        try:
            # Do not pipe stdin: interactive target requests must still reach the child console.
            child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace", env=env)
        except OSError as exc:
            emit(f"[session] launch failed: {exc}\n[session] exit_code: 1\n")
            return 1
        try:
            for line in child.stdout:
                emit(line)
            code = child.wait()
        except KeyboardInterrupt:
            emit("[session] interrupted; waiting for camera/video cleanup\n")
            try:
                code = child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.terminate()
                code = child.wait()
                emit("[session] forced stop; video may be incomplete\n")
            for line in child.stdout:
                emit(line)
        finally:
            child.stdout.close()
        emit(f"[session] UTC {datetime.now(timezone.utc).isoformat()} exit_code: {code}\n")
        return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    return run_logged(command, args.log)


if __name__ == "__main__":
    raise SystemExit(main())
