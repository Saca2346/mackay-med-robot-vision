"""Paired camera/view photos with status, timing and source provenance."""
from datetime import datetime, timezone
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import cv2


class PhotoEvidence:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run = str(time.time_ns())
        self.sequence = 0
        self._executor = None
        self._pending = deque()

    def submit(self, frame, view, event, metadata):
        """Write away from the detector; bounded backlog, no silently dropped photos."""
        while self._pending and (self._pending[0].done() or len(self._pending) >= 4):
            self._pending.popleft().result()
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="evidence-photo")
        self._pending.append(self._executor.submit(self.save, frame.copy(), view.copy(), event, dict(metadata)))

    def close(self):
        """Finish every queued photo and surface write failures before session success."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        while self._pending:
            self._pending.popleft().result()

    def save(self, frame, view, event, metadata):
        self.sequence += 1
        stem = f"{self.run}_{self.sequence:04d}_{event.lower()}"
        paths = {}
        for name, image in (("camera", frame), ("view", view)):
            path = self.directory / f"{stem}_{name}.png"
            if not cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                raise OSError(f"Could not save evidence photo: {path}")
            paths[name] = path.name
        row = {**metadata, "event": event, "saved_utc": datetime.now(timezone.utc).isoformat(),
               "photos": paths, "photo_use": "current camera/display frame; reader crops are in verification/"}
        (self.directory / f"{stem}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        with (self.directory / "index.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(row) + "\n")
        print(f"[photos] {event}: {self.directory / paths['view']}", flush=True)
        return row
