"""Bounded annotated-video encoding away from the focused camera loop."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2


class ViewRecorder:
    """Keep every submitted frame and its real timestamp; surface encoder errors."""
    def __init__(self, path, fps, max_pending=4, max_width=None):
        self.path = Path(path)
        self.fps = fps
        self.limit = max(1, max_pending)
        self.max_width = max_width
        self.pending = deque()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="target-video")
        self.writer = None
        self.times = []

    def _write(self, frame, seconds):
        if self.max_width and frame.shape[1] > self.max_width:
            height = max(2, round(frame.shape[0] * self.max_width / frame.shape[1] / 2) * 2)
            frame = cv2.resize(frame, (self.max_width, height), interpolation=cv2.INTER_AREA)
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
                                          (frame.shape[1], frame.shape[0]))
            if not self.writer.isOpened():
                raise OSError(f"Could not open annotated video recording: {self.path}")
        self.writer.write(frame)
        self.times.append(seconds)

    def write(self, frame, seconds):
        while self.pending and (self.pending[0].done() or len(self.pending) >= self.limit):
            self.pending.popleft().result()
        self.pending.append(self.executor.submit(self._write, frame.copy(), seconds))

    def close(self):
        self.executor.shutdown(wait=True)
        try:
            while self.pending:
                self.pending.popleft().result()
        finally:
            if self.writer is not None:
                self.writer.release()
        with Path(str(self.path) + ".times.csv").open("w", encoding="utf-8") as log:
            log.write("frame,t\n")
            for i, seconds in enumerate(self.times):
                log.write(f"{i},{seconds:.6f}\n")
