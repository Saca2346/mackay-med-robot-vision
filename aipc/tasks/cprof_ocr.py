"""cProfile of 10 GPU RapidOCR calls on a real spine tile (Task 1 box 2): which functions take the time."""
import cProfile
import pstats
import sys

sys.argv += ["--imgsz", "544x960", "--workers", "1", "--tag", "profile"]
import aipc_tasks as A  # noqa: E402
from src.box_evidence import _cap, _tiles  # noqa: E402

frame = A.cv2.imread(A.SETS["t1_drive"][0])
sl, box, far = A.prepare_crops(frame, A.boxes(frame)[2], A.cfg)
spine = _tiles(_cap(box))[0]
r = A.SizeTextReader(retry_flip_below=0.8, threads=2, use_cuda=True, backend="onnxruntime")
for _ in range(3):
    r._engine(spine)
pr = cProfile.Profile()
pr.enable()
for _ in range(10):
    r._engine(spine)
pr.disable()
pstats.Stats(pr).sort_stats("cumulative").print_stats(28)
