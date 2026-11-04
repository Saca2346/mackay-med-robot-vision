"""Det / cls / rec time inside one RapidOCR call on a real spine crop (Task 1 box 2), GPU and CPU readers."""
import sys
import time

sys.argv += ["--imgsz", "544x960", "--workers", "1", "--tag", "profile"]
import aipc_tasks as A  # noqa: E402
from src.box_evidence import _cap, _tiles  # noqa: E402

frame = A.cv2.imread(A.SETS["t1_drive"][0])
g = A.boxes(frame)[2]
sl, box, far = A.prepare_crops(frame, g, A.cfg)
spine = _tiles(_cap(box))[0]
import ocr_worker  # noqa: E402

for cuda, cls_cpu in ((True, False), (True, True), (False, False)):
    r = A.SizeTextReader(retry_flip_below=0.8, threads=2, use_cuda=cuda, backend="onnxruntime")
    if cls_cpu:
        ocr_worker.cls_on_cpu(r._engine)
    for name, img in (("size label", sl[0] if sl else spine), ("spine tile", spine)):
        for _ in range(3):
            r._engine(img)
        el = []
        t0 = time.perf_counter()
        for _ in range(10):
            out = r._engine(img)
            el.append(getattr(out, "elapse_list", None))
        ms = (time.perf_counter() - t0) / 10 * 1000
        det, cls, rec = [round(1000 * sum(e[i] for e in el) / len(el), 1) for i in range(3)] if el[0] else (0, 0, 0)
        print(f"cuda={cuda} cls_cpu={cls_cpu} {name} {img.shape[:2]}: {ms:.0f} ms per call (det {det}, cls {cls}, rec {rec} ms), "
              f"{len(out.txts) if out.txts else 0} texts", flush=True)
