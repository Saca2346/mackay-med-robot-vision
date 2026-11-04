"""One OCR process of the AI PC reader (aipc_tasks.py --procs N): its own RapidOCR engine on the GPU, so N processes
run OCR at the same time without sharing Python's interpreter lock. Same engine settings as src/box_ocr.py."""
import os
import sys

_R = None


def cls_on_cpu(engine, threads=2):
    """The text-direction classifier (same model file, same session options) on the CPU: on the A5000 it took ~100 ms
    per call through the CUDA provider against ~6 ms on the CPU (profile_ocr.py, 4 Nov). Det and rec stay on the GPU."""
    import onnxruntime as o
    import rapidocr
    so = o.SessionOptions()
    so.log_severity_level = 4
    so.enable_cpu_mem_arena = False
    so.graph_optimization_level = o.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
    path = os.path.join(os.path.dirname(rapidocr.__file__), "models", "ch_ppocr_mobile_v2.0_cls_mobile.onnx")
    engine.text_cls.session.session = o.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def init(code_dir, device="cuda", threads=2, cls_cpu=False):
    global _R
    sys.path.insert(0, code_dir)
    os.chdir(code_dir)
    import onnxruntime
    onnxruntime.preload_dlls()
    import rapidocr

    class _Arena(rapidocr.RapidOCR):
        """GPU memory arena grown by what is asked, not to the next power of two: 24 processes reached 23 GB of the
        shared 24 GB card (4 Nov evening). Allocation only; the computation is the same."""
        def __init__(self, params=None, **kw):
            p = dict(params or {})
            if p.get("EngineConfig.onnxruntime.use_cuda") and os.environ.get("OCR_ARENA", "same") == "same":
                p["EngineConfig.onnxruntime.cuda_ep_cfg.arena_extend_strategy"] = "kSameAsRequested"
            super().__init__(params=p, **kw)

    rapidocr.RapidOCR = _Arena
    from src.box_ocr import SizeTextReader
    _R = SizeTextReader(retry_flip_below=0.8, threads=threads, use_cuda=device == "cuda", backend="onnxruntime")
    if cls_cpu and device == "cuda":
        cls_on_cpu(_R._engine, threads)


def run(img, use_cls=True):
    """-> [(text, conf, poly)] exactly as SizeTextReader._run returns them."""
    return [(t.text, t.conf, t.poly) for t in _R._run(img, use_cls)]


def ping(_=None):
    return os.getpid()
