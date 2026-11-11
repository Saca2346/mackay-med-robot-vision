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


def _models(code_dir):
    return os.path.join(os.path.dirname(os.path.abspath(code_dir)), "ocr_models")


# OCR_VARIANT (6 Nov, the OCR comparison; the decision code is the same for all): the text detector / recogniser files
# given to RapidOCR. v6s = the laptop's (PP-OCRv6 small det + rec, the default); v5det = PaddleOCR PP-OCRv5 server
# detector + v6 small recogniser; v5srv = PP-OCRv5 server det + rec; v5en = PP-OCRv5 server det + PP-OCRv5 English rec.
# Files: ocr_models/ (PaddleOCR's models converted to ONNX, huggingface.co/monkt/paddleocr-onnx). easy / tess: EasyOCR
# 1.7 (CRAFT + English recogniser, GPU) and Tesseract 5 (CPU) in place of RapidOCR.
def _variant_params(code_dir):
    m = _models(code_dir)
    det = {"Det.model_path": os.path.join(m, "v5_server_det.onnx")}
    return {"v6s": {}, "v5det": det,
            "v5srv": dict(det, **{"Rec.model_path": os.path.join(m, "v5_server_rec.onnx"),
                                  "Rec.rec_keys_path": os.path.join(m, "v5_server_dict.txt")}),
            "v5en": dict(det, **{"Rec.model_path": os.path.join(m, "v5_en_rec.onnx"),
                                 "Rec.rec_keys_path": os.path.join(m, "v5_en_dict.txt")})}


def init(code_dir, device="cuda", threads=2, cls_cpu=False):
    global _R
    sys.path.insert(0, code_dir)
    os.chdir(code_dir)
    import onnxruntime
    onnxruntime.preload_dlls()
    import rapidocr
    variant = os.environ.get("OCR_VARIANT", "v6s")
    extra = _variant_params(code_dir).get(variant, {})

    class _Arena(rapidocr.RapidOCR):
        """GPU memory arena grown by what is asked, not to the next power of two: 24 processes reached 23 GB of the
        shared 24 GB card (4 Nov evening). Allocation only; the computation is the same."""
        def __init__(self, params=None, **kw):
            p = dict(params or {}, **extra)
            if p.get("EngineConfig.onnxruntime.use_cuda") and os.environ.get("OCR_ARENA", "same") == "same":
                p["EngineConfig.onnxruntime.cuda_ep_cfg.arena_extend_strategy"] = "kSameAsRequested"
            # 8 Nov: aipc_tasks --algo reached only the main process; these spawned processes kept RapidOCR's default
            # EXHAUSTIVE cuDNN search, which tries every algorithm (big workspaces) for each new crop shape. With 2
            # detector views loaded that ran the shared card out of memory (BFCArena errors, reads lost). OCR_ALGO=
            # HEURISTIC sets it here; unset = as before.
            if p.get("EngineConfig.onnxruntime.use_cuda") and os.environ.get("OCR_ALGO"):
                p["EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search"] = os.environ["OCR_ALGO"]
            super().__init__(params=p, **kw)

    rapidocr.RapidOCR = _Arena
    from src.box_ocr import SizeTextReader
    _R = SizeTextReader(retry_flip_below=0.8, threads=threads, use_cuda=device == "cuda", backend="onnxruntime")
    if cls_cpu and device == "cuda":
        cls_on_cpu(_R._engine, threads)
    if variant in ("easy", "tess"):
        _R._run = _easy_run(device == "cuda", code_dir) if variant == "easy" else _tess_run


def _tokens(items):
    """[(poly, text, conf)] -> OcrTokens in the reading order of SizeTextReader._run (top to bottom, left to right)."""
    import numpy as np
    from src.box_ocr import OcrToken
    items = sorted(items, key=lambda r: (round(np.mean([p[1] for p in r[0]]) / 20), np.mean([p[0] for p in r[0]])))
    return [OcrToken(str(t).strip(), float(c), np.asarray(p, np.float32)) for p, t, c in items if str(t).strip()]


def _easy_run(gpu, code_dir):
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(code_dir)), "pylib_easy"))
    import easyocr
    reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    def run(img, use_cls=True):
        return _tokens([(p, t, c) for p, t, c in reader.readtext(img)])
    return run


def _tess_run(img, use_cls=True):
    import cv2
    import pytesseract
    d = pytesseract.image_to_data(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), config="--psm 11",
                                  output_type=pytesseract.Output.DICT)
    out = []
    for t, c, x, y, w, h in zip(d["text"], d["conf"], d["left"], d["top"], d["width"], d["height"]):
        if str(t).strip() and float(c) >= 0:
            out.append(([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], t, float(c) / 100.0))
    return _tokens(out)


def run(img, use_cls=True):
    """-> [(text, conf, poly)] exactly as SizeTextReader._run returns them (or the OCR_VARIANT engine's tokens).
    7 Nov: an engine error (seen once: onnxruntime BiasSoftmax on the shared GPU) gives NO tokens for that one image
    (nothing read, never a guess) and is logged, instead of ending the whole run (the 12 h v5srv hang of 6-7 Nov)."""
    try:
        return [(t.text, t.conf, t.poly) for t in _R._run(img, use_cls)]
    except Exception as e:                       # noqa: BLE001
        lines = [ln.strip() for ln in str(e).splitlines() if ln.strip()]
        print(f"OCR_ENGINE_ERROR pid {os.getpid()} shape {getattr(img, 'shape', None)}: {type(e).__name__}: "
              f"{(lines[-1] if lines else '')[:400]}", file=sys.stderr, flush=True)
        # 8 Nov: the close-camera recordings gave 13 such errors (wide 800 px crops); one retry of the same image at
        # 0.75 scale (outlines mapped back), logged either way; still nothing read if it fails again
        try:
            import cv2
            small = cv2.resize(img, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
            out = [(t.text, t.conf, t.poly / 0.75) for t in _R._run(small, use_cls)]
            print(f"OCR_ENGINE_RETRY_OK pid {os.getpid()} tokens {len(out)}", file=sys.stderr, flush=True)
            return out
        except Exception as e2:                  # noqa: BLE001
            print(f"OCR_ENGINE_RETRY_FAIL pid {os.getpid()}: {type(e2).__name__}", file=sys.stderr, flush=True)
            return []


def ping(_=None):
    return os.getpid()
