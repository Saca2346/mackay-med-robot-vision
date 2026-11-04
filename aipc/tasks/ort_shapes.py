"""Cost of NEW input shapes on the CUDA provider: rec and cls models, cudnn_conv_algo_search options."""
import os
import time

import numpy as np
import onnxruntime as o

o.preload_dlls()
import rapidocr  # noqa: E402

M = os.path.join(os.path.dirname(rapidocr.__file__), "models")
rng = np.random.default_rng(0)
for model, h in (("PP-OCRv6_rec_small.onnx", 48), ("ch_ppocr_mobile_v2.0_cls_mobile.onnx", 48)):
    for algo in ("EXHAUSTIVE", "HEURISTIC", "DEFAULT"):
        s = o.InferenceSession(os.path.join(M, model), providers=[("CUDAExecutionProvider",
                               {"cudnn_conv_algo_search": algo}), "CPUExecutionProvider"])
        name = s.get_inputs()[0].name
        fixed_w = 192 if "cls" in model else None
        new, seen = [], []
        for w in list(range(100, 900, 37)) * 2:
            ww = fixed_w or w
            x = rng.standard_normal((5, 3, h, ww)).astype(np.float32)
            t = time.perf_counter()
            s.run(None, {name: x})
            (seen if len(new) >= 22 else new).append((time.perf_counter() - t) * 1000)
        print(f"{model[:24]:24s} {algo:10s} first time a width is seen {np.median(new):6.1f} ms, "
              f"same width again {np.median(seen):6.1f} ms", flush=True)
