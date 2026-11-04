"""Rec model on CUDA: output size, time of run() with the full output copied to the CPU, and of the numpy argmax."""
import os
import time

import numpy as np
import onnxruntime as o

o.preload_dlls()
import rapidocr  # noqa: E402

M = os.path.join(os.path.dirname(rapidocr.__file__), "models")
s = o.InferenceSession(os.path.join(M, "PP-OCRv6_rec_small.onnx"), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
x = np.random.default_rng(0).standard_normal((5, 3, 48, 320)).astype(np.float32)
name = s.get_inputs()[0].name
for _ in range(3):
    y = s.run(None, {name: x})[0]
t = time.perf_counter()
for _ in range(20):
    y = s.run(None, {name: x})[0]
run_ms = (time.perf_counter() - t) / 20 * 1000
t = time.perf_counter()
for _ in range(20):
    a, m = y.argmax(axis=2), y.max(axis=2)
arg_ms = (time.perf_counter() - t) / 20 * 1000
io = s.io_binding()
io.bind_cpu_input(name, x)
io.bind_output(s.get_outputs()[0].name, "cuda")
for _ in range(3):
    s.run_with_iobinding(io)
t = time.perf_counter()
for _ in range(20):
    s.run_with_iobinding(io)
gpu_only_ms = (time.perf_counter() - t) / 20 * 1000
print(f"output {y.shape} {y.nbytes / 1e6:.1f} MB; run with copy to CPU {run_ms:.1f} ms; output left on GPU "
      f"{gpu_only_ms:.1f} ms; numpy argmax+max {arg_ms:.1f} ms")
