from __future__ import annotations

import time
import numpy as np
import onnxruntime as ort


def benchmark(session: ort.InferenceSession, iters: int = 300) -> float:
    """
    Measure average inference latency (ms) for an ONNX Runtime session.
    """
    # Dummy input matching Fashion-MNIST shape
    x = np.random.randn(1, 1, 28, 28).astype(np.float32)
    input_name = session.get_inputs()[0].name

    # Warm-up runs (important for fair timing)
    for _ in range(30):
        session.run(None, {input_name: x})

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        session.run(None, {input_name: x})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    return float(np.mean(times))


def main() -> None:
    providers = ["CPUExecutionProvider"]

    # FP32 model (shape-stripped)
    fp32_sess = ort.InferenceSession(
        "models/baseline_fp32_valueinfo_stripped.onnx",
        providers=providers,
    )

    # INT8 quantized model
    int8_sess = ort.InferenceSession(
        "models/baseline_int8.onnx",
        providers=providers,
    )

    fp32_ms = benchmark(fp32_sess)
    int8_ms = benchmark(int8_sess)

    print(f"[ONNXRuntime] FP32 avg latency: {fp32_ms:.3f} ms")
    print(f"[ONNXRuntime] INT8 avg latency: {int8_ms:.3f} ms")


if __name__ == "__main__":
    main()







