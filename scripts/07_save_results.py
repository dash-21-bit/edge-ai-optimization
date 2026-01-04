from __future__ import annotations

import os
import pandas as pd


def size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def main() -> None:
    fp32_path = "models/baseline_fp32_valueinfo_stripped.onnx"
    int8_path = "models/baseline_int8.onnx"

    # Use the benchmark numbers you got from the terminal
    fp32_latency_ms = 0.031
    int8_latency_ms = 0.039

    df = pd.DataFrame(
        [
            {
                "model": "onnx_fp32",
                "path": fp32_path,
                "size_mb": round(size_mb(fp32_path), 4),
                "avg_latency_ms": fp32_latency_ms,
            },
            {
                "model": "onnx_int8_dynamic_matmul_gemm",
                "path": int8_path,
                "size_mb": round(size_mb(int8_path), 4),
                "avg_latency_ms": int8_latency_ms,
            },
        ]
    )

    os.makedirs("reports", exist_ok=True)
    out_csv = "reports/metrics_onnxruntime.csv"
    df.to_csv(out_csv, index=False)

    print(f"[DONE] Saved -> {out_csv}")
    print(df)


if __name__ == "__main__":
    main()
