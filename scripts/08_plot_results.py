from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    df = pd.read_csv("reports/metrics_onnxruntime.csv")

    os.makedirs("reports/figures", exist_ok=True)

    plt.figure()
    plt.scatter(df["size_mb"], df["avg_latency_ms"])

    for _, r in df.iterrows():
        plt.text(r["size_mb"], r["avg_latency_ms"], r["model"])

    plt.xlabel("Model size (MB)")
    plt.ylabel("Avg latency (ms)")
    plt.title("Edge Optimization: Size vs Latency (ONNXRuntime CPU)")

    out_path = "reports/figures/size_vs_latency_onnxruntime.png"
    plt.savefig(out_path, dpi=150)
    print(f"[DONE] Saved -> {out_path}")


if __name__ == "__main__":
    main()
