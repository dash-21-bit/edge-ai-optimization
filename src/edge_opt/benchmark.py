from __future__ import annotations
import os
import time
import torch
import numpy as np


def file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


@torch.no_grad()
def benchmark_pytorch(model: torch.nn.Module, device: torch.device, iters: int = 200) -> float:
    """
    Returns average inference latency (ms) on dummy input.
    """
    model.to(device)
    model.eval()

    x = torch.randn(1, 1, 28, 28).to(device)

    # Warmup
    for _ in range(20):
        _ = model(x)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = model(x)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return float(np.mean(times))
