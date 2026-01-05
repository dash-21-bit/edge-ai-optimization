# Edge AI Optimization — Pruning + Quantization + Latency Benchmarks (PyTorch)

**Repo:** edge-ai-optimization  
**Author:** Adarsh Ravi  
**Tech:** Python, PyTorch, TorchVision, ONNX, ONNX Runtime, Matplotlib

## Overview
This project demonstrates practical "edge AI" optimization techniques on a small CNN:
- **Baseline FP32 model**
- **Unstructured pruning** (removing low-importance weights)
- **Dynamic quantization** (INT8 for Linear layers)
- **Model size + CPU latency benchmarking**
- **Accuracy comparison across variants**
- Exports baseline model to **ONNX** for edge-friendly inference pipelines

## Dataset
- **Fashion-MNIST** via TorchVision (auto-downloads into `data/`)

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/01_train_baseline.py
python scripts/02_prune_model.py
python scripts/03_quantize_model.py
python scripts/04_export_onnx.py
python scripts/05_benchmark_all.py
```
## Results (ONNXRuntime CPU)

### Model sizes
- FP32 ONNX: `models/baseline_fp32_valueinfo_stripped.onnx` (~0.80 MB)
- INT8 ONNX: `models/baseline_int8.onnx` (~0.22 MB)

✅ Size reduction: ~72% smaller (818 KB → 226 KB)

### Latency benchmark (dummy input 1×1×28×28)
- FP32 avg latency: **0.031 ms**
- INT8 avg latency: **0.039 ms**

**Notes**
- INT8 is smaller and more edge-friendly for storage and distribution.
- Latency may not improve for very small models because FP32 is already highly optimized and the overhead of INT8 operators can dominate.
- This quantization run targets `MatMul`/`Gemm` ops (FC layers), which is a safe and common first step on CPU.
