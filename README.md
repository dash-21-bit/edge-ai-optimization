# ai-optimization
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
