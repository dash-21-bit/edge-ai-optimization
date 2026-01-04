from __future__ import annotations

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


def main() -> None:
    # Use the stripped model if you created it; otherwise you can point to baseline_fp32.onnx
    fp32_path = "models/baseline_fp32_valueinfo_stripped.onnx"
    int8_path = "models/baseline_int8.onnx"

    # Quantize only MatMul/Gemm first (safe and usually gives size+speed gains)
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
        extra_options={
            # If ORT can't infer some tensor types, assume FLOAT by default
            "DefaultTensorType": onnx.TensorProto.FLOAT
        },
    )

    print(f"[DONE] Quantized ONNX saved -> {int8_path}")


if __name__ == "__main__":
    main()
