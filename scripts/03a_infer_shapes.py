from __future__ import annotations

import onnx
from onnx import shape_inference


def main() -> None:
    in_path = "models/baseline_fp32.onnx"
    out_path = "models/baseline_fp32_inferred.onnx"

    model = onnx.load(in_path)

    # Run ONNX shape inference in-memory
    inferred = shape_inference.infer_shapes(model)

    # Save inferred model
    onnx.save(inferred, out_path)
    print(f"[DONE] Saved shape-inferred ONNX -> {out_path}")


if __name__ == "__main__":
    main()

