from __future__ import annotations

import onnx


def _clear_tensor_shape(type_proto) -> None:
    """
    Remove shape info from a Tensor type (keeps dtype but clears dims).
    """
    if type_proto.HasField("tensor_type") and type_proto.tensor_type.HasField("shape"):
        type_proto.tensor_type.shape.dim.clear()


def main() -> None:
    in_path = "models/baseline_fp32.onnx"
    out_path = "models/baseline_fp32_stripped.onnx"

    model = onnx.load(in_path)
    graph = model.graph

    # 1) Remove intermediate value_info entries (common source of conflicts)
    graph.value_info.clear()

    # 2) Remove shapes from graph inputs/outputs (keep dtype only)
    for v in list(graph.input):
        _clear_tensor_shape(v.type)
    for v in list(graph.output):
        _clear_tensor_shape(v.type)

    # 3) Save cleaned model
    onnx.save(model, out_path)
    print(f"[DONE] Saved shape-stripped ONNX -> {out_path}")


if __name__ == "__main__":
    main()
