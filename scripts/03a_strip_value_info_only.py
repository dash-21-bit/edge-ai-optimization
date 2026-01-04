from __future__ import annotations

import onnx


def main() -> None:
    in_path = "models/baseline_fp32.onnx"
    out_path = "models/baseline_fp32_valueinfo_stripped.onnx"

    model = onnx.load(in_path)
    graph = model.graph

    # Remove ONLY intermediate value_info (where conflicts usually live)
    graph.value_info.clear()

    # Keep graph.input and graph.output as-is (do NOT clear their shapes)
    onnx.save(model, out_path)
    print(f"[DONE] Saved value_info-stripped ONNX -> {out_path}")


if __name__ == "__main__":
    main()

