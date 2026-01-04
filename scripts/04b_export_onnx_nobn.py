from __future__ import annotations

import torch

from edge_opt.model import SmallCNN
from edge_opt.model_nobn import SmallCNNNoBN


def main() -> None:
    # 1) Load your trained baseline model (with BN) weights
    src = SmallCNN()
    src.load_state_dict(torch.load("models/baseline_fp32.pt", map_location="cpu"))
    src.eval()

    # 2) Create NoBN model
    dst = SmallCNNNoBN()

    # 3) Copy weights for layers that exist in both models
    #    (conv1, conv2, fc1, fc2). BN layers are ignored.
    dst.load_state_dict(
        {
            "conv1.weight": src.conv1.weight,
            "conv1.bias": src.conv1.bias,
            "conv2.weight": src.conv2.weight,
            "conv2.bias": src.conv2.bias,
            "fc1.weight": src.fc1.weight,
            "fc1.bias": src.fc1.bias,
            "fc2.weight": src.fc2.weight,
            "fc2.bias": src.fc2.bias,
        },
        strict=False,
    )
    dst.eval()

    # 4) Dummy input for tracing the graph
    x = torch.randn(1, 1, 28, 28)

    # 5) Export to ONNX (use opset 18 to avoid version conversion)
    torch.onnx.export(
        dst,
        x,
        "models/baseline_nobn_fp32.onnx",
        input_names=["input"],
        output_names=["logits"],
        opset_version=18,
    )

    print("[DONE] Exported ONNX (NoBN) -> models/baseline_nobn_fp32.onnx")


if __name__ == "__main__":
    main()
