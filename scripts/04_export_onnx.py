from __future__ import annotations

import torch
from edge_opt.model import SmallCNN


def main() -> None:
    model = SmallCNN()
    model.load_state_dict(torch.load("models/baseline_fp32.pt", map_location="cpu"))
    model.eval()

    x = torch.randn(1, 1, 28, 28)

    torch.onnx.export(
        model,
        x,
        "models/baseline_fp32.onnx",
        input_names=["input"],
        output_names=["logits"],
        opset_version=18,
    )

    print("[DONE] Exported ONNX -> models/baseline_fp32.onnx")


if __name__ == "__main__":
    main()
