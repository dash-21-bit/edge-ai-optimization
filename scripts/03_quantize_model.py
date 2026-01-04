from __future__ import annotations

import torch
from edge_opt.model import SmallCNN
from edge_opt.optimize import dynamic_quantize


def main() -> None:
    model = SmallCNN()
    model.load_state_dict(torch.load("models/baseline_fp32.pt", map_location="cpu"))
    model.eval()

    qmodel = dynamic_quantize(model)

    # Save full quantized model (not just state_dict)
    torch.save(qmodel, "models/quant_dynamic.pt")
    print("[DONE] Saved dynamic quantized model -> models/quant_dynamic.pt")


if __name__ == "__main__":
    main()
