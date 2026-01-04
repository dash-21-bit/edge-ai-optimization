from __future__ import annotations

import torch
from edge_opt.model import SmallCNN
from edge_opt.optimize import apply_unstructured_pruning, remove_pruning_reparam


def main() -> None:
    model = SmallCNN()
    model.load_state_dict(torch.load("models/baseline_fp32.pt", map_location="cpu"))

    # Prune 30% of weights
    model = apply_unstructured_pruning(model, amount=0.30)

    # Make pruning permanent
    model = remove_pruning_reparam(model)

    torch.save(model.state_dict(), "models/pruned_fp32.pt")
    print("[DONE] Saved pruned model -> models/pruned_fp32.pt")


if __name__ == "__main__":
    main()
