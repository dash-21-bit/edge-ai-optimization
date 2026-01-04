from __future__ import annotations

import torch
import torch.nn.utils.prune as prune
import torch.nn as nn


def apply_unstructured_pruning(model: nn.Module, amount: float = 0.3) -> nn.Module:
    """
    Unstructured pruning: sets a % of weights to zero in selected layers.
    amount=0.3 => 30% weights pruned.
    """
    # Prune weights of conv and linear layers
    layers_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layers_to_prune.append((module, "weight"))

    for (module, param_name) in layers_to_prune:
        prune.l1_unstructured(module, name=param_name, amount=amount)

    return model


def remove_pruning_reparam(model: nn.Module) -> nn.Module:
    """
    Makes pruning permanent by removing pruning hooks/reparameterization.
    """
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            try:
                prune.remove(module, "weight")
            except ValueError:
                pass
    return model


def dynamic_quantize(model: nn.Module) -> nn.Module:
    """
    Dynamic quantization (INT8) for CPU inference.
    On macOS/Apple Silicon, ensure qnnpack is selected.
    """
    # 1) Pick a quantization engine that exists on this machine
    #    qnnpack is commonly available on macOS.
    if "qnnpack" in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = "qnnpack"
    elif torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = torch.backends.quantized.supported_engines[0]

    # 2) Convert Linear layers to dynamic quantized versions
    qmodel = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )
    return qmodel
