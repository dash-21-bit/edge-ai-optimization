from __future__ import annotations

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from edge_opt.model import SmallCNN
from edge_opt.train import train_model


def main() -> None:
    device = torch.device("cpu")

    # Transform: convert image to tensor [0..1]
    tfm = transforms.Compose([transforms.ToTensor()])

    # Dataset auto-downloads into ./data
    ds = datasets.FashionMNIST(root="data", train=True, download=True, transform=tfm)

    # Split into train/val
    train_size = int(0.9 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    model = SmallCNN()

    res = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=5,
        lr=1e-3,
        save_path="models/baseline_fp32.pt",
    )

    print(f"[DONE] best_val_acc={res.best_acc:.4f} saved={res.best_path}")


if __name__ == "__main__":
    main()
