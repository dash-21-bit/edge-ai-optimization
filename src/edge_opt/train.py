from __future__ import annotations

from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
import torch.nn as nn


@dataclass
class TrainResult:
    best_acc: float
    best_path: str


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    save_path: str,
) -> TrainResult:
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        accs = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                accs.append(accuracy(logits, yb))

        val_acc = sum(accs) / len(accs)

        print(f"[EPOCH {epoch}] train_loss={total_loss/len(train_loader):.4f} val_acc={val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"[SAVE] best model -> {save_path}")

    return TrainResult(best_acc=best_acc, best_path=save_path)
