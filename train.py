import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


from dataloader import create_dataloaders
from model import (
    CutoffPredictorCNN_GRU,
    CutoffPredictorCNN,
    CutoffPredictorGRU,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0

    for x, Fs, lengths, y, fnames, ratios in loader:
        x = x.to(DEVICE)
        Fs = Fs.to(DEVICE)
        lengths = lengths.to(DEVICE)
        y = y.to(DEVICE)
        ratios = ratios.to(DEVICE).view(-1)  # (B, 1) → (B,)

        optimizer.zero_grad()
        pred = model(x, Fs, lengths)  # (B,) or (B,1)

        loss_elem = loss_fn(pred, y)  # shape: (B,) or (B,1)
        if loss_elem.dim() > 1:
            loss_elem = loss_elem.mean(dim=1)  # (B,)

        w = torch.ones_like(loss_elem, device=DEVICE)

        loss = (w * loss_elem).mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    preds, trues = [], []

    with torch.no_grad():
        for x, Fs, lengths, y, fnames in loader:
            x = x.to(DEVICE)
            Fs = Fs.to(DEVICE)
            lengths = lengths.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x, Fs, lengths)
            loss_elem = loss_fn(pred, y)
            if loss_elem.dim() > 1:
                loss_elem = loss_elem.mean(dim=1)
            loss = loss_elem.mean()

            total_loss += loss.item()
            preds.extend(pred.cpu().numpy())
            trues.extend(y.cpu().numpy())

    return total_loss / len(loader), np.array(preds), np.array(trues)


def main():
    train_path = "processed/train_aug.npz"
    val_path = "processed/val_aug.npz"
    ckpt_dir = "checkpoints_base_onlineAug"
    os.makedirs(ckpt_dir, exist_ok=True)

    train_dl, val_dl, y_mean, y_std = create_dataloaders(
        train_path, val_path, batch_size=16
    )

    model_dict = {
        "cnn_gru": CutoffPredictorCNN_GRU,
        "cnn": CutoffPredictorCNN,
        "gru": CutoffPredictorGRU,
    }

    results = []
    num_epochs = 150

    loss_fn = nn.MSELoss(reduction="none")

    for model_name, ModelClass in model_dict.items():
        print("\n==============================")
        print(f"🔹 Training model: {model_name}")
        print("==============================")

        model = ModelClass().to(DEVICE)
        total_params, trainable_params = count_params(model)
        print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        best_val_loss = float("inf")
        best_epoch = -1

        for epoch in range(1, num_epochs + 1):
            train_loss = train_one_epoch(model, train_dl, optimizer, loss_fn)
            val_loss, preds, trues = validate(model, val_dl, loss_fn)

            print(
                f"[{model_name}] Epoch {epoch:03d}/{num_epochs} | "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch

                ckpt = {
                    "model_state": model.state_dict(),
                    "y_mean": y_mean,
                    "y_std": y_std,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "model_name": model_name,
                    "total_params": total_params,
                    "trainable_params": trainable_params,
                }

                ckpt_path = os.path.join(ckpt_dir, f"{model_name}_best.pth")
                torch.save(ckpt, ckpt_path)
                print(
                    f"[{model_name}] Saved new best model @ epoch {epoch} "
                    f"(val_loss={val_loss:.6f}) → {ckpt_path}"
                )

        print(
            f"Finished training {model_name} | "
            f"Best val loss = {best_val_loss:.6f} @ epoch {best_epoch}"
        )

        results.append(
            {
                "model_name": model_name,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
            }
        )

    csv_path = os.path.join(ckpt_dir, "results_baseline_models.csv")
    fieldnames = [
        "model_name",
        "total_params",
        "trainable_params",
        "best_val_loss",
        "best_epoch",
    ]

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print("\n📄 Saved summary CSV:", csv_path)
    for row in results:
        print(
            f"{row['model_name']:8s} | "
            f"params: {row['total_params']:,} | "
            f"best_val: {row['best_val_loss']:.6f} @ epoch {row['best_epoch']}"
        )


if __name__ == "__main__":
    main()
