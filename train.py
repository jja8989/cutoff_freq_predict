import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataloader import create_dataloaders
from model import CutoffPredictorCNN_GRU
import numpy as np
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for x, Fs, lengths, y, fn in loader:
        x, Fs, lengths, y = (
            x.to(DEVICE),
            Fs.to(DEVICE),
            lengths.to(DEVICE),
            y.to(DEVICE),
        )
        
        optimizer.zero_grad()
        pred = model(x, Fs, lengths)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, loss_fn):
    model.eval()
    total_loss = 0
    preds, trues = [], []
    with torch.no_grad():
        for x, Fs, lengths, y, fn in loader:
            x, Fs, lengths, y = (
                x.to(DEVICE),
                Fs.to(DEVICE),
                lengths.to(DEVICE),
                y.to(DEVICE),
            )
            pred = model(x, Fs, lengths)
            total_loss += loss_fn(pred, y).item()
            preds.extend(pred.cpu().numpy())
            trues.extend(y.cpu().numpy())
    return total_loss / len(loader), np.array(preds), np.array(trues)

# --------------------------
# 🔹 Main
# --------------------------
def main():
    # 1️⃣ Paths
    train_path = "processed2/train_aug.npz"
    val_path = "processed2/val_aug.npz"
    os.makedirs("checkpoints3", exist_ok=True)

    # 2️⃣ Data — train 기준 global mean/std 사용
    train_dl, val_dl, y_mean, y_std = create_dataloaders(train_path, val_path, batch_size=16)

    # 3️⃣ Model / Loss / Optimizer
    model = CutoffPredictorCNN_GRU().to(DEVICE)
    # loss_fn = nn.SmoothL1Loss(beta=0.05)   # ✅ 안정적 회귀용 loss
    loss_fn = nn.MSELoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 4️⃣ Training loop
    best_val_loss = float("inf")
    num_epochs = 150

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_dl, optimizer, loss_fn)
        val_loss, preds, trues = validate(model, val_dl, loss_fn)

        print(f"Epoch {epoch:03d}/{num_epochs} | "
              f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # 5️⃣ Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = {
                "model_state": model.state_dict(),
                "y_mean": y_mean,
                "y_std": y_std,
                "epoch": epoch,
                "val_loss": val_loss,
            }
            torch.save(ckpt, "checkpoints3/best_model2.pth")
            print(f"✅ Saved new best model @ epoch {epoch} (val_loss={val_loss:.6f})")

    print(f"\n✅ Training complete! Best val loss = {best_val_loss:.6f}")


if __name__ == "__main__":
    main()