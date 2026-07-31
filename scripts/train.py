"""
Train the NNUE model on the binary cache produced by prepare_data.py.

Example:
    python train.py --train ../data/train.bin --val ../data/val.bin \
        --epochs 20 --batch-size 8192 --lr 1e-3
"""

import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import NNUE
from nnue_dataset import NNUEDataset, nnue_collate


def _to_device(features, device):
    white_idx, black_idx, offsets, stm = features
    return (
        white_idx.to(device),
        black_idx.to(device),
        offsets.to(device),
        stm.to(device),
    )


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n = 0
    loss_fn = nn.MSELoss(reduction="sum")
    with torch.no_grad():
        for features, y in loader:
            white_idx, black_idx, offsets, stm = _to_device(features, device)
            y = y.to(device)
            pred = torch.sigmoid(model(white_idx, offsets, black_idx, offsets, stm))
            total_loss += loss_fn(pred, y).item()
            n += y.size(0)
    model.train()
    return total_loss / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="../data/train.bin")
    ap.add_argument("--val", default="../data/val.bin")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="../checkpoints/nnue.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = NNUEDataset(args.train)
    val_ds = NNUEDataset(args.val)
    print(f"Train positions: {len(train_ds)}  Val positions: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        drop_last=True, collate_fn=nnue_collate,
        persistent_workers=(args.workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=nnue_collate,
        persistent_workers=(args.workers > 0),
    )

    model = NNUE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, args.epochs // 4), gamma=0.3)
    loss_fn = nn.MSELoss()

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        running = 0.0
        seen = 0
        for features, y in train_loader:
            white_idx, black_idx, offsets, stm = _to_device(features, device)
            y = y.to(device)

            opt.zero_grad()
            pred = torch.sigmoid(model(white_idx, offsets, black_idx, offsets, stm))
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
            seen += y.size(0)
        sched.step()

        val_loss = evaluate(model, val_loader, device)
        dt = time.time() - t0
        print(
            f"epoch {epoch:3d}  train_mse {running/seen:.6f}  "
            f"val_mse {val_loss:.6f}  ({dt:.1f}s)"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "val_mse": val_loss}, args.out)
            print(f"  -> saved new best checkpoint to {args.out}")

    print("Done. Best val MSE:", best_val)


if __name__ == "__main__":
    main()