"""
Train the NNUE model on the binary cache produced by prepare_data.py.

Example:
    python train.py --train ../data/train.bin --val ../data/val.bin \
        --epochs 20 --batch-size 8192 --lr 1e-3

Perf notes
----------
NNUE nets are tiny -- a forward+backward pass is microseconds of GPU work.
That means the GPU is almost always waiting on the CPU (data loading /
Python overhead), not the other way around, so the fixes here are aimed at
keeping the GPU continuously fed rather than at the model math itself:

  - loss.item() is no longer called every batch. That forces a CPU<->GPU
    sync and stalls the pipeline; the running loss is now accumulated as a
    GPU tensor and only pulled to CPU every --log-every steps (and once at
    epoch end), so the GPU can keep several batches queued ahead of the CPU.
  - .to(device, non_blocking=True) + pin_memory on BOTH loaders (the val
    loader was missing pin_memory before), so H2D copies overlap with compute.
  - persistent_workers + a real prefetch_factor so worker processes aren't
    torn down and respawned every epoch and stay ahead of the training loop.
  - TF32 matmuls enabled (free ~1.5-2x on Ampere+ for this kind of small
    dense-matmul workload) and optional torch.compile / AMP.
  - optimizer.zero_grad(set_to_none=True) -- skips a memset per param.

If the GPU is still near-idle after this, the bottleneck is almost
certainly in NNUEDataset.__getitem__ (e.g. a per-sample file.seek()/
struct.unpack() instead of an in-memory / memmapped read) -- that's worth
looking at next.
"""

import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False

from model import NNUE
from nnue_dataset import NNUEDataset


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n = 0
    loss_fn = nn.MSELoss(reduction="sum")
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = torch.sigmoid(model(x))
            total_loss += loss_fn(pred, y).item()
            n += x.size(0)
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
    ap.add_argument("--prefetch-factor", type=int, default=4,
                     help="Batches each worker prefetches ahead of time.")
    ap.add_argument("--out", default="../checkpoints/nnue.pt")
    ap.add_argument("--out-last", default="../checkpoints/last.pt")
    ap.add_argument("--no-progress", action="store_true",
                     help="Disable the live per-batch progress bar.")
    ap.add_argument("--log-every", type=int, default=50,
                     help="How often (in batches) to sync loss to CPU for "
                          "the progress display. Higher = fewer stalls.")
    ap.add_argument("--amp", action="store_true",
                     help="Use mixed precision (fp16 autocast + GradScaler). "
                          "Usually not the bottleneck for NNUE-sized models "
                          "but free to try.")
    ap.add_argument("--compile", action="store_true",
                     help="Wrap the model in torch.compile() (PyTorch 2.x). "
                          "Helps most when the GPU is the bottleneck; won't "
                          "fix a data-loading-bound pipeline.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        # TF32 matmuls: essentially free precision-for-speed tradeoff on
        # Ampere+ GPUs, and NNUE's dense layers are exactly the op this helps.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    train_ds = NNUEDataset(args.train)
    val_ds = NNUEDataset(args.val)
    print(f"Train positions: {len(train_ds)}  Val positions: {len(val_ds)}")

    use_persistent = args.workers > 0

    print("Loading train data...")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=use_persistent,
        prefetch_factor=args.prefetch_factor if use_persistent else None,
    )
    print("Loaded train data...")

    print("Loading val data...")
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=use_persistent,
        prefetch_factor=args.prefetch_factor if use_persistent else None,
    )
    print("Loaded val data...")

    model = NNUE().to(device)
    print(f"model size:\ninput={model.input_size}\nl1={model.l1_size}\nl2={model.l2_size}\nl3={model.l3_size}")

    if args.compile:
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception as e:
            print(f"torch.compile failed ({e}), continuing uncompiled")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, args.epochs // 4), gamma=0.3)
    loss_fn = nn.MSELoss()

    scaler = torch.amp.GradScaler("cuda",enabled=(args.amp and device.type == "cuda"))

    use_bar = HAVE_TQDM and not args.no_progress
    n_batches = len(train_loader)

    best_val = float("inf")
    print("Training...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # Accumulated on-device; only pulled to CPU every --log-every steps
        # (and at epoch end) to avoid a sync on every single batch.
        running_gpu = torch.zeros((), device=device)
        seen = 0
        running_cpu = 0.0  # last CPU-visible snapshot, for the progress bar

        if use_bar:
            bar = tqdm(
                train_loader,
                total=n_batches,
                desc=f"epoch {epoch}/{args.epochs}",
                unit="batch",
                leave=False,
            )
        else:
            bar = train_loader

        for i, (x, y) in enumerate(bar, 1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            if scaler.is_enabled():
                with torch.cuda.amp.autocast():
                    pred = torch.sigmoid(model(x))
                    loss = loss_fn(pred, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                pred = torch.sigmoid(model(x))
                loss = loss_fn(pred, y)
                loss.backward()
                opt.step()

            # Stays on-device -- no sync here.
            running_gpu += loss.detach() * x.size(0)
            seen += x.size(0)

            if i % args.log_every == 0 or i == n_batches:
                running_cpu = (running_gpu / seen).item()  # single sync
                if use_bar:
                    bar.set_postfix(
                        loss=f"{running_cpu:.6f}",
                        lr=f"{opt.param_groups[0]['lr']:.2e}",
                        refresh=False,
                    )
                elif not use_bar:
                    elapsed = time.time() - t0
                    rate = i / elapsed if elapsed > 0 else 0
                    eta_s = (n_batches - i) / rate if rate > 0 else float("inf")
                    print(
                        f"  epoch {epoch} batch {i}/{n_batches} "
                        f"loss={running_cpu:.6f} "
                        f"rate={rate:.1f} batch/s eta={eta_s:.0f}s",
                        flush=True,
                    )

        if use_bar:
            bar.close()

        sched.step()

        val_loss = evaluate(model, val_loader, device)
        dt = time.time() - t0
        train_mse = (running_gpu / seen).item()
        print(
            f"epoch {epoch:3d}  train_mse {train_mse:.6f}  "
            f"val_mse {val_loss:.6f}  ({dt:.1f}s, "
            f"{n_batches/dt:.1f} batch/s, {seen/dt:,.0f} pos/s)"
        )

        # torch.compile wraps the model in an OptimizedModule; unwrap so the
        # checkpoint's state_dict keys match the plain NNUE module.
        state_dict = getattr(model, "_orig_mod", model).state_dict()

        torch.save(
            {"model": state_dict, "val_mse": val_loss, "epoch": epoch},
            args.out_last,
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": state_dict, "val_mse": val_loss}, args.out)
            print(f"  -> saved new best checkpoint to {args.out}")

    print("Done. Best val MSE:", best_val)


if __name__ == "__main__":
    main()