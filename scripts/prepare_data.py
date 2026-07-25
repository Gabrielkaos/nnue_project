"""
Stream Lichess/chess-position-evaluations from Hugging Face and write a
compact fixed-record binary cache for fast NNUE training.

This needs internet access to huggingface.co, so run it on your own
machine (not inside a sandboxed tool environment):

    pip install datasets tqdm
    python prepare_data.py --n-train 8000000 --n-val 50000

It writes:
    data/train.bin
    data/val.bin

Each record is 68 bytes (see fen_utils.py for the exact layout).
"""

import argparse
import os
import random
import time

from fen_utils import pack_record, RECORD_SIZE

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False


def iter_filtered_rows(min_depth: int, max_abs_cp: int, keep_mate: bool):
    from datasets import load_dataset

    ds = load_dataset(
        "Lichess/chess-position-evaluations", split="train", streaming=True
    )

    # ds = ds.shuffle(seed=2312,buffer_size=100_000)
    for row in ds:
        if row["depth"] is not None and row["depth"] < min_depth:
            continue
        cp = row["cp"]
        mate = row["mate"]
        if mate is not None:
            if not keep_mate:
                continue
        else:
            if cp is None:
                continue
            if abs(cp) > max_abs_cp:
                # Very lopsided positions add little signal for training a
                # small eval net and can dominate the loss; skip them.
                continue
        yield row["fen"], cp, mate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="../data")
    ap.add_argument("--n-train", type=int, default=8_000_000)
    ap.add_argument("--n-val", type=int, default=50_000)
    ap.add_argument("--min-depth", type=int, default=10)
    ap.add_argument("--max-abs-cp", type=int, default=1500)
    ap.add_argument("--keep-mate", action="store_true",
                     help="Include forced-mate rows (stored as +/-3000cp).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=50_000,
                     help="If tqdm isn't installed, print a status line "
                          "every N rows scanned.")
    args = ap.parse_args()

    random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    train_path = f"{args.out_dir}/train.bin"
    val_path = f"{args.out_dir}/val.bin"

    n_train_written = 0
    n_val_written = 0
    n_scanned = 0
    start = time.time()

    # Progress bar tracks n_train_written against the requested target,
    # since that's the dominant stopping condition and gives a real ETA.
    # n_scanned (rows pulled from the stream, including filtered-out ones)
    # is shown in the postfix so you can see the filter's reject rate.
    pbar = tqdm(total=args.n_train, unit="rec", desc="train") if HAVE_TQDM else None

    with open(train_path, "wb") as f_train, open(val_path, "wb") as f_val:
        for fen, cp, mate in iter_filtered_rows(
            args.min_depth, args.max_abs_cp, args.keep_mate
        ):
            n_scanned += 1

            if n_train_written >= args.n_train and n_val_written >= args.n_val:
                break

            record = pack_record(fen, cp, mate)
            # ~0.6% of rows go to validation until it's full
            if n_val_written < args.n_val and random.random() < 0.006:
                f_val.write(record)
                n_val_written += 1
            elif n_train_written < args.n_train:
                f_train.write(record)
                n_train_written += 1
                if pbar is not None:
                    pbar.update(1)
                    if n_train_written % 1000 == 0:
                        pbar.set_postfix(
                            val=f"{n_val_written}/{args.n_val}",
                            scanned=n_scanned,
                            refresh=False,
                        )

            if pbar is None and n_scanned % args.progress_every == 0:
                elapsed = time.time() - start
                rate = n_train_written / elapsed if elapsed > 0 else 0
                remaining = args.n_train - n_train_written
                eta_s = remaining / rate if rate > 0 else float("inf")
                print(
                    f"scanned={n_scanned:,} "
                    f"train={n_train_written:,}/{args.n_train:,} "
                    f"val={n_val_written:,}/{args.n_val:,} "
                    f"rate={rate:,.0f} rec/s "
                    f"eta={eta_s/60:,.1f} min",
                    flush=True,
                )

    if pbar is not None:
        pbar.close()

    elapsed = time.time() - start
    print(f"Wrote {n_train_written} train records to {train_path}")
    print(f"Wrote {n_val_written} val records to {val_path}")
    print(f"Record size: {RECORD_SIZE} bytes")
    print(f"Total time: {elapsed/60:.1f} min "
          f"({n_train_written/elapsed:,.0f} train rec/s)")


if __name__ == "__main__":
    main()