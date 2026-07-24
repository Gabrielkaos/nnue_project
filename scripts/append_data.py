"""
Fetch N additional, deduplicated, randomized positions from
Lichess/chess-position-evaluations and (optionally) merge them into an
existing fixed-record train.bin.

Why this is safe to just "top up" instead of re-running the full
prepare_data.py with a bigger --n-train:

  - Each 68-byte record's first 65 bytes (64 board squares + side-to-move)
    are a deterministic function of the FEN alone (see fen_utils.py). Two
    rows that describe the same position always pack to the same 65-byte
    prefix, regardless of eval/depth, so we can dedupe on that slice
    without ever touching the raw dataset again.
  - We reshuffle the HF stream with a different seed (and optionally skip
    ahead) so the new sample isn't just re-picking the same early rows
    your first run already consumed.

Usage:
    pip install datasets tqdm
    python append_data.py --existing-bin ../data/train.bin \
        --out-bin ../data/train_new.bin --n-new 10000000 --seed 1337 \
        --merged-out ../data/train_merged.bin

Run on your own machine (needs internet access to huggingface.co).
"""

import argparse
import os
import time

from fen_utils import pack_record, RECORD_SIZE

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False

KEY_SIZE = 65  # bytes[0:65] = board (64) + side-to-move (1)


def load_existing_keys(path: str) -> set:
    """Read an existing fixed-record .bin and return the set of position
    keys (record[:KEY_SIZE]) it already contains."""
    keys = set()
    if not path or not os.path.exists(path):
        print(f"No existing file at {path!r} - starting with an empty key set.")
        return keys

    size = os.path.getsize(path)
    if size % RECORD_SIZE != 0:
        print(f"WARNING: {path} size ({size}) is not a multiple of "
              f"RECORD_SIZE ({RECORD_SIZE}); file may be truncated/corrupt.")

    n_records = size // RECORD_SIZE
    print(f"Loading {n_records:,} existing keys from {path} ...")
    with open(path, "rb") as f:
        while True:
            chunk = f.read(RECORD_SIZE)
            if len(chunk) < RECORD_SIZE:
                break
            keys.add(chunk[:KEY_SIZE])
    print(f"Loaded {len(keys):,} unique existing keys.")
    return keys


def iter_filtered_rows(min_depth: int, max_abs_cp: int, keep_mate: bool,
                        seed: int, shuffle_buffer: int, skip_rows: int):
    from datasets import load_dataset

    ds = load_dataset(
        "Lichess/chess-position-evaluations", split="train", streaming=True
    )
    if skip_rows > 0:
        ds = ds.skip(skip_rows)
    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

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
                continue
        yield row["fen"], cp, mate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--existing-bin", default="../data/train.bin",
                     help="Existing train.bin to dedupe against.")
    ap.add_argument("--out-bin", default="../data/train_new.bin",
                     help="Where to write the new unique records.")
    ap.add_argument("--merged-out", default=None,
                     help="If set, write existing-bin + out-bin concatenated "
                          "here (fixed-size records, so this is a plain "
                          "byte concat, no reprocessing).")
    ap.add_argument("--n-new", type=int, default=10_000_000)
    ap.add_argument("--min-depth", type=int, default=10)
    ap.add_argument("--max-abs-cp", type=int, default=1500)
    ap.add_argument("--keep-mate", action="store_true")
    ap.add_argument("--seed", type=int, default=1337,
                     help="Use a DIFFERENT seed than your original run.")
    ap.add_argument("--shuffle-buffer", type=int, default=100_000)
    ap.add_argument("--skip-rows", type=int, default=0,
                     help="Skip this many rows of the raw stream before "
                          "shuffling. Useful if your first run already "
                          "scanned deep into the dataset - pass roughly "
                          "the 'scanned=' count it ended on to reduce "
                          "how often you re-hit already-used positions.")
    ap.add_argument("--progress-every", type=int, default=50_000)
    args = ap.parse_args()

    existing_keys = load_existing_keys(args.existing_bin)

    os.makedirs(os.path.dirname(args.out_bin) or ".", exist_ok=True)

    n_written = 0
    n_scanned = 0
    n_dupe = 0
    start = time.time()

    pbar = tqdm(total=args.n_new, unit="rec", desc="new") if HAVE_TQDM else None

    with open(args.out_bin, "wb") as f_out:
        for fen, cp, mate in iter_filtered_rows(
            args.min_depth, args.max_abs_cp, args.keep_mate,
            args.seed, args.shuffle_buffer, args.skip_rows
        ):
            n_scanned += 1
            if n_written >= args.n_new:
                break

            record = pack_record(fen, cp, mate)
            key = record[:KEY_SIZE]

            if key in existing_keys:
                n_dupe += 1
                continue

            existing_keys.add(key)  # also guards against dupes within this run
            f_out.write(record)
            n_written += 1

            if pbar is not None:
                pbar.update(1)
                if n_written % 1000 == 0:
                    pbar.set_postfix(
                        scanned=n_scanned, dupes=n_dupe, refresh=False
                    )
            elif n_scanned % args.progress_every == 0:
                elapsed = time.time() - start
                rate = n_written / elapsed if elapsed > 0 else 0
                remaining = args.n_new - n_written
                eta_s = remaining / rate if rate > 0 else float("inf")
                print(
                    f"scanned={n_scanned:,} written={n_written:,}/{args.n_new:,} "
                    f"dupes={n_dupe:,} rate={rate:,.0f} rec/s "
                    f"eta={eta_s/60:,.1f} min",
                    flush=True,
                )

    if pbar is not None:
        pbar.close()

    elapsed = time.time() - start
    print(f"Wrote {n_written:,} new unique records to {args.out_bin}")
    print(f"Scanned {n_scanned:,} rows, skipped {n_dupe:,} duplicates "
          f"({n_dupe / n_scanned * 100:.1f}% dupe rate)" if n_scanned else "")
    print(f"Total time: {elapsed/60:.1f} min "
          f"({n_written/elapsed:,.0f} rec/s)" if elapsed > 0 else "")

    if args.merged_out:
        print(f"Merging {args.existing_bin} + {args.out_bin} -> {args.merged_out}")
        with open(args.merged_out, "wb") as fout:
            if os.path.exists(args.existing_bin):
                with open(args.existing_bin, "rb") as f1:
                    while True:
                        chunk = f1.read(1024 * 1024)
                        if not chunk:
                            break
                        fout.write(chunk)
            with open(args.out_bin, "rb") as f2:
                while True:
                    chunk = f2.read(1024 * 1024)
                    if not chunk:
                        break
                    fout.write(chunk)
        merged_size = os.path.getsize(args.merged_out)
        print(f"Merged file has {merged_size // RECORD_SIZE:,} total records "
              f"at {args.merged_out}")


if __name__ == "__main__":
    main()