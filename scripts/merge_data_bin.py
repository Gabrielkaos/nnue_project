"""
Merge two or more fixed-record .bin files (as produced by prepare_data.py)
into one, removing duplicate records.

Usage:
    python merge_dedup.py train_run1.bin train_run2.bin -o merged_train.bin
    python merge_dedup.py train_run1.bin train_run2.bin -o merged_train.bin --key position

Dedup keys:
    full      (default) - the entire 68-byte record must match exactly
                (identical board, side-to-move, AND eval/mate flag).
                Safest: never drops two records of the same position that
                happen to have different stored evals.
    position  - board + side-to-move (first 65 bytes) must match.
                Any second occurrence of the same position is dropped,
                keeping only the first one seen (first file, then in
                file order). Use this if you expect the same position to
                repeat across your two runs (very likely, since streaming
                the same HF dataset twice mostly re-visits the same rows)
                and you just want one eval per position.

Memory: instead of keeping raw keys in memory (68 or 65 bytes each, times
tens of millions of rows), we keep a 16-byte BLAKE2b digest per unique key
in a set. Collision probability across even 100M rows is astronomically
low (~1e-15), so this is safe in practice and uses far less RAM.

Place this file next to fen_utils.py (or anywhere on PYTHONPATH) so it can
import RECORD_SIZE from it; falls back to the hardcoded 68 if not found.
"""

import argparse
import hashlib
import os
import sys

try:
    from fen_utils import RECORD_SIZE
except ImportError:
    RECORD_SIZE = 68

POSITION_KEY_LEN = 65  # 64 board bytes + 1 side-to-move byte

READ_CHUNK_RECORDS = 100_000  # records per buffered read


def key_for(record: bytes, mode: str) -> bytes:
    if mode == "full":
        return record
    elif mode == "position":
        return record[:POSITION_KEY_LEN]
    else:
        raise ValueError(f"unknown key mode: {mode}")


def iter_records(path: str, chunk_records: int = READ_CHUNK_RECORDS):
    chunk_bytes = RECORD_SIZE * chunk_records
    with open(path, "rb") as f:
        leftover = b""
        while True:
            data = f.read(chunk_bytes)
            if not data:
                break
            data = leftover + data
            n_whole = len(data) // RECORD_SIZE
            usable = n_whole * RECORD_SIZE
            leftover = data[usable:]
            for i in range(0, usable, RECORD_SIZE):
                yield data[i:i + RECORD_SIZE]
        if leftover:
            print(
                f"warning: {path} ends with {len(leftover)} leftover bytes "
                f"(not a multiple of RECORD_SIZE={RECORD_SIZE}); ignoring.",
                file=sys.stderr,
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="input .bin files, in priority order "
                                               "(earlier files' records win on duplicates)")
    ap.add_argument("-o", "--out", required=True, help="output .bin path")
    ap.add_argument("--key", choices=["full", "position"], default="full",
                     help="dedup key (default: full)")
    args = ap.parse_args()

    for p in args.inputs:
        if not os.path.isfile(p):
            ap.error(f"input file not found: {p}")
    if os.path.abspath(args.out) in [os.path.abspath(p) for p in args.inputs]:
        ap.error("output path must differ from all input paths")

    seen = set()
    n_in = 0
    n_out = 0
    n_dupe = 0

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    with open(args.out, "wb") as fout:
        for path in args.inputs:
            file_in = 0
            file_out = 0
            for record in iter_records(path):
                n_in += 1
                file_in += 1
                key = key_for(record, args.key)
                digest = hashlib.blake2b(key, digest_size=16).digest()
                if digest in seen:
                    n_dupe += 1
                    continue
                seen.add(digest)
                fout.write(record)
                n_out += 1
                file_out += 1
            print(f"{path}: read {file_in:,} records, kept {file_out:,}")

    print()
    print(f"Total records read:    {n_in:,}")
    print(f"Duplicates removed:    {n_dupe:,}")
    print(f"Records written to {args.out}: {n_out:,}")
    print(f"Dedup key mode: {args.key}")


if __name__ == "__main__":
    main()