"""
PyTorch Dataset over the binary record cache produced by prepare_data.py.

Feature set: 768 = 12 piece planes (6 piece types x 2 colors) x 64 squares,
ABSOLUTE / White-fixed orientation. No mirroring by side to move: plane
assignment and square numbering are always "White's pawn is on e4", never
"my pawn is on e4". This matters because the C engine maintains this same
768-wide layer as an INCREMENTAL accumulator (only a couple of add/subtract
operations per piece moved), and that only works if a piece's feature index
never changes just because the side to move changed — with the old
mirror-by-stm scheme, the entire input flipped every ply, which made
incremental updates impossible.

Target scaling: cp values (already White-relative in the cache) are used
directly, squashed with a sigmoid:
    target = sigmoid(cp_white / CP_SCALE)
so the network always predicts a White-relative score. The C side flips
the sign at the very end based on side to move (exactly how the engine's
classical eval already does it via `pos->side==WHITE ? score : -score`).
CP_SCALE=410 is the classic nnue-pytorch default and works well as a
starting point.

Note: because there's no side-to-move signal anywhere in the features,
the net can't learn tempo-type effects (a small bonus for "it's my move").
That's an intentional simplicity/speed trade-off — your classical eval
already adds its own explicit `tempo` bonus on top separately.

Perf notes
----------
This is the actual bottleneck in the training pipeline: with a tiny NNUE
model, the GPU finishes a batch almost instantly, so throughput is set
entirely by how fast the CPU can hand over batches. The original
__getitem__ built one 768-float feature vector at a time with a Python
`for sq in nz: ...` loop -- called once per training sample, i.e. millions
of times per epoch, each doing several small numpy calls whose per-call
overhead dominates over the actual (tiny) amount of work being done.

The fix is __getitems__ (note the plural): PyTorch's DataLoader fetcher
checks for this method and, if present, calls it once per *batch* with the
whole list of indices instead of calling __getitem__ once per sample. That
turns "N python-level numpy calls" into "N/batch_size python-level numpy
calls", with the actual feature construction done as a single vectorized
scatter over the whole batch. Nothing else about how you use NNUEDataset
or DataLoader needs to change -- shuffling, num_workers, collation into
(x, y) batch tensors, all of it stays exactly as before, this just changes
how the data underneath gets built.

Requires torch >= 2.0 (older torch DataLoader fetchers ignore
__getitems__ and silently fall back to the slow per-sample path via
__getitem__, which is kept below for that fallback and for any code that
indexes the dataset directly, e.g. dataset[i] in a REPL).
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from fen_utils import RECORD_SIZE

CP_SCALE = 410.0

# Structured view of the 68-byte record so field extraction is a plain
# strided numpy read instead of manual byte-shift arithmetic. '<i2' is
# explicit little-endian int16 to match struct.Struct("<64sBhB") in
# fen_utils.py regardless of host byte order.
RECORD_DTYPE = np.dtype([
    ("board", np.uint8, (64,)),
    ("stm", np.uint8),
    ("eval", "<i2"),
    ("flags", np.uint8),
])
assert RECORD_DTYPE.itemsize == RECORD_SIZE, (
    f"RECORD_DTYPE size {RECORD_DTYPE.itemsize} != RECORD_SIZE {RECORD_SIZE}; "
    "keep this in sync with fen_utils.RECORD_STRUCT"
)


class NNUEDataset(Dataset):
    def __init__(self, bin_path: str):
        self.path = bin_path
        with open(bin_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
        assert size % RECORD_SIZE == 0, "corrupt cache file"
        self.n = size // RECORD_SIZE
        # Memory-map lazily per-worker to play nicely with DataLoader workers.
        self._mmap = None

    def _ensure_mmap(self):
        if self._mmap is None:
            self._mmap = np.memmap(
                self.path, dtype=RECORD_DTYPE, mode="r", shape=(self.n,)
            )

    def __len__(self):
        return self.n

    @staticmethod
    def _boards_to_features(boards: np.ndarray) -> np.ndarray:
        """Vectorized board(s) -> one-hot 768-feature array. boards can be
        shape (64,) for a single position or (B, 64) for a batch; returns
        shape (768,) or (B, 768) to match."""
        single = boards.ndim == 1
        if single:
            boards = boards[None, :]

        rows, cols = np.nonzero(boards)
        pieces = boards[rows, cols].astype(np.int64)
        feat_idx = (pieces - 1) * 64 + cols  # plane*64 + square

        features = np.zeros((boards.shape[0], 768), dtype=np.float32)
        features[rows, feat_idx] = 1.0
        return features[0] if single else features

    def __getitems__(self, indices):
        """Batch path -- used automatically by DataLoader when available
        (torch >= 2.0). Does the whole batch's worth of work in one shot
        with no Python-level loop over squares or samples."""
        self._ensure_mmap()
        idx = np.asarray(indices)
        recs = self._mmap[idx]  # structured array, shape (B,)

        features = self._boards_to_features(recs["board"])
        cp_white = recs["eval"].astype(np.float32)
        targets = 1.0 / (1.0 + np.exp(-cp_white / CP_SCALE))

        feat_t = torch.from_numpy(features)
        target_t = torch.from_numpy(targets)
        # Split back into a list of (x, y) samples -- default_collate then
        # stacks these into the same (B, 768) / (B,) batch tensors the
        # training loop already expects. This is cheap: plain tensor
        # views/slices, no per-element numpy work.
        return list(zip(feat_t, target_t))

    def __getitem__(self, idx):
        """Single-sample fallback: used if something indexes the dataset
        directly, or on torch < 2.0 where __getitems__ isn't recognized."""
        self._ensure_mmap()
        rec = self._mmap[idx]
        features = self._boards_to_features(rec["board"])
        cp_white = float(rec["eval"])
        target = 1.0 / (1.0 + np.exp(-cp_white / CP_SCALE))
        return torch.from_numpy(features), torch.tensor(target, dtype=torch.float32)