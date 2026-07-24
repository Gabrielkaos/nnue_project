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
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from fen_utils import RECORD_SIZE

CP_SCALE = 410.0


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
                self.path, dtype=np.uint8, mode="r", shape=(self.n, RECORD_SIZE)
            )

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        self._ensure_mmap()
        rec = self._mmap[idx]
        board = rec[:64]
        # stm byte (rec[64]) is intentionally unused now: features are
        # always absolute/White-fixed, regardless of whose move it is.
        eval_raw = int(rec[65]) | (int(rec[66]) << 8)
        if eval_raw >= 32768:
            eval_raw -= 65536
        cp_white = eval_raw

        features = np.zeros(768, dtype=np.float32)
        nz = np.nonzero(board)[0]
        for sq in nz:
            piece = int(board[sq])  # 1..12
            plane = piece - 1  # 0..11
            features[plane * 64 + sq] = 1.0

        target = 1.0 / (1.0 + np.exp(-cp_white / CP_SCALE))
        return torch.from_numpy(features), torch.tensor(target, dtype=torch.float32)