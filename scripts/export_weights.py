"""
Export a trained NNUE checkpoint to a binary weight file the C engine
can load directly with fread().

Format v2 (bumped from v1 because of two breaking changes: absolute/
White-fixed features instead of stm-mirrored, and a quantized fc1 layer):

    char[4]   magic       = b"NNUE"
    int32     version     = 2
    int32     input_size  = 768
    int32     l1_size     = 256
    int32     l2_size     = 32
    int32     l3_size     = 32
    float32   cp_scale            (see nnue_dataset.py, default 410.0)
    int32     qa_scale            (fc1 fixed-point scale, e.g. 64)
    int16[l1_size * input_size]   fc1.weight, quantized: round(w * qa_scale)
    int32[l1_size]                fc1.bias,   quantized: round(b * qa_scale)
    float32[l2_size * l1_size]    fc2.weight  (kept as plain float)
    float32[l2_size]              fc2.bias
    float32[l3_size * l2_size]    fc3.weight
    float32[l3_size]              fc3.bias
    float32[1 * l3_size]          fc4.weight
    float32[1]                    fc4.bias

Why only fc1 is quantized: fc1 is the layer the C engine maintains as an
incremental accumulator (one int32 add/subtract per active feature per
piece moved), so quantizing it gives real memory/cache/speed benefits.
fc2/fc3/fc4 are tiny (~9k multiply-adds total) and recomputed fully on
every eval call either way, so quantizing them further buys very little
extra speed for real added complexity — not done here.

Usage:
    python export_weights.py --checkpoint ../checkpoints/nnue.pt \
        --out ../checkpoints/nnue.bin
"""

import argparse
import struct

import numpy as np
import torch

from model import NNUE
from nnue_dataset import CP_SCALE

QA_SCALE = 64  # fc1 fixed-point scale factor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="../checkpoints/nnue.pt")
    ap.add_argument("--out", default="../checkpoints/nnue.bin")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = NNUE()
    model.load_state_dict(ckpt["model"])
    model.eval()

    with open(args.out, "wb") as f:
        f.write(b"NNUE")
        f.write(struct.pack("<i", 2))
        f.write(struct.pack("<i", model.input_size))
        f.write(struct.pack("<i", model.l1_size))
        f.write(struct.pack("<i", model.l2_size))
        f.write(struct.pack("<i", model.l3_size))
        f.write(struct.pack("<f", CP_SCALE))
        f.write(struct.pack("<i", QA_SCALE))

        # fc1: quantized int16 weight / int32 bias
        w1 = model.fc1.weight.detach().numpy().astype(np.float64)
        b1 = model.fc1.bias.detach().numpy().astype(np.float64)
        w1_q = np.round(w1 * QA_SCALE)
        b1_q = np.round(b1 * QA_SCALE)
        max_w = np.max(np.abs(w1_q))
        if max_w > 32767:
            raise ValueError(
                f"fc1 weight overflow at qa_scale={QA_SCALE}: max |w*scale|={max_w:.0f} "
                f"exceeds int16 range. Retrain with weight decay or lower QA_SCALE."
            )
        f.write(w1_q.astype("<i2").tobytes())  # int16, little-endian
        f.write(b1_q.astype("<i4").tobytes())  # int32, little-endian

        # fc2/fc3/fc4: plain float32
        for layer in (model.fc2, model.fc3, model.fc4):
            w = layer.weight.detach().numpy().astype("<f4")
            b = layer.bias.detach().numpy().astype("<f4")
            f.write(w.tobytes())
            f.write(b.tobytes())

    print(f"Exported v2 weights to {args.out} (fc1 quantized at scale={QA_SCALE})")


if __name__ == "__main__":
    main()