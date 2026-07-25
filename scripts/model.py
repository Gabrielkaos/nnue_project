"""
NNUE architecture: 768 -> 256 -> 32 -> 32 -> 1

This mirrors the classic small NNUE topology (feature transformer +
3 small linear layers), but simplified to a single perspective as
requested. ClippedReLU (clamp to [0, 1]) is used instead of plain ReLU
because it keeps activations bounded, which matters once you quantize
this to fixed point for the C engine later.

Forward pass returns a raw scalar in roughly [-8, 8] logit space; apply
sigmoid to get the same [0, 1] "win probability-like" scale used for
training targets. Multiply logit by CP_SCALE (see nnue_dataset.py) and
you get back something in centipawn units if you want a human-readable
number, i.e. eval_cp ~= CP_SCALE * raw_output.
"""

import torch
import torch.nn as nn


class ClippedReLU(nn.Module):
    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0)


class NNUE(nn.Module):
    def __init__(self, input_size=768, l1=512, l2=32, l3=32):
        super().__init__()
        self.input_size = input_size
        self.l1_size = l1
        self.l2_size = l2
        self.l3_size = l3

        self.fc1 = nn.Linear(input_size, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, l3)
        self.fc4 = nn.Linear(l3, 1)
        self.act = ClippedReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.fc4(x)
        return x.squeeze(-1)
