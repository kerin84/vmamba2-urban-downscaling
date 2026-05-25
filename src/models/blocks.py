"""Shared building blocks — faithful port of v1 src/tf_engine/blocks/cnn.py."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def td(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Time-distributed: apply a 2-D module to (B, T, C, H, W).
    Merges batch+time, runs the module, then splits back.
    """
    B, T = x.shape[:2]
    out = module(x.flatten(0, 1))       # (B*T, ...)
    return out.unflatten(0, (B, T))     # (B, T, ...)


class ConvBlock(nn.Module):
    """
    Time-distributed Conv→BN→LeakyReLU × 2.
    Exact port of v1 conv_block (cnn.py).
    Input/output: (B, T, in_ch, H, W) → (B, T, out_ch, H, W)
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return td(self.net, x)


class ConvLSTMCell(nn.Module):
    """Standard ConvLSTM cell. Input/hidden: (B, C, H, W)."""
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, x: torch.Tensor, state):
        B, _, H, W = x.shape
        if state is None:
            h = x.new_zeros(B, self.hidden_channels, H, W)
            c = x.new_zeros(B, self.hidden_channels, H, W)
        else:
            h, c = state
        i, f, g, o = self.gates(torch.cat([x, h], 1)).chunk(4, dim=1)
        c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_new = torch.sigmoid(o) * torch.tanh(c_new)
        return h_new, (h_new, c_new)
