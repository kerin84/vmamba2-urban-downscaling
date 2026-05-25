"""
ConvLSTM — temporal downscaling model.

Architecture:
  1. Static features are encoded with a UNet encoder → skip connections at
     three spatial scales + bottleneck (31×31).
  2. Each ERA5 frame is encoded to the same bottleneck resolution by a
     lightweight shared CNN (time-distributed, no weight sharing with static).
  3. The ERA5 frame features are concatenated with the static bottleneck and
     fed sequentially through a ConvLSTM cell at the bottleneck resolution.
  4. The final hidden state is decoded via the UNet decoder, which uses the
     static skip connections to recover HR spatial detail.

Input shapes:
  era5_seq : (B, T, DYN_CHANNELS, 4, 5)
  static   : (B, STATIC_CHANNELS, 251, 251)

Output: (B, 251, 251)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import (
    DYN_CHANNELS, STATIC_CHANNELS,
    UNET_BASE_FILTERS, CONVLSTM_HIDDEN_DIM,
)
from .blocks import DoubleConv, upsample_cat, ConvLSTMCell


class ConvLSTMDownscaler(nn.Module):
    def __init__(
        self,
        dyn_channels: int = DYN_CHANNELS,
        static_channels: int = STATIC_CHANNELS,
        base_filters: int = UNET_BASE_FILTERS,
        hidden_dim: int = CONVLSTM_HIDDEN_DIM,
    ):
        super().__init__()
        C = base_filters

        # --- Static UNet encoder (provides spatial structure) ---
        self.senc1 = DoubleConv(static_channels, C)       # (B, C,   251, 251)
        self.senc2 = DoubleConv(C,               C * 2)   # (B, 2C,  125, 125)
        self.senc3 = DoubleConv(C * 2,           C * 4)   # (B, 4C,   62,  62)
        self.sbot  = DoubleConv(C * 4,           C * 8)   # (B, 8C,   31,  31)
        self.pool  = nn.MaxPool2d(2)

        # --- ERA5 frame encoder (time-distributed, LR → bottleneck) ---
        # Projects each ERA5 frame from (DYN, 4, 5) → (C*8, 31, 31)
        self.era5_enc = nn.Sequential(
            nn.Conv2d(dyn_channels, C * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(C * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C * 2, C * 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(C * 4),
            nn.ReLU(inplace=True),
        )

        # --- ConvLSTM at bottleneck resolution (31×31) ---
        # Input per step: cat(era5_bot, static_bot) = C*4 + C*8 = C*12
        self.lstm = ConvLSTMCell(
            in_channels=C * 4 + C * 8,
            hidden_channels=hidden_dim,
        )

        # --- Decoder (uses static skip connections) ---
        self.dec3 = DoubleConv(hidden_dim + C * 4, C * 4)
        self.dec2 = DoubleConv(C * 4    + C * 2,  C * 2)
        self.dec1 = DoubleConv(C * 2    + C,       C)
        self.head = nn.Conv2d(C, 1, 1)

    def forward(self, era5_seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        B, T, _, H, W = era5_seq.shape[0], era5_seq.shape[1], None, static.shape[-2], static.shape[-1]

        # --- Encode static ---
        s1 = self.senc1(static)                   # (B, C,   H, W)
        s2 = self.senc2(self.pool(s1))            # (B, 2C,  H/2, W/2)
        s3 = self.senc3(self.pool(s2))            # (B, 4C,  H/4, W/4)
        sb = self.sbot(self.pool(s3))             # (B, 8C,  H/8, W/8)  ≈31×31

        # --- Time-distributed ERA5 encoding ---
        era5_flat = era5_seq.view(B * T, *era5_seq.shape[2:])          # (B*T, DYN, 4, 5)
        era5_bot  = self.era5_enc(era5_flat)                            # (B*T, 4C, 4, 5)
        era5_bot  = F.interpolate(era5_bot, size=sb.shape[-2:],
                                  mode="bilinear", align_corners=False) # (B*T, 4C, 31, 31)
        era5_bot  = era5_bot.view(B, T, *era5_bot.shape[1:])            # (B, T, 4C, 31, 31)

        # --- ConvLSTM over temporal sequence ---
        state = None
        for t in range(T):
            inp_t = torch.cat([era5_bot[:, t], sb], dim=1)  # (B, 4C+8C, 31, 31)
            h, state = self.lstm(inp_t, state)

        # --- Decode using static skips ---
        d3 = self.dec3(upsample_cat(h,  s3))     # (B, 4C,  ~62,  ~62)
        d2 = self.dec2(upsample_cat(d3, s2))     # (B, 2C,  ~125, ~125)
        d1 = self.dec1(upsample_cat(d2, s1))     # (B, C,   251,  251)

        return self.head(d1).squeeze(1)           # (B, 251, 251)
