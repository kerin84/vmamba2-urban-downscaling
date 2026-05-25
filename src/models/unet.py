"""
UNet — purely spatial downscaling baseline.

Uses only the LAST ERA5 timestep (no temporal modeling).  The ERA5 frame is
bilinearly interpolated to the HR grid and concatenated with the static
features before entering the encoder, following the standard SR-UNet approach.

Input shapes:
  era5_seq : (B, T, DYN_CHANNELS, 4, 5)   — only last frame used
  static   : (B, STATIC_CHANNELS, 251, 251)

Output: (B, 251, 251) — predicted T2m at HR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import DYN_CHANNELS, STATIC_CHANNELS, UNET_BASE_FILTERS
from .blocks import DoubleConv, upsample_cat


class UNet(nn.Module):
    def __init__(
        self,
        dyn_channels: int = DYN_CHANNELS,
        static_channels: int = STATIC_CHANNELS,
        base_filters: int = UNET_BASE_FILTERS,
    ):
        super().__init__()
        C = base_filters
        in_ch = dyn_channels + static_channels

        # Encoder  (251 → 125 → 62 → 31)
        self.enc1 = DoubleConv(in_ch, C)
        self.enc2 = DoubleConv(C,     C * 2)
        self.enc3 = DoubleConv(C * 2, C * 4)
        self.bot  = DoubleConv(C * 4, C * 8)
        self.pool = nn.MaxPool2d(2)

        # Decoder  (31 → 62 → 125 → 251)
        self.dec3 = DoubleConv(C * 8 + C * 4, C * 4)
        self.dec2 = DoubleConv(C * 4 + C * 2, C * 2)
        self.dec1 = DoubleConv(C * 2 + C,     C)
        self.head = nn.Conv2d(C, 1, 1)

    def forward(self, era5_seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        H, W = static.shape[-2:]

        # Upsample last ERA5 frame to HR and cat with static
        era5_hr = F.interpolate(era5_seq[:, -1], size=(H, W),
                                mode="bilinear", align_corners=False)
        x = torch.cat([era5_hr, static], dim=1)    # (B, DYN+STATIC, H, W)

        # Encoder
        e1 = self.enc1(x)                           # (B, C,   251, 251)
        e2 = self.enc2(self.pool(e1))               # (B, 2C,  125, 125)
        e3 = self.enc3(self.pool(e2))               # (B, 4C,   62,  62)
        b  = self.bot(self.pool(e3))                # (B, 8C,   31,  31)

        # Decoder
        d3 = self.dec3(upsample_cat(b,  e3))        # (B, 4C,   62,  62)
        d2 = self.dec2(upsample_cat(d3, e2))        # (B, 2C,  125, 125)
        d1 = self.dec1(upsample_cat(d2, e1))        # (B, C,   251, 251)

        return self.head(d1).squeeze(1)             # (B, 251, 251)
