"""
MambaUNet — temporal downscaling model using state-space sequence modeling.

Architecture:
  1. Static features encoded via UNet backbone → skip connections + bottleneck.
  2. Each ERA5 frame encoded to the bottleneck spatial resolution (time-distributed).
  3. At the bottleneck, the T spatial feature maps are spatially averaged to give
     T token vectors of shape (B, T, D_model).  Mamba processes this compact
     temporal sequence to capture synoptic dynamics.
  4. The Mamba output is projected and spatially broadcast back to the bottleneck
     grid, then fused with the static bottleneck features.
  5. UNet decoder with static skip connections recovers full HR resolution.

Why this design:
  Running Mamba over raw spatial tokens (31×31 = 961 positions × T) is feasible
  but redundant — most spatial positions share the same synoptic signal.  Instead
  we compress to T compact tokens and let the static branch carry spatial detail.
  This reduces Mamba sequence length from ~6000 to 6 while preserving all temporal
  information, and avoids the positional encoding complexity of ViM-style models.

Input shapes:
  era5_seq : (B, T, DYN_CHANNELS, 4, 5)
  static   : (B, STATIC_CHANNELS, 251, 251)

Output: (B, 251, 251)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba as _MambaSSM
    _HAS_MAMBA = True
except Exception:
    _MambaSSM = None
    _HAS_MAMBA = False

from config.config import (
    DYN_CHANNELS, STATIC_CHANNELS,
    UNET_BASE_FILTERS, MAMBA_D_MODEL, MAMBA_D_STATE, MAMBA_D_CONV, MAMBA_EXPAND,
)
from .blocks import DoubleConv, upsample_cat


class _FallbackMamba(nn.Module):
    """CPU/MPS fallback: depthwise conv + gated MLP, same (B, L, D) contract."""
    def __init__(self, d_model, **kwargs):
        super().__init__()
        k = 7
        self.norm  = nn.LayerNorm(d_model)
        self.dconv = nn.Conv1d(d_model, d_model, k, padding=k // 2, groups=d_model)
        self.gate  = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU())
        self.proj  = nn.Linear(d_model, d_model)

    def forward(self, x):
        r = x
        x = self.norm(x)
        t = self.dconv(x.transpose(1, 2)).transpose(1, 2)
        g, v = self.gate(t).chunk(2, dim=-1)
        return r + self.proj(g * v)


def _make_mamba(d_model, d_state, d_conv, expand):
    if _HAS_MAMBA:
        return _MambaSSM(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    return _FallbackMamba(d_model)


class MambaUNet(nn.Module):
    def __init__(
        self,
        dyn_channels: int = DYN_CHANNELS,
        static_channels: int = STATIC_CHANNELS,
        base_filters: int = UNET_BASE_FILTERS,
        d_model: int = MAMBA_D_MODEL,
        d_state: int = MAMBA_D_STATE,
        d_conv: int = MAMBA_D_CONV,
        expand: int = MAMBA_EXPAND,
        n_mamba_layers: int = 2,
    ):
        super().__init__()
        C = base_filters

        # --- Static UNet encoder ---
        self.senc1 = DoubleConv(static_channels, C)
        self.senc2 = DoubleConv(C,               C * 2)
        self.senc3 = DoubleConv(C * 2,           C * 4)
        self.sbot  = DoubleConv(C * 4,           C * 8)
        self.pool  = nn.MaxPool2d(2)

        # --- ERA5 frame encoder (time-distributed) ---
        self.era5_enc = nn.Sequential(
            nn.Conv2d(dyn_channels, C * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(C * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C * 2, C * 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(C * 4),
            nn.ReLU(inplace=True),
        )

        # --- Temporal Mamba: (B, T, d_model) → (B, T, d_model) ---
        # Project C*4 spatial avg-pooled features → d_model
        self.to_tokens = nn.Linear(C * 4, d_model)
        self.mamba_layers = nn.ModuleList([
            _make_mamba(d_model, d_state, d_conv, expand)
            for _ in range(n_mamba_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Project d_model back to C*8 and broadcast to spatial grid
        self.from_tokens = nn.Linear(d_model, C * 8)

        # --- Decoder ---
        self.dec3 = DoubleConv(C * 8 + C * 4, C * 4)
        self.dec2 = DoubleConv(C * 4 + C * 2, C * 2)
        self.dec1 = DoubleConv(C * 2 + C,     C)
        self.head = nn.Conv2d(C, 1, 1)

    def forward(self, era5_seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        B, T = era5_seq.shape[:2]

        # --- Encode static ---
        s1 = self.senc1(static)
        s2 = self.senc2(self.pool(s1))
        s3 = self.senc3(self.pool(s2))
        sb = self.sbot(self.pool(s3))                          # (B, 8C, 31, 31)

        # --- Time-distributed ERA5 encoding → compact tokens ---
        era5_flat = era5_seq.view(B * T, *era5_seq.shape[2:])  # (B*T, DYN, 4, 5)
        era5_bot  = self.era5_enc(era5_flat)                   # (B*T, 4C, 4, 5)
        # Global average pool over spatial dims → (B*T, 4C)
        tokens = era5_bot.mean(dim=(-2, -1))                   # (B*T, 4C)
        tokens = self.to_tokens(tokens)                        # (B*T, D)
        tokens = tokens.view(B, T, -1)                         # (B, T, D)

        # --- Mamba temporal sequence ---
        for layer in self.mamba_layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)                             # (B, T, D)

        # Aggregate temporal context (take last step; Mamba is causal)
        ctx = tokens[:, -1]                                    # (B, D)
        ctx = self.from_tokens(ctx)                            # (B, 8C)

        # Broadcast to spatial bottleneck grid and add to static bottleneck
        H_bot, W_bot = sb.shape[-2:]
        ctx_spatial = ctx[:, :, None, None].expand(B, -1, H_bot, W_bot)  # (B, 8C, 31, 31)
        fused = sb + ctx_spatial                                           # (B, 8C, 31, 31)

        # --- Decode ---
        d3 = self.dec3(upsample_cat(fused, s3))
        d2 = self.dec2(upsample_cat(d3,   s2))
        d1 = self.dec1(upsample_cat(d2,   s1))

        return self.head(d1).squeeze(1)                        # (B, 251, 251)
