"""
DownsrUNet — faithful PyTorch port of v1 src/models/downsr_unet.py.

Architecture (identical encoder + decoder for all variants):

  1. Bridge:  ERA5 bilinear-upsampled to HR, concatenated with static per frame.
             x: (B, T, DYN+STATIC, H, W)

  2. Encoder (time-distributed, shared):
             e1 : ConvBlock(DYN+STATIC → 32)     skip, H × W
       pool → e2 : ConvBlock(32 → 64)             skip, H/2 × W/2
       pool → e3 : ConvBlock(64 → 128)            skip, H/4 × W/4
       pool → p3 : (B, T, 128, H/8, W/8)          bottleneck input

  3. Bottleneck (ONLY difference between the three models):
       'unet'     : ConvBlock(128 → 256) + Dropout(0.3)
       'convlstm' : ConvLSTM2D(128 → 256, return_sequences=True) + BN + LeakyReLU
       'mamba'    : flatten T×H/8×W/8 → 2×Mamba(128) → unflatten → Conv1×1(128→256)

  4. Decoder (time-distributed, shared):
       up3 : resize to e3, cat(256+128 → 128)
       up2 : resize to e2, cat(128+64  → 64)
       up1 : resize to e1, cat(64+32   → 32)
       head: Conv1×1(32 → 1)

  5. Output: (B, T, 1, H, W) → squeeze → (B, T, H, W)
     Loss supervised on [:, -1] only (last timestep) — matches v1 training.

Notes on v1 fidelity:
  - v1 Mamba ran at 128ch (d_model = c_enc = 128) with NO channel expansion.
    We add a 1×1 conv to project 128→256 so the decoder is identical for all three.
    This is a minor improvement over v1 (removes an inconsistency) and is documented.
  - v1 used SimpleMambaBlock (lite/fallback); v2 uses real mamba_ssm when available.
  - Conv/BN/LeakyReLU(0.1) matches v1 cnn.py exactly.
"""

from __future__ import annotations

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
    MAMBA_D_STATE, MAMBA_D_CONV, MAMBA_EXPAND,
)
from .blocks import ConvBlock, ConvLSTMCell, td


# ---------------------------------------------------------------------------
# Fallback Mamba block (CPU / MPS / no mamba_ssm install)
# ---------------------------------------------------------------------------

class _FallbackMamba(nn.Module):
    """Depthwise-conv + gated MLP, same (B, L, D) contract as mamba_ssm.Mamba."""
    def __init__(self, d_model: int, **_):
        super().__init__()
        k = 7
        self.norm  = nn.LayerNorm(d_model)
        self.dconv = nn.Conv1d(d_model, d_model, k, padding=k // 2, groups=d_model)
        self.gate  = nn.Linear(d_model, d_model * 2)
        self.proj  = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.norm(x)
        t = self.dconv(x.transpose(1, 2)).transpose(1, 2)
        g, v = self.gate(t).chunk(2, dim=-1)
        return r + self.proj(F.gelu(g) * v)


def _make_mamba(d_model: int, d_state: int, d_conv: int, expand: int) -> nn.Module:
    if _HAS_MAMBA:
        return _MambaSSM(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    return _FallbackMamba(d_model)


# ---------------------------------------------------------------------------
# Bottleneck modules
# ---------------------------------------------------------------------------

class _UNetBottleneck(nn.Module):
    """v1 'standard CNN' bottleneck: ConvBlock + Dropout."""
    def __init__(self, in_ch: int = 128, out_ch: int = 256, dropout: float = 0.3):
        super().__init__()
        self.block   = ConvBlock(in_ch, out_ch)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return td(self.dropout, self.block(x))


class _ConvLSTMBottleneck(nn.Module):
    """v1 ConvLSTM bottleneck: return_sequences=True + BN + LeakyReLU."""
    def __init__(self, in_ch: int = 128, out_ch: int = 256):
        super().__init__()
        self.cell = ConvLSTMCell(in_ch, out_ch)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 128, H, W)
        B, T = x.shape[:2]
        state = None
        outs = []
        for t in range(T):
            h, state = self.cell(x[:, t], state)
            outs.append(h)
        # stack → (B, T, out_ch, H, W), apply BN+act time-distributed
        out = torch.stack(outs, dim=1)
        return td(nn.Sequential(self.bn, self.act), out)


class _MambaBottleneck(nn.Module):
    """
    v1 Mamba bottleneck, ported to real mamba_ssm.
    Flatten (B, T, C, H, W) → (B, T*H*W, C) → 2×Mamba → unflatten.
    d_model = in_ch (128), matching v1 where c_enc was passed directly.
    1×1 conv projects to out_ch (256) so decoder is identical across all variants.
    """
    def __init__(
        self,
        in_ch:  int = 128,
        out_ch: int = 256,
        d_state: int = MAMBA_D_STATE,
        d_conv:  int = MAMBA_D_CONV,
        expand:  int = MAMBA_EXPAND,
    ):
        super().__init__()
        self.mamba1 = _make_mamba(in_ch, d_state, d_conv, expand)
        self.mamba2 = _make_mamba(in_ch, d_state, d_conv, expand)
        self.norm   = nn.LayerNorm(in_ch)
        # project to out_ch so decoder is identical to UNet/ConvLSTM variants
        self.proj   = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        # permute to (B, T, H, W, C) then flatten sequence dim
        x_flat = x.permute(0, 1, 3, 4, 2).reshape(B, T * H * W, C)
        x_flat = self.mamba1(x_flat)
        x_flat = self.mamba2(x_flat)
        x_flat = self.norm(x_flat)
        # unflatten back to (B, T, H, W, C) → (B, T, C, H, W)
        out = x_flat.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3).contiguous()
        # project 128 → 256 (time-distributed via B*T merge)
        return td(self.proj, out)


class _VMamba2DBottleneck(nn.Module):
    """
    VMamba-style 2D scanning bottleneck — v1: spatial only.
    Row + column scanning per timestep, NO temporal integration.
    Baseline for ablation vs v2.
    """
    def __init__(
        self,
        in_ch:  int = 128,
        out_ch: int = 256,
        d_state: int = MAMBA_D_STATE,
        d_conv:  int = MAMBA_D_CONV,
        expand:  int = MAMBA_EXPAND,
    ):
        super().__init__()
        self.mamba_row1 = _make_mamba(in_ch, d_state, d_conv, expand)
        self.mamba_row2 = _make_mamba(in_ch, d_state, d_conv, expand)
        self.mamba_col1 = _make_mamba(in_ch, d_state, d_conv, expand)
        self.mamba_col2 = _make_mamba(in_ch, d_state, d_conv, expand)
        self.norm       = nn.LayerNorm(in_ch)
        self.proj       = nn.Conv2d(in_ch, out_ch, 1)

    def _scan_row(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_r = x.permute(0, 2, 3, 1).reshape(B * H, W, C)
        x_r = self.mamba_row1(x_r)
        x_r = self.mamba_row2(x_r)
        return x_r.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    def _scan_col(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_c = x.permute(0, 2, 3, 1).permute(0, 2, 1, 3).reshape(B * W, H, C)
        x_c = self.mamba_col1(x_c)
        x_c = self.mamba_col2(x_c)
        return x_c.reshape(B, W, H, C).permute(0, 2, 1, 3).permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        outs = []
        for t in range(T):
            frame = x[:, t]
            row_out = self._scan_row(frame)
            col_out = self._scan_col(frame)
            outs.append((row_out + col_out) / 2.0)
        out = torch.stack(outs, dim=1)
        out = self.norm(out.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3).contiguous()
        return td(self.proj, out)


class _VMambaTemporalBottleneck(nn.Module):
    """
    VMamba 2D scanning bottleneck — v2: spatial + temporal.

    Pipeline por timestep:
      1. Row scan espacial (por frame): (B*H, W, C) → Mamba
      2. Col scan espacial (por frame): (B*W, H, C) → Mamba
      3. Merge espacial → (B, T, C, H, W)
      4. Temporal scan (por pixel): (B*H*W, T, C) → Mamba

    Orden: espacial primero (estructura del frame), temporal después (evolución).
    """
    def __init__(
        self,
        in_ch:  int = 128,
        out_ch: int = 256,
        d_state: int = MAMBA_D_STATE,
        d_conv:  int = MAMBA_D_CONV,
        expand:  int = MAMBA_EXPAND,
    ):
        super().__init__()
        # Espacial: row + col (2 capas cada uno, como v1)
        self.mamba_row1 = _make_mamba(in_ch, d_state, d_conv, expand)
        self.mamba_row2 = _make_mamba(in_ch, d_state, d_conv, expand)
        self.mamba_col1 = _make_mamba(in_ch, d_state, d_conv, expand)
        self.mamba_col2 = _make_mamba(in_ch, d_state, d_conv, expand)
        # Temporal: 1 capa Mamba sobre secuencia T
        self.mamba_time = _make_mamba(in_ch, d_state, d_conv, expand)
        self.norm       = nn.LayerNorm(in_ch)
        self.proj       = nn.Conv2d(in_ch, out_ch, 1)

    def _scan_row(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_r = x.permute(0, 2, 3, 1).reshape(B * H, W, C)
        x_r = self.mamba_row1(x_r)
        x_r = self.mamba_row2(x_r)
        return x_r.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    def _scan_col(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_c = x.permute(0, 2, 3, 1).permute(0, 2, 1, 3).reshape(B * W, H, C)
        x_c = self.mamba_col1(x_c)
        x_c = self.mamba_col2(x_c)
        return x_c.reshape(B, W, H, C).permute(0, 2, 1, 3).permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape

        # --- Fase 1: Espacial (row + col scan por frame) ---
        outs = []
        for t in range(T):
            frame = x[:, t]
            row_out = self._scan_row(frame)
            col_out = self._scan_col(frame)
            merged = (row_out + col_out) / 2.0
            outs.append(merged)
        out = torch.stack(outs, dim=1)                     # (B, T, C, H, W)

        # --- Fase 2: Temporal scan por posición espacial ---
        # (B, T, C, H, W) → (B*H*W, T, C) → Mamba → (B*H*W, T, C) → reshape
        out_t = out.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)
        out_t = self.mamba_time(out_t)
        out = out_t.reshape(B, H, W, T, C).permute(0, 3, 4, 1, 2).contiguous()  # (B, T, C, H, W)

        # LayerNorm + project
        out = self.norm(out.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3).contiguous()
        return td(self.proj, out)


_BOTTLENECKS = {
    "unet":     _UNetBottleneck,
    "convlstm": _ConvLSTMBottleneck,
    "mamba":    _MambaBottleneck,
    "vmamba":   _VMamba2DBottleneck,
    "vmamba2":  _VMambaTemporalBottleneck,
}


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class DownsrUNet(nn.Module):
    """
    Shared encoder + decoder; only the bottleneck changes.
    bottleneck_type: 'unet' | 'convlstm' | 'mamba'
    """
    def __init__(
        self,
        bottleneck_type: str = "mamba",
        dyn_channels:    int = DYN_CHANNELS,
        static_channels: int = STATIC_CHANNELS,
    ):
        super().__init__()
        assert bottleneck_type in _BOTTLENECKS, \
            f"bottleneck_type must be one of {list(_BOTTLENECKS)}"
        in_ch = dyn_channels + static_channels

        # Encoder
        self.enc1 = ConvBlock(in_ch, 32)
        self.enc2 = ConvBlock(32,    64)
        self.enc3 = ConvBlock(64,   128)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = _BOTTLENECKS[bottleneck_type]()  # 128 → 256

        # Decoder
        self.dec3 = ConvBlock(256 + 128, 128)
        self.dec2 = ConvBlock(128 +  64,  64)
        self.dec1 = ConvBlock( 64 +  32,  32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, era5_seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """
        era5_seq : (B, T, DYN, 4, 5)
        static   : (B, STATIC, H, W)
        returns  : (B, T, H, W)  — predict all T steps; supervise on [:, -1]
        """
        B, T, _, H, W = *era5_seq.shape[:2], None, *static.shape[-2:]

        # ---- bridge: upsample ERA5 to HR, cat with static ----
        era5_flat = era5_seq.flatten(0, 1)                          # (B*T, DYN, 4, 5)
        era5_hr   = F.interpolate(era5_flat, size=(H, W),
                                  mode="bilinear", align_corners=False)
        era5_hr   = era5_hr.unflatten(0, (B, T))                   # (B, T, DYN, H, W)
        static_t  = static.unsqueeze(1).expand(-1, T, -1, -1, -1)  # (B, T, STATIC, H, W)
        x = torch.cat([era5_hr, static_t], dim=2)                  # (B, T, DYN+STATIC, H, W)

        # ---- encoder ----
        e1 = self.enc1(x)                                           # (B, T, 32,  H,   W)
        e2 = self.enc2(td(self.pool, e1))                           # (B, T, 64,  H/2, W/2)
        e3 = self.enc3(td(self.pool, e2))                           # (B, T, 128, H/4, W/4)
        p3 = td(self.pool, e3)                                      # (B, T, 128, H/8, W/8)

        # ---- bottleneck ----
        neck = self.bottleneck(p3)                                  # (B, T, 256, H/8, W/8)

        # ---- decoder ----
        def _up_cat(x, skip):
            x = td(lambda t: F.interpolate(t, size=skip.shape[-2:],
                                           mode="bilinear", align_corners=False), x)
            return torch.cat([x, skip], dim=2)

        d3 = self.dec3(_up_cat(neck, e3))                          # (B, T, 128, H/4, W/4)
        d2 = self.dec2(_up_cat(d3,   e2))                          # (B, T, 64,  H/2, W/2)
        d1 = self.dec1(_up_cat(d2,   e1))                          # (B, T, 32,  H,   W)

        # ---- output ----
        out = td(self.head, d1)                                     # (B, T, 1, H, W)
        return out.squeeze(2)                                       # (B, T, H, W)
