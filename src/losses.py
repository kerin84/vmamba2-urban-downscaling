"""
Hybrid loss: (1-alpha)*MSE + alpha*(1-SSIM).
Port of v1 src/losses.py TorchHybridLoss. alpha=0.8 matches v1 default.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(size: int, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """SSIM on (B, C, H, W) tensors. Exact port of v1 ssim_torch."""
    C = img1.size(1)
    w1d = _gaussian_window(window_size, 1.5).unsqueeze(1)
    w2d = w1d.mm(w1d.t()).unsqueeze(0).unsqueeze(0)
    win = w2d.expand(C, 1, window_size, window_size).to(img1.device, img1.dtype)

    pad = window_size // 2
    mu1 = F.conv2d(img1, win, padding=pad, groups=C)
    mu2 = F.conv2d(img2, win, padding=pad, groups=C)

    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = F.conv2d(img1 * img1, win, padding=pad, groups=C) - mu1_sq
    s2 = F.conv2d(img2 * img2, win, padding=pad, groups=C) - mu2_sq
    s12 = F.conv2d(img1 * img2, win, padding=pad, groups=C) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * s12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return ssim_map.mean()


class HybridLoss(nn.Module):
    """(1-alpha)*MSE + alpha*(1-SSIM). Inputs: (B, 1, H, W)."""
    def __init__(self, alpha: float = 0.8, window_size: int = 11):
        super().__init__()
        self.alpha = alpha
        self.window_size = window_size

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse  = F.mse_loss(pred, target)
        ssim = _ssim(pred, target, self.window_size)
        return (1 - self.alpha) * mse + self.alpha * (1 - ssim)
