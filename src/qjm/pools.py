from __future__ import annotations
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TemporalKernelPool(nn.Module):
    """Multi-kernel depthwise temporal filtering, followed by channel mixing."""

    def __init__(self, channels: int, width: int=16):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.kernels = nn.ModuleList((nn.Conv1d(channels, channels, kernel_size=k, padding=k // 2, groups=channels, bias=False) for k in (3, 5)))
        self.mix = nn.Sequential(nn.Conv1d(2 * channels, width, 1, bias=False), nn.GroupNorm(4, width), nn.GELU())
        self.score = nn.Conv1d(width, 1, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError('Expected [batch, time, channels] tokens')
        x = self.norm(tokens).transpose(1, 2)
        x = self.mix(torch.cat([layer(x) for layer in self.kernels], dim=1))
        weights = self.score(x).softmax(dim=-1)
        return (x * weights).sum(dim=-1)


class EEGResponseKernelPool(nn.Module):
    """Shared temporal kernels for the three existing EEG response scales."""

    def __init__(self, width: int=16):
        super().__init__()
        self.norm = nn.GroupNorm(4, 8)
        self.kernels = nn.ModuleList((nn.Conv2d(8, 8, kernel_size=(1, k), padding=(0, k // 2), groups=8, bias=False) for k in (3, 5)))
        self.mix = nn.Sequential(nn.Conv2d(16, width, 1, bias=False), nn.GroupNorm(4, width), nn.GELU())
        self.electrode_score = nn.Conv1d(width, 1, 1)

    def forward(self, scale: torch.Tensor) -> torch.Tensor:
        if scale.ndim != 5 or scale.shape[1:] != (2, 4, 59, 8):
            raise ValueError('Expected EEG scale [batch, 2, 4, 59, 8]')
        x = self.norm(scale.flatten(1, 2))
        x = self.mix(torch.cat([layer(x) for layer in self.kernels], dim=1))
        x = x.mean(dim=-1)
        weights = self.electrode_score(x).softmax(dim=-1)
        return (x * weights).sum(dim=-1)
