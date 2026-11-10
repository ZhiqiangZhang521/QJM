from __future__ import annotations
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _ResidualBlock(nn.Module):

    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * 2
        self.net = nn.Sequential(nn.Conv1d(width, width, 5, padding=padding, dilation=dilation, groups=width, bias=False), nn.GroupNorm(1, width), nn.GELU(), nn.Conv1d(width, width, 1, bias=False), nn.GroupNorm(1, width), nn.Dropout(dropout))

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(x + self.net(x))


class RawEOGEmotionNet(nn.Module):
    """EOG-only multi-scale temporal network for independent trials."""

    def __init__(self, channels: int=6, classes: int=7, width: int=48, dropout: float=0.15):
        super().__init__()
        self.channels = int(channels)
        self.stem = nn.Sequential(nn.Conv1d(channels, width, 7, stride=1, padding=3, bias=False), nn.BatchNorm1d(width), nn.GELU())
        self.scales = nn.ModuleList([nn.Sequential(nn.Conv1d(width, width, kernel, padding=kernel // 2, groups=width, bias=False), nn.BatchNorm1d(width), nn.GELU()) for kernel in (5, 11, 21)])
        self.mix = nn.Sequential(nn.Conv1d(width * 3, width, 1, bias=False), nn.BatchNorm1d(width), nn.GELU())
        self.blocks = nn.Sequential(_ResidualBlock(width, 1, dropout), _ResidualBlock(width, 2, dropout), _ResidualBlock(width, 4, dropout))
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.head = nn.Sequential(nn.LayerNorm(width * 8), nn.Linear(width * 8, width * 2), nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(width * 2, classes)
        self.projection = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, width))

    def encode(self, signal: Tensor) -> Tensor:
        if signal.ndim != 3 or signal.shape[1] != self.channels:
            raise ValueError(f'expected [B,{self.channels},T], got {tuple(signal.shape)}')
        x = self.stem(signal)
        x = self.mix(torch.cat([branch(x) for branch in self.scales], dim=1))
        x = self.blocks(x)
        return self.head(self.pool(x).flatten(1))

    def encode_tokens(self, signal: Tensor) -> Tensor:
        """Return the pooled temporal token map before the final flatten/head."""
        if signal.ndim != 3 or signal.shape[1] != self.channels:
            raise ValueError(f'expected [B,{self.channels},T], got {tuple(signal.shape)}')
        x = self.stem(signal)
        x = self.mix(torch.cat([branch(x) for branch in self.scales], dim=1))
        x = self.blocks(x)
        return self.pool(x).transpose(1, 2).contiguous()

    def forward(self, signal: Tensor, return_features: bool=False):
        features = self.encode(signal)
        logits = self.classifier(features)
        if return_features:
            return (logits, F.normalize(self.projection(features), dim=1))
        return logits
