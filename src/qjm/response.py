from __future__ import annotations
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ResponseStatisticsEncoder(nn.Module):

    def __init__(self, channels=59, filters_per_scale=4, spatial_filters=2, bins=8, dropout=0.3):
        super().__init__()
        self.temporal_filters = nn.ModuleList([nn.Conv2d(1, filters_per_scale, (1, k), padding=(0, k // 2), bias=False) for k in (15, 31, 63)])
        self.filter_count = 3 * filters_per_scale
        self.spatial_filters = spatial_filters
        self.spatial_weight = nn.Parameter(torch.randn(self.filter_count, spatial_filters, channels) * channels ** (-0.5))
        maps = self.filter_count * spatial_filters
        self.smoothing_logits = nn.Parameter(torch.zeros(maps, 1, 63))
        self.compression_logits = nn.Parameter(torch.full((maps,), -2.0))
        self.pool = nn.AdaptiveAvgPool1d(bins)
        self.dropout = nn.Dropout(dropout)
        self.feature_dim = 2 * maps * bins

    def forward(self, x, *, return_clean=False):
        response = torch.cat([layer(x[:, None]) for layer in self.temporal_filters], 1)
        spatial = F.normalize(self.spatial_weight, dim=-1)
        response = torch.einsum('bkct,krc->bkrt', response, spatial).flatten(1, 2)
        power = response.float().square()
        weights = self.smoothing_logits.softmax(-1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            energy = F.conv1d(F.pad(power, (31, 31), mode='replicate'), weights.float(), stride=8, groups=power.shape[1])
            exponent = 0.1 + 0.9 * self.compression_logits.sigmoid()[None, :, None]
            strength = torch.expm1(exponent * torch.log1p(energy)) / exponent
            contrast = strength - strength.mean(1, keepdim=True)
            features = self.pool(torch.cat((strength, contrast), 1)).flatten(1)
        return features if return_clean else self.dropout(features)


class ChannelResponseEncoder(nn.Module):
    """Measure learned local responses BEFORE combining electrodes.

    Classical per-electrode energy does not mix unrelated electrode phases.
    Squaring a spatially summed waveform introduces cross terms; this order
    avoids them and retains every electrode until the final task readout.
    The cross-electrode layer acts on learned response strengths, not raw
    oscillations. No canonical spectral bands or fixed spectral descriptors.
    """

    def __init__(self, channels=59, filters_per_scale=4, bins=8, dropout=0.3, smoothing_kernel=63, spatial_mixing=True):
        super().__init__()
        self.channels = channels
        self.filter_count = 3 * filters_per_scale
        if smoothing_kernel < 1 or smoothing_kernel % 2 != 1:
            raise ValueError('positive odd response smoothing kernel required')
        self.smoothing_kernel = smoothing_kernel
        self.temporal_filters = nn.ModuleList([nn.Conv2d(1, filters_per_scale, (1, k), padding=(0, k // 2), bias=False) for k in (15, 31, 63)])
        self.smoothing_logits = nn.Parameter(torch.zeros(self.filter_count, 1, smoothing_kernel))
        self.compression_logits = nn.Parameter(torch.full((self.filter_count,), -2.0))
        if spatial_mixing:
            self.spatial_weight = nn.Parameter(torch.zeros(channels, channels))
        else:
            self.register_buffer('spatial_weight', torch.zeros(channels, channels))
        self.pool = nn.AdaptiveAvgPool1d(bins)
        self.dropout = nn.Dropout(dropout)
        self.feature_dim = 2 * self.filter_count * channels * bins

    def forward(self, x, *, return_clean=False):
        response = torch.cat([layer(x[:, None]) for layer in self.temporal_filters], 1)
        batch, filters, channels, time = response.shape
        power = response.float().square().flatten(1, 2)
        weights = self.smoothing_logits.softmax(-1).repeat_interleave(channels, dim=0)
        with torch.autocast(device_type=x.device.type, enabled=False):
            pad = self.smoothing_kernel // 2
            energy = F.conv1d(F.pad(power, (pad, pad), mode='replicate'), weights.float(), stride=8, groups=filters * channels)
            energy = energy.reshape(batch, filters, channels, -1)
            exponent = 0.1 + 0.9 * self.compression_logits.sigmoid()[None, :, None, None]
            strength = torch.expm1(exponent * torch.log1p(energy)) / exponent
            relative = strength - strength.mean(1, keepdim=True)
            evidence = torch.cat((strength, relative), 1)
            evidence = evidence + 0.1 * torch.einsum('bdct,oc->bdot', evidence, self.spatial_weight.float())
            features = self.pool(evidence.flatten(1, 2)).flatten(1)
        return features if return_clean else self.dropout(features)


class ConsensusResponseEEG(nn.Module):

    def __init__(self, channels=59, samples=500, *, filters=4, spatial_filters=2, bins=8, dropout=0.3, retain_electrodes=False, clean_consensus=False, smoothing_kernel=63, trial_channel_normalize=False, spatial_mixing=True):
        super().__init__()
        if channels < 2 or samples < 8 or (not 1 <= bins <= (samples - 1) // 8 + 1):
            raise ValueError('invalid electrodes, trial duration or non-upsampling readout bins')
        self.channels, self.samples = (channels, samples)
        self.clean_consensus = clean_consensus
        self.trial_channel_normalize = trial_channel_normalize
        self.register_buffer('input_center', torch.zeros(channels))
        self.register_buffer('input_scale', torch.ones(channels))
        self.encoder = ChannelResponseEncoder(channels, filters, bins, dropout, smoothing_kernel, spatial_mixing) if retain_electrodes else ResponseStatisticsEncoder(channels, filters, spatial_filters, bins, dropout)
        self.classifier = nn.Linear(self.encoder.feature_dim, 7)

    def encode(self, raw, *, clean=False):
        if raw.ndim != 3 or raw.shape[1:] != (self.channels, self.samples):
            raise ValueError('one raw trial with the fixed electrode/time layout required')
        x = raw - raw.mean(1, keepdim=True)
        x = (x - self.input_center[None, :, None]) / self.input_scale[None, :, None]
        x = x.clamp(-8.0, 8.0)
        if self.trial_channel_normalize:
            x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True, unbiased=False).clamp_min(1e-05)
        return self.encoder(x, return_clean=clean)

    def forward(self, raw, *, return_features=False):
        if self.clean_consensus and return_features:
            z = self.encode(raw, clean=True)
            logits = self.classifier(self.encoder.dropout(z))
        else:
            z = self.encode(raw)
            logits = self.classifier(z)
        return (logits, z) if return_features else logits
