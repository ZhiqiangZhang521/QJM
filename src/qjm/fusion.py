"""Fixed grouped mapper and the full six-qubit QJM inference head."""
from __future__ import annotations
import math
import torch
from torch import nn
from .pools import EEGResponseKernelPool, TemporalKernelPool
from .circuit import ThreeBranchQJM


class ResponsePool62(EEGResponseKernelPool):
    def forward(self, scale):
        if scale.shape[1:] != (2, 4, 62, 8):
            raise ValueError('Expected [B,2,4,62,8]')
        x = self.norm(scale.flatten(1, 2))
        x = self.mix(torch.cat([layer(x) for layer in self.kernels], dim=1))
        x = x.mean(dim=-1)
        return (x * self.electrode_score(x).softmax(dim=-1)).sum(dim=-1)


class FixedMapper(nn.Module):
    def __init__(self, electrodes=59, context_patches=39, width=16):
        super().__init__()
        if (electrodes, context_patches) not in ((59,39), (62,31)):
            raise ValueError('Unsupported checkpoint geometry')
        self.electrodes, self.context_patches = electrodes, context_patches
        self.response_dim = 2*12*electrodes*8
        self.response_pool = (EEGResponseKernelPool(width) if electrodes==59 else ResponsePool62(width))
        self.response_heads = nn.ModuleList(nn.Linear(width,3) for _ in range(3))
        self.context_pool = TemporalKernelPool(64,width)
        self.context_head = nn.Linear(width,3)
        self.eog_temporal_pool = TemporalKernelPool(48,width)
        self.eog_temporal_head = nn.Linear(width,3)
        self.eog_global_head = nn.Sequential(nn.LayerNorm(96),nn.Linear(96,width),nn.GELU(),nn.Linear(width,3))
        self.angle_scale = nn.Parameter(torch.ones(6,3))
        self.angle_bias = nn.Parameter(torch.zeros(6,3))
        self.register_buffer('calibration_strength',torch.tensor(.9))

    def forward(self, eeg, tokens, global_eog):
        if eeg.shape[1:] != (self.response_dim+self.context_patches*64,) or tokens.shape != (len(eeg),8,48) or global_eog.shape != (len(eeg),96):
            raise ValueError('Frozen feature geometry differs from checkpoint')
        r = eeg[:,:self.response_dim].reshape(len(eeg),2,12,self.electrodes,8)
        groups = [head(self.response_pool(r[:,:,4*i:4*(i+1)])) for i,head in enumerate(self.response_heads)]
        groups += [self.context_head(self.context_pool(eeg[:,self.response_dim:].reshape(-1,self.context_patches,64))),
                   self.eog_temporal_head(self.eog_temporal_pool(tokens)), self.eog_global_head(global_eog)]
        raw = math.pi*torch.tanh(torch.stack(groups,dim=1))
        scale = 1+self.calibration_strength*(self.angle_scale-1)
        bias = self.calibration_strength*self.angle_bias
        return math.pi*torch.tanh(scale*(raw/math.pi)+bias)


class CompleteHead(ThreeBranchQJM):
    def __init__(self):
        super().__init__(head_dim=48,cross_floor=.8,dropout=.1,eog_feature_dim=96,
                         intra_entangler_init=.2,cross_entangler_init=.55)
        del self.angle_encoders
        del self.eog_horizontal
        del self.eog_vertical

    def forward(self, angles, return_aux=False):
        return self.forward_angles(angles,return_aux=return_aux)
