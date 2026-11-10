from __future__ import annotations
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .response import ConsensusResponseEEG
from .eegpt import EEGPTTrialHead


class ResponseContextEEG(nn.Module):
    gradient_audit_step = 1

    def __init__(self, channels=59, samples=500, patches=31, *, use_response=True, context_width=64, context_dropout=0.5, response_dropout=0.3, filters=4, bins=8, spatial_mixing=True):
        super().__init__()
        self.channels, self.samples, self.patches = (channels, samples, patches)
        seed_head = EEGPTTrialHead(patches, bottleneck=context_width, dropout=context_dropout)
        self.context_projection = seed_head.patch_projection
        self.context_dropout = seed_head.dropout
        self.response = None
        raw_dim = 0
        if use_response:
            self.response = ConsensusResponseEEG(channels, samples, filters=filters, bins=bins, dropout=response_dropout, retain_electrodes=True, clean_consensus=True, trial_channel_normalize=True, spatial_mixing=spatial_mixing)
            raw_dim = self.response.encoder.feature_dim
            del self.response.classifier
        self.classifier = nn.Linear(raw_dim + patches * context_width, 7)
        with torch.no_grad():
            self.classifier.weight.zero_()
            self.classifier.weight[:, raw_dim:].copy_(seed_head.classifier.weight)
            self.classifier.bias.copy_(seed_head.classifier.bias)
        self.raw_feature_dim = raw_dim

    def raw_features(self, raw):
        if raw.ndim != 3 or raw.shape[1:] != (self.channels, self.samples):
            raise ValueError('raw EEG shape differs')
        return None if self.response is None else self.response.encode(raw, clean=True)

    def combine(self, raw_features, context, *, stochastic):
        if context.ndim != 4 or context.shape[1:] != (self.patches, 4, 512):
            raise ValueError('context features must be from the SAME original trial')
        c = context.flatten(2)
        if stochastic:
            c = self.context_dropout(c)
        c = self.context_projection(c).flatten(1)
        if self.response is None:
            return c
        r = raw_features
        if stochastic:
            r = self.response.encoder.dropout(r)
        return torch.cat((r, c), 1)

    def forward(self, inputs):
        raw, context = inputs
        z = self.combine(self.raw_features(raw), context, stochastic=self.training)
        return self.classifier(z)
