"""EEGPT encoder transfer for independent short EEG trials.

Architectural reference: Wang et al., EEGPT, NeurIPS 2024 (CCF-A).
https://proceedings.neurips.cc/paper_files/paper/2024/hash/4540d267eeec4e5dbd9dae9448f0b739-Abstract-Conference.html
Original implementation (Apache-2.0): https://github.com/BINE022/EEGPT
Weight-compatible implementation/reference (BSD-3-Clause):
https://github.com/braindecode/braindecode/blob/master/braindecode/models/eegpt.py

This small dependency-free implementation keeps the released encoder's tensor
names and operations. It does NOT include pretrained reconstruction/prediction
heads or unrelated downstream channel projection/classification weights.
Each input is exactly one trial; patches and summary tokens never mix trials.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


WEIGHT_SHA256 = 'fb34c20609983324679b9534f9d17a2289a232d0276dd2b8b7d5b088f876f621'
WEIGHT_REPOSITORY = 'braindecode/eegpt-pretrained'
WEIGHT_REVISION = 'e41cb3ae2ce4fd9eb736862292c91f8128d15618'
CHANNELS = (
    'FP1','FPZ','FP2','AF7','AF3','AF4','AF8',
    'F7','F5','F3','F1','FZ','F2','F4','F6','F8',
    'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8',
    'T7','C5','C3','C1','CZ','C2','C4','C6','T8',
    'TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8',
    'P7','P5','P3','P1','PZ','P2','P4','P6','P8',
    'PO7','PO5','PO3','POZ','PO4','PO6','PO8','O1','OZ','O2')
# The original mixed pretraining setup used these 58 channels. Restricting the
# frozen transfer to their intersection avoids using untrained position rows.
PRETRAIN_CHANNELS = tuple(c for c in CHANNELS if c not in {'AF7','AF8','PO5','PO6'})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_safe_tensor_subset(path: Path, prefix: str, expected_sha256: str | None = None) -> dict[str, Tensor]:
    """Read audited F32/F16/F64/I64/I32 safetensors only; never unpickle.

    The released encoder has F32 weights. Metadata, shapes, byte bounds and
    exact tensor lengths are checked before copying each tensor to torch.
    """
    path = Path(path)
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError('pretrained checkpoint SHA256 mismatch')
    dtypes = {'F32': np.dtype('<f4'), 'F16': np.dtype('<f2'), 'F64': np.dtype('<f8'),
              'I64': np.dtype('<i8'), 'I32': np.dtype('<i4')}
    result = {}
    file_size = path.stat().st_size
    with path.open('rb') as stream:
        leading = stream.read(8)
        if len(leading) != 8:
            raise ValueError('truncated safetensors header')
        header_size = struct.unpack('<Q', leading)[0]
        if header_size < 2 or header_size > min(16*1024*1024, file_size-8):
            raise ValueError('invalid safetensors header length')
        header = json.loads(stream.read(header_size))
        start = 8 + header_size
        for key, spec in header.items():
            if key == '__metadata__' or not key.startswith(prefix):
                continue
            if spec['dtype'] not in dtypes:
                raise ValueError(f'unsupported safe tensor dtype: {spec["dtype"]}')
            dtype = dtypes[spec['dtype']]
            shape = spec['shape']
            if not all(isinstance(v, int) and v >= 0 for v in shape):
                raise ValueError('invalid tensor shape')
            begin, end = spec['data_offsets']
            if not (0 <= begin <= end <= file_size-start):
                raise ValueError('tensor offsets outside file')
            if end-begin != math.prod(shape) * dtype.itemsize:
                raise ValueError('tensor byte length mismatch')
            stream.seek(start+begin)
            data = stream.read(end-begin)
            if len(data) != end-begin:
                raise ValueError('truncated tensor payload')
            array = np.frombuffer(data, dtype=dtype).reshape(shape).copy()
            result[key[len(prefix):]] = torch.from_numpy(array)
    if not result:
        raise ValueError(f'no tensors with prefix {prefix!r}')
    return result


class _MLP(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4*dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4*dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3*dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        self.attention_dropout = dropout

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b,n,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        y = F.scaled_dot_product_attention(q,k,v,
            dropout_p=self.attention_dropout if self.training else 0.)
        return self.proj_drop(self.proj(y.transpose(1,2).reshape(b,n,d)))


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _Attention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _MLP(dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class _PatchEmbed(nn.Module):
    def __init__(self, dim: int, patch_size: int, stride: int):
        super().__init__()
        self.proj = nn.Conv2d(1, dim, (1,patch_size), stride=(1,stride))

    def forward(self, x):
        return self.proj(x[:,None]).transpose(1,3)


class EEGPTEncoder(nn.Module):
    """Return [trial, temporal patch, 4 learned summary tokens, 512].

    The encoder mixes electrodes within each patch. It does not mix different
    trials and does not itself attend across temporal patches; downstream
    readout must keep/learn the within-trial temporal organization.
    """
    def __init__(self, channel_names: Sequence[str], *, patch_size=64,
                 patch_stride=32, embed_dim=512, embed_num=4, depth=8,
                 num_heads=8, dropout=0.):
        super().__init__()
        names = tuple(str(c).upper() for c in channel_names)
        if not names or len(set(names)) != len(names) or any(c not in CHANNELS for c in names):
            raise ValueError('EEGPT requires known unique channel names')
        if min(patch_size,patch_stride,embed_dim,embed_num,depth,num_heads) < 1 or embed_dim % num_heads:
            raise ValueError('invalid EEGPT dimensions')
        self.channel_names = names
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.embed_dim = embed_dim
        self.embed_num = embed_num
        self.patch_embed = _PatchEmbed(embed_dim,patch_size,patch_stride)
        self.chan_embed = nn.Embedding(len(CHANNELS),embed_dim)
        self.summary_token = nn.Parameter(torch.zeros(1,embed_num,embed_dim))
        self.blocks = nn.ModuleList([_Block(embed_dim,num_heads,dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim,eps=1e-6)
        self.register_buffer('channel_ids',torch.tensor([CHANNELS.index(c) for c in names]),persistent=False)
        self.pretrained = False

    def load_pretrained(self, path: Path):
        state = load_safe_tensor_subset(path,'target_encoder.',WEIGHT_SHA256)
        self.load_state_dict(state,strict=True)
        self.pretrained = True
        return {'tensor_count':len(state),'sha256':WEIGHT_SHA256,
                'repository':WEIGHT_REPOSITORY,'revision':WEIGHT_REVISION}

    def forward_prefix(self,x:Tensor,stop_block:int=6)->Tensor:
        """Unpooled per-channel/summary tokens after a frozen block prefix."""
        if not 0<=stop_block<=len(self.blocks):raise ValueError('invalid prefix block')
        if x.ndim != 3 or x.shape[1] != len(self.channel_names) or x.shape[-1] < self.patch_size:
            raise ValueError('EEGPT input must be [B,known_channels,T>=patch_size]')
        z = self.patch_embed(x)
        b, n, c, d = z.shape
        z = z + self.chan_embed(self.channel_ids)[None,None]
        z = z.reshape(b*n,c,d)
        summary = self.summary_token.expand(b*n,-1,-1)
        z = torch.cat((z,summary),dim=1)
        for block in self.blocks[:stop_block]:
            z = block(z)
        return z.reshape(b,n,c+self.embed_num,d)

    def forward_from_prefix(self,prefix:Tensor,start_block:int=6)->Tensor:
        if not 0<=start_block<=len(self.blocks):raise ValueError('invalid suffix block')
        if prefix.ndim!=4 or prefix.shape[2:]!=(len(self.channel_names)+self.embed_num,self.embed_dim):
            raise ValueError('invalid unpooled EEGPT prefix shape')
        b,n,c,d=prefix.shape
        z=prefix.reshape(b*n,c,d)
        for block in self.blocks[start_block:]:z=block(z)
        return self.norm(z[:,-self.embed_num:]).reshape(b,n,self.embed_num,d)

    def forward(self,x:Tensor)->Tensor:
        return self.forward_from_prefix(self.forward_prefix(x,0),0)


class EEGPTTrialHead(nn.Module):
    """One supervised prediction per trial; shared patch readout then ordered FC."""
    def __init__(self, patches: int, *, bottleneck=32, dropout=.3, classes=7):
        super().__init__()
        self.patches = patches
        self.dropout = nn.Dropout(dropout)
        self.patch_projection = nn.Linear(4*512,bottleneck)
        self.classifier = nn.Linear(patches*bottleneck,classes)

    def forward(self, features):
        if features.ndim != 4 or features.shape[1:] != (self.patches,4,512):
            raise ValueError('invalid EEGPT trial feature shape')
        z = self.patch_projection(self.dropout(features.flatten(2)))
        return self.classifier(z.flatten(1))


def rdrop_trial_loss(logits_a: Tensor, logits_b: Tensor, labels: Tensor,
                     alpha: float = 1.0) -> tuple[Tensor, dict[str, Tensor]]:
    """R-Drop (NeurIPS 2021), two dropout passes of the SAME original trials.

    Bidirectional KL is summed over classes and averaged over trials. Both
    paths receive gradients. It never combines different trials or labels.
    https://proceedings.neurips.cc/paper_files/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html
    """
    if logits_a.ndim!=2 or logits_a.shape!=logits_b.shape or len(labels)!=len(logits_a):
        raise ValueError('paired trial logits/labels must match')
    if not math.isfinite(alpha) or alpha<0:raise ValueError('alpha must be finite nonnegative')
    ce=.5*(F.cross_entropy(logits_a,labels)+F.cross_entropy(logits_b,labels))
    log_a=F.log_softmax(logits_a,dim=-1);log_b=F.log_softmax(logits_b,dim=-1)
    kl=.5*(F.kl_div(log_a,log_b.exp(),reduction='batchmean')+
            F.kl_div(log_b,log_a.exp(),reduction='batchmean'))
    return ce+alpha*kl,{'ce':ce,'symmetric_kl':kl}
