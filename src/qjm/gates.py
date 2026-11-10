from __future__ import annotations
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _complex_dtype(real: Tensor) -> torch.dtype:
    return torch.complex128 if real.dtype == torch.float64 else torch.complex64


def _rx(theta: Tensor) -> Tensor:
    """Batched RX matrices; theta may have any leading dimensions."""
    dtype = _complex_dtype(theta)
    c = torch.cos(theta / 2).to(dtype)
    s = (-1j * torch.sin(theta / 2)).to(dtype)
    out = torch.zeros(theta.shape + (2, 2), dtype=dtype, device=theta.device)
    out[..., 0, 0] = c
    out[..., 1, 1] = c
    out[..., 0, 1] = s
    out[..., 1, 0] = s
    return out


def _ry(theta: Tensor) -> Tensor:
    dtype = _complex_dtype(theta)
    c = torch.cos(theta / 2).to(dtype)
    s = torch.sin(theta / 2).to(dtype)
    out = torch.zeros(theta.shape + (2, 2), dtype=dtype, device=theta.device)
    out[..., 0, 0] = c
    out[..., 1, 1] = c
    out[..., 0, 1] = -s
    out[..., 1, 0] = s
    return out


def _rz(theta: Tensor) -> Tensor:
    dtype = _complex_dtype(theta)
    out = torch.zeros(theta.shape + (2, 2), dtype=dtype, device=theta.device)
    out[..., 0, 0] = torch.exp((-0.5j * theta).to(dtype))
    out[..., 1, 1] = torch.exp((0.5j * theta).to(dtype))
    return out
