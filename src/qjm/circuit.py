from __future__ import annotations
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .gates import _rx, _ry, _rz


def apply_gate(state: torch.Tensor, gate: torch.Tensor, wires: tuple[int, ...], n_qubits: int) -> torch.Tensor:
    """Apply a batched local gate to a [batch, 2**n_qubits] state vector."""
    if len(wires) != len(set(wires)) or any((w < 0 or w >= n_qubits for w in wires)):
        raise ValueError('Invalid qubit wires')
    batch = state.shape[0]
    others = tuple((w for w in range(n_qubits) if w not in wires))
    order = (0,) + tuple((w + 1 for w in wires + others))
    inverse = tuple((order.index(i) for i in range(n_qubits + 1)))
    block = state.reshape(batch, *(2,) * n_qubits).permute(order)
    block = block.reshape(batch, 2 ** len(wires), -1)
    if gate.ndim == 2:
        block = torch.einsum('ij,bjk->bik', gate, block)
    elif gate.ndim == 3:
        block = torch.einsum('bij,bjk->bik', gate, block)
    else:
        raise ValueError('Gate must be shared or batch-specific')
    return block.reshape(batch, *(2,) * n_qubits).permute(inverse).reshape(batch, -1)


class EOGAxisEncoder(nn.Module):
    """Learn an axis-specific representation from two leads and their difference."""

    def __init__(self, width: int=24, output_dim: int=64):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(3, width, 7, padding=3, bias=False), nn.GroupNorm(4, width), nn.GELU())
        self.scales = nn.ModuleList((nn.Conv1d(width, width, k, padding=k // 2, groups=width, bias=False) for k in (5, 11, 21)))
        self.mix = nn.Sequential(nn.Conv1d(3 * width, width, 1, bias=False), nn.GroupNorm(4, width), nn.GELU())
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.head = nn.Sequential(nn.LayerNorm(width * 8), nn.Linear(width * 8, output_dim), nn.GELU())

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        h = self.stem(signal)
        h = self.mix(torch.cat([layer(h) for layer in self.scales], dim=1))
        return self.head(self.pool(h).flatten(1))


class ThreeBranchQJM(nn.Module):
    EEG_RESPONSE_DIM = 11328
    EEG_CONTEXT_DIM = 1984
    EEG_SCALE_DIM = 3776
    EEG_EDGES = ((0, 1), (1, 2), (1, 3))
    EOG_EDGES = ((4, 5),)
    CROSS_EDGES = tuple(((e, o) for e in range(4) for o in (4, 5)))

    def __init__(self, hidden: int=64, head_dim: int=48, eog_width: int=24, cross_floor: float=0.6, dropout: float=0.1, intra_entangler_init: float=0.0, cross_entangler_init: float=0.0, eog_feature_dim: int | None=None):
        super().__init__()
        if not 0 < cross_floor < 1:
            raise ValueError('Cross-branch floor must lie in (0,1)')
        self.cross_floor = float(cross_floor)
        self.eog_feature_dim = eog_feature_dim
        self.eog_horizontal = EOGAxisEncoder(eog_width, hidden) if eog_feature_dim is None else nn.Identity()
        self.eog_vertical = EOGAxisEncoder(eog_width, hidden) if eog_feature_dim is None else nn.Identity()
        self.angle_encoders = nn.ModuleList((nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 3)) for dim in (self.EEG_SCALE_DIM,) * 3 + (self.EEG_CONTEXT_DIM,) + ((hidden,) * 2 if eog_feature_dim is None else (eog_feature_dim,) * 2)))
        self.eeg_entanglers = nn.Parameter(torch.full((2, len(self.EEG_EDGES)), intra_entangler_init))
        self.eog_entanglers = nn.Parameter(torch.full((2, len(self.EOG_EDGES)), intra_entangler_init))
        self.cross_entanglers = nn.Parameter(torch.full((2, len(self.CROSS_EDGES)), cross_entangler_init))
        self.cross_between = nn.Parameter(torch.zeros(6))
        self.readout_angles = nn.Parameter(torch.zeros(6))
        self.cross_logit = nn.Parameter(torch.tensor(0.0))
        self.eeg_eog_split_logit = nn.Parameter(torch.tensor(0.0))
        self.eeg_readout = nn.Sequential(nn.Linear(16, head_dim), nn.GELU())
        self.eog_readout = nn.Sequential(nn.Linear(4, head_dim), nn.GELU())
        self.cross_readout = nn.Sequential(nn.Linear(64, head_dim), nn.GELU())
        self.classifier = nn.Sequential(nn.Linear(head_dim, head_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(head_dim, 7))
        self.cross_classifier = nn.Linear(head_dim, 7)
        eye = torch.eye(4, dtype=torch.complex64)
        x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
        self.register_buffer('eye4', eye, persistent=False)
        self.register_buffer('xx', torch.kron(x, x), persistent=False)

    def _eeg_groups(self, eeg: torch.Tensor) -> list[torch.Tensor]:
        if eeg.ndim != 2 or eeg.shape[1] != self.EEG_RESPONSE_DIM + self.EEG_CONTEXT_DIM:
            raise ValueError('Expected frozen EEG feature vector of length 13312')
        response = eeg[:, :self.EEG_RESPONSE_DIM].reshape(len(eeg), 2, 12, 59, 8)
        groups = [response[:, :, 4 * scale:4 * (scale + 1)].reshape(len(eeg), -1) for scale in range(3)]
        return groups + [eeg[:, self.EEG_RESPONSE_DIM:]]

    def _angles(self, eeg: torch.Tensor, eog: torch.Tensor) -> torch.Tensor:
        groups = self._eeg_groups(eeg)
        if self.eog_feature_dim is None:
            if eog.ndim != 3 or eog.shape[1] != 6:
                raise ValueError('Expected normalized six-channel EOG waveform')
            groups += [self.eog_horizontal(eog[:, (0, 1, 4)]), self.eog_vertical(eog[:, (2, 3, 5)])]
        else:
            if eog.ndim != 2 or eog.shape[1] != self.eog_feature_dim:
                raise ValueError('Expected frozen EOG feature vector')
            groups += [eog, eog]
        return math.pi * torch.tanh(torch.stack([encoder(group) for encoder, group in zip(self.angle_encoders, groups)], dim=1))

    def _rxx(self, raw_angle: torch.Tensor) -> torch.Tensor:
        angle = 0.5 * math.pi * torch.tanh(raw_angle)
        return torch.cos(angle / 2).to(torch.complex64) * self.eye4 - 1j * torch.sin(angle / 2).to(torch.complex64) * self.xx

    def _entangle(self, state: torch.Tensor, edges: tuple[tuple[int, int], ...], parameters: torch.Tensor, layer: int) -> torch.Tensor:
        for edge, raw in zip(edges, parameters[layer]):
            state = apply_gate(state, self._rxx(raw), edge, 6)
        return state

    def _terminal(self, state: torch.Tensor) -> torch.Tensor:
        for wire in range(6):
            angle = 0.5 * math.pi * torch.tanh(self.readout_angles[wire])
            state = apply_gate(state, _ry(angle), (wire,), 6)
        return state

    def forward(self, eeg: torch.Tensor, eog: torch.Tensor, *, return_aux: bool=False, disable_cross_gates: bool=False):
        angles = self._angles(eeg, eog)
        return self.forward_angles(angles, return_aux=return_aux, disable_cross_gates=disable_cross_gates)

    def forward_angles(self, angles: torch.Tensor, *, return_aux: bool=False, disable_cross_gates: bool=False):
        """Apply the unchanged circuit/readout to already prepared angles."""
        if angles.ndim != 3 or angles.shape[1:] != (6, 3):
            raise ValueError('Expected local rotation angles [batch, 6, 3]')
        batch = len(angles)
        state = torch.zeros((batch, 64), device=angles.device, dtype=torch.complex64)
        state[:, 0] = 1
        for wire in range(6):
            state = apply_gate(state, _ry(angles[:, wire, 0]), (wire,), 6)
            state = apply_gate(state, _rz(angles[:, wire, 1]), (wire,), 6)
        for layer in range(2):
            state = self._entangle(state, self.EEG_EDGES, self.eeg_entanglers, layer)
            state = self._entangle(state, self.EOG_EDGES, self.eog_entanglers, layer)
            if layer == 0:
                for wire in range(6):
                    state = apply_gate(state, _rx(angles[:, wire, 2]), (wire,), 6)
        local_joint = self._terminal(state).abs().square().reshape(batch, 16, 4)
        p_eeg, p_eog = (local_joint.sum(-1), local_joint.sum(1))
        cross_state = state
        if not disable_cross_gates:
            for layer in range(2):
                cross_state = self._entangle(cross_state, self.CROSS_EDGES, self.cross_entanglers, layer)
                if layer == 0:
                    for wire in range(6):
                        angle = 0.5 * math.pi * torch.tanh(self.cross_between[wire])
                        cross_state = apply_gate(cross_state, _ry(angle), (wire,), 6)
        p_cross = self._terminal(cross_state).abs().square()
        u_eeg = F.normalize(self.eeg_readout(p_eeg), dim=-1)
        u_eog = F.normalize(self.eog_readout(p_eog), dim=-1)
        u_cross = F.normalize(self.cross_readout(p_cross), dim=-1)
        cross_weight = self.cross_floor + (1 - self.cross_floor) * torch.sigmoid(self.cross_logit)
        eeg_fraction = torch.sigmoid(self.eeg_eog_split_logit)
        eeg_weight = (1 - cross_weight) * eeg_fraction
        eog_weight = (1 - cross_weight) * (1 - eeg_fraction)
        fused = cross_weight * u_cross + eeg_weight * u_eeg + eog_weight * u_eog
        logits = self.classifier(fused)
        cross_logits = self.cross_classifier(u_cross)
        if return_aux:
            return (logits, dict(cross_logits=cross_logits, p_eeg=p_eeg, p_eog=p_eog, p_cross=p_cross, cross_weight=cross_weight, eeg_weight=eeg_weight, eog_weight=eog_weight))
        return logits

    def counts(self):
        return dict(parameters=sum((p.numel() for p in self.parameters())), trainable=sum((p.numel() for p in self.parameters() if p.requires_grad)), qubits=6, eeg_intra_edges=len(self.EEG_EDGES), eog_intra_edges=len(self.EOG_EDGES), cross_edges=len(self.CROSS_EDGES))
