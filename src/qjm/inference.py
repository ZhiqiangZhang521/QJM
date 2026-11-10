"""Load tensor-only fold parameters. No private data paths are needed."""
from __future__ import annotations
from contextlib import nullcontext
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .eeg import ResponseContextEEG
from .eog import RawEOGEmotionNet
from .eegpt import EEGPTEncoder, PRETRAIN_CHANNELS
from .fusion import FixedMapper, CompleteHead


class FrozenFold(nn.Module):
    def __init__(self, folder, device='cpu'):
        super().__init__()
        folder=Path(folder)
        self.config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
        c=self.config
        self.eeg=ResponseContextEEG(**c['eeg_encoder'])
        self.eog=RawEOGEmotionNet(**c['eog_encoder'])
        self.mapper=FixedMapper(c['eeg_channels_count'],c['context_patches'])
        self.head=CompleteHead()
        for name,model in [('eeg_encoder',self.eeg),('eog_encoder',self.eog),('mapper',self.mapper),('qjm_head',self.head)]:
            model.load_state_dict(torch.load(folder/(name+'.pt'),map_location='cpu',weights_only=True),strict=True)
        normalizers=torch.load(folder/'normalizers.pt',map_location='cpu',weights_only=True)
        for group,state in normalizers.items():
            for name,value in state.items():
                self.register_buffer(group+'__'+name,value)
        self.to(device).eval().requires_grad_(False)

    def _norm(self, x, group, clip=None):
        mean=getattr(self,group+'__mean');scale=getattr(self,group+'__scale')
        x=(x-mean)/scale
        return x if clip is None else x.clamp(-clip,clip)

    @torch.inference_mode()
    def from_features(self,eeg,tokens,global_eog,return_aux=False):
        eeg=self._norm(eeg,'eeg')
        tokens=self._norm(tokens,'tokens',8.)
        global_eog=self._norm(global_eog,'global_eog')
        angles=self.mapper(eeg,tokens,global_eog)
        return self.head(angles,return_aux=return_aux)

    @torch.inference_mode()
    def forward(self,eeg,eog,context,return_aux=False):
        """Input: preprocessed 125Hz EEG/EOG and matching EEGPT context grid.

        SEED-VII originally used FP16 encoder inference and FP16 context storage.
        Supply matching precision for exact numerical reproduction; CPU uses FP32.
        """
        if tuple(eeg.shape[1:]) != tuple(self.config['eeg_input_shape']) or tuple(eog.shape[1:]) != (6,self.config['samples']):
            raise ValueError('Input signal shape differs from checkpoint')
        eog=self._norm(eog,'raw_eog',float(self.raw_eog__clip))
        if self.config['dataset']=='SEED-VII':
            eog=torch.nan_to_num(eog,nan=0.,posinf=0.,neginf=0.)
            context=context.half().float()
        use_amp=self.config['encoder_precision']=='cuda_fp16' and eeg.device.type=='cuda'
        with torch.autocast('cuda',dtype=torch.float16) if use_amp else nullcontext():
            z_eeg=self.eeg.combine(self.eeg.raw_features(eeg),context,stochastic=False)
            tokens=self.eog.encode_tokens(eog)
            z_eog=self.eog.head(tokens.transpose(1,2).contiguous().flatten(1))
        return self.from_features(z_eeg.float(),tokens.float(),z_eog.float(),return_aux=return_aux)

    @torch.inference_mode()
    def extract_context(self,eeg,pretrained_path):
        """Extract EEGPT input context from preprocessed 125Hz EEG (microvolts).

        This is not acquisition-file cleaning, event slicing or eye-signal extraction.
        """
        from scipy.signal import resample_poly
        names=self.config['eeg_channels']
        selected=[i for i,c in enumerate(names) if c.upper() in PRETRAIN_CHANNELS]
        model=EEGPTEncoder([names[i] for i in selected]).to(eeg.device).eval().requires_grad_(False)
        model.load_pretrained(Path(pretrained_path))
        raw=eeg.detach().cpu().numpy().astype(np.float32,copy=True)
        raw-=raw.mean(axis=1,keepdims=True)
        raw=resample_poly(raw[:,selected],256,125,axis=-1).astype(np.float32)*.001
        out=[]
        for start in range(0,len(raw),8):
            n=min(8,len(raw)-start)
            x=torch.zeros((8,len(selected),raw.shape[-1]),device=eeg.device)
            x[:n]=torch.from_numpy(raw[start:start+n]).to(eeg.device)
            with torch.autocast('cuda',dtype=torch.float16) if eeg.device.type=='cuda' else nullcontext():
                z=model(x)[:n].float()
            out.append(z.half().float() if self.config['dataset']=='SEED-VII' else z)
        return torch.cat(out)


def load_fold(folder,device='cpu'):
    return FrozenFold(folder,device)
