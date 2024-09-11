import os
import json
import torch
import torch.nn.functional as F
import torch.nn as nn
from open_clip.model import _build_vision_tower

class CLIPConvNextModel(nn.Module):
    def __init__(self, pretrained_model_path, *args, **kwargs):
        super().__init__(*args, **kwargs)

        config_path = os.path.join(pretrained_model_path, 'open_clip_config.json')
        config = self.load_config(config_path)
        self.visual = _build_vision_tower(
            embed_dim=config['embed_dim'], 
            vision_cfg=config['vision_cfg'], 
            quick_gelu=False
        )
        model_path = os.path.join(pretrained_model_path, 'open_clip_pytorch_model.bin')
        self.load_checkpoint(model_path)

        self.freeze()
        self.eval()

    @classmethod
    def from_pretrained(cls, pretrained_model_path):
        return cls(pretrained_model_path)

    def load_config(self, config_json):
        with open(config_json) as f:
            config = json.load(f)
        return config['model_cfg']

    def load_checkpoint(self, model_path):
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict, strict=False)

    def freeze(self):
        for param in self.visual.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        results = dict()
        x = self.visual.trunk.stem(x)
        results['stem'] = x

        hidden_states = []
        for i in range(4):
            x = self.visual.trunk.stages[i](x)
            hidden_states.append(x)
        results['hidden_states'] = hidden_states
        
        x = self.visual.trunk.norm_pre(x)
        results['hidden_states_normed'] = x
        return results

    @property
    def dtype(self):
        param = next(self.parameters())
        return param.dtype

    @property
    def device(self):
        param = next(self.parameters())
        return param.device