# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import random
from torch import nn
from torch.nn import MSELoss
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from .configuration_sync_tuner import SyncTunerConfig

def cosine_loss(rec, target):
    target = target / target.norm(dim=-1, keepdim=True)
    rec = rec / rec.norm(dim=-1, keepdim=True)
    rec_loss = (1 - (target * rec).sum(-1)).mean()
    return rec_loss

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=256):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class SyncTunerAttentionBlcok(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(SyncTunerAttentionBlcok, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        # Self-attention with residual connection and layer norm
        residual = x
        x = self.layernorm1(x)
        attn_output, _ = self.multihead_attn(x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        x = residual + self.dropout(attn_output)
        
        # Feed-forward network with residual connection and layer norm
        residual = x
        x = self.layernorm2(x)
        ffn_output = self.ffn(x)
        x = residual + self.dropout(ffn_output)
        return x

class SyncTunerModel(PreTrainedModel):
    _auto_class = 'AutoModel'
    config_class = SyncTunerConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True

    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """

    def __init__(self, config: SyncTunerConfig):
        super().__init__(config)
        self.gradient_checkpointing = False
        self.config = config

        num_queries = config.num_queries
        d_input = config.d_input
        d_model = config.d_model
        n_heads = config.num_heads
        dropout = config.dropout
        d_ffn = config.d_ffn
        d_output = config.output_dim

        # models
        self.in_proj = nn.Linear(d_input, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=0, max_len=num_queries)
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            self.layers.append(
                SyncTunerAttentionBlcok(
                    d_model,
                    n_heads,
                    d_ffn,
                    dropout
                )
            )
        self.out_proj = nn.Linear(d_model, d_output)
        self.criterion = MSELoss(reduction='none')

        # # self.image_pool = redis.StrictRedis(host='localhost', port=6379, db=0)
        # if image_pool is not None:
        #     with jsonlines.open(image_pool, 'r') as f:
        #         self.image_pool_idx2path = [data for data in f]
        #         self.image_pool_path2idx = {path: index for index, path in \
        #             enumerate(self.image_pool_idx2path)}
        # else:
        #     self.image_pool_idx2path = None
        #     self.image_pool_path2idx = None
        
        # self.register_buffer(
        #     "image_pool_used_idx", 
        #     torch.zeros(len(self.image_pool_idx2path)).bool(), 
        #     persistent=False
        # )

    def enable_input_require_grads(self):

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        self.out_proj.register_forward_hook(make_inputs_require_grad)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, SyncTunerConfig):
            module.gradient_checkpointing = value
    
    def upsample(self, x):
        # upsample with factor '2'
        b, l, c = x.shape
        grid_size = int(math.sqrt(l))
        target_size = grid_size * 2
        x = x.view(b, grid_size, grid_size, c)
        x = x.permute(0, 3, 1, 2) # b, c, h, w
        x = F.interpolate(
            x, 
            size=(target_size, target_size),
            mode="bicubic",
            align_corners=False
        )
        x = x.permute(0, 2, 3, 1).view(b, -1, c) # b, l, c
        return x
    
    def get_loss(self, logits, image, image_path):
        rec_flags = []
        # for path in image_path:
            # if path == '':
            #     rec_flag = False # fake image
            # # elif self.image_pool.sismember("processed_images", path):
            # elif self.image_pool_used_idx[self.image_pool_path2idx[path]]:
            #     rec_flag = False
            # else:
            #     rec_flag = np.random.uniform(0, 1) < self.config.ratio
            #     self.image_pool_used_idx[self.image_pool_path2idx[path]] = rec_flag
            #     # if rec_flag:
            #     #     # self.image_pool.sadd("processed_images", path)
        
        for idx in range(logits.shape[0]):
            if image_path[idx] == '':
                rec_flag = False
            else:
                rec_flag = np.random.uniform(0, 1) < self.config.ratio
            rec_flags.append(rec_flag)
        if not any(rec_flags):
            idx = random.randint(0, len(rec_flags) - 1)
            rec_flags[idx] = True

        b, c, h, w = image.shape
        mask = torch.Tensor(rec_flags).expand(c, h, w, b).permute(3, 0, 1, 2).bool() # b, c, h, w
        
        # transform pred to target shape
        b, l, c = logits.shape
        grid_size = int(math.sqrt(l))
        if logits.shape[1] != image.shape[-1] * image.shape[-2]:
            logits = logits.view(b, grid_size, grid_size, c).permute(0, 3, 1, 2) # b, c, h, w
            logits = F.interpolate(
                logits,
                size=image.shape[2:],
                mode="bicubic",
                align_corners=False
            )

        # ignore loss when 'rec_flag = False'
        logits = logits[mask]
        target = image[mask].to(logits.dtype)

        # # ignore padding value
        # ignore_mask = target == 1
        # pred = pred[ignore_mask]
        # target = target[ignore_mask]

        loss_rec = self.criterion(logits, target)
        return loss_rec.mean()

    def forward(self, x, metas, mode='loss'):
        # Add position embedding
        x = self.in_proj(x)
        hidden_states = [x]

        x = self.positional_encoding(x)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x)
            else:
                x = layer(x)
            hidden_states.append(x)
            x = self.upsample(x)
        
        # output projection
        last_hidden_state = self.out_proj(x)
        hidden_states.append(last_hidden_state)

        loss = None
        if mode == 'loss':
            loss = self.get_loss(
                last_hidden_state,
                image = metas['ori_image'],
                image_path = metas['image_path'],
            )
        
        return dict(
            loss = loss,
            hidden_states = hidden_states,
            last_hidden_state = last_hidden_state
        )